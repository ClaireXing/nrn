import tensorflow as tf
import numpy as np
import random
import cv2
import os
import matplotlib.pyplot as plt
import pickle
from scipy import misc
import math

import config as cfg
from resnet import *
from input_fn import Data
#np.random.seed(1)
#tf.set_random_seed(1)
#random.seed(1)

batch_size = cfg.BATCH_SIZE
keep_prob = 0.8
regular_weight = 0.001

def get_sess():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    return tf.Session(config=tf_config)

def get_img2(pic0):
    input_pics = []
    y_data = []
    y_rank = []
    for p in pic0:
        pic_name, ppl = p
        im = cv2.imread(os.path.join(cfg.FILE_PATH, pic_name))
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (cfg.IMG_SIZE, cfg.IMG_SIZE))
        input_pics.append(im)
        y_data.append(ppl)
        y_rank.append(math.floor(ppl+2.5))
        
    input_pics = np.reshape(input_pics, (-1, cfg.IMG_SIZE, cfg.IMG_SIZE, 3))
    y_data = np.reshape(y_data, (-1, 1))
    y_rank = np.reshape(y_rank, (-1))
    return input_pics, y_data, y_rank

data = Data()

# Preprocessing
input_shape = (batch_size, cfg.IMG_SIZE, cfg.IMG_SIZE, 3)
imgs0 = tf.placeholder(tf.uint8, shape=[batch_size, cfg.IMG_SIZE, cfg.IMG_SIZE, 3])
batch_imgs = []
for i in range(imgs0.shape[0]):
    image = imgs0[i]
    height = cfg.IMG_SIZE
    width = cfg.IMG_SIZE
    with tf.name_scope('eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        if height and width:
            image = tf.div(image, 255.0)
            image = tf.expand_dims(image,0)
            if i == 0:
                batch_imgs = image
            else:
                batch_imgs = tf.concat([batch_imgs, image], 0)
        images = batch_imgs

# Initialize
y_value = tf.placeholder(dtype=tf.float32, shape=[batch_size,1])
is_train = tf.placeholder(tf.bool, name="is_train")

arg_scope = resnet_arg_scope(weight_decay=regular_weight,
                     batch_norm_decay=0.9,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     activation_fn=tf.nn.relu,
                     use_batch_norm=True)
                     
# ResNet
with slim.arg_scope(arg_scope):
    net, end_points = resnet_v2_50(images, is_training=is_train)

# Final Conv
with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d]):
    with tf.variable_scope('Logits_out'):
        # (12 x 12) x 2048 
        net = slim.avg_pool2d(net, kernel_size=[4, 4], stride=4, padding='VALID', scope='AvgPool_7x7')
        # 3 x 3 x 256
        net = slim.conv2d(net, 1, [3, 3], activation_fn=None, padding='VALID', weights_regularizer=slim.l2_regularizer(regular_weight), scope='Last_conv')
        # 1 x 1 x 1
        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

# Variable To Train and Restore 
checkpoint_exclude_scopes = "Logits_out"
exclusions = []
if checkpoint_exclude_scopes:
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
print (exclusions)
variables_to_restore = []
variables_to_train = []
for var in slim.get_model_variables():
    excluded = False
    print(var.op.name)
    for exclusion in exclusions:
        if var.op.name.startswith(exclusion):
            excluded = True
            variables_to_train.append(var)
            print ("ok")
            print (var.op.name)
            break
    if not excluded:
        variables_to_restore.append(var)

# Loss
reg_loss = tf.reduce_mean(tf.abs(net-y_value), name='reg_loss')  # regression loss
tf.summary.scalar('loss', reg_loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
reg_total_loss = tf.losses.get_regularization_loss()   # l2_regularizer

total_loss = reg_loss + reg_total_loss

# Optimization
with tf.control_dependencies(update_ops):
    optimizer = tf.train.MomentumOptimizer(learning_rate=cfg.LEARNING_RATE, momentum=0.9, name='Momentum', use_nesterov=True).minimize(total_loss)

# Training
train_step = 0

test_file = './loss_128_ave1_pre_e4.txt'
merged = tf.summary.merge_all()
with get_sess() as sess:
    writer = tf.summary.FileWriter("loss_128_ave1_pre_e4/", sess.graph)
    tf.logging.info("Creating the model...")
    sess.run(tf.global_variables_initializer())
    saver2 = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    #saver2.restore(sess, './savenet/loss_128_ave1_pre_e4')
    model_path = './savenet/loss_128_ave1_pre_e4'
    
    # net_vars = variables_to_restore
    # saver_net = tf.train.Saver(net_vars)
    # checkpoint_path = './savenet/resnet_v2_50.ckpt'
    # saver_net.restore(sess, checkpoint_path)

    for i in range(cfg.EPISODE):
        for j in range(int(len(data.train_data)/batch_size)):
            input_pics, y_data, name_batch = data.obtain_batch_data(j)
            los, reg, _, pred = sess.run([total_loss, reg_loss, optimizer, net], feed_dict={imgs0:input_pics, y_value: y_data, is_train:True})
            print("TRAINER >> label: {} >> pred: {} >> loss: {}\n".format(y_data, pred, reg))
            if train_step % 100 == 0:
                saver2.save(sess, model_path, global_step=train_step, write_meta_graph=False)
                
            if train_step % 20 == 0:
                train_input_pics, train_y_data, train_name = data.obtain_batch_data(j, random1=True) # training
                valid_input_pics, valid_y_data, valid_name = data.obtain_batch_data(j, valid=True) # validation
                
                # train acc
                train_loss = sess.run(reg_loss, feed_dict={imgs0:train_input_pics, y_value:train_y_data, is_train:False})
                # valid acc
                valid_loss, summary = sess.run([reg_loss, merged], feed_dict={imgs0:valid_input_pics, y_value:valid_y_data, is_train:False})
                writer.add_summary(summary, train_step)
                
                print("###### TESTER>> step: %d >> train_set loss: %.4f >> valid_set loss: %.4f\n"%(train_step, train_loss, valid_loss))
                
                with open(test_file,'a+') as test_f:
                    test_f.write("TESTER>> step: %d >> train_set loss: %.4f >> valid_set loss: %.4f\n"%(train_step, train_loss, valid_loss))
            train_step += 1
        random.shuffle(data.train_data)
        