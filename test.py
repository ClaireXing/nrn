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

# file = './output/LOG_res_e4_pre.txt'
epoch = 1800
with get_sess() as sess:
    tf.logging.info("Creating the model...")
    sess.run(tf.global_variables_initializer())
    saver2 = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    saver2.restore(sess, './savenet/loss_128_ave1_pre_e4/loss_128_ave1_pre_e4-%s'%epoch)
    
    ############## test set in study cities #################
    city_names = ['Beijing', 'Guangzhou', 'Shanghai', 'Harbin', 'Kunming', 'Wuhan', 'Lasa', 'Lanzhou']
    for city_name in city_names:
        print(city_name)
        test_file = open('./output/loss_128_ave1_pre_e4/test_%s_%s.txt'%(epoch, city_name),'w')
        train_city_data = data.generate_train_city_data(city_name)
        l = len(train_city_data)
        for k in range(l):
            test_input_pics, test_y_data, test_name = data.obtain_train_city_data(k)
            # test acc
            pred = sess.run(net, feed_dict={imgs0:test_input_pics, y_value:test_y_data, is_train:False})

            test_file.write("####### step:%d >> pic_name: %s >> real_value: %s >> pred: %s\n"%(k,test_name[0], test_y_data[0][0], pred[0][0]))
            
    ############## additional test cities ##################
    # Shenyang, Jinan, Tianjin, Shijiazhuang, Hefei, Changsha, Nanchang, Shenzhen, Luoyang, Dalian
    city_names = [['Shenyang',12315,12364,4170,4194]] # [[city, min_lon, max_lon, min_lat, max_lat]]
    for city_name in city_names:
        print(city_name)
        test_file = open('./output/loss_128_ave1_pre_e4/test_%s_%s.txt'%(epoch, city_name[0]),'w')
        test_city_key = data.generate_test_city_data(file='./data/%s.txt'%city_name[0], lon1=city_name[1], lon2=city_name[2], lat1=city_name[3], lat2=city_name[4])
        l = len(test_city_key)
        for k in range(l):
            test_input_pics, test_y_data, test_name = data.obtain_test_city_data(k)
            
            # test acc
            pred = sess.run(net, feed_dict={imgs0:test_input_pics, y_value:test_y_data, is_train:False})
            test_file.write("####### step:%d >> pic_name: %s >> real_value: %s >> pred: %s\n"%(k,test_name[0], test_y_data[0][0], pred[0][0]))