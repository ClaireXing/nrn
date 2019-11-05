FILE_PATH = './../../0.01/'
data_file = './data/shuffle_ppl.txt'

LEARNING_RATE = 1e-4
EPISODE = 1000

total = 66581
TOTAL_TRAIN_SIZE = 38400
TOTAL_VALID_SIZE = int(total*0.2)
TOTAL_TEST_SIZE = int(total*0.2)
BATCH_SIZE = 32
EACH_SIZE = 128
IMG_SIZE = EACH_SIZE*3

train_ratio = 0.6
valid_ratio = 0.2
test_ratio = 0.2
