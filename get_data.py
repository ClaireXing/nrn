from itertools import islice
import cv2
import random
import os
import tensorflow as tf
import numpy as np
import math
import config as cfg

# np.random.seed(1)
# tf.set_random_seed(1)
# random.seed(1)

class Data:
    def __init__(self, **kwargs):
        """
        file_name
        """
        self.__dict__.update(kwargs)
        self.train_data = []
        self.valid_data = []
        self.test_data = []
        
    def get_data(self, test=False):
        line_num = 0
        with open(self.file_name, 'r') as f:
            for line in f.readlines():
                line_num += 1
                image_path, lon, lat, ppl, dense_ppl = line.split(' ')
                lon = float(lon)
                lat = float(lat)
                dense_ppl = float(dense_ppl)
                if dense_ppl > 0.00:
                    log_ppl = float(math.log(dense_ppl))
                    pic_lon = int(round(lon * 100))
                    pic_lat = int(round(lat * 100))
                    pic_name = image_path + '/' + str(pic_lon)+'_'+str(pic_lat)+'.jpg'
                    if len(self.train_data) < cfg.TOTAL_TRAIN_SIZE:  
                        self.train_data.append((pic_name, log_ppl))
                    elif len(self.valid_data) < cfg.TOTAL_VALID_SIZE:
                        self.valid_data.append((pic_name, log_ppl))
                    else:
                        self.test_data.append((pic_name, log_ppl))
        return self.train_data, self.valid_data, self.test_data
    
    def get_test_data(self, test=False):
        line_num = 0
        with open(self.file_name, 'r') as f:
            for line in f.readlines():
                line_num += 1
                image_path, lon, lat, ppl, dense_ppl = line.split(' ')
                lon = float(lon)
                lat = float(lat)
                dense_ppl = float(dense_ppl)
                if dense_ppl > 0.00:
                    log_ppl = float(math.log(dense_ppl))
                    pic_lon = int(round(lon * 100))
                    pic_lat = int(round(lat * 100))
                    pic_name = image_path + '/' + str(pic_lon)+'_'+str(pic_lat)+'.jpg'
                    self.test_data.append((pic_name, log_ppl))
        return self.test_data
    
if __name__ == "__main__":
    data = Data(file_name='./super_ppl.txt')
    super_data = data.get_data()
    print(super_data)
    print(len(super_data))
