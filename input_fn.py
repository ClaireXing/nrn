import tensorflow as tf
import math
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import config as cfg

class Data:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.file_name = cfg.data_file
        self.total_data, self.no_check_data = self._get_total_data()  # dict: {'lon_lat': city, ppl, image}
        self.train_data, self.valid_data, self.test_data = self._generate_dataset()  # keys list: ['lon_lat1', 'lon_lat2', ...]

    def _check_boundary(self, lon, lat):
        """
        Beijing:116.0500000,39.47000,117.050000,40.47000
        Guangzhou:112.420000,22.600000,113.420000,23.600000
        Harbin:126.2,45.14,127.2,46.14
        Kunming:102.22,24.62,103.22,25.62
        Lanzhou:103.28,35.75,104.28,36.75
        Lasa:90.81,29.19,91.81,30.19
        Shanghai:120.40000,30.790000,121.59000,31.35000
        Wuhan:113.750000,30.250000,114.750000,31.25000
        :return:
        """
        flag = 1
        lon_list = [11605, 11704, 11242, 11341, 12620, 12719, 10222, 10321, 10328, 10427, 9081, 9180, 12040, 12158,
                    11375, 11474]
        lat_list = [3947, 4046, 2260, 2359, 4514, 4613, 2462, 2561, 3575, 3674, 2919, 3018, 3079, 3134, 3025, 3124]
        for mg_lon in lon_list:
            if lon <= mg_lon + 1e-4 and lon >= mg_lon - 1e-4:
                flag = 0
        for mg_lat in lat_list:
            if lat <= mg_lat + 1e-4 and lat >= mg_lat - 1e-4:
                flag = 0
        return flag

    def _get_total_data(self):
        """
        Create dictionary for all data
        (except for boundary areas)
        :return: dict: {'lon_lat': city, ppl, image}
        """
        line_num = 0
        no_check_data = {}
        tdata = {}
        with open(self.file_name, 'r') as f:
            for line in f.readlines():
                line_num += 1
                image_path, lon, lat, ppl, dense_ppl = line.split(' ')
                lon = float(lon)
                lat = float(lat)
                dense_ppl = float(dense_ppl)
                pic_lon = int(round(lon * 100))
                pic_lat = int(round(lat * 100))
                pic_loc = str(pic_lon) + '_' + str(pic_lat)
                no_check_data[pic_loc] = [image_path, dense_ppl]
                if dense_ppl > 0.00 and self._check_boundary(pic_lon, pic_lat):  # check for boundary areas
                    ## ppl
                    log_ppl = float(math.log(dense_ppl))
                    #log_ppl = dense_ppl
                    tdata[pic_loc] = [image_path, log_ppl]

        self.total_data = tdata
        self.no_check_data = no_check_data
        return self.total_data, self.no_check_data

    def _generate_dataset(self):
        self.loc_list = list(self.total_data.keys())
        total_num = len(self.loc_list)
        train_index = int(total_num * cfg.train_ratio)
        valid_index = int(total_num * (cfg.train_ratio + cfg.valid_ratio))
        train_data = self.loc_list[:train_index]
        valid_data = self.loc_list[train_index:valid_index]
        test_data = self.loc_list[valid_index:]
        return train_data, valid_data, test_data

    def _neighbor_merging(self, patch):
        patch = {}  # patch(dict): {'-1,-1':image0, ...}
        # add neighbor
        lon = int(k.split('_')[0])
        lat = int(k.split('_')[1])
        ppl = self.total_data[k][1]
        for m in delta:
            for n in delta:
                delta_lon = lon + m
                delta_lat = lat + n
                str_loc = str(delta_lon) + '_' + str(delta_lat)
                
                ## get image
                pic_name = self.no_check_data[str_loc][0] + '/' + str(str_loc) + '.jpg'
                name_batch.append(pic_name)
                im = cv2.imread(os.path.join(cfg.FILE_PATH, pic_name))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, (cfg.EACH_SIZE, cfg.EACH_SIZE))
                patch['{},{}'.format(str(m), str(n))] = im
        each_image = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3))
        each_image[0:cfg.EACH_SIZE, 0:cfg.EACH_SIZE, :] = patch['-1,1']
        each_image[0:cfg.EACH_SIZE, cfg.EACH_SIZE:2*cfg.EACH_SIZE, :] = patch['0,1']
        each_image[0:cfg.EACH_SIZE, 2*cfg.EACH_SIZE:3*cfg.EACH_SIZE, :] = patch['1,1']
        
        each_image[cfg.EACH_SIZE:2*cfg.EACH_SIZE, 0:cfg.EACH_SIZE, :] = patch['-1,0']
        each_image[cfg.EACH_SIZE:2*cfg.EACH_SIZE, cfg.EACH_SIZE:2*cfg.EACH_SIZE, :] = patch['0,0']
        each_image[cfg.EACH_SIZE:2*cfg.EACH_SIZE, 2*cfg.EACH_SIZE:3*cfg.EACH_SIZE, :] = patch['1,0']
        
        each_image[2*cfg.EACH_SIZE:3*cfg.EACH_SIZE, 0:cfg.EACH_SIZE, :] = patch['-1,-1']
        each_image[2*cfg.EACH_SIZE:3*cfg.EACH_SIZE, cfg.EACH_SIZE:2*cfg.EACH_SIZE, :] = patch['0,-1']
        each_image[2*cfg.EACH_SIZE:3*cfg.EACH_SIZE, 2*cfg.EACH_SIZE:3*cfg.EACH_SIZE, :] = patch['1,-1']
        return each_image

    def obtain_batch_data(self, batch_num, valid=False, test=False, random1=False, random_num=1, imlist=False):
        train_batch = []
        ppl_batch = []
        name_batch = []
        delta = [0, -1, 1]
        if valid:
            train_origin = random.sample(self.valid_data, cfg.BATCH_SIZE) 
        elif test:
            train_origin = [self.test_data[batch_num]]
        elif random1:
            train_origin = random.sample(self.train_data, random_num) 
        elif imlist:
            train_origin = self.test_data[:batch_num]
        else:
            train_origin = self.train_data[cfg.BATCH_SIZE*batch_num:cfg.BATCH_SIZE*batch_num+cfg.BATCH_SIZE]

        for k in train_origin:  # for each training data
            ppl_batch.append(ppl)
            patch = {}  # patch(dict): {'-1,-1':image0, ...}
            # add neighbor
            lon = int(k.split('_')[0])
            lat = int(k.split('_')[1])
            ppl = self.total_data[k][1]
            for m in delta:
                for n in delta:
                    delta_lon = lon + m
                    delta_lat = lat + n
                    str_loc = str(delta_lon) + '_' + str(delta_lat)
                    
                    ## get image
                    pic_name = self.no_check_data[str_loc][0] + '/' + str(str_loc) + '.jpg'
                    name_batch.append(pic_name)
                    im = cv2.imread(os.path.join(cfg.FILE_PATH, pic_name))
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im, (cfg.EACH_SIZE, cfg.EACH_SIZE))
                    patch['{},{}'.format(str(m), str(n))] = im
        
            each_image = _neighbor_merging(patch)        
            train_batch.append(each_image)
            
        train_batch = np.reshape(train_batch, (-1, cfg.IMG_SIZE, cfg.IMG_SIZE, 3))
        ppl_batch = np.reshape(ppl_batch, (-1, 1))
        
        return train_batch, ppl_batch, name_batch
    
    def generate_train_city_data(self, train_city):
        self.train_city_test = []
        for k in self.test_data:
            city = self.total_data[k][0]
            if city == train_city:
                self.train_city_test.append(k)
        return self.train_city_test
    
    def obtain_train_city_data(self, n):
        train_batch = []
        ppl_batch = []
        name_batch = []
        delta = [0, -1, 1]
        train_origin = [self.train_city_test[n]]
        
        for k in train_origin:  # for each testing data
            patch = {}  # patch(dict): {'-1,-1':image0, ...}
            # add neighbor
            lon = int(k.split('_')[0])
            lat = int(k.split('_')[1])
            ppl = self.total_data[k][1]
            ppl_batch.append(ppl)
            for m in delta:
                for n in delta:
                    delta_lon = lon + m
                    delta_lat = lat + n
                    str_loc = str(delta_lon) + '_' + str(delta_lat)
                    ## get image
                    pic_name = self.no_check_data[str_loc][0] + '/' + str(str_loc) + '.jpg'
                    name_batch.append(pic_name)
                    im = cv2.imread(os.path.join(cfg.FILE_PATH, pic_name))
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    #im = im / 255.0
                    im = cv2.resize(im, (cfg.EACH_SIZE, cfg.EACH_SIZE))
                    patch['{},{}'.format(str(m), str(n))] = im

            each_image = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3))
            each_image[0:cfg.EACH_SIZE, 0:cfg.EACH_SIZE, :] = patch['-1,1']
            each_image[0:cfg.EACH_SIZE, cfg.EACH_SIZE:2*cfg.EACH_SIZE, :] = patch['0,1']
            each_image[0:cfg.EACH_SIZE, 2*cfg.EACH_SIZE:3*cfg.EACH_SIZE, :] = patch['1,1']

            each_image[cfg.EACH_SIZE:2*cfg.EACH_SIZE, 0:cfg.EACH_SIZE, :] = patch['-1,0']
            each_image[cfg.EACH_SIZE:2*cfg.EACH_SIZE, cfg.EACH_SIZE:2*cfg.EACH_SIZE, :] = patch['0,0']
            each_image[cfg.EACH_SIZE:2*cfg.EACH_SIZE, 2*cfg.EACH_SIZE:3*cfg.EACH_SIZE, :] = patch['1,0']

            each_image[2*cfg.EACH_SIZE:3*cfg.EACH_SIZE, 0:cfg.EACH_SIZE, :] = patch['-1,-1']
            each_image[2*cfg.EACH_SIZE:3*cfg.EACH_SIZE, cfg.EACH_SIZE:2*cfg.EACH_SIZE, :] = patch['0,-1']
            each_image[2*cfg.EACH_SIZE:3*cfg.EACH_SIZE, 2*cfg.EACH_SIZE:3*cfg.EACH_SIZE, :] = patch['1,-1']

            train_batch.append(each_image)

        train_batch = np.reshape(train_batch, (-1, cfg.IMG_SIZE, cfg.IMG_SIZE, 3))
        ppl_batch = np.reshape(ppl_batch, (-1, 1))
        
        return train_batch, ppl_batch, name_batch
    
    def obtain_batch_ppl_data(self):
        train_origin = random.sample(self.train_data, cfg.BATCH_SIZE)  # 1 for target area
        batch_image = []
        batch_ppl = []
        for loc in train_origin:
            ## get image
            pic_name = self.total_data[loc][0] + '/' + str(loc) + '.jpg'
            im = cv2.imread(os.path.join(cfg.FILE_PATH, pic_name))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im / 255.0
            im = cv2.resize(im, (cfg.EACH_SIZE, cfg.EACH_SIZE))
            batch_image.append(im)
            ppl = self.total_data[loc][1]
            batch_ppl.append(ppl)
        return batch_image, batch_ppl
    
    def generate_test_city_data(self, file, lon1, lon2, lat1, lat2):
        flag = 1
        line_num = 0
        self.nocheck_test_city = {}
        self.check_test_city = {}
        with open(file, 'r') as f:
            for line in f.readlines():
                line_num += 1
                image_path, lon, lat, ppl, dense_ppl = line.split(' ')
                lon = float(lon)
                lat = float(lat)
                dense_ppl = float(dense_ppl)
                pic_lon = int(round(lon * 100))
                pic_lat = int(round(lat * 100))
                pic_loc = str(pic_lon) + '_' + str(pic_lat)
                self.nocheck_test_city[pic_loc] = [image_path, dense_ppl]
                if pic_lon > lon1 and pic_lon < lon2 and pic_lat > lat1 and pic_lat < lat2:
                    if dense_ppl > 0.00:  # check for boundary areas
                        ## ppl
                        log_ppl = float(math.log(dense_ppl))
                        #log_ppl = dense_ppl
                        self.check_test_city[pic_loc] = [image_path, log_ppl]
        self.test_city_key = list(self.check_test_city.keys())
        return self.test_city_key
    
    def obtain_test_city_data(self, n):
        train_batch = []
        ppl_batch = []
        name_batch = []
        delta = [0, -1, 1]
        train_origin = [self.test_city_key[n]]
        
        for k in train_origin:  # for each training data
            patch = {}  # patch(dict): {'-1,-1':image0, ...}
            # add neighbor
            lon = int(k.split('_')[0])
            lat = int(k.split('_')[1])
            ppl = self.check_test_city[k][1]
            ppl_batch.append(ppl)
            for m in delta:
                for n in delta:
                    delta_lon = lon + m
                    delta_lat = lat + n
                    str_loc = str(delta_lon) + '_' + str(delta_lat)
                    
                    ## get image
                    pic_name = self.nocheck_test_city[str_loc][0] + '/' + str(str_loc) + '.jpg'
                    name_batch.append(pic_name)
                    im = cv2.imread(os.path.join(cfg.FILE_PATH, pic_name))
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    #im = im / 255.0
                    im = cv2.resize(im, (cfg.EACH_SIZE, cfg.EACH_SIZE))
                    patch['{},{}'.format(str(m), str(n))] = im
        
            each_image = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3))
            each_image[0:cfg.EACH_SIZE, 0:cfg.EACH_SIZE, :] = patch['-1,1']
            each_image[0:cfg.EACH_SIZE, cfg.EACH_SIZE:2*cfg.EACH_SIZE, :] = patch['0,1']
            each_image[0:cfg.EACH_SIZE, 2*cfg.EACH_SIZE:3*cfg.EACH_SIZE, :] = patch['1,1']
            
            each_image[cfg.EACH_SIZE:2*cfg.EACH_SIZE, 0:cfg.EACH_SIZE, :] = patch['-1,0']
            each_image[cfg.EACH_SIZE:2*cfg.EACH_SIZE, cfg.EACH_SIZE:2*cfg.EACH_SIZE, :] = patch['0,0']
            each_image[cfg.EACH_SIZE:2*cfg.EACH_SIZE, 2*cfg.EACH_SIZE:3*cfg.EACH_SIZE, :] = patch['1,0']
            
            each_image[2*cfg.EACH_SIZE:3*cfg.EACH_SIZE, 0:cfg.EACH_SIZE, :] = patch['-1,-1']
            each_image[2*cfg.EACH_SIZE:3*cfg.EACH_SIZE, cfg.EACH_SIZE:2*cfg.EACH_SIZE, :] = patch['0,-1']
            each_image[2*cfg.EACH_SIZE:3*cfg.EACH_SIZE, 2*cfg.EACH_SIZE:3*cfg.EACH_SIZE, :] = patch['1,-1']
        
            train_batch.append(each_image)
        
        train_batch = np.reshape(train_batch, (-1, cfg.IMG_SIZE, cfg.IMG_SIZE, 3))
        ppl_batch = np.reshape(ppl_batch, (-1, 1))
        return train_batch, ppl_batch, name_batch

    def obtain_valid_data(self):
        valid_batch = []
        delta = [0, -1, 1]
        valid_origin = random.sample(self.valid_data, cfg.BATCH_SIZE)  # 1 for target area

        for k in valid_origin:  # for each valid data
            # add neighbor
            lon = int(k.split('_')[0])
            lat = int(k.split('_')[1])
            for m in delta:
                for n in delta:
                    delta_lon = lon + m
                    delta_lat = lat + n
                    str_loc = str(delta_lon) + '_' + str(delta_lat)
                    valid_batch.append(str_loc)

            # add negative sample
            flag = 0
            while flag < 8:
                neg = random.sample(self.valid_data, 1)  # random chosen negative sample
                neg_lon = int(neg[0].split('_')[0])
                neg_lat = int(neg[0].split('_')[1])
                if neg_lon > lon + cfg.buffer or neg_lon < lon - cfg.buffer or neg_lat > lat + cfg.buffer or neg_lat < lat - cfg.buffer:
                    str_neg_loc = str(neg_lon) + '_' + str(neg_lat)
                    valid_batch.append(str_neg_loc)
                    flag += 1

        batch_image = []
        for loc in valid_batch:
            ## get image
            pic_name = self.no_check_data[loc][0] + '/' + str(loc) + '.jpg'
            im = cv2.imread(os.path.join(cfg.FILE_PATH, pic_name))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im / 255.0
            im = cv2.resize(im, (cfg.EACH_SIZE, cfg.EACH_SIZE))
            batch_image.append(im)
        return batch_image

    def obtain_visual_data(self):
        batch_image = []
        batch_label = []
        visual_data = self.valid_data[:900]
        for loc in visual_data:
            ## get image
            pic_name = self.total_data[loc][0] + '/' + str(loc) + '.jpg'
            im = cv2.imread(os.path.join(cfg.FILE_PATH, pic_name))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im / 255.0
            im = cv2.resize(im, (cfg.EACH_SIZE, cfg.EACH_SIZE))
            batch_image.append(im)
            batch_label.append(self.total_data[loc][1])
        return batch_image, batch_label

######## test ########
if __name__ == '__main__':
    data = Data(file_name='../data/test_shuffle_ppl.txt', json_path='../experiments/batch_all/params.json')
    total_data = data.total_data
    print(total_data)
    print(data.loc_list, data.train_data, data.valid_data, data.test_data)
