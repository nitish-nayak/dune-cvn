from utils import *
import pandas as pd
import sys
import threading
import subprocess
import os
import errno
from multiprocessing import Queue
import glob
import time
import copy

test_values = []
TEST_PARAMS = {'batch_size':1, #batch size for network evaluation (model.predict)
               'images_path':'dataset_info', # folder where all the images lie
               'shuffle':False, # whether we want to shuffle the files (doesn't matter for us, since we're just evaluating them one by one)
               'test_values':test_values}

def get_labels(flav):
  #print('Reading dataset from serialized file...')
  filename = TEST_PARAMS['images_path']+'/'+flav+'/partition_'+flav+'.p'
  with open(filename, 'rb') as partition_file:
      labels = pk.load(partition_file)
      IDs = list(labels.keys())
  #print('Loaded. Number of test examples for flavor %s : %d'%(flav,len(IDs)))
  return IDs, labels

import zlib
# this function reads the .gz image file given an argument like '0' or '1' or '2' and so on
def get_pixelmap(key, flav): # with a default value
    path = TEST_PARAMS['images_path']+'/'+flav
    with open(path+'/event'+key+'.gz', 'rb') as image_file:
        pixels = np.fromstring(zlib.decompress(image_file.read()), dtype=np.uint8, sep='').reshape(3, 500, 500)
        return pixels

def get_eventinfo(key, flav):
    path = TEST_PARAMS['images_path']+'/'+flav
    ret = {}
    with open(path+'/event'+key+'.info', 'rb') as info_file:
        info = info_file.readlines()
        ret['NuPDG'] = int(info[7].strip())
        ret['NuEnergy'] = float(info[1])
        ret['LepEnergy'] = float(info[2])
        ret['Interaction'] = int(info[0].strip()) % 4
        ret['NProton'] = int(info[8].strip())
        ret['NPion'] = int(info[9].strip())
        ret['NPiZero'] = int(info[10].strip())
        ret['NNeutron'] = int(info[11].strip())
        #ret['OscWeight'] = float(info[6])
    return ret

def convert_pixelmap(pm):
    views = len(pm)
    planes = pm.shape[1]
    cells = pm.shape[2]
                    
    X = [None]*views
    for view in range(views):
        X[view] = np.zeros((1, planes, cells, 1), dtype='float32')
    for view in range(views):
        X[view][0, :, :, :] = pm[view, :, :].reshape(planes, cells, 1)
    return X

# loads the already trained neural network model for evaluation
def get_model(print_model=False): 
    with open('saved_model/model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights('saved_model/weights.h5')

    if(print_model):
        model.summary()
    return model

def get_scores(pm, model): 
    score = model.predict(convert_pixelmap(pm))
    return score[1]

import matplotlib.pylab as plt

# useful variable which we use in the code later on
flav_keys=['numucc', 'nuecc', 'nutaucc', 'NC']
# function meant to draw an image. Input here is the image that we obtain from the previous function get_pixelmap
def draw_single_pm(pm):
    fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    fig.suptitle('Pixel Maps')        
    titles = ['U', 'V', 'Z']
    for i in range(3):
        maps = np.swapaxes(pm[i], 0, 1)
        axs[i].imshow(maps, interpolation='none', cmap='twilight')
        axs[i].set_xlabel('Wire')
        axs[i].set_ylabel('TDC')
        axs[i].title.set_text(titles[i])
    plt.show()

# print results of network evaluation on input image
def print_pminfo(pm, ID, flav, model):
    print('Results of network evaluation on pixel map')
    scores = get_scores(pm, model)
    IDs, labels = get_labels(flav)
    flav_score = np.max(scores, axis=1)
    flav_pred = np.argmax(scores, axis=1)
    print('CVN score for pred label : %f, True Label : %s, Pred Label : %s'%
      (flav_score[0], flav_keys[labels[ID][1]], flav_keys[flav_pred[0]]))
    print('CVN score for true label : ', scores[0][labels[ID][1]])
    print('All scores :', scores)

# get results of network evaluation on input image in list format
def get_pmresults(pm, info, flav, model):
    
    scores = get_scores(pm, model)
    ret = {'true_flav':flav}
    ret.update(info)
    ret['numu_score'] = scores[0][0]
    ret['nue_score'] = scores[0][1]
    ret['nutau_score'] = scores[0][2]
    ret['nc_score'] = scores[0][3]

    return ret

# draw image and also print results of network evaluation
# flip = True, False (to flip each image horizontally)
# turnoff = 0, 1 or 2 (to turn off particular images before evaluation)
def show_pminfo(model, key, flav, flip=False, turnoff=None):
    
    pm = get_pixelmap(key, flav)
    info = get_eventinfo(key, flav)
    print('shape of image : ', pm.shape)
    print('Drawing pixel map')
    pm2 = pm
    if flip:
        #pm2 = np.empty(pm.shape, dtype=np.uint8)
        # flips the image
        for view in range(3):
            pm2[view] = np.flip(pm[view], axis=1)
            
    if turnoff is not None:
        assert (turnoff < 3 and turnoff >= 0), "turnoff can only be 0, 1 or 2"
        pm2[turnoff] = np.zeros(pm[turnoff].shape, dtype=np.uint8)

    #  draw_single_pm(pm2)
    print_pminfo(pm2, key, flav, model)
    print("Other info : \n", info)


def turnoffPixel(px, py, pm, info, flav, model):
    "hadd a list of files, will be dumped in to a thread"

    pm2 = copy.copy(pm)
    pm2[0][px][py] = 0
    pm2[1][px][py] = 0
    pm2[2][px][py] = 0
    return get_pmresults(pm2, info, flav, model) 
