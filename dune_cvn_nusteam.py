#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('git clone https://github.com/nitish-nayak/dune-cvn.git')


# In[3]:


get_ipython().run_line_magic('cd', 'dune-cvn')


# In[1]:


get_ipython().run_line_magic('ls', '')


# The `dataset` folder contains 20 events in .gz files. Each .gz file is a compressed binary made up of three 500x500 images for the DUNE HD design, with 2 induction planes and 1 collection plane. 
# 
# There's also a `dataset_highstats` folder which we will use here that contains 100 events for each of `nueCC`, `numuCC` and `NC`, in the same .gz format
# 
# To access the truth labels for these events, we will need to read in pickle (`.p`) files. 

# In[2]:


# import some useful modules
get_ipython().system('cat utils.py')
from utils import *


# In[3]:


# some useful variables for our code
test_values = []
TEST_PARAMS = {'batch_size':1, #batch size for network evaluation (model.predict)
               'images_path':'dataset_info', # folder where all the images lie
               'shuffle':False, # whether we want to shuffle the files (doesn't matter for us, since we're just evaluating them one by one)
               'test_values':test_values}


# Here, we'll load the truth labels from the `.p` files. 

# In[4]:


# load dataset into IDs, labels 
def get_labels(flav):
  #print('Reading dataset from serialized file...')
  filename = TEST_PARAMS['images_path']+'/'+flav+'/partition_'+flav+'.p'
  with open(filename, 'rb') as partition_file:
      labels = pk.load(partition_file)
      IDs = list(labels.keys())
  #print('Loaded. Number of test examples for flavor %s : %d'%(flav,len(IDs)))
  return IDs, labels


# In[5]:


IDs, labels = get_labels('nue') # load pickle file for nueCC events


# In[6]:


print(list(labels.keys())[:10])


# In[7]:


# lookup value for key = '1'
print(labels['1'])


# `labels` is a dictionary that contains the truth information for each event indexed by keys `'0' - '99'`. Remember the CNN has been trained to predict a bunch of information about the topology and not just the flavor. We will only focus on the flavor tagging aspect for now, which is given by the second element of the list

# In[8]:


print(labels['1'][1])


# The second element is enumerated as: 
# 
# 
# *   0 - numuCC
# *   1 - nueCC
# *   2 - nutauCC
# *   3 - NC
# 
# 
# 
# 

# In[9]:


import zlib
# this function reads the .gz image file given an argument like '0' or '1' or '2' and so on
def get_pixelmap(key, flav): # with a default value
    path = TEST_PARAMS['images_path']+'/'+flav
    with open(path+'/event'+key+'.gz', 'rb') as image_file:
        pixels = np.fromstring(zlib.decompress(image_file.read()), dtype=np.uint8, sep='').reshape(3, 500, 500)
        return pixels


# In[10]:


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


# In[11]:


get_pixelmap('0', 'numu')


# The pixel map is a `3x500x500` array of integers. Each pixel represents the energy deposited, but digitized into a 16-bit integer. Lets try to plot them to see how it looks

# In[12]:


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


# In[13]:


pm = get_pixelmap('3', 'nue')
draw_single_pm(pm)


# Here, the x-axis is the `Wire` number associated with that anode-plane. `U` and `V` denoting the induction planes and `Z` denoting the collection. TDC refers to the time tick which is related to the time it takes for the drift electrons to travel to the anode plane from the ionization point. 

# We can also look at some relevant information about the event that makes up the image above.

# In[14]:


get_eventinfo('3', 'nue')


# NuEnergy and LepEnergy denotes the neutrino and lepton energy respectively
# The PDG codes denote the type of neutrino where : 
# 
#  -  12 (-12) : nue (anti-nue)
#  -  14 (-14) : numu (anti-numu)
#  -  16 (-16) : nutau (anti-nutau)
#  -  1 : NC (since distinguishing the flavor is not possible)

# The Interaction codes denote the type of interaction the neutrino had with the nucleus where : 
# 
#  - 0 : QuasiElastic (QE)
#  - 1 : Delta-Resonance (RES)
#  - 2 : Deep Inelastic Scattering (DIS)
#  - 3 : Other
#  
# There is also information about the type and number of important final state particles. Bear in mind, that these numbers have a certain threshold cut, so if for eg a proton in the final state doesn't have enough energy deposits it will not count towards the total

# ![Screen%20Shot%202022-09-15%20at%2015.57.41.png](attachment:Screen%20Shot%202022-09-15%20at%2015.57.41.png)

# Now let's try to evaluate our trained network on these images. For that we have to first re-jig the pixel array we got from the `.gz` file into a numpy array format that the network expects. 

# In[15]:


# convert image from .gz file (3 images, 500x500 pixels) to some format that the neural network understands
def convert_pm(pm):
    views = len(pm)
    planes = pm.shape[1]
    cells = pm.shape[2]
    
    X = [None]*views
    for view in range(views):
        X[view] = np.zeros((1, planes, cells, 1), dtype='float32')
    for view in range(views):
        X[view][0, :, :, :] = pm[view, :, :].reshape(planes, cells, 1)
    return X


# In[16]:


# loads the already trained neural network model for evaluation
def get_model(print_model=False): 
    with open('saved_model/model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights('saved_model/weights.h5')

    if(print_model):
        model.summary()
    return model
model = get_model() # get the neural network model


# In[17]:


# return list of flavor scores for each pixel map
def get_scores(pm): 
    scores = model.predict(convert_pm(pm))
    return scores[1]


# Let's try to print the results of the network evaluation along with the plot above for a nice summary on what's happening for each event. 
# 
# The following code also allows us to manipulate images in ways like : 
# 
# 
# *   Flipping - flip the images horizontally and then evaluate the network
# *   TurnOff - Turning off one image out of the 3 and see how the network responds to having lesser information. 
# 
# But for now, we won't have to worry about this. 
# 
# 

# In[18]:


# print results of network evaluation on input image
def print_pminfo(pm, ID, flav):
    print('Results of network evaluation on pixel map')
    scores = model.predict(convert_pm(pm))
    IDs, labels = get_labels(flav)
    flav_score = np.max(scores[1], axis=1)
    flav_pred = np.argmax(scores[1], axis=1)
    print('CVN score for pred label : %f, True Label : %s, Pred Label : %s'%
      (flav_score[0], flav_keys[labels[ID][1]], flav_keys[flav_pred[0]]))
    print('CVN score for true label : ', scores[1][0][labels[ID][1]])
    print('All scores :', scores[1])

# draw image and also print results of network evaluation
# flip = True, False (to flip each image horizontally)
# turnoff = 0, 1 or 2 (to turn off particular images before evaluation)
def show_pminfo(key, flav, flip=False, turnoff=None):
    
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

    draw_single_pm(pm2)
    print_pminfo(pm2, key, flav)
    print("Other info : \n", info)


# In[19]:


show_pminfo('3', 'nue', flip=True, turnoff=0)


# Here we have: 
# 
# 
# *   The predicted label and the true label along with the "likelihood" score the network gives for its prediction
# *   In cases where the network gets it wrong, its also useful to see what the score was for the true label
# *   Finally, a list of all scores for each of the 4 labels

# In[20]:


import time
import copy
t = time.time()
scores_orig = get_scores(pm)
print(time.time() - t)
print(scores_orig)


# In[107]:


import sys
import threading
import subprocess                                                                                                                                                                                                                                                                        
import os                                                                                                                                                                                                                                                                                
import errno                                                                                                                                                                                                                                                                             
from multiprocessing import Queue                                                                                                                                                                                                                                                        
import glob                                                                                                                                                                                                                                                                              
import time                                                                                                                                                                                                                                                                              
import copy                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                         
workQueue = Queue()                                                                                                                                                                                                                                                                      
queueLock = threading.Lock()                                                                                                                                                                                                                                                             
threads = []                                                                                                                                                                                                                                                                             
exitFlag = 0


# In[108]:


scores = {}                                                                                                                                                                                                                                                                              
cvn_model = model
inList = []
nThreads = 3
cvn_pm = pm


# In[109]:


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
                                                                                                                                                                                                                                                                                         
def get_cvnscores(pm):
    t0 = time.time()
    score = cvn_model.predict(convert_pixelmap(pm))
    print(time.time()-t0)
    return score[1]                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                         
def turnoffPixel(px, py, pm):                                                                                                                                                                                                                                                            
    "hadd a list of files, will be dumped in to a thread"                                                                                                                                                                                                                                
    global scores                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                         
    pm2 = copy.copy(pm)                                                                                                                                                                                                                                                                  
    pm2[0][px][py] = 0                                                                                                                                                                                                                                                                   
    pm2[1][px][py] = 0                                                                                                                                                                                                                                                                   
    pm2[2][px][py] = 0
    scores[(px, py)] = get_cvnscores(pm2)
    
class evalThread(threading.Thread):                                                                                                                                                                                                                                                      
    def __init__(self, threadId, wQ):                                                                                                                                                                                                                                                    
        threading.Thread.__init__(self)                                                                                                                                                                                                                                                  
        self.wQ = wQ
        self.threadId = threadId
     
    def run(self):
        worker(self.wQ, self.threadId)
       

def worker(q, i):
    while not exitFlag:
        queueLock.acquire()
        if not workQueue.empty():
            item = q.get()
            queueLock.release()
            turnoffPixel( item[0], item[1], item[2])
        else:
            queueLock.release()
        #time.sleep(1)


# In[110]:


#for ix in range(5):
#        for iy in range(5):
for ix in range(cvn_pm[0].shape[0]):
    for iy in range(cvn_pm[0].shape[1]):
        check1 = (cvn_pm[0][ix][iy] == 0)
        check2 = (cvn_pm[1][ix][iy] == 0)
        check3 = (cvn_pm[2][ix][iy] == 0)
        if check1 or check2 or check3: continue
        inList.append((ix, iy))
nTotal = len(inList)

threadIds = 1
for t in range(nThreads):
    t = evalThread(threadIds, workQueue)
    t.start()
    threads.append(t)
    threadIds += 1


# In[111]:


t2 = time.time()
queueLock.acquire()
for cx, cy in inList:
    workQueue.put((cx, cy, cvn_pm))
queueLock.release()


# In[112]:


while True:
    if workQueue.empty(): 
        break
        
exitFlag = 1
for t in threads:
    t.join()
print ("Finished all chunks!")
print(time.time()-t2)


# In[113]:


workQueue.close()


# First attempt to rotate images and evaluate impact on CVN. A bit dirty and doesn't handle all cases. Will update

# In[201]:


from math import *

get_offset = lambda a: min(a)
get_step = lambda a: abs(sorted(a)[1] - min(a))
def get_index(val, step, offset, min_offset):
    o = round(offset-min_offset)
    steps = round((val-offset)/step)
    return o+steps

def rotate_pm(pm, angle):
    X = []
    Y = []
    for i in range(500):
        for j in range(500):
            X.append(i*cos(angle) - j*sin(angle))
            Y.append(i*sin(angle) + j*cos(angle))
    stepX = get_step(X[::500])
    stepY = get_step(Y[:500])
    
    offsetX = []
    offsetY = []
    for i in range(500):
        offsetX.append(get_offset(X[i::500]))
        offsetY.append(get_offset(Y[i*500:(i+1)*500]))
    X_i = []
    Y_i = []
    minOx = min(offsetX)
    minOy = min(offsetY)
    for i in range(len(X)):
        osetX = offsetX[i%500]
        osetY = offsetY[i%500]
        X_i.append(get_index(X[i], stepX, osetX, minOx))
        Y_i.append(get_index(Y[i], stepY, osetY, minOy))
    
    pm2 = [None]*3
    for i in range(3):
        pm2[i] = np.zeros(pm[i].shape, dtype=np.uint8)
    for i in range(len(X_i)):
        if (X_i[i] >= 0) and (X_i[i] < 500):
            if (Y_i[i] >=0) and (Y_i[i] < 500):
                x = i // 500
                y = i % 500
                for view in range(3):
                    #pm2[view][X_i[i], Y_i[i]] += pm[view][x, y]
                    pm2[view][x, y] = pm[view][X_i[i], Y_i[i]]
    return pm2


# In[203]:


pm = get_pixelmap('0', 'nue')
pm2 = rotate_pm(pm, pi/8)
draw_single_pm(pm2)
np.all(pm2 == pm)


# I encourage you to play around with the inputs and see what kind of results you get and try to get a feel for whether the network is behaving accurately and as intended and where its getting confused if at all. But we'll move on to evaluate the network performance more quantitatively. 
# 
# 
# 
# 

# In[32]:


# function to get the predicted label for input event
def get_pred(key, flav_event):
    pm = get_pixelmap(key, flav_event)
    scores = model.predict(convert_pm(pm))
    flav_pred = np.argmax(scores[1], axis=1)[0]

    return flav_pred

# function to get the given label score for each event. 
# For eg, one can ask it to provide the nueCC score for an NC event and so on
def get_flav_score(key, flav, flav_event):
    tags = np.array(['numu', 'nue', 'nutau', 'nc'])
    pm = get_pixelmap(key, flav_event)
    scores = model.predict(convert_pm(pm))
    flav_score = scores[1][0][np.where(tags == flav)[0][0]]

    return flav_score


# Typically, the first set of numbers we look at in the test dataset are so-called confusion matrices. Essentially a matrix of `True Label` vs `Predicted Label` that tells us where the network gets things wrong and predominantly which label is the most confusing. Here we will ignore `nuTauCC` and just concentrate on `nueCC`, `numuCC` and `NC`

# In[33]:


confusion_mat = np.zeros((3,3), dtype=np.uint8)
tags = ['numu', 'nue', 'nc']
for f in range(3):
  for i in range(100):
    flav_pred = get_pred(str(i), tags[f])
    if(flav_pred == 2) : continue
    if(flav_pred == 3) : flav_pred = 2
    confusion_mat[f][flav_pred] += 1


# Here, rows are given by True Label, columns by Predicted Label. So one reads this off as, for `100` true numuCC events, the network is able to predict `95` of them accurately. `1` out of `100` is predicted as `nueCC` and `4` as NC. Similarly, one can read off for the other rows as well. 

# In[34]:


print(confusion_mat)


# We can normalize them by row to get an "efficiency" matrix. Also referred to as "Sensitivity" in other fields. Essentially, for given true label, we ask what fraction is predicted as signal or other background channels. 

# In[35]:


eff_mat = confusion_mat/confusion_mat.sum(axis=1,keepdims=True)
print(eff_mat)


# The other side of this is a "purity" matrix. Or "Specificity". Essentially, for a given predicted label, we ask what fraction is actually signal or other background. Sometimes a network can get very good efficiency by predicting a particular label all the time, but that's not a good thing and this metric is designed to catch that. So for eg, we see that the network is able to predict `98%` of `nueCC` accurately and also `95%` of events that it predicts as `nueCC` are actually `nueCC`. 

# In[36]:


pur_mat = confusion_mat/confusion_mat.sum(axis=0,keepdims=True)
print(pur_mat)


# We can visualize these numbers better in the following plots. Highly diagonal matrices are a sign that the network is able to generalize to a test dataset (made up of events that it has never seen) pretty well. 

# In[37]:


fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(eff_mat, interpolation=None)

ax.set_xticks(np.arange(len(tags)))
ax.set_xticklabels(tags)
ax.set_ylabel('True Label')
ax.set_xlabel('Pred Label')
ax.set_yticks(np.arange(len(tags)))
ax.set_yticklabels(tags)
for i in range(len(tags)):
  for j in range(len(tags)):
    text = ax.text(j, i, round(eff_mat[i, j], 2), ha='center', va='center', color='r')

plt.title('Efficiency by Label')
plt.show()


# In[38]:


fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(pur_mat, interpolation=None)

ax.set_xticks(np.arange(len(tags)))
ax.set_xticklabels(tags)
ax.set_ylabel('True Label')
ax.set_xlabel('Pred Label')
ax.set_yticks(np.arange(len(tags)))
ax.set_yticklabels(tags)
for i in range(len(tags)):
  for j in range(len(tags)):
    text = ax.text(j, i, round(pur_mat[i, j], 2), ha='center', va='center', color='r')

plt.title('Purity by Label')
plt.show()


# Let's try to plot some histograms of the network scores to see how much its able to separate signal and background. Here we will look at only `nueCC` scores but as homework, one can make similar plots for `numuCC` and `NC` scores as well. 

# In[39]:


# loop over events and append nueCC scores based on whether its signal or background
sig_scores = []
numu_scores = []
nc_scores = []
tags = ['numu', 'nue', 'nc']
for f in range(3):
  for i in range(100):
    flav_score = get_flav_score(str(i), 'nue', tags[f])
    if(f == 1): sig_scores.append(flav_score)
    if(f == 0): numu_scores.append(flav_score)
    if(f == 2): nc_scores.append(flav_score)


# In[40]:


# make the plot
b = np.arange(0, 1.05, 0.05)
fig, ax = plt.subplots(figsize=(10, 10))
ax.hist(np.array(sig_scores), bins=b, histtype='step',color='red',linestyle='solid',label='Signal nueCC')
ax.hist(np.array(numu_scores), bins=b, histtype='step',color='blue',linestyle='solid',label='NumuCC Bkg')
ax.hist(np.array(nc_scores), bins=b, histtype='step',color='green',linestyle='solid',label='NC Bkg')
ax.hist(np.array(nc_scores+numu_scores), bins=b, histtype='step',color='black',linestyle='solid',label='Total Bkg')
ax.legend(loc='best')
ax.set_ylabel('Events')
ax.set_xlabel('CNN Score')
plt.show()


# This shows that in terms of the `nueCC` scores, the true `nueCC` events peak near 1 while the background peaks near 0. This shows that the network is behaving appropriately and is able to distinguish signal and background. The better the separation in this plot, the better the performance. 

# We can even try to quantify the separation using a "ROC" Curve. 
# 
# 
# *   ROC - Receiver Operating Characteristic 
# 
# Essentially, it scans across the CNN scores and at each point gathers the fraction of background selected and the fraction of signal selected. Then, it just plots them together. It's instructive to compare this to an extremely dumb classifier which just predicts everything as nueCC without any skill, no matter which event is thrown at it. 
# 
# 
# 

# In[41]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

lr_fpr, lr_tpr, _ = roc_curve(200*[0] + 100*[1], numu_scores+nc_scores+sig_scores)
ns_fpr, ns_tpr, _ = roc_curve(200*[0] + 100*[1], 300*[1])
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(lr_fpr, lr_tpr, marker='.', label='CNN for NueCC')
ax.plot(ns_fpr, ns_tpr, marker='.', label='No Skill')
ax.legend(loc='best')
ax.set_title('ROC Curve')
ax.set_xlabel('Background Efficiency')
ax.set_ylabel('Signal Efficiency')
plt.show()
lr_auc = roc_auc_score(200*[0] + 100*[1], numu_scores+nc_scores+sig_scores)
ns_auc = roc_auc_score(200*[0] + 100*[1], 300*[1])
print('AUC scores for (CNN, No Skill) : %0.03f, %0.03f'%(lr_auc, ns_auc))


# One can clearly see that the CNN is obviously far superior to the "No Skill" network that just predicts `nueCC` all the time. The way to quantify this better is to estimate the "Area under the Curve" (AUC) which is one of many possible metrics. The "No Skill" one has an AUC of 0.5 while a CNN has an AUC of 0.998. 
# 
# A perfect network will have an AUC of exactly 1, so we can see our CNN is actually very very good! (Of course, there's some influence here of limited statistics, but point still stands.)
# 
# One can develop a different network and also compare with this CNN and the No Skill network to see how it performs. The AUC is one metric that can tell us which network is better.

# In[ ]:




