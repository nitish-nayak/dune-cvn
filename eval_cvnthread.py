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
scores = []
cvn_model = None

def set_cvnmodel(model):
    global cvn_model
    cvn_model = model

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
    score = cvn_model.predict(convert_pixelmap(pm))
    return score[1]

def turnoffPixel(px, py, pm):
    "hadd a list of files, will be dumped in to a thread"
    global scores

    pm2 = copy.copy(pm)
    pm2[0][px][py] = 0
    pm2[1][px][py] = 0
    pm2[2][px][py] = 0
    scores.append(get_cvnscores(pm2))

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
        time.sleep(1)


class evalCVN:
    def __init__(self, pm, nThreads):
        self.pm = pm
        self.nThreads = nThreads
        self.inList = []
    
    def Init(self):
        #  for ix in range(self.pm[0].shape[0]):
        #      for iy in range(self.pm[0].shape[1]):
                #  if self.pm[0][ix][iy] == 0: continue
                #  if self.pm[1][ix][iy] == 0: continue
                #  if self.pm[2][ix][iy] == 0: continue
        for ix in range(5):
            for iy in range(5):
                self.inList.append((ix, iy))
        self.nTotal = len(self.inList)

    def Run(self):
        global threads, exitFlag, workQueue
        threadIds = 1
        for t in range(self.nThreads):
            t = evalThread(threadIds, workQueue)
            t.start()
            threads.append(t)
            threadIds += 1

        queueLock.acquire()
        for cx, cy in self.inList:
            workQueue.put((cx, cy, self.pm))
        queueLock.release()

        while not workQueue.empty():
            pass

        exitFlag = 1
        for t in threads:
            t.join()

        print ("Finished all chunks!")
