#!/usr/bin/env python
# script inspired by A. Aurisano's script in NOvA
# modified to allow more control for number of output files
# also, make running threads access new chunks on demand if available 

import sys
import threading
import subprocess
import argparse
import os
import errno
import Queue
import glob
import time

workQueue = Queue.Queue()
exitFlag = 0

def chunks(l, n):
    return [l[i::n] for i in xrange(n)]

def haddF(fcl, ofName = None, ifList = []):
    "hadd a list of files, will be dumped in to a thread"

    print ofName
    FNULL = open(os.devnull, 'w')

    cmd = ['lar']
    #  if fcl:
    #      cmd.append('-f')
    cmd.append('-c')
    cmd.append(fcl)


    for f in ifList:
        cmd.append('-s')
        cmd.append("%s"%(f))
    
    cmd.append('-o')
    cmd.append(ofName)
    #  print(cmd)
    #  subprocess.list2cmdline(cmd)
    subprocess.call(cmd, stdout=FNULL)

class haddThread(threading.Thread):
  
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
            #  print "Processing thread ", i, " for ", item[0], "\n"
            haddF( item[0], item[1], item[2] )
        else:
            queueLock.release()
        time.sleep(1)


parser = argparse.ArgumentParser(description="Multi threaded hadd.")
parser.add_argument("--outPrefix", "-p", type = str,  dest='outPrefix', default="out_hadd")
parser.add_argument("--outDir",    "-o", type = str,  dest='outDir',    default=".")
parser.add_argument("--nThreads",  "-j", type = int,  dest="nThreads",  default=5)
parser.add_argument("--nOutputs",  "-n", type = int,  dest="nOutputs",  default=1)
parser.add_argument("--fcl",       "-f", type = str,  dest='fcl')
parser.add_argument("--inputf",    "-i", nargs='+')

args = parser.parse_args()

inList = []
if type(args.inputf) == str:
  inList = glob.glob(args.inputf)
if type(args.inputf) == list:
  inList = args.inputf
else:
  sys.exit(1)

fcl = args.fcl

nTotal = len(inList)

nChunks = args.nOutputs
endHadd = False

if args.nOutputs == 1:
    endHadd = True
    maxFiles = 50
    nChunks = int(nTotal/maxFiles)

print nTotal, "files to hadd into ", nChunks, " file(s) using ", args.nThreads, " thread(s)"

partList = []
jList = chunks( inList, nChunks )

queueLock = threading.Lock()
threads = []

threadIds = 1
for t in range(args.nThreads):
    t = haddThread(threadIds, workQueue)
    t.start()
    threads.append(t)
    threadIds += 1

queueLock.acquire()
for i,c in enumerate(jList):
    outFile = args.outDir+"/"+args.outPrefix+"_part%i.root"%(i)
    workQueue.put( (fcl, outFile, c) )
queueLock.release()

while not workQueue.empty():
    pass

exitFlag = 1
for t in threads:
    t.join()

print ("Finished all chunks!")

#  if endHadd:
#
#      print("Hadding into single file")
#      haddF(args.outDir+"/"+args.outPrefix+".root", partList, fcl)
#
#      print("Deleting part files")
#      for partFile in partList:
#          os.unlink( partFile )
