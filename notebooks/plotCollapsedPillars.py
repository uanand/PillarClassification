import numpy
import pandas
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import shuffle
plt.style.use('UMM_ISO9001')

############################################################
# PLOT FOR PLASMA + RESTORE EXPERIMENTS WITH WATER
############################################################
# numSamples = 100
# dataFile = '/home/utkarsh/Desktop/exSitu/Plasma+RestoreByWater/Plasma+RestoreByWater.dat'
# df = pandas.read_csv(dataFile,delimiter='\t')
# tags = numpy.unique(df['Tag'])
# percentStanding = {}
# avgList,stdList = [],[]
# for tag in tqdm(tags):
    # data = df[df['Tag']==tag]['ClassificationClass']
    # percentStanding[tag] = []
    # for i in range(1000):
        # percentStanding[tag].append(100.0*numpy.sum(shuffle(data)[:numSamples])/numSamples)
    # avgList.append(numpy.mean(percentStanding[tag]))
    # stdList.append(numpy.std(percentStanding[tag]))
    
# fig = plt.figure(figsize=(2.8,2.0))
# ax = fig.add_axes([0,0,1,1])
# ax.fill_between((-1,4.5),y1=avgList[0]-stdList[0],y2=avgList[0]+stdList[0],ec='None',fc='r',alpha=0.25)
# ax.axhline(avgList[0],c='r',label='water dry')
# ax.fill_between((-1,4.5),y1=avgList[1]-stdList[1],y2=avgList[1]+stdList[1],ec='None',fc='k',alpha=0.25)
# ax.axhline(avgList[1],c='k',label='IPA dry')
# ax.errorbar(x=[-0.5],y=avgList[2],yerr=stdList[2],c='b',marker='o',ls='None',capsize=2,label='dry box')
# ax.errorbar(x=[0,1,2,4],y=avgList[3:],yerr=stdList[3:],c='g',marker='o',ls='None',capsize=2,label='dry box + vacuum')
# ax.set_xlim(-1,4.5)
# ax.set_ylim(0,100)
# ax.set_xticks([0,1,2,3,4])
# ax.set_xlabel('t (h)')
# ax.set_ylabel('% Standing NP')
# ax.legend(numpoints=1)
# plt.savefig('Plasma+RestoreByWater.png',format='png')
# plt.savefig('Plasma+RestoreByWater.pdf',format='pdf')
# plt.close()
############################################################

############################################################
# PLOT FOR RESTORE EXPERIMENTS WITH WATER
############################################################
# numSamples = 100
# dataFile = '/home/utkarsh/Desktop/exSitu/Plasma+RestoreByWater/Plasma+RestoreByWater.dat'
# df = pandas.read_csv(dataFile,delimiter='\t')
# tags = [0,1]
# percentStanding = {}
# avgList,stdList = [],[]
# for tag in tqdm(tags):
    # data = df[df['Tag']==tag]['ClassificationClass']
    # percentStanding[tag] = []
    # for i in range(1000):
        # percentStanding[tag].append(100.0*numpy.sum(shuffle(data)[:numSamples])/numSamples)
    # avgList.append(numpy.mean(percentStanding[tag]))
    # stdList.append(numpy.std(percentStanding[tag]))
    
# dataFile = '/home/utkarsh/Desktop/exSitu/RestoreByWater/RestoreByWater.dat'
# df = pandas.read_csv(dataFile,delimiter='\t')
# tags = numpy.unique(df['Tag'])
# for tag in tqdm(tags):
    # data = df[df['Tag']==tag]['ClassificationClass']
    # percentStanding[tag] = []
    # for i in range(1000):
        # percentStanding[tag].append(100.0*numpy.sum(shuffle(data)[:numSamples])/numSamples)
    # avgList.append(numpy.mean(percentStanding[tag]))
    # stdList.append(numpy.std(percentStanding[tag]))
    
# fig = plt.figure(figsize=(2.8,2.0))
# ax = fig.add_axes([0,0,1,1])
# ax.fill_between((-1,2.5),y1=avgList[0]-stdList[0],y2=avgList[0]+stdList[0],ec='None',fc='r',alpha=0.25)
# ax.axhline(avgList[0],c='r',label='water dry')
# ax.errorbar(x=[0,1./6,2./6,3./6,1,2],y=avgList[1:],yerr=stdList[1:],c='b',marker='o',ls='None',capsize=2,label='dry box')
# ax.set_xlim(-0.2,2.2)
# ax.set_ylim(0,100)
# ax.set_xticks([0,0.5,1,1.5,2])
# ax.set_xlabel('t (h)')
# ax.set_ylabel('% Standing NP')
# ax.legend(numpoints=1)
# plt.savefig('RestoreByWater.png',format='png')
# plt.savefig('RestoreByWater.pdf',format='pdf')
# plt.close()
############################################################

############################################################
# PLOT FOR RESTORE EXPERIMENTS WITH 1.00% HF
############################################################
numSamples = 100
dataFile = 'RestoreBy1.00%HF.dat'
df = pandas.read_csv(dataFile,delimiter='\t')
tags = [0,1]
percentStanding = {}
avgList,stdList = [],[]

tags = numpy.unique(df['Tag'])
for tag in tqdm(tags):
    data = df[df['Tag']==tag]['ClassificationClass']
    percentStanding[tag] = []
    for i in range(1000):
        percentStanding[tag].append(100.0*numpy.sum(shuffle(data)[:numSamples])/numSamples)
    avgList.append(numpy.mean(percentStanding[tag]))
    stdList.append(numpy.std(percentStanding[tag]))
    
fig = plt.figure(figsize=(2.8,1.0))
ax = fig.add_axes([0,0,1,1])
ax.errorbar(x=[0,10,20,30,40,50,60],y=avgList,yerr=stdList,c='g',marker='o',ls='None',capsize=2,label='1.00% HF')
ax.set_xlim(-10,310)
ax.set_ylim(0,100)
ax.set_xticks([0,30,60,90,120,150,180,210,240,270,300])
ax.set_xlabel('t (s)')
ax.set_ylabel('% Standing NP')
# ax.legend(numpoints=1)
plt.savefig('RestoreBy1.00%HF.png',format='png')
plt.savefig('RestoreBy1.00%HF.pdf',format='pdf')
plt.close()
############################################################

############################################################
# PLOT FOR RESTORE EXPERIMENTS WITH 0.10% HF
############################################################
numSamples = 100
dataFile = 'RestoreBy0.10%HF.dat'
df = pandas.read_csv(dataFile,delimiter='\t')
tags = [0,1]
percentStanding = {}
avgList,stdList = [],[]

tags = numpy.unique(df['Tag'])
for tag in tqdm(tags):
    if not(tag==2 or tag==4):
        data = df[df['Tag']==tag]['ClassificationClass']
        percentStanding[tag] = []
        for i in range(1000):
            percentStanding[tag].append(100.0*numpy.sum(shuffle(data)[:numSamples])/numSamples)
        avgList.append(numpy.mean(percentStanding[tag]))
        stdList.append(numpy.std(percentStanding[tag]))
    
fig = plt.figure(figsize=(2.8,1.0))
ax = fig.add_axes([0,0,1,1])
ax.errorbar(x=[0,10,30,50,60,120,300],y=avgList,yerr=stdList,c='g',marker='o',ls='None',capsize=2,label='1.00% HF')
ax.set_xlim(-10,310)
ax.set_ylim(0,100)
ax.set_xticks([0,30,60,90,120,150,180,210,240,270,300])
ax.set_xlabel('t (s)')
ax.set_ylabel('% Standing NP')
# ax.legend(numpoints=1)
plt.savefig('RestoreBy0.10%HF.png',format='png')
plt.savefig('RestoreBy0.10%HF.pdf',format='pdf')
plt.close()
############################################################

############################################################
# PLOT FOR RESTORE EXPERIMENTS WITH 0.30% HF
############################################################
numSamples = 100
dataFile = 'RestoreBy0.30%HF.dat'
df = pandas.read_csv(dataFile,delimiter='\t')
tags = [0,1]
percentStanding = {}
avgList,stdList = [],[]

tags = numpy.unique(df['Tag'])
for tag in tqdm(tags):
    data = df[df['Tag']==tag]['ClassificationClass']
    percentStanding[tag] = []
    for i in range(1000):
        percentStanding[tag].append(100.0*numpy.sum(shuffle(data)[:numSamples])/numSamples)
    avgList.append(numpy.mean(percentStanding[tag]))
    stdList.append(numpy.std(percentStanding[tag]))
    
fig = plt.figure(figsize=(2.8,1.0))
ax = fig.add_axes([0,0,1,1])
ax.errorbar(x=[0,30,60,90,120],y=avgList,yerr=stdList,c='g',marker='o',ls='None',capsize=2,label='1.00% HF')
ax.set_xlim(-10,310)
ax.set_ylim(0,100)
ax.set_xticks([0,30,60,90,120,150,180,210,240,270,300])
ax.set_xlabel('t (s)')
ax.set_ylabel('% Standing NP')
# ax.legend(numpoints=1)
plt.savefig('RestoreBy0.30%HF.png',format='png')
plt.savefig('RestoreBy0.30%HF.pdf',format='pdf')
plt.close()
############################################################
