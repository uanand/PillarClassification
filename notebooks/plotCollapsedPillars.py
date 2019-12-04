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
numSamples = 100
dataFile = '/home/utkarsh/Desktop/exSitu/Plasma+RestoreByWater/Plasma+RestoreByWater.dat'
df = pandas.read_csv(dataFile,delimiter='\t')
tags = [0,1]
percentStanding = {}
avgList,stdList = [],[]
for tag in tqdm(tags):
    data = df[df['Tag']==tag]['ClassificationClass']
    percentStanding[tag] = []
    for i in range(1000):
        percentStanding[tag].append(100.0*numpy.sum(shuffle(data)[:numSamples])/numSamples)
    avgList.append(numpy.mean(percentStanding[tag]))
    stdList.append(numpy.std(percentStanding[tag]))
    
dataFile = '/home/utkarsh/Desktop/exSitu/RestoreByWater/RestoreByWater.dat'
df = pandas.read_csv(dataFile,delimiter='\t')
tags = numpy.unique(df['Tag'])
for tag in tqdm(tags):
    data = df[df['Tag']==tag]['ClassificationClass']
    percentStanding[tag] = []
    for i in range(1000):
        percentStanding[tag].append(100.0*numpy.sum(shuffle(data)[:numSamples])/numSamples)
    avgList.append(numpy.mean(percentStanding[tag]))
    stdList.append(numpy.std(percentStanding[tag]))
    
fig = plt.figure(figsize=(2.8,2.0))
ax = fig.add_axes([0,0,1,1])
ax.fill_between((-1,2.5),y1=avgList[0]-stdList[0],y2=avgList[0]+stdList[0],ec='None',fc='r',alpha=0.25)
ax.axhline(avgList[0],c='r',label='water dry')
ax.errorbar(x=[0,1./6,2./6,3./6,1,2],y=avgList[1:],yerr=stdList[1:],c='b',marker='o',ls='None',capsize=2,label='dry box')
ax.set_xlim(-0.2,2.2)
ax.set_ylim(0,100)
ax.set_xticks([0,0.5,1,1.5,2])
ax.set_xlabel('t (h)')
ax.set_ylabel('% Standing NP')
ax.legend(numpoints=1)
plt.savefig('RestoreByWater.png',format='png')
plt.savefig('RestoreByWater.pdf',format='pdf')
plt.close()
############################################################
