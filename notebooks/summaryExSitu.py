import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
# from matplotlib.ticker import MultipleLocator

plt.style.use('UMM_ISO9001')
plt.rcParams['ytick.right'] = False
plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.direction'] = 'out'

############################################################
# CALCULATING STATISTICS FOR DIFFERENT EXPERIMENTS AND CONSOLIDATING
# INTO A SINGLE FILE
############################################################
# numSamples = 100
# extList = ['ALL','CNN','DNN','VGG']
# prefixList = ['Original','Plasma+RestoreByWater','RestoreByWater','RestoreBy0.10%HF','RestoreBy0.30%HF','RestoreBy1.00%HF']
# inputDir = '/scratch/utkur/utkarsh/RestorePillars/exSitu'

# outFile = open('summaryExSitu.dat','w')
# outFile.write('Experiment\tModel\tTag\tAverage\tStandardDeviation\n')
# for prefix in prefixList:
    # for ext in extList:
        # fileName = inputDir+'/'+prefix+'_'+ext+'.dat'
        # df = pandas.read_csv(fileName,delimiter='\t')
        # tagList = numpy.unique(df['Tag'])
        # N = df.shape[0]
        # for tag in tagList:
            # data = df[df['Tag']==tag]['ClassificationClass']
            # percentStanding = []
            # for i in range(1000):
                # percentStanding.append(100.0*numpy.sum(shuffle(data,random_state=i)[:numSamples])/numSamples)
            # try:
                # outFile.write('%s\t%s\t%s\t%f\t%f\n' %(prefix,ext,tag,numpy.mean(percentStanding),numpy.std(percentStanding)))
            # except:
                # outFile.write('%s\t%s\t%d\t%f\t%f\n' %(prefix,ext,tag,numpy.mean(percentStanding),numpy.std(percentStanding)))
# outFile.close()
############################################################


############################################################
# BAR PLOT FOR FIGURE S12
############################################################
x = [1,2,3]
y = [95.974,31.561,90.881]
yerr = [1.926999,4.669077,2.882853]
fig = plt.figure(figsize=(1.1811,1.1811))
ax = fig.add_axes([0,0,1,1])
ax.bar(x,y,width=0.9,yerr=yerr,color='#FFBFBF',edgecolor='k',capsize=4)
ax.set_xticks([1,2,3])
ax.set_xticklabels(['I','II','III'])
ax.set_ylabel('% Standing')
ax.set_ylim(0,100)
ax.set_xlim(0.5,3.5)
plt.savefig('collapse_water_ipa.png',format='png')
plt.savefig('collapse_water_ipa.pdf',format='pdf')
plt.close()
############################################################


############################################################
# LINE PLOT FOR FIGURE S13
# NP DO NOT RESTORE WITH WATER
############################################################
hLine1,hLine1Err = 90.941,2.789537
hLine2,hLine2Err = 31.401,4.616297
x = [30,60,120]
y = [21.215,28.064,21.751]
yErr = [3.98833,4.490201,4.098658]
fig = plt.figure(figsize=(2.5,2))
ax = fig.add_axes([0,0,1,1])
ax.fill_between((20,130),y1=hLine1-hLine1Err,y2=hLine1+hLine1Err,ec='None',fc='k',alpha=0.25)
ax.axhline(hLine1,c='k',label='IPA')
ax.fill_between((20,130),y1=hLine2-hLine2Err,y2=hLine2+hLine2Err,ec='None',fc='r',alpha=0.25)
ax.axhline(hLine2,c='r',label='Water')
ax.errorbar(x=x,y=y,yerr=yErr,c='g',marker='o',ls='None',capsize=2,label='Restore with Water')
ax.set_xticks([30,60,120])
ax.set_xlabel('Drying time (min)')
ax.set_ylabel('% Standing')
ax.set_ylim(0,100)
ax.set_xlim(20,130)
plt.savefig('restore_water.png',format='png')
plt.savefig('restore_water.pdf',format='pdf')
plt.close()


############################################################
# BAR PLOT FOR FIGURE S2
############################################################
# x = [1,2,3,4]
# y = [95.974,31.561,

# 1.926999,4.669077]


# # fileNameList = [\
# # '/scratch/utkur/utkarsh/RestorePillars/exSitu/Original.dat',\
# # '/scratch/utkur/utkarsh/RestorePillars/exSitu/Plasma+RestoreByWater.dat',\
# # '/scratch/utkur/utkarsh/RestorePillars/exSitu/RestoreByWater.dat',\
# # '/scratch/utkur/utkarsh/RestorePillars/exSitu/RestoreBy0.10%HF.dat'
# # ]
# # tagList = ['Original',0,5,8]

# # x = [1,2,3,4]
# # y,yerr = [],[]
# # for fileName,tag in zip(fileNameList,tagList):
    # # df = pandas.read_csv(fileName,delimiter='\t')
    # # data = df[df['Tag']==tag]['ClassificationClass']
    # # print (data.shape)
    # # percentStanding = []
    # # for i in range(1000):
        # # percentStanding.append(100.0*numpy.sum(shuffle(data)[:numSamples])/numSamples)
    # # y.append(numpy.mean(percentStanding))
    # # yerr.append(numpy.std(percentStanding))
    
# # fig = plt.figure(figsize=(2.16535,1.10236))
# # ax = fig.add_axes([0,0,1,1])
# # ax.bar(x,y,width=0.9,yerr=yerr,color='#FFBFBF',edgecolor='k',capsize=5)
# # ax.set_xticks([1,2,3,4])
# # ax.set_xticklabels(['I','II','III','IV'])
# # ax.set_ylabel('% Standing')
# # ax.set_ylim(0,100)
# # ax.set_xlim(0.35,4.5)
# # plt.savefig('plotBarFigure2.png',format='png')
# # plt.savefig('plotBarFigure2.pdf',format='pdf')
# # plt.close()
