import numpy
import pandas
import hyperspy.api as hs
import matplotlib.pyplot as plt

outputDir = '/mnt/cbis/home/utkarsh/codes/pillarCollapseClassification/dataset/allImages'

####################################################
# READING THE EXCEL WORKBOOK FOR USER INPUTS
####################################################
excelBook = pandas.ExcelFile('prepareCropImages.xlsx')
sheetName = excelBook.sheet_names[0]
inputInfo = excelBook.parse(sheetName)
####################################################

####################################################
for line in inputInfo.values:
    inputFile = line[0]
    rowTopLeft = line[1]
    colTopLeft = line[2]
    rowTopRight = line[3]
    colTopRight = line[4]
    rowBottomLeft = line[5]
    colBottomLeft = line[6]
    numPillarsInRow = line[7]
    numPillarsInCol = line[8]
    cropSize = line[9]
    
    
    rCenterList = 
