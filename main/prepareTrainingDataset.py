import os
import sys
import numpy
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath('../lib'))
import imageProcess
import imageDraw

inputDir = '../dataset/allImages'

