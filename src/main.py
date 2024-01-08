import os
import numpy as np
from skimage.io import imread
import pylab as pl

os.system('wget https://raw.githubusercontent.com/aplyer/gefolki/master/datasets/QB.tif')
os.system('wget https://raw.githubusercontent.com/aplyer/gefolki/master/datasets/WV.tif')

print("Read WV Image \n")
WV = imread("WV.tif")
pl.figure()
pl.imshow(WV,vmin=0,vmax=700,cmap='gray')
pl.title('Image WV')

print("Read QB Image \n")
QB = imread("QB.tif")
pl.figure()
pl.imshow(QB,vmin=0,vmax=500,cmap='gray')
pl.title('Image QB')
