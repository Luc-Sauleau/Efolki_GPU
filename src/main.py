import os
import sys
import numpy as np
from skimage.io import imread
import pylab as pl

os.system('git clone https://github.com/aplyer/gefolki.git')
sys.path.append('gefolki/python')
from algorithm import EFolki

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

mask = WV>0
WV = WV.astype(np.float32)
QB = QB.astype(np.float32)
WV=WV*mask

u, v = EFolki(WV, QB, iteration=4, radius=[16,8], rank=4, levels=5)
QB=QB*mask

N = np.sqrt(u**2+v**2)
pl.figure()
pl.imshow(N,vmin=0,vmax=60)
pl.title('Norm of Optic to optic registration')
pl.colorbar()
