
from spectral.io import envi
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from osgeo import gdal
import cv2 as cv

files = np.genfromtxt('emit_targets.txt',dtype=str, delimiter=',')




for _f in range(len(files)):

    locfile=f'/beegfs/scratch/jchapman/EMIT_CMF_08102022/EMIT_cmf/{files[_f,0]}_l1b_ch4mf_ort_b0105_v01'
    if not os.path.isfile(locfile):
        continue
    trans = gdal.Open(locfile).GetGeoTransform()
    dat = envi.open(f'{locfile}.hdr').open_memmap(interleave='bip')
    

    coord_x = int((float(files[_f,2]) - trans[0])/trans[1])
    coord_y = int((float(files[_f,1]) - trans[3])/trans[5])
    print(coord_x, coord_y)

    meth = dat[...,-1].copy()
    meth[meth==-9999] = np.nan

    clahe = meth.copy()
    clahe[clahe <=0] = 0.001

    clahe /= 500
    clahe[clahe <= 0] = 1
    clahe[np.isnan(clahe)] = 0
    clahe *= 255
    clahe[clahe > 255] = 255
    clahe = clahe.astype('uint8')

    #clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #cl1 = clahe.apply(clahe)
    clahe = cv.equalizeHist(clahe)/255.



    #meth -= np.nanpercentile(meth,75)
    #meth /= np.nanpercentile(meth,100)
    #meth[meth <= 0] = 0.001
    #meth[meth >1] = 1
    #meth *=255

    meth -= 500
    meth /= 500
    meth[meth <= 0] = 0.001
    meth[meth >1] = 1
    meth *=255
    meth[np.isnan(meth)] = 0
    meth = meth.astype(int)

    out = np.zeros((meth.shape[0],meth.shape[1],3))
    #out[:,:,0] = meth
    out[:,:,0] = clahe
    
    plt.figure(figsize=(10,10))
    plt.imshow(out)
    plt.scatter(coord_x,coord_y, c='green', s=2)
    plt.savefig(f'figs/{files[_f,0]}_ch4.png',dpi=200)





