

import os
import subprocess
import argparse
import numpy as np
import glob
import time
from spectral.io import envi
import subprocess

def main(rawargs=None):

    path = os.environ['PATH']
    path = path.replace('\Library\\bin;',':')
    os.environ['PATH'] = path

    parser = argparse.ArgumentParser(description="Daily Tiler")
    parser.add_argument('file_list',type=str)
    parser.add_argument('output_dest', type=str)
    parser.add_argument('--dynamic', action='store_true')
    args = parser.parse_args(rawargs)

    files = np.genfromtxt(args.file_list,dtype=str)

    for _f, fi in enumerate(files):
        count = 0
        success=False
        
        #g2tmain([fi, args.output_dest,'-z','4-12','--srcnodata','0','--processes','40','-r','antialias'])
        if args.dynamic:
            subprocess.call(f'python /beegfs/scratch/brodrick/emit/MMGIS/auxiliary/gdal2customtiles/gdal2tiles_3.5.2.py --dem {fi} {args.output_dest} -r near-composite -z 5-11 --srcnodata=-9999 --processes=40 && echo {fi} >> {os.path.splitext(args.file_list)[0] + "_completed.txt"}',shell=True)
        else:
            subprocess.call(f'gdal2tiles.py {fi} {args.output_dest} -z 4-12 --srcnodata 0 --processes=40 -r antialias -x && echo {fi} >> {os.path.splitext(args.file_list)[0] + "_completed.txt"}',shell=True)
        #subprocess.call(f'python /beegfs/scratch/brodrick/emit/MMGIS/auxiliary/gdal2customtiles/gdal2tiles_3.5.2.py --dem {fi} {args.output_dest} -r near-composite -z 2-16 --srcnodata=-9999 --processes=40 && echo {fi} > {os.path.splitext(args.file_list)[0] + "_completed.txt"}',shell=True)
        #while success is False and count < 3:
        #    try:
        #        g2tmain([args.output_dest,fi,'-z','4-12','--srcnodata','0','--processes','40','-r','antialias'])
        #        success=True
        #    except:
        #        count+=1






if __name__ == "__main__":
    main()






