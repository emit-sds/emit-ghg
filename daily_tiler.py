#! /usr/bin/env python
#
#  Copyright 2023 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov

import os
import subprocess
import argparse
import numpy as np
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
        
        if args.dynamic:
            subprocess.call(f'python /beegfs/scratch/brodrick/emit/MMGIS/auxiliary/gdal2customtiles/gdal2tiles_3.5.2.py --dem {fi} {args.output_dest} -r near-composite -z 5-11 --srcnodata=-9999 --processes=40 && echo {fi} >> {os.path.splitext(args.file_list)[0] + "_completed.txt"}',shell=True)
        else:
            subprocess.call(f'gdal2tiles.py {fi} {args.output_dest} -z 4-12 --srcnodata 0 --processes=40 -r antialias -x && echo {fi} >> {os.path.splitext(args.file_list)[0] + "_completed.txt"}',shell=True)



if __name__ == "__main__":
    main()






