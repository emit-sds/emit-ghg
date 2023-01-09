#! /usr/bin/env python
#
#  Copyright 2022 California Institute of Technology
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
# ISOFIT: Imaging Spectrometer Optimal FITting
# Authors: Philip G Brodrick, philip.brodrick@jpl.nasa.gov

import argparse
import subprocess

import os
from utils import envi_header
from glob import glob
import time



def main(input_args=None):
    parser = argparse.ArgumentParser(description="Robust MF")
    parser.add_argument('date', type=str,  metavar='INPUT', help='path to input image')   
    parser.add_argument('output_dir', type=str,  metavar='OUTPUT', help='path to input image')   
    args = parser.parse_args(input_args)

    if args.date == 'all':
        rdn_files = glob(f'/beegfs/store/emit/ops/data/acquisitions/*/emit*/l1b/*_rdn_b0106_v01.img')
    else:
        rdn_files = glob(f'/beegfs/store/emit/ops/data/acquisitions/{args.date}/emit*/l1b/*_rdn_b0106_v01.img')
    obs_files = [x.replace('rdn','obs') for x in rdn_files]
    loc_files = [x.replace('rdn','loc') for x in rdn_files]
    glt_files = [x.replace('rdn','glt') for x in rdn_files]

    state_files = [x.replace('l1b','l2a').replace('rdn','statesubs') for x in rdn_files]
    state_files = [x if os.path.isfile(x) else None for x in state_files]

    out_files = [os.path.join(args.output_dir, os.path.basename(x).split('_')[0]) for x in rdn_files]

    n=0
    for _r in range(len(rdn_files)):
    #for _r in range(2):

      ch4_mf_kmz_file = f'{out_files[_r]}_ch4_mf_color.kmz'

      if os.path.isfile(ch4_mf_kmz_file) is False:
        #n+=1
        cmd_str=f'sbatch -N 1 -c 40 -p standard --mem=180G --wrap="python ghg_process.py {rdn_files[_r]} {obs_files[_r]} {loc_files[_r]} {glt_files[_r]} {out_files[_r]}'
        if state_files[_r] is not None:
            cmd_str += f' --state_subs {state_files[_r]}"'
        else:
            cmd_str += f'"'

        #subprocess.call(cmd_str,shell=True)
        #print(os.environ.copy())
        env=os.environ.copy()
        subprocess.call(cmd_str,shell=True,env=env)
        #print(cmd_str)
        #time.sleep(0.1)


if __name__ == '__main__':
    main()
