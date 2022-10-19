
import argparse
import subprocess

import target_generation
import parallel_mf
import local_surface_control
import logging
from spectral.io import envi
import numpy as np
import os
from utils import envi_header
from glob import glob
import time



def main(input_args=None):
    parser = argparse.ArgumentParser(description="Robust MF")
    parser.add_argument('date', type=str,  metavar='INPUT', help='path to input image')   
    parser.add_argument('output_dir', type=str,  metavar='OUTPUT', help='path to input image')   
    args = parser.parse_args(input_args)

    rdn_files = glob(f'/beegfs/store/emit/ops/data/acquisitions/{args.date}/emit*/l1b/*_rdn_*.img')
    obs_files = [x.replace('rdn','obs') for x in rdn_files]
    loc_files = [x.replace('rdn','loc') for x in rdn_files]
    glt_files = [x.replace('rdn','glt') for x in rdn_files]

    state_files = [x.replace('l1b','l2a').replace('rdn','statesubs') for x in rdn_files]
    state_files = [x if os.path.isfile(x) else None for x in state_files]

    out_files = [os.path.join(args.output_dir, os.path.basename(x).split('_')[0]) for x in rdn_files]

    for _r in range(len(rdn_files)):
    #for _r in range(2):

        cmd_str=f'sbatch -N 1 -c 40 -p standard,debug --mem=180G --wrap="python ghg_process.py {rdn_files[_r]} {obs_files[_r]} {loc_files[_r]} {glt_files[_r]} {out_files[_r]}'
        if state_files[_r] is not None:
            cmd_str += f' --state_subs {state_files[_r]}"'
        else:
            cmd_str += f'"'

        subprocess.call(cmd_str,shell=True)
        time.sleep(0.1)
    


if __name__ == '__main__':
    main()
