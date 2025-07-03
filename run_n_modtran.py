



import subprocess
import glob
import os, sys
import time
import random
import argparse
import multiprocessing
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('lut_dir',type=str)
parser.add_argument('-n_cpu',type=int, default=-1)
parser.add_argument('-n_jobs',type=int, default=-1)
parser.add_argument('-modtran_exe',type=str,default='/beegfs/store/shared/MODTRAN6/MODTRAN6.0.0/bin/linux/mod6c_cons')
args = parser.parse_args()




def run_single_modtran(modtran_exe,json_file,check_files,lock_file):
    completed = True
    for _f in check_files:
        if (os.path.isfile(_f) is False):
            completed = False
    if completed:
        print('already completed: {}'.format(json_file))
        return

    underway = os.path.isfile(lock_file)
    if underway:
        print('already underway: {}'.format(lock_file))
        return
    
    print('starting: '+json_file)

    cmd_str = 'touch {}'.format(lock_file)
    subprocess.call(cmd_str,shell=True)
    time.sleep(random.randint(1, 10) / 10.)

    cmd_str = '{} {}'.format(modtran_exe, json_file)
    subprocess.call(cmd_str,shell=True)
    cmd_str = 'rm {}'.format(lock_file)
    subprocess.call(cmd_str,shell=True)
    for name in ['r_k',
                 "t_k",
                 "wrn",
                 "psc",
                 "plt",
                 "7sc",
                 "acd"]:
        subprocess.call('rm {}.{}'.format(os.path.splitext(check_files[0])[0], name), shell=True)


    return

subprocess.call('rm -f /dev/shm/*',shell=True)
#subprocess.call('cp run_one_modtran.csh {}'.format(args.lut_dir),shell=True)
os.chdir(args.lut_dir)
subprocess.call('mkdir logs',shell=True)

run_files = glob.glob('./*.json', recursive=True)

lock_files = [os.path.splitext(x)[0].replace('LUT_','') + '.lock' for x in run_files]
tp6_files = [os.path.splitext(x)[0].replace('LUT_','') + '.tp6' for x in run_files]
chn_files = [os.path.splitext(x)[0].replace('LUT_','') + '.chn' for x in run_files]

if (args.n_cpu == -1):
    args.n_cpu = multiprocessing.cpu_count()

pool = multiprocessing.Pool(processes=args.n_cpu)
results = []
if args.n_jobs == -1:
    max_runs = 100000
else:
    max_runs = args.n_jobs
run=0
for _f in range(len(run_files)):
    if np.all([os.path.isfile(cf) is False for cf in [lock_files[_f], tp6_files[_f], chn_files[_f]]]):
        results.append(pool.apply_async(run_single_modtran, args=(args.modtran_exe,run_files[_f],[tp6_files[_f],chn_files[_f]],lock_files[_f],)))
        run+=1

    if run > max_runs:
        break

        
#results = [p.get() for p in results]
pool.close()
pool.join()





