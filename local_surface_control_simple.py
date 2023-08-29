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
# ISOFIT: Imaging Spectrometer Optimal FITting
# Authors: David R. Thompson
#          Philip G. Brodrick, philip.brodrick@jpl.nasa.gov

from spectral.io import envi 
import scipy as s
from sklearn.cross_decomposition import PLSRegression
from sklearn import linear_model
import argparse
from utils import envi_header
import ray
import numpy as np


@ray.remote
def local_model(x_full, y_full, good_full, start_l, end_l, start_c, end_c, model_type='linear'):

    x = x_full[start_l:end_l,start_c:end_c,:]
    y = y_full[start_l:end_l,start_c:end_c]
    ret_y = y.copy()
    good = good_full[start_l:end_l,start_c:end_c]

    if np.sum(good) > 10:
        x = x.reshape((x.shape[0]*x.shape[1],x.shape[2]))
        y = y.reshape((y.shape[0]*y.shape[1],1))
        good = good.reshape((good.shape[0]*good.shape[1]))
        good[np.any(np.isfinite(x) == False,axis=1)] = False
        good[np.isfinite(y.flatten()) == False] = False
        
        if model_type == 'linear':
            reg = linear_model.LinearRegression()
            reg.fit(x[good,:],y[good,:])
            pred = reg.predict(x)
        elif model_type == 'random_forest':
            reg = linear_model.LinearRegression()
            reg.fit(x[good,:],y[good,:].reshape(np.sum(good),))
            pred = reg.predict(x).flatten()

        pred = pred.reshape((end_l-start_l,end_c-start_c))
        good = good.reshape((end_l-start_l,end_c-start_c))
        print(np.mean(pred))

        #ret_y[good] -= pred[good]
        ret_y[good] -= np.maximum(pred[good],0)
    else:
        print(f'no good found: {np.sum(good)}')

    return ret_y, start_l, end_l, start_c, end_c


def subtract_local_model(ray_x, ray_y, ray_good, shape, l_chunk=160, c_chunk=160, model_type='linear'):


    jobs = []
    for line_start in range(0,shape[0],l_chunk):
        for col_start in range(0,shape[1],c_chunk):
            jobs.append(local_model.remote(ray_x, ray_y, ray_good, line_start, min(line_start + l_chunk,shape[0]), col_start, min(col_start + c_chunk,shape[1]),model_type=model_type))
            
    rreturn = [ray.get(jid) for jid in jobs]


    output = np.zeros(shape)
    for ret, start_line, stop_line, start_col, stop_col in rreturn:
        output[start_line:stop_line, start_col:stop_col] = ret

    return output



def main(input_args=None):
  parser = argparse.ArgumentParser(description="Control for surface")
  parser.add_argument('cmf', type=str,  metavar='CMF',
                      help='path to input image')   
  parser.add_argument('rdnfile', type=str,  
                      help='path to radiance file')
  parser.add_argument('maskfile', type=str,  
                      help='path to mask file')   
  parser.add_argument('output', type=str,  metavar='OUTPUT',
                      help='path for revised output image (mf ch4 ppm)')    
  parser.add_argument('--n_cores', type=int,  default=-1, metavar='num_cores',
                      help='number of cores to use')    
  parser.add_argument('--model_type', type=str,  default='linear', choices=['linear','random_forest'],
                      help='type of internal model to use') 
  parser.add_argument('--type', type=str,  default='ch4', choices=['ch4','co2'])
  args = parser.parse_args(input_args)


  if args.n_cores == -1:
    import multiprocessing
    args.n_cores = multiprocessing.cpu_count() - 1 
    
  rayargs = {'ignore_reinit_error': True, 'num_cpus': args.n_cores, 'include_dashboard': False}
  ray.init(**rayargs)
    
    
  cmf_ds = envi.open(envi_header(args.cmf))
  cmf = np.squeeze(cmf_ds.open_memmap(interleave='bip').copy())

  

  wl = s.array([float(f) for f in envi.open(envi_header(args.rdnfile)).metadata['wavelength']])

  if args.type == 'ch4':
    active = s.where(s.logical_or(s.logical_and(wl>380,wl<1250), 
                        s.logical_or(s.logical_and(wl>1500,wl<1610),
                          s.logical_and(wl>2030,wl<2140))))[0]
  elif args.type == 'co2':

    active = s.where(s.logical_or(s.logical_and(wl>380,wl<=1190), 
                        s.logical_or(s.logical_and(wl>=1630,wl<=1700),
                          s.logical_and(wl>2130,wl<2500))))[0]
  else:
    raise AttributeError('Invalid type')
    
  
  mask = np.sum(envi.open(envi_header(args.maskfile)).open_memmap(interleave='bip')[...,:3],axis=-1) > 0
  rdn = envi.open(envi_header(args.rdnfile)).open_memmap(interleave='bip')[...,active].copy()
  print(mask.shape, cmf.shape, rdn.shape)  


  good = np.logical_and.reduce((cmf != -9999, np.logical_not(mask)))


  rdn_id = ray.put(rdn)
  cmf_id = ray.put(cmf)
  good_id = ray.put(good)

  subtracted_cmf = subtract_local_model(rdn_id, cmf_id, good_id, cmf.shape, model_type=args.model_type)

  subtracted_cmf[mask == 1] = 0
  #subtracted_cmf[np.logical_and(subtracted_cmf != -9999, subtracted_cmf < 0)] = 0
  subtracted_cmf[np.logical_and(subtracted_cmf != -9999, subtracted_cmf < -200)] = -200

  outmeta = cmf_ds.metadata
  outmeta['description'] = 'masked  / loc filtered matched filter results'
  outimg = envi.create_image(envi_header(args.output),outmeta,force=True,ext='')
  out_mm = outimg.open_memmap(interleave='bip', writable=True)
  out_mm[...,0] = subtracted_cmf
  del out_mm

if __name__ == "__main__":
  main()
