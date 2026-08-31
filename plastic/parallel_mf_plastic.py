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
# Authors: David R. Thompson
#          Brian Bue
#          Philip G. Brodrick, philip.brodrick@jpl.nasa.gov

import argparse
from spectral.io import envi

from os import makedirs
from os.path import join as pathjoin, exists as pathexists
import scipy
import numpy as np
import pylab as plt

from sklearn.cluster import MiniBatchKMeans
import logging

import os
os.environ['SPECTRAL_DATA'] = '/restrela/Documents/emit-ghg-main/'

ppmscaling = 1
CH4_WL = [2137, 2493]
CO2_WL = [1922, 2337]

PLASTIC_WL_lowertrim = [380,1300]
PLASTIC_WL_midtrim = [1400,1790] 
PLASTIC_WL_uppertrim = [2000,2400] 

# 1320-1440 nm and 1770-1970 nm


def main(input_args=None):
    parser = argparse.ArgumentParser(description="Robust MF")
    parser.add_argument('-k', '--kmodes', type=int, default=1, help='number of columnwise modes (k-means clusters)')
    parser.add_argument('-r', '--reject', action='store_true', help='enable multimodal covariance outlier rejection')
    parser.add_argument('-f', '--full', action='store_true', help='regularize multimodal estimates with the full column covariariance')    
    parser.add_argument('--pcadim', default=6, type=int, help='number of PCA dimensions to use')
    parser.add_argument('-m', '--metadata', action='store_true', help='save metadata image')    
    parser.add_argument('-R', '--reflectance', action='store_true', help='reflectance signature')
    parser.add_argument('-M', '--model', type=str, default='looshrinkage', help='model name (looshrinkage (default)|empirical)')    
    parser.add_argument('-n', '--num_cores', type=int, default=-1, help='number of cores (-1 (default))')    
    parser.add_argument('--ray_temp_dir', type=str, default=None, help='ray temp directory (None (default))')    
    parser.add_argument('--loglevel', type=str, default='DEBUG', help='logging verbosity')    
    parser.add_argument('--logfile', type=str, default=None, help='output file to write log to')    
    parser.add_argument('--threshold', type=float, default=None, help='Max-value threshold for connectivity and vis.')    
    parser.add_argument('--connectivity', type=int, default=None, help='Number of connected components for plume detection.')    
    parser.add_argument('--use_ace_filter', action='store_true', help='Use the Adaptive Cosine Estimator (ACE) Filter')    
    parser.add_argument('--output_whitened_spectra', default='', help='Use the Adaptive Cosine Estimator (ACE) Filter')    
    parser.add_argument('radiance_file', type=str,  metavar='INPUT', help='path to input image')   
    parser.add_argument('library', type=str,  metavar='LIBRARY', help='path to target library file')
    parser.add_argument('output', type=str,  metavar='OUTPUT', help='path for output image (mf ch4 ppm)')    
    args = parser.parse_args(input_args)


    logging.basicConfig(format='%(levelname)s:%(asctime)s ||| %(message)s', level=args.loglevel,
                        filename=args.logfile, datefmt='%Y-%m-%d,%H:%M:%S')
    
    
    radiance_file = args.radiance_file
    if radiance_file.endswith('.img'):
        radiance_filehdr = radiance_file.replace('.img','.hdr')
    else:
        radiance_filehdr = radiance_file + '.hdr'

    baseoutfile = args.output 
    if baseoutfile.endswith('.img'):
        baseoutfilehdr = baseoutfile.replace('.img','.hdr')
    else:
        baseoutfilehdr = baseoutfile + '.hdr'

    # columnwise spectral averaging function
    colavgfn = np.mean

    logging.info('Started processing input file: "%s"'%str(radiance_file))
    img = envi.open(radiance_filehdr,image=radiance_file)
    img_mm = img.open_memmap(interleave='source',writeable=False)
    logging.debug('Memmap openned: "%s"'%str(radiance_file))
    nrows,nbands,ncols = img_mm.shape

    # define active channels wrt target gas + measurement units
    if 'wavelength' not in img.metadata:
        sys.exit(0)
    wavelengths = np.array([float(x) for x in img.metadata['wavelength']])
    # cond_wav = ((wavelengths < 1320) | (wavelengths > 1440)) & ((wavelengths < 1770) | (wavelengths > 1970))
    cond_wav = (wavelengths > 1440) | (wavelengths < 1770)
    #wavelenghts = wavelengths[cond_wav]
    if 'ch4' in args.library:
        active = [np.argmin(np.abs(wavelengths - x)) for x in CH4_WL]
        logging.debug(f'CH4 active chanels: {active}')
    elif 'co2' in args.library:
        active = [np.argmin(np.abs(wavelengths - x)) for x in CO2_WL]
        logging.debug(f'CO2 active chanels: {active}')
    # active = [np.argmin(np.abs(wavelengths - x)) for x in PLASTIC_WL]    
    elif 'Plastic' in args.library:
        active = [np.argmin(np.abs(wavelengths - x)) for x in PLASTIC_WL_lowertrim]
        active2 = [np.argmin(np.abs(wavelengths - x)) for x in PLASTIC_WL_midtrim]
        active3 = [np.argmin(np.abs(wavelengths - x)) for x in PLASTIC_WL_uppertrim]
        logging.debug(f'Plastic active chanels: {active}')
        logging.debug(f'Plastic active chanels: {active2}')
        logging.debug(f'Plastic active chanels: {active3}')
    else:
        logging.error('could not set active range - neither co2 nor ch4 found in library name')
        sys.exit(0)

    img_mm1 = img.open_memmap(interleave='source',writeable=False)[:,active[0]-1:active[1],:]
    img_mm2 = img.open_memmap(interleave='source',writeable=False)[:,active2[0]-1:active2[1],:]
    img_mm3 = img.open_memmap(interleave='source',writeable=False)[:,active3[0]-1:active3[1],:]
    img_mm = np.hstack((img_mm1,img_mm2,img_mm3))

    # load the gas spectrum
    libdata = np.float64(np.loadtxt(args.library))
    abscf1=libdata[active[0]-1:active[1],2]
    abscf2 = libdata[active2[0]-1:active2[1],2]
    abscf3 = libdata[active3[0]-1:active3[1],2]
    abscf = np.concatenate((abscf1,abscf2,abscf3))

    # want bg modes to have at least as many samples as 120% x (# features)
    #bgminsamp = int((active[1]-active[0])*1.2)
    bgminsamp = int(((active[1]-active[0])+(active2[1]-active2[0])+(active3[1]-active3[0]))*1.2)
    bgmodel = 'unimodal' if args.kmodes==1 else 'multimodal'
    
    # alphas for leave-one-out cross validation shrinkage
    if args.model == 'looshrinkage':
        astep,aminexp,amaxexp = 0.05,-10.0,0.0
        alphas=(10.0 ** np.arange(aminexp,amaxexp+astep,astep))
        nll=np.zeros(len(alphas))
        

    # Get header info
    outmeta = img.metadata
    outmeta['lines'] = nrows
    outmeta['data type'] = np2envitype(np.float64)
    outmeta['bands'] = 1
    outmeta['description'] = 'matched filter results'
    outmeta['band names'] = 'mf'
    
    outmeta['interleave'] = 'bip'    
    for kwarg in ['smoothing factors','wavelength','wavelength units','fwhm']:
        outmeta.pop(kwarg,None)
        
    nodata = float(outmeta.get('data ignore value',-9999))
    if nodata > 0:
        raise Exception('nodata value=%f > 0, values will not be masked'%nodata)

    modelparms  = 'modelname={args.model}, bgmodel={bgmodel}'
    if args.kmodes > 1:
        modelparms += ', bgmodes={args.kmodes}, pcadim={args.pcadim}, reject={args.reject}'
        if args.model == 'looshrinkage':
            modelparms += ', regfull={args.full}'

    if args.model == 'looshrinkage':
        modelparms += ', aminexp={aminexp}, amaxexp={amaxexp}, astep={astep}'

    modelparms += ', reflectance={args.reflectance}, active_bands={active}'    

    outdict = locals()
    outdict.update(globals())
    outmeta['model parameters'] = '{ %s }'%(modelparms.format(**outdict))

    # Create output image
    outimg = envi.create_image(baseoutfilehdr,outmeta,force=True,ext='')
    outimg_mm = outimg.open_memmap(interleave='source',writable=True)
    assert((outimg_mm.shape[0]==nrows) & (outimg_mm.shape[1]==ncols))
    # Set values to nodata
    outimg_mm[...] = nodata
    
    if args.metadata:
        # output image of bgster membership labels per column
        bgfile=baseoutfile+'_bg'
        bgfilehdr=envi_header(bgfile)
        bgmeta = outmeta
        bgmeta['bands'] = 2
        bgmeta['data type'] = np2envitype(np.uint16)
        bgmeta['num alphas'] = len(alphas)
        bgmeta['alphas'] = '{%s}'%(str(alphas)[1:-1])
        bgmeta['band names'] = '{cluster_count, alpha_index}'
        bgimg = envi.create_image(bgfilehdr,bgmeta,force=True,ext='')
        bgimg_mm = bgimg.open_memmap(interleave='source',writable=True)

    outimg_shp = (outimg_mm.shape[0],1,outimg_mm.shape[2])
    bgimg_shp = None
    if args.metadata:
        bgimg_shp = (bgimg_mm.shape[0],1,bgimg_mm.shape[2])
    del outimg_mm


    # Run jobs in parallel
   #rayargs = {'_temp_dir': args.ray_temp_dir, 'ignore_reinit_error': True, 'include_dashboard': False}
   #if args.num_cores != -1:
   #    rayargs['num_cpus'] = args.num_cores
   #else:
   #    import multiprocessing
   #    rayargs['num_cpus'] = multiprocessing.cpu_count() - 1
    #ray.init(**rayargs)
    img_mm_id = img_mm.copy()#ray.put(img_mm.copy())
    abscf_id = abscf.copy() #ray.put(abscf)

    #jobs = [mf_one_column.remote(col,img_mm_id, bgminsamp, outimg_shp, bgimg_shp, abscf_id, args) for col in np.arange(ncols)]
    #rreturn = [ray.get(jid) for jid in jobs]
    rreturn = [mf_one_column(col,img_mm_id, bgminsamp, outimg_shp, bgimg_shp, abscf_id, args) for col in np.arange(ncols)]
    
    outimg_mm = outimg.open_memmap(interleave='source',writable=True)
    for ret in rreturn:
        if ret[0] is not None:
            outimg_mm[:, ret[2],-1] = np.squeeze(ret[0])
            if args.metadata:
                bgimg_mm[:, ret[2] ,-1] = np.squeeze(ret[1])

    logging.info('Complete')

def randperm(*args):
    n = args[0]
    k = n if len(args) < 2 else args[1] 
    return np.random.permutation(n)[:k]

def np2envitype(np_dtype):
    _dtype = np.dtype(np_dtype).char
    return envi.dtype_to_envi[_dtype]

def cov(A,**kwargs):
    """
    cov(A,**kwargs)
    
    Summary: computes covariance that matches matlab covariance function (ddof=1)
    
    Arguments:
    - A: n x m array of n samples with m features per sample
    
    Keyword Arguments:
    - same as numpy.cov
    
    Output:
    m x m covariance matrix
    """

    kwargs.setdefault('ddof',1)
    return np.cov(A.T,**kwargs)

def inv(A,**kwargs):
    kwargs.setdefault('overwrite_a',False)
    kwargs.setdefault('check_finite',False)
    return scipy.linalg.inv(A,**kwargs)

def eig(A,**kwargs):
    kwargs.setdefault('overwrite_a',False)    
    kwargs.setdefault('check_finite',False)
    kwargs.setdefault('left',False)
    kwargs.setdefault('right',True)
    return scipy.linalg.eig(A,**kwargs)

def det(A,**kwargs):
    kwargs.setdefault('overwrite_a',False)
    kwargs.setdefault('check_finite',False)    
    return scipy.linalg.det(A,**kwargs)

#@ray.remote
def par_looshrinkage(I_zm,alpha,nll,n,I_reg=[]):
    # loocv shrinkage estimation via Theiler et al.
    print(f'starting {alpha}')
    stability_scaling=100.0 
    nchan = I_zm.shape[1]
    
    X = I_zm*stability_scaling
    S = cov(X)
    T = np.diag(np.diag(S)) if len(I_reg)==0 else cov(I_reg*stability_scaling)
        
    nchanlog2pi = nchan*np.log(2.0*np.pi)

    # Closed form for leave one out cross validation error
    try:
        # See Theiler, "The Incredible Shrinking Covariance Estimator",
        # Proc. SPIE, 2012. eqn. 29
        beta = (1.0-alpha) / (n-1.0)
        G_alpha = n * (beta*S) + (alpha*T)
        G_det = det(G_alpha)
        if G_det==0:
            return np.nan
        r_k  = (X.dot(inv(G_alpha)) * X).sum(axis=1)
        q = 1.0 - beta * r_k
        print(f'completed {alpha}')
        return 0.5*(nchanlog2pi+np.log(G_det))+1.0/(2.0*n) * \
                 (np.log(q)+(r_k/q)).sum()
    except np.linalg.LinAlgError:
        logging.warning('looshrinkage encountered a LinAlgError')
        return np.nan
        

def looshrinkage(I_zm,alphas,nll,n,I_reg=[]):
    # loocv shrinkage estimation via Theiler et al.
    stability_scaling=100.0 
    nchan = I_zm.shape[1]

    C = cov(I_zm)+np.eye(nchan)*1e-10
    return C,1
    
    X = I_zm*stability_scaling
    S = cov(X)
    T = np.diag(np.diag(S)) if len(I_reg)==0 else cov(I_reg*stability_scaling)
        
    nchanlog2pi = nchan*np.log(2.0*np.pi)
    nll[:] = np.inf

    # Closed form for leave one out cross validation error
    for i,alpha in enumerate(alphas):
        try:
            # See Theiler, "The Incredible Shrinking Covariance Estimator",
            # Proc. SPIE, 2012. eqn. 29
            beta = (1.0-alpha) / (n-1.0)
            G_alpha = n * (beta*S) + (alpha*T)
            G_det = det(G_alpha)
            if G_det==0:
                continue
            r_k  = (X.dot(inv(G_alpha)) * X).sum(axis=1)
            q = 1.0 - beta * r_k
            nll[i] = 0.5*(nchanlog2pi+np.log(G_det))+1.0/(2.0*n) * \
                     (np.log(q)+(r_k/q)).sum()
        except np.linalg.LinAlgError:
            logging.warning('looshrinkage encountered a LinAlgError')

    mindex = np.argmin(nll)
    
    if nll[mindex]!=np.inf:
        alpha = alphas[mindex]
    else:
        mindex = -1
        alpha = 0.0

    # Final nonregularized covariance and shrinkage target
    S = cov(I_zm)
    T = np.diag(np.diag(S)) if len(I_reg)==0 else cov(I_reg)
        
    # Final covariance 
    C = (1.0 - alpha) * S + alpha * T

    return C,mindex


#@ray.remote
def mf_one_column(col, img_mm, bgminsamp, outimg_mm_shape, bgimg_mm_shape, abscf, args):


    logging.basicConfig(format='%(levelname)s:%(asctime)s ||| %(message)s', level=args.loglevel,
                        filename=args.logfile, datefmt='%Y-%m-%d,%H:%M:%S')

    logging.debug(f'Col: {col}')
    bgmodes = args.kmodes
    reject = args.reject
    regfull = args.full
    reflectance = args.reflectance
    savebgmeta = args.metadata
    modelname = args.model
    pcadim = args.pcadim
    use_ace_filter = args.use_ace_filter

    # alphas for leave-one-out cross validation shrinkage
    if args.model == 'looshrinkage':
        astep,aminexp,amaxexp = 0.05,-10.0,0.0
        alphas=(10.0 ** np.arange(aminexp,amaxexp+astep,astep))
        nll=np.zeros(len(alphas))
     

    outimg_mm = np.zeros((outimg_mm_shape))
    if savebgmeta:
        bgimg_mm = np.zeros((outimg_mm_shape))
    else:
        bgimg_mm = None

    # exclude nonfinite + negative spectra in covariance estimates
    #useidx = lambda Icolz: np.where(((~(Icolz<0)) & np.isfinite(Icolz)).all(axis=1))[0]
    # columnwise spectral averaging function
    colavgfn = np.mean

    Icol_full=img_mm[...,col]
    use = np.where(np.all(np.logical_and(np.isfinite(Icol_full), Icol_full > -0.05), axis=1))[0]
    Icol = np.float64(Icol_full[use,:].copy())
    nuse = Icol.shape[0]

    if nuse == 0:
        return None, None, None
    if len(use) == 1:
        bglabels = np.ones(nuse)
        bgulab = np.array([1])
        return None, None, None
    if bgmodes > 1:
        # PCA projection down to a smaller number of dimensions 
        # then apply K-means to separate spatially into clusters        
        Icol_zm = Icol-colavgfn(Icol,axis=0)
        evals,evecs = eig(cov(Icol_zm))
        Icol_pca = Icol_zm.dot(evecs[:,:pcadim]) 
        cmodel = MiniBatchKMeans(n_clusters=bgmodes)
        if np.iscomplexobj(Icol_pca):
            return None, None, None
        bglabels = cmodel.fit(Icol_pca).labels_
        bgulab = np.unique(bglabels)
        bgcounts = []
        bgulabn = np.zeros(len(bgulab))
        for i,l in enumerate(bgulab):
            lmask = bglabels==l
            bgulabn[i] = lmask.sum()
            if reject and bgulabn[i] < bgminsamp:
                logging.debug('Flagged outlier cluster %d (%d samples)'%(l,bgulabn[i]))
                bglabels[lmask] = -l
                bgulab[i] = -l                                    
            bgcounts.append("%d: %d"%(l,bgulabn[i]))

            if savebgmeta:
                bgimg_mm[use[lmask],0,0] = bgulabn[i]

        logging.debug('bg cluster counts:',', '.join(bgcounts))
        if (bgulab<0).all():
            logging.warning('all clusters rejected, proceeding without rejection (beware!)')
            bglabels,bgulab = abs(bglabels),abs(bgulab)
            
    else: # bgmodes==1
        bglabels = np.ones(nuse)
        bgulab = np.array([1])
    # operate independently on each columnwise partition
    for ki in bgulab:
        # if bglabel<0 (=rejected), estimate using all (nonrejected) modes
        kmask = bglabels==ki if ki >= 0 else bglabels>=0

        # need to recompute mu and associated vars wrt this cluster
        Icol_ki = (Icol if bgmodes == 1 else Icol[kmask,:]).copy()     
        
        Icol_sub = Icol_ki.copy()
        mu = colavgfn(Icol_sub,axis=0)
        # reinit model/modelfit here for each column/cluster instance
        if modelname == 'empirical':
            modelfit = lambda I_zm: cov(I_zm)
        elif modelname == 'looshrinkage':
            # optionally use the full zero mean column as a regularizer
            Icol_reg = Icol-mu if (regfull and bgmodes>1) else []
            modelfit = lambda I_zm: looshrinkage(I_zm,alphas,nll,
                                                 nuse,I_reg=Icol_reg)
            
        try:                            
            Icol_sub = Icol_sub-mu
            Icol_model = modelfit(Icol_sub)
            if modelname=='looshrinkage':
                C,alphaidx = Icol_model
                Cinv=inv(C)
                if savebgmeta:
                    bgimg_mm[use[kmask],0,1] = alphaidx
            elif modelname=='empirical':
                Cinv = inv(Icol_model)
            else:
                Cinv = Icol_model
                
        except np.linalg.LinAlgError:
            logging.warn('singular matrix. skipping this column mode.')
            outimg_mm[use[kmask],0,-1] = 0
            return None, None, None

        # Classical matched filter
        Icol_ki = Icol_ki-mu # = fully-sampled column mode
        target = abscf.copy()
        target = target-mu if reflectance else target*mu
        normalizer = target.dot(Cinv).dot(target.T)

        if use_ace_filter:
            # Self Mahalanobis distance
            rx = np.sum(Icol_ki @ Cinv * Icol_ki, axis = 1)
            # ACE filter normalization
            #normalizer = np.sqrt(normalizer * rx)
            normalizer = normalizer * rx

        mf = (Icol_ki.dot(Cinv).dot(target.T)) / normalizer

        if reflectance:
            outimg_mm[use[kmask],0,-1] = mf 
        else:
            outimg_mm[use[kmask],0,-1] = mf*ppmscaling

    colmu = outimg_mm[use[bglabels>=0],0,-1].mean()
    logging.debug('Column %i mean: %e'%(col,colmu))
    return outimg_mm, bgimg_mm, col




if __name__ == '__main__':
    main()
    #ray.shutdown()




