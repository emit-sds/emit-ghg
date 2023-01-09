# Brian D. Bue
# David R. Thompson
# Philip G. Brodrick
# Copyright 2017, by the California Institute of Technology. ALL RIGHTS 
# RESERVED. United States Government Sponsorship acknowledged. Any commercial 
# use must be negotiated with the Office of Technology Transfer at the 
# California Institute of Technology.  This software is controlled under the 
# U.S. Export Regulations and may not be released to foreign persons without 
# export authorization, e.g., a license, license exception or exemption.


import argparse
from spectral.io import envi

from os import makedirs
from os.path import join as pathjoin, exists as pathexists
import scipy
import numpy as np
from utils import envi_header

from sklearn.cluster import MiniBatchKMeans
import ray
import logging

ppmscaling = 100000.0
CH4_WL = [2137, 2493]
CO2_WL = [1922, 2337]


def main(input_args=None):
    parser = argparse.ArgumentParser(description="Robust MF")
    parser.add_argument('-r', '--reject', action='store_true', help='enable multimodal covariance outlier rejection')   
    parser.add_argument('-M', '--model', type=str, default='looshrinkage', help='model name (looshrinkage (default)|empirical)')    
    parser.add_argument('-n', '--num_cores', type=int, default=-1, help='number of cores (-1 (default))')
    parser.add_argument('--n_mc', type=int, default=10, help='number of monte carlo runs')
    parser.add_argument('--mc_bag_fraction',type=float, default=0.7, help='fraction of data to use in each MC instance')
    parser.add_argument('--ray_temp_dir', type=str, default=None, help='ray temp directory (None (default))')    
    parser.add_argument('--loglevel', type=str, default='DEBUG', help='logging verbosity')    
    parser.add_argument('--logfile', type=str, default=None, help='output file to write log to')         

    parser.add_argument('radiance_file', type=str,  metavar='INPUT', help='path to input image')   
    parser.add_argument('library', type=str,  metavar='LIBRARY', help='path to target library file')
    parser.add_argument('output', type=str,  metavar='OUTPUT', help='path for output image (mf ch4 ppm)')    
    args = parser.parse_args(input_args)


    logging.basicConfig(format='%(levelname)s:%(asctime)s ||| %(message)s', level=args.loglevel,
                        filename=args.logfile, datefmt='%Y-%m-%d,%H:%M:%S')
    
    
    radiance_file = args.radiance_file
    radiance_filehdr = envi_header(radiance_file)

    baseoutfile = args.output 
    baseoutfilehdr = envi_header(baseoutfile)

    # columnwise spectral averaging function
    colavgfn = np.mean

    logging.info('Started processing input file: "%s"'%str(radiance_file))
    img = envi.open(radiance_filehdr,image=radiance_file)
    img_mm = img.open_memmap(interleave='source',writeable=False)
    logging.debug('Memmap openned: "%s"'%str(radiance_file))
    nrows,nbands,ncols = img_mm.shape

    # define active channels wrt target gas + measurement units
    if 'wavelength' not in img.metadata:
        logging.error('wavelength field not found in input header')
        sys.exit(0)
    wavelengths = np.array([float(x) for x in img.metadata['wavelength']])
    if 'ch4' in args.library:
        active = [np.argmin(np.abs(wavelengths - x)) for x in CH4_WL]
        logging.debug(f'CH4 active chanels: {active}')
    elif 'co2' in args.library:
        active = [np.argmin(np.abs(wavelengths - x)) for x in CO2_WL]
        logging.debug(f'CO2 active chanels: {active}')
    else:
        logging.error('could not set active range - neither co2 nor ch4 found in library name')
        sys.exit(0)
    active = np.where(np.logical_or(np.logical_and(wavelengths > 2137, wavelengths < 2493),
                                    np.logical_and(wavelengths > 1610, wavelengths < 1840) ))[0]
    active = np.where(np.logical_and(wavelengths > 2137, wavelengths < 2493))[0]
    img_mm = img.open_memmap(interleave='source',writeable=False)[:,active,:]
    #img_mm = img.open_memmap(interleave='source',writeable=False)[:,active[0]-1:active[1],:]

    # load the gas spectrum
    libdata = np.float64(np.loadtxt(args.library))
    #abscf=libdata[active[0]-1:active[1],2]
    abscf=libdata[active,2]

    # want bg modes to have at least as many samples as 120% x (# features)
    #bgminsamp = int((active[1]-active[0])*1.2)
    bgminsamp = len(active)*1.2
    bgmodel = 'unimodal'
    
    # alphas for leave-one-out cross validation shrinkage
    if args.model == 'looshrinkage':
        astep,aminexp,amaxexp = 0.05,-10.0,0.0
        alphas=(10.0 ** np.arange(aminexp,amaxexp+astep,astep))
        nll=np.zeros(len(alphas))
        
    # a_noise = sqrt(mf.T @ S_e @ mf)
    # a_total = sqrt(a_noise**2 + alpha_bootstrap_std**2)
    
    # compare w/ background std of mf results
    
    # Get header info
    outmeta = img.metadata
    outmeta['lines'] = nrows
    outmeta['data type'] = np2envitype(np.float64)
    outmeta['bands'] = args.n_mc
    outmeta['description'] = 'matched filter results'
    outmeta['band names'] = 'mf'
    
    outmeta['interleave'] = 'bip'    
    for kwarg in ['smoothing factors','wavelength','wavelength units','fwhm']:
        outmeta.pop(kwarg,None)
        
    nodata = float(outmeta.get('data ignore value',-9999))
    if nodata > 0:
        raise Exception('nodata value=%f > 0, values will not be masked'%nodata)

    modelparms  = 'modelname={args.model}, bgmodel={bgmodel}'

    if args.model == 'looshrinkage':
        modelparms += ', aminexp={aminexp}, amaxexp={amaxexp}, astep={astep}'

    modelparms += ', active_bands={active}'    

    outdict = locals()
    outdict.update(globals())
    outmeta['model parameters'] = '{ %s }'%(modelparms.format(**outdict))

    # Create output image
    outimg = envi.create_image(baseoutfilehdr,outmeta,force=True,ext='')
    outimg_mm = outimg.open_memmap(interleave='source',writable=True)
    assert((outimg_mm.shape[0]==nrows) & (outimg_mm.shape[1]==ncols))
    # Set values to nodata
    outimg_mm[...] = nodata
    
    outimg_shp = (outimg_mm.shape[0],1,outimg_mm.shape[2])
    del outimg_mm


    # Run jobs in parallel
    rayargs = {'_temp_dir': args.ray_temp_dir, 'ignore_reinit_error': True, 'include_dashboard': False}
    if args.num_cores != -1:
        rayargs['num_cpus'] = args.num_cores
    else:
        import multiprocessing
        rayargs['num_cpus'] = multiprocessing.cpu_count() - 1
    ray.init(**rayargs)
    img_mm_id = ray.put(img_mm.copy())
    abscf_id = ray.put(abscf)

    jobs = [mf_one_column.remote(col,img_mm_id, bgminsamp, outimg_shp, abscf_id, args) for col in np.arange(ncols)]
    
    rreturn = [ray.get(jid) for jid in jobs]
    outimg_mm = outimg.open_memmap(interleave='source',writable=True)
    for ret in rreturn:
        if ret[0] is not None:
            #outimg_mm[:, ret[1],-1] = np.squeeze(ret[0])
            outimg_mm[:, ret[1],:] = ret[0][:,0,:]

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

@ray.remote
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


@ray.remote
def mf_one_column(col, img_mm, bgminsamp, outimg_mm_shape, abscf, args):


    logging.basicConfig(format='%(levelname)s:%(asctime)s ||| %(message)s', level=args.loglevel,
                        filename=args.logfile, datefmt='%Y-%m-%d,%H:%M:%S')

    logging.debug(f'Col: {col}')
    reject = args.reject
    modelname = args.model

    # alphas for leave-one-out cross validation shrinkage
    if args.model == 'looshrinkage':
        astep,aminexp,amaxexp = 0.05,-10.0,0.0
        alphas=(10.0 ** np.arange(aminexp,amaxexp+astep,astep))
        nll=np.zeros(len(alphas))
        
    rdn = np.float64(img_mm[...,col].copy())
    use = np.where(np.all(np.logical_and(np.isfinite(rdn), rdn > -0.05), axis=1))[0]
    nuse = len(use)
    
    outimg_mm = np.zeros((outimg_mm_shape))
    mf_mc = np.zeros((rdn.shape[0],args.n_mc))

    if nuse < 10:
        return None, None
    
    np.random.seed(13)
    for _mc in range(args.n_mc):
        
        perm = np.random.permutation(len(use))
        subset_size = int(args.mc_bag_fraction*len(use))
        cov_subset = use[perm[:subset_size]]
        
        
        if modelname == 'empirical':
            modelfit = lambda I_zm: cov(I_zm)
        elif modelname == 'looshrinkage':
            # optionally use the full zero mean column as a regularizer
            modelfit = lambda I_zm: looshrinkage(I_zm,alphas,nll,subset_size,I_reg=[])
                        
        try:                            
            Icol_model = modelfit(rdn[cov_subset,:] - np.mean(rdn[cov_subset,:],axis=0))
            if modelname=='looshrinkage':
                C, alphaidx = Icol_model
                Cinv = inv(C)
            elif modelname=='empirical':
                Cinv = inv(Icol_model)
            else:
                Cinv = Icol_model
                
        except np.linalg.LinAlgError:
            logging.warn('singular matrix. skipping this column mode.')
            outimg_mm[subset,0,-1] = 0
            return None, None

        #logging.debug(f'{col}, {_mc}: {C[0,:5]}')
        # Classical matched filter
        target = abscf.copy() * np.mean(rdn[cov_subset,:],axis=0)
        normalizer = target.dot(Cinv).dot(target.T)
        mf = ((rdn[use,:] - np.mean(rdn[cov_subset,:], axis=0)).dot(Cinv).dot(target.T)) / normalizer
        #logging.debug(f'{col}, {_mc}: {mf[:5]}')
        
        mf_mc[use,_mc] = mf * ppmscaling
    
    #outimg_mm[use,0,-1] = np.mean(mf_mc[use,:], axis=1)
    outimg_mm[use,0,:] = mf_mc[use,:]                  

    colmu = np.mean(outimg_mm[use,0,:],axis=0)
    logging.debug(f'Column {col} mean: {colmu}')
    return outimg_mm, col




if __name__ == '__main__':
    main()
    ray.shutdown()




