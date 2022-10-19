# Brian D. Bue
# David R. Thompson
# Philip G. Brodrick (parallelization only)
# Copyright 2017, by the California Institute of Technology. ALL RIGHTS 
# RESERVED. United States Government Sponsorship acknowledged. Any commercial 
# use must be negotiated with the Office of Technology Transfer at the 
# California Institute of Technology.  This software is controlled under the 
# U.S. Export Regulations and may not be released to foreign persons without 
# export authorization, e.g., a license, license exception or exemption.




from __future__ import division, print_function

import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)
warn = warnings.warn

import argparse
from spectral.io import envi

from os import makedirs
from os.path import join as pathjoin, exists as pathexists
import scipy
import numpy as np

from sklearn.cluster import MiniBatchKMeans
import ray

modelname='looshrinkage' # 'empirical' 

ncalrows = 200
ppmscaling = 100000.0

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
    return scipy.linagl.eig(A,**kwargs)

def det(A,**kwargs):
    kwargs.setdefault('overwrite_a',False)
    kwargs.setdefault('check_finite',False)    
    return scipy.linalg.det(A,**kwargs)

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
            warn('looshrinkage encountered a LinAlgError')

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
def mf_one_column(col, img_mm, bgmodes, pcadim, bgminsamp, outimg_mm_shape, bgimg_mm_shape, abscf, active):

    outimg_mm = np.zeros((outimg_mm_shape))
    if savebgmeta:
        bgimg_mm = np.zeros((outimg_mm_shape))
    else:
        bgimg_mm = None


    Icol_full=img_mm[:,active[0]-1:active[1],col]
    use = useidx(Icol_full)
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
                print('flagged outlier cluster %d (%d samples)'%(l,bgulabn[i]))
                bglabels[lmask] = -l
                bgulab[i] = -l                                    
            bgcounts.append("%d: %d"%(l,bgulabn[i]))

            if savebgmeta:
                bgimg_mm[use[lmask],0,0] = bgulabn[i]

        print('bg cluster counts:',', '.join(bgcounts))
        if (bgulab<0).all():
            warn('all clusters rejected, proceeding without rejection (beware!)')
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
            print('singular matrix. skipping this column mode.')
            outimg_mm[use[kmask],0,-1] = 0
            return None, None, None

        # Classical matched filter
        Icol_ki = Icol_ki-mu # = fully-sampled column mode
        target = abscf.copy()
        target = target-mu if reflectance else target*mu
        normalizer = target.dot(Cinv).dot(target.T)
        mf = (Icol_ki.dot(Cinv).dot(target.T)) / normalizer

        if reflectance:
            outimg_mm[use[kmask],0,-1] = mf 
        else:
            outimg_mm[use[kmask],0,-1] = mf*ppmscaling

    colmu = outimg_mm[use[bglabels>=0],0,-1].mean()
    print('Column %i mean: %e'%(col,colmu))
    return outimg_mm, bgimg_mm, col




if __name__ == '__main__':



    parser = argparse.ArgumentParser(description="Robust MF")
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='verbose output')    
    parser.add_argument('-k', '--kmodes', type=int, default=1,
                        help='number of columnwise modes (k-means clusters)')
    parser.add_argument('-r', '--reject', action='store_true',
                        help='enable multimodal covariance outlier rejection')
    parser.add_argument('-f', '--full', action='store_true', 
                        help='regularize multimodal estimates with the full column covariariance')    
    parser.add_argument('--rgb_bands', default='40,28,15',
                        help='comma-separated list of RGB channels') #changed for EMIT
    parser.add_argument('-m', '--metadata', action='store_true', 
                        help='save metadata image')    
    parser.add_argument('-R', '--reflectance', action='store_true',
                        help='reflectance signature')
    parser.add_argument('-M', '--model', type=str, default='looshrinkage',
                        help='model name (looshrinkage (default)|empirical)')    
    parser.add_argument('-n', '--num_cores', type=int, default=-1,
                        help='number of cores (-1 (default))')    
    parser.add_argument('--ray_temp_dir', type=str, default=None,
                        help='ray temp directory (None (default))')    
    parser.add_argument('input', type=str,  metavar='INPUT',
                        help='path to input image')   
    parser.add_argument('library', type=str,  metavar='LIBRARY',
                        help='path to target library file')
    parser.add_argument('output', type=str,  metavar='OUTPUT',
                        help='path for output image (mf ch4 ppm)')    
    
    args = parser.parse_args()
    bgmodes = args.kmodes
    reject = args.reject
    regfull = args.full
    reflectance = args.reflectance
    savebgmeta = args.metadata
    modelname = args.model
    pcadim = 6
    
    infile = args.input
    infilehdr = infile.replace('.img','.hdr')
    outfile = args.output 
    outfilehdr = outfile+'.hdr'

    # define active channels (AVIRIS-NG clipped radiances) wrt target gas + measurement units
    if reflectance and 'ch4' in args.library:
      active = [5,420] 
    elif 'ch4' in args.library:
      active = [236,284]#[351,422] #changed for EMIT
    elif 'co2' in args.library:
      active = [207,263]#[309,391] #changed for EMIT
    else:
      print('could not set active range')
      sys.exit(0)

    # columnwise spectral averaging function
    colavgfn = np.mean

    # want bg modes to have at least as many samples as 120% x (# features)
    bgminsamp = int((active[1]-active[0])*1.2)
    bgmodel = 'unimodal' if bgmodes==1 else 'multimodal'


    print('started processing input file: "%s"'%str(infile))

    img = envi.open(infilehdr,image=infile)
    img_mm = img.open_memmap(interleave='source',writeable=False)
    nrows,nbands,ncols = img_mm.shape

    outmeta = img.metadata
    outmeta['lines'] = nrows
    outmeta['data type'] = np2envitype(np.float64)
    
    rgb_bands = args.rgb_bands
    rgb_bands = [] if rgb_bands=='[]' else [int(x) for x in rgb_bands.split(',')]
    if len(rgb_bands)==3:
        outmeta['bands'] = 4
        if 'co2' in args.library:
            outmeta['band names'] = ['Red Radiance (uW/nm/sr/cm2)',
                                     'Green Radiance (uW/nm/sr/cm2)',
                                     'Blue Radiance (uW/nm/sr/cm2)',
                                     'CO2 Absorption (ppm x m)']
        else:
            outmeta['band names'] = ['Red Radiance (uW/nm/sr/cm2)',
                                     'Green Radiance (uW/nm/sr/cm2)',
                                     'Blue Radiance (uW/nm/sr/cm2)',
                                     'CH4 Absorption (ppm x m)']
    elif len(rgb_bands)==0:
        outmeta['bands'] = 1
        outmeta['band names'] = ['CH4 Absorption (ppm x m)']
    else:
        raise Exception('invalid value of rgb_bands argument: %s'%args.rgb_bands)
        
    outmeta['interleave'] = 'bip'    
    for kwarg in ['smoothing factors','wavelength','wavelength units','fwhm']:
        outmeta.pop(kwarg,None)
        
    nodata = float(outmeta.get('data ignore value',-9999))
    if nodata > 0:
        raise Exception('nodata value=%f > 0, values will not be masked'%nodata)

    # load the gas spectrum
    libdata = np.float64(np.loadtxt(args.library))
    abscf=libdata[active[0]-1:active[1],2]
    
    # alphas for leave-one-out cross validation shrinkage
    if modelname == 'looshrinkage':
        astep,aminexp,amaxexp = 0.05,-10.0,0.0
        alphas=(10.0 ** np.arange(aminexp,amaxexp+astep,astep))
        nll=np.zeros(len(alphas))
        
    modelparms  = 'modelname={modelname}, bgmodel={bgmodel}'
    if bgmodes > 1:
        modelparms += ', bgmodes={bgmodes}, pcadim={pcadim}, reject={reject}'
        if modelname == 'looshrinkage':
            modelparms += ', regfull={regfull}'

    if modelname == 'looshrinkage':
        modelparms += ', aminexp={aminexp}, amaxexp={amaxexp}, astep={astep}'

    modelparms += ', reflectance={reflectance}, active_bands={active}'    

    outdict = locals()
    outdict.update(globals())
    outmeta['model parameters'] = '{ %s }'%(modelparms.format(**outdict))

    outimg = envi.create_image(outfilehdr,outmeta,force=True,ext='')
    outimg_mm = outimg.open_memmap(interleave='source',writable=True)
    assert((outimg_mm.shape[0]==nrows) & (outimg_mm.shape[1]==ncols))
    
    # outimage mf values = nodata by default
    outimg_mm[:,:,-1] = nodata
    
    if savebgmeta:
        # output image of bgster membership labels per column
        bgfile=outfile+'_bg'
        bgfilehdr=bgfile+'.hdr'
        bgmeta = outmeta
        bgmeta['bands'] = 2
        bgmeta['data type'] = np2envitype(np.uint16)
        bgmeta['num alphas'] = len(alphas)
        bgmeta['alphas'] = '{%s}'%(str(alphas)[1:-1])
        bgmeta['band names'] = '{cluster_count, alpha_index}'
        bgimg = envi.create_image(bgfilehdr,bgmeta,force=True,ext='')
        bgimg_mm = bgimg.open_memmap(interleave='source',writable=True)

    # exclude nonfinite + negative spectra in covariance estimates
    useidx = lambda Icolz: np.where(((~(Icolz<0)) & np.isfinite(Icolz)).all(axis=1))[0]

    if args.num_cores == 1:
        a = 1
    else:
        rayargs = {'_temp_dir': args.ray_temp_dir}
        if args.num_cores != -1:
            rayargs['num_cpus'] = args.num_cores
        ray.init(**rayargs)

    img_mm_id = ray.put(img_mm.copy())
    abscf_id = ray.put(abscf)

    outimg_shp = (outimg_mm.shape[0],1,outimg_mm.shape[2])
    if savebgmeta:
        bgimg_shp = (bgimg_mm.shape[0],1,bgimg_mm.shape[2])
    else:
        bgimg_shp = None

    jobs = [mf_one_column.remote(col,img_mm_id, bgmodes, pcadim, bgminsamp, outimg_shp, bgimg_shp, abscf_id, active) for col in np.arange(ncols)]
    
    rreturn = [ray.get(jid) for jid in jobs]
    for ret in rreturn:
        if ret[0] is not None:
            outimg_mm[:, ret[2],-1] = np.squeeze(ret[0])[:,-1]
            if savebgmeta:
                bgimg_mm[:, ret[2] ,-1] = np.squeeze(ret[1])

    if len(rgb_bands)==3:
        print('copying rgb bands to outfile')    
        outimg_mm[:,:,0] = img_mm[:,rgb_bands[0],:]
        outimg_mm[:,:,1] = img_mm[:,rgb_bands[1],:]
        outimg_mm[:,:,2] = img_mm[:,rgb_bands[2],:]

    print('done')

