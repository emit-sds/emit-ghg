#! /usr/bin/env python
#
#  Copyright 2025 California Institute of Technology
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
# Authors: Brian D. Bue

import sys,os
import numpy as np
import pylab as pl

from parallel_mf import savgol

dmax = 3
ppm_scaling = 100000.0

def plot_zmw_rdntgt(npzpath):
    fid = os.path.split(npzpath)[1].split('_')[0]
    npz = np.load(npzpath)
    loc = npz['loc'] # (i,j) location of this pixel in the fid
    rdn = npz['rdn'].squeeze() # radiance pixel
    tgt = npz['cmftgt'].squeeze() # abscf for this fid
    wvl = npz['cmfwvl'] # abscf / radiance wavelengths
    
    I_mu = npz['I_mu'].squeeze() # mean of column j radiances
    S_inv = npz['S_inv'] # inverse standarization matrix = sqrtm(inv(covj))

    rdnzm = rdn-I_mu
    rdnzmw = np.dot(S_inv,rdnzm) # whitened radiance pixel (p_w)
    
    tgtzm = tgt*I_mu
    tgtzmw = np.dot(S_inv,tgtzm) # whitened ghg target (t_w)

    
    figrows,figcols = 2,1 
    figscale,figpad = 3.5,0.05 
    figaspect = 2
    figsize = figcols*figaspect*figscale, figrows*figscale
    fig,ax  = pl.subplots(figrows,figcols,figsize=figsize,sharex=True,
                          sharey=False)
    figbuf  = dict(bottom=2*figpad,top=1-figpad,left=figpad,right=1-figpad,
    	           hspace=figpad,wspace=figpad)
    
    c = ['r','violet','c']

    ymaxtgt = ymaxrdn = -np.inf
    drdnzmw = rdnzmw.copy()
    dtgtzmw = tgtzmw.copy()
    for d in range(dmax):
        if d>0:
            drdnzmw = savgol(rdnzmw,deriv=d)
            dtgtzmw = savgol(tgtzmw,deriv=d)

        # drdnxtgt = mf numer = elementwise product (p_w * t_w) 
        drdnxtgt = drdnzmw * dtgtzmw
        # dtgtxtgt = mf denom = elementwise product (t_w * t_w) 
        dtgtxtgt = dtgtzmw * dtgtzmw 

        dretr = drdnxtgt.sum()/dtgtxtgt.sum() * ppm_scaling

        ax[0].plot(wvl,drdnxtgt,label=f'd={d} ({dretr:.1f} ppm-m)',
                   c=c[d], zorder=dmax-d)
        ax[1].plot(wvl,dtgtxtgt,c=c[d],zorder=dmax-d)

        ymaxrdn = max(ymaxrdn,np.abs(drdnxtgt).max())
        ymaxtgt = max(ymaxtgt,np.abs(dtgtxtgt).max())

    ax[0].set_ylim(-ymaxrdn,ymaxrdn)
    ax[1].set_ylim(-ymaxtgt,ymaxtgt)
    ax[1].set_xlabel('Wavelength (nm)')
    ax[0].set_ylabel('${\\bf p}_w$ * ${\\bf t}_w$')
    ax[1].set_ylabel('${\\bf t}_w$ * ${\\bf t}_w$')
    ax[0].legend(loc='best',fontsize='small')
    
    pl.subplots_adjust(**figbuf)
    pl.tight_layout() 
    
    pl.show()
    pl.close(fig)        
        
if __name__ == '__main__':
    plot_zmw_rdntgt(sys.argv[1])        
           
