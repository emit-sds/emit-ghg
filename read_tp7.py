from sys import platform
import os
import logging
from os.path import split
import re
import json
from copy import deepcopy
import scipy as s
from scipy.stats import norm as normal
from scipy.interpolate import interp1d
import numpy as np
import pdb

def load_data(infile, return_specrad = False):
 
    with open(infile) as f:
        rads, freqs = [], []
      
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i < 11:
                continue
            toks = line.strip().split(' ')
            
           # print(len(toks))
            if len(toks) >= 10:
                
                toks = re.findall(r"[\S]+", line.strip())
                freq, rad = float(toks[0]), float(toks[9])  # nm
                
                freqs.append(freq)
                rads.append(rad)

    return s.array(rads),s.array(freqs)



  
    