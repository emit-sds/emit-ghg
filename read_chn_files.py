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

def load_chn(infile, coszen, return_specrad = False):
    """Load a '.chn' output file and parse critical coefficient vectors. 

        These are:
            wl      - wavelength vector
            sol_irr - solar irradiance
            sphalb  - spherical sky albedo at surface
            transm  - diffuse and direct irradiance along the 
                        sun-ground-sensor path
            transup - transmission along the ground-sensor path only 

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            Be careful with these! They are to be used only by the modtran_tir functions because
            MODTRAN must be run with a reflectivity of 1 for them to be used in the RTM defined
            in radiative_transfer.py.
            thermal_upwelling - atmospheric path radiance
            thermal_downwelling - sky-integrated thermal path radiance reflected off the ground
                                and back into the sensor.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        We parse them one wavelength at a time."""
    with open(infile) as f:
        sols, transms, sphalbs, wls, rhoatms, transups, spec_rads = [], [], [], [], [], [], []
        rdnatms = []
        centers, fwhms=[], []
        thermal_upwellings, thermal_downwellings = [], []
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i < 5:
                continue
            toks = line.strip().split(' ')
            toks = re.findall(r"[\S]+", line.strip())
            wl, wid = float(toks[0]), float(toks[8])  # nm
            solar_irr = float(toks[18]) * 1e6 * \
                np.pi / wid / coszen  # uW/nm/sr/cm2
            rdnatm = float(toks[4]) * 1e6  # uW/nm/sr/cm2
            rhoatm = rdnatm * np.pi / (solar_irr * coszen)
            #sphalb = float(toks[23])
            sphalb = 1
            transm = float(toks[22]) + float(toks[21])
            transup = float(toks[24])

            spec_rad = float(toks[4])

            # Be careful with these! See note in function comments above
            thermal_emission = float(toks[11])
            thermal_scatter = float(toks[12])
            thermal_upwelling = (thermal_emission + thermal_scatter) / wid * 1e6  # uW/nm/sr/cm2

            # Be careful with these! See note in function comments above
            grnd_rflt = float(toks[16])
            thermal_downwelling = grnd_rflt / wid * 1e6  # uW/nm/sr/cm2
            center=float(toks[-5])
            fwhm=float(toks[-2])
           

            sols.append(solar_irr)
            transms.append(transm)
            sphalbs.append(sphalb)
            rhoatms.append(rhoatm)
            transups.append(transup)
            rdnatms.append(rdnatm)
            centers.append(center)
            fwhms.append(fwhm)

            spec_rads.append(spec_rad)

            thermal_upwellings.append(thermal_upwelling)
            thermal_downwellings.append(thermal_downwelling)

            wls.append(wl)

    params = [np.array(i) for i in [wls, sols, rhoatms, transms, sphalbs, transups, thermal_upwellings, thermal_downwellings, spec_rads]]

    return tuple(params), np.array(wls),np.array(rdnatms), np.array(centers), np.array(fwhms)

def change_temperature_profile(input_modtran_json_filename, output_modtran_json_filename, deltaT_K, tropopause_km = 17.):
    with open(input_modtran_json_filename) as f:
        js_in = json.load(f)
    
    nprof = len(js_in['MODTRAN'][0]['MODTRANINPUT']['ATMOSPHERE']['PROFILES'])
    for i in range(nprof):
        if js_in['MODTRAN'][0]['MODTRANINPUT']['ATMOSPHERE']['PROFILES'][i]['TYPE'] == 'PROF_TEMPERATURE':
            prof_temperature = np.array(js_in['MODTRAN'][0]['MODTRANINPUT']['ATMOSPHERE']['PROFILES'][i]['PROFILE'])

        if js_in['MODTRAN'][0]['MODTRANINPUT']['ATMOSPHERE']['PROFILES'][i]['TYPE'] == 'PROF_ALTITUDE':
            prof_altitude = np.array(js_in['MODTRAN'][0]['MODTRANINPUT']['ATMOSPHERE']['PROFILES'][i]['PROFILE'])

    new_prof_temperature = np.where(prof_altitude <= tropopause_km, prof_temperature + deltaT_K, prof_temperature)

    for i in range(nprof):
        if js_in['MODTRAN'][0]['MODTRANINPUT']['ATMOSPHERE']['PROFILES'][i]['TYPE'] == 'PROF_TEMPERATURE':
            js_in['MODTRAN'][0]['MODTRANINPUT']['ATMOSPHERE']['PROFILES'][i]['PROFILE'] = new_prof_temperature.tolist()
    

    with open(output_modtran_json_filename, 'w') as f:
        json.dump(js_in, f, sort_keys = True, indent = 4)


  
    