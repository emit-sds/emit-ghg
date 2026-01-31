#  Five degrees lookup table
import numpy as np
import os 
import read_chn_files
import scipy.ndimage
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Or 'MacOSX'

invertflag=1    # 1: convert zenith angle from 120-180 to 0-60 degrees
useexist=1  #  1: use existing LUT; 0: create new LUT from .chn files
actualpath=0 # 1: calculate the actual path length that light travels through the atmosphere, given the solar zenith angle and the sensor zenith angle

def get_5deg_lookup_index(zenith, ground, water, solarz,conc):
    idx = np.asarray([[zenith],[ground],[water],[solarz],[conc]])
   # idx = np.asarray([zenith, ground, water, solarz, conc])
    return idx

def spline_5deg_lookup(grid_data, zenith, ground, water, solarz, conc):
    order=1
    coords = get_5deg_lookup_index(
        zenith=zenith, ground=ground, water=water, solarz=solarz, conc=conc)
    coords_fractional_part, coords_whole_part = np.modf(coords)
   # print(coords_whole_part)
    coords_near_slice = tuple((slice(int(cc.item()), int(cc.item()+2)) for cc in coords_whole_part))
    near_grid_data = grid_data[coords_near_slice]
    new_coord = np.concatenate((coords_fractional_part * np.ones((1, near_grid_data.shape[-1])),
                                np.arange(near_grid_data.shape[-1])[None, :]), axis=0)
    if order == 1:
        lookup = scipy.ndimage.map_coordinates(near_grid_data, coordinates=new_coord, order=1, mode='nearest')
    elif order == 3:
        lookup = np.asarray([scipy.ndimage.map_coordinates(
            im, coordinates=coords_fractional_part, order=order, mode='nearest') for im in np.moveaxis(near_grid_data, 5, 0)])
    return lookup.squeeze()

def generate_library(gas_concentration_vals, zenith, ground, water, solarz, grid):
    rads = np.empty((len(gas_concentration_vals), grid.shape[-1]))
    for i, ppmm in enumerate(gas_concentration_vals):
        rads[i, :] = spline_5deg_lookup(
            grid, zenith=zenith, ground=ground, water=water, solarz=solarz, conc=ppmm)
    return rads

# Generate LUT using .chn files
def create_4d_matrix(num):
    #path = '/Users/xiang/Desktop/data/c21'
    current_folder = os.getcwd()
    path = os.path.join(current_folder, 'chn_files')
    all_r=np.zeros((num, num, num, num, num,285))

    z_ind=0
    for z in np.linspace(120,180,num):
      z_ind=z_ind+1
      g_ind=0
      for g in np.linspace(0,3,num):
       g_ind=g_ind+1
       w_ind=0
       for w in np.linspace(0,6,num):
        w_ind=w_ind+1
        s_ind=0
        for s in np.linspace(0,60,num):
            s_ind=s_ind+1  
            c_ind=0 
            for c in np.linspace(0,120,num):
                 c_ind=c_ind+1
                 
                 base_name=f'z{z_ind}_g{g_ind}_w{w_ind}_s{s_ind}_c{c_ind}'
                 fn=base_name+'.chn'
                 filename= os.path.join(path, fn)
                 
                 if os.path.isfile(filename):  # radiance is already resampled in .chn files
                    t, wl, r, center, fwhm = read_chn_files.load_chn(filename, 1)
                    #logr = np.log(r, out=np.zeros_like(wl), where=r > 0)
                    all_r[z_ind-1][g_ind-1][w_ind-1][s_ind-1][c_ind-1] = r
                 else:
                    print('Missed file')
    return all_r, center

def find_ind(num_table,value_min,value_max,value): # convert value of parameter to its index in LUT
    ind=(value-value_min)/(value_max-value_min)*(num_table-1)
    return ind


num=11  # number of lines (parameter values) in the plot
num_table=5 # number of values for each parameter in the LUT

if (os.path.exists('all_r.npy')) & useexist == 1:   # If LUT exists, load the data from the file
    all_r = np.load('all_r.npy') # LUT
    center = np.load('center.npy') # corresponding wavelength
else:   # If LUT doesn't exist, generate LUT and save it to the file
    print('Generating new lookup table')
    all_r, center = create_4d_matrix(num_table)
    np.save('all_r.npy', all_r)
    np.save('center.npy', center)

smaller_range=1
maxz, minz, maxg, ming, maxw, minw, maxs, mins = 180, 120, 3, 0, 6, 0, 60, 0  # range of sensor zenith angle, ground altitude, water vapor and solar zenith angle
SCALING = 1e5 
concentrations_ind=range(1,4,1)   
c_list=np.linspace(0,120,num_table)   # Attendiont: make it consistant with modify_json.py used to generate LUT
concentrations = c_list[concentrations_ind]
#cl = [c * 0.5 * 1000 for c in concentrations]  # convert concentration to concentration length

# Generate plots for each of the four parameters—zenith, ground altitude, water content, and solar zenith angle.
# Each plot shows how the spectrum varies with one specific parameter, while keeping the remaining parameters fixed at their median value.
for para in [0,1,2,3]:
    plt.figure(dpi=150)
    
    if para==0: # sensor zenith angle
        g=1.5
        w=3
        s=30
        g_ind=find_ind(num_table,ming,maxg, g)
        w_ind=find_ind(num_table,minw,maxw, w)
        s_ind = find_ind(num_table, mins, maxs, s)
        para_list=np.linspace(minz,maxz,num)

    if para==1: # ground altitude
        z=150
        w=3
        s=30
        z_ind = find_ind(num_table, minz, maxz, z)
        w_ind=find_ind(num_table,minw, maxw, w)
        s_ind = find_ind(num_table, mins, maxs, s)
        para_list=np.linspace(ming, maxg, num)

    if para==2: # w
        z=150
        g=1.5
        s=30
        z_ind = find_ind(num_table, minz, maxz, z)
        g_ind=find_ind(num_table,ming,maxg, g)
        s_ind=find_ind(num_table,mins,maxs, s)
        para_list=np.linspace(minw, maxw, num)

    if para==3: # sensor zenith angle
        z=150
        g=1.5
        w=3
        z_ind = find_ind(num_table, minz, maxz, z)
        g_ind=find_ind(num_table,ming,maxg,g)
        w_ind=find_ind(num_table,minw,maxw,w)
        para_list=np.linspace(mins,maxs,num)

    spectra_matrix = np.empty((len(para_list), len(center)))
    j=0

    if (para == 0) & (invertflag == 1):
        para_list = para_list[::-1]
        para_list2 = 180 - para_list
    else:
        para_list2 = para_list


    for i in para_list:
        if para==0:
         z_ind = find_ind(num_table, minz, maxz, i)
         z=i
        if para==1:
         g_ind=find_ind(num_table,ming,maxg,i)
         g=i
        if para==2:
         w_ind=find_ind(num_table,minw,maxw,i) 
         w=i
        if para == 3:
         s_ind=find_ind(num_table,mins,maxs,i)
         s=i
         
        rads= generate_library(concentrations_ind, z_ind, g_ind, w_ind, s_ind, all_r)
        logr = np.log(rads, out=np.zeros_like(rads), where=rads > 0)
        
        layer_thickness=0.5*1000  
        if actualpath==1: # calculate the actual path length that light travels through the atmosphere, given the solar zenith angle and the sensor zenith angle
            path_length=layer_thickness/np.cos(s*np.pi/180)+layer_thickness/np.cos((180-z)*np.pi/180)
        else:
            path_length=layer_thickness
        cl = [c * path_length for c in concentrations]  # Multiply each element in the list

        slope, _, _, _ = np.linalg.lstsq(np.stack((np.ones_like(cl), cl)).T, logr, rcond=None)
        spectrum = slope[1, :] * SCALING
        spectra_matrix[j, :] = spectrum
        j=j+1
        
        cmap = plt.get_cmap('Blues')
        minc=min(para_list2)
        minc=minc-0.3 #increase the shade of the lightest line
        maxc=max(para_list2)
        color = cmap((para_list2[j-1]-minc)/ (maxc-minc))
        plt.plot(center, spectrum, label=f'Line {j}', c=color)
        plt.xlabel('Wavelength')
        plt.ylabel(r'$dlog(L)/dl_{\mathrm{CH4}}$')
        if smaller_range==1: # plot methane absorption region
           plt.xlim(2100,2500)

    degree_symbol = "\u00B0"
    names = [f"Sensor zenith angle ({degree_symbol})", "Ground altitude (km)", "Water vapor (g)",
             f"Solar zenith angle ({degree_symbol})"]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=minc, vmax=maxc))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.1)
    cbar.set_label(f'{names[para]}')    
   # plt.ylim(-0.1,1e-5)
   # plt.title('New Lookup Table')

    plt.tight_layout()
    plt.show()