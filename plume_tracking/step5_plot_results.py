# David R Thompson
# David R Thompson

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
import numpy as np
import pylab as plt
import pandas as pd

fig, ax1 = plt.subplots()
k=2

# noise estimate for LIDAR
a = 0.5

# Read the Excel file into a DataFrame
df = pd.read_excel('data/windspeed_estimates.xlsx')

for field, color in [('LIDAR 40m', '#4444ff'),
                     ('10m wind', '#4444ff'),
                     ('LIDAR 100m','#4444ff')]:

    # Fit a GP 
    gpk = ConstantKernel(1.0) * RBF(50.0)
    gp40m = GaussianProcessRegressor(kernel=gpk, alpha=a,normalize_y = True)
    x = df['Time index'].to_numpy()[:,np.newaxis]
    y = df[field].to_numpy()
    use = np.logical_and(np.all(np.isfinite(x),axis=1),np.isfinite(y))
    gp40m.fit(x[use,:],y[use])
    y_pred, y_sigma = gp40m.predict(x[use,:], return_std=True)

    # Plot the results
    plt.plot(x[use,:],y_pred,color=color)
    plt.fill_between(x[use,:].ravel(), y_pred - k*y_sigma, y_pred + k*y_sigma,
                     alpha=0.1, color=color, label='±2 std. dev.')

plt.text(15,3.75,'LIDAR\n40 m', color='#4444ff')
plt.text(30,2,'In Situ\n10 m', color='#4444ff')
plt.text(2,4.5,'LIDAR\n100 m', color='#4444ff')



# ERA5
plt.text(100,2.8,'ERA5',color='#448844')
plt.plot(x,np.ones_like(x)*3.15,color='#448844')

# HRRR
plt.text(100,3.6,'HRRR',color='#448844')
plt.plot(x,np.ones_like(x)*3.446246676,color='#448844')

# Metered value
factor = 37 # scaling factor relating windspeed to flux (via IME method)
metered_value = 150
plt.text(45,4.15,'Metered',color='k')
ival = np.logical_and(x>=48,x<=66) 
plt.plot(x[ival],np.ones_like(x[ival])*metered_value/factor,color='k')

# Remote observations
plt.plot(60,4.48,'o',color='#bb4444')
plt.plot(66,3.46,'o',color='#bb4444')
plt.text(50,3.7,'Remote',color='#bb4444')

# Set up the plot & axes
lo = 0
hi = 8
plt.xlim(0,120)
plt.ylim([lo,hi])
plt.grid(False)
plt.box(True)
plt.xlabel('UTC Time')
plt.ylabel('Wind or Plume Velocity (m s$^{-1}$)')
ax2 = ax1.twinx()
plt.xticks([0,20,40,60,80,100,120],['18:31:45','18:33:25','18:35:05','18:36:45','18:38:25','18:40:05','18:41:45'])
plt.plot([0,120],[lo*factor,hi*factor],color='w',alpha=0)
plt.ylim(lo*factor,hi*factor)
plt.ylabel('Source Emission (kg hr$^{-1}$)')
plt.savefig('output/timeseries_revised.pdf')