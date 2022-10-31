<h1 align="center"> emit-ghg </h1>

Welcome to the EMIT GHG codebase.  This is research code to support point-source mapping from EMIT, and stands slightly outside of the main SDS - if you're looking for info on the full SDS, please see [the repository guide](https://github.jpl.nasa.gov/emit-sds/emit-main/wiki/Repository-Guide).

Please note that this is research code, made available as it is being developed and deployed in the interest of open science and open applications.  In particular, some poor coding practices like hardcoded paths are still in place, though should fade with time.  The main call for a particular scene is:

```
python ghg_process.py 
       radiance_file
       obs_file
       loc_file
       glt_file
       output_base
       --state_subs  
       --loglevel
       --logfile
```

This will generate both ch4 and co2 matched filter results, along with some scaling and visualization products to accomany each. 

This code uses a classical matched filter applied independently along each pushbroom column (Thompson et al., 2015, 2016; Frankenberg et al., 2016).  Signatures are calculated on a scene-specific basis to account for local water vapor, elevation and solar position as in Foote et al. (2020). Statistical control for surface reflectance is used as in Elder et al. (2020).  

Relevent references include, but are not limited to:

Thompson, D. R., Leifer, I., Bovensmann, H., Eastwood, M., Fladeland, M., Frankenberg, C., Gerilowski, K., Green, R.O., Kratwurst, S., Krings, T. and Luna, B., (2015). Real-time remote detection and measurement for airborne imaging spectroscopy: a case study with methane. Atmospheric Measurement Techniques, 8(10), pp.4383-4397.

Frankenberg, C., Thorpe, A.K., Thompson, D.R., Hulley, G., Kort, E.A., Vance, N., Borchardt, J., Krings, T., Gerilowski, K., Sweeney, C. and Conley, S., (2016). Airborne methane remote measurements reveal heavy-tail flux distribution in Four Corners region. Proceedings of the national academy of sciences, 113(35), pp.9734-9739.

Thompson, D.R., Thorpe, A.K., Frankenberg, C., Green, R.O., Duren, R., Guanter, L., Hollstein, A., Middleton, E., Ong, L. and Ungar, S., (2016). Space‐based remote imaging spectroscopy of the Aliso Canyon CH4 superemitter. Geophysical Research Letters, 43(12), pp.6571-6578.

Foote, M.D., Dennison, P.E., Thorpe, A.K., Thompson, D.R., Jongaramrungruang, S., Frankenberg, C. and Joshi, S.C., (2020). Fast and accurate retrieval of methane concentration from imaging spectrometer data using sparsity prior. IEEE Transactions on Geoscience and Remote Sensing, 58(9), pp.6480-6492.

Elder, C. D., Thompson, D. R., Thorpe, A. K., Hanke, P., Walter Anthony, K. M., & Miller, C. E. (2020). Airborne mapping reveals emergent power law of arctic methane emissions. Geophysical Research Letters, 47(3), e2019GL085707.

Thorpe, A. K., Duren, R. M., Conley, S., Prasad, K. R., Bue, B. D., Yadav, V., ... & Miller, C. E. (2020). Methane emissions from underground gas storage in California. Environmental Research Letters, 15(4), 045005.




