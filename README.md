<h1 align="center"> emit-ghg </h1>

Welcome to the EMIT GHG codebase.  This is research code to support point-source mapping from EMIT, and stands slightly outside of the main SDS - if you're looking for info on the full SDS, please see [the repository guide](https://github.jpl.nasa.gov/emit-sds/emit-main/wiki/Repository-Guide).

Please note that this is research code, made available as it is being developed and deployed in the interest of open science and open applications.  In particular, some poor coding practices like hardcoded paths are still in place, though should fade with time.  This code builds off of effort described in:

Elder, C. D., Thompson, D. R., Thorpe, A. K., Hanke, P., Walter Anthony, K. M., & Miller, C. E. (2020). Airborne mapping reveals emergent power law of arctic methane emissions. Geophysical Research Letters, 47(3), e2019GL085707.

Thorpe, A. K., Duren, R. M., Conley, S., Prasad, K. R., Bue, B. D., Yadav, V., ... & Miller, C. E. (2020). Methane emissions from underground gas storage in California. Environmental Research Letters, 15(4), 045005.

among others.

The main call for a particular scene is:

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


