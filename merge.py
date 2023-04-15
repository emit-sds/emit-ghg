import argparse

import os
import scipy
import numpy as np
import json
import glob
import datetime

class SerialEncoder(json.JSONEncoder):
    """Encoder for json to help ensure json objects can be passed to the workflow manager.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super(SerialEncoder, self).default(obj)

def main(input_args=None):
    parser = argparse.ArgumentParser(description="merge jsons")
    parser.add_argument('input_dir', type=str,  metavar='INPUT_DIR', help='input directory')   
    args = parser.parse_args(input_args)

    source_files = sorted(glob.glob(os.path.join(args.input_dir, 'emit*ch4_plumes.json')))

    first_file = source_files.pop(0)

    outdict = json.load(open(first_file,'r'))

    plume_count = 0 
    for sf in source_files:
        loc = json.load(open(sf,'r'))
        for lf in loc['features']:
            lf['properties']['plume_complex_count'] = plume_count + 1
            plume_count +=1
        #for _feat, feat in enumerate(loc['features']):
        #    feat['properties']['style'] = {'color': 'green', 'opacity': 1, 'weight': 2, 'fillOpacity': 0, 'maxZoom': 10, 'minZoom': 0}
        #    #del feat['vis_style']
        #    loc[_feat] = feat
            
        outdict['features'].extend(loc['features'])
    outdict['crs']['properties']['latest_update'] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    
    with open(os.path.join(args.input_dir, 'combined_plume_metadata.json'), 'w') as fout:
        fout.write(json.dumps(outdict, cls=SerialEncoder, indent=2, sort_keys=True)) 

 
if __name__ == '__main__':
    main()

