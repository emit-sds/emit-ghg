#! /usr/bin/env python
#
#  Copyright 2023 California Institute of Technology
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
# Authors: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov

import argparse

import os
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

    for sf in source_files:
        loc = json.load(open(sf,'r'))
        outdict['features'].extend(loc['features'])
        #for _feat, feat in enumerate(loc['features']):
        #    feat['properties']['style'] = {'color': 'green', 'opacity': 1, 'weight': 2, 'fillOpacity': 0, 'maxZoom': 10, 'minZoom': 0}
        #    #del feat['vis_style']
        #    loc[_feat] = feat

    plume_count = 0 
    for lf in outdict['features']:
        if lf['geometry']['type'] == 'Polygon':
            lf['properties']['plume_complex_count'] = plume_count + 1
            plume_count +=1
        else:
            lf['properties']['plume_complex_count'] = plume_count
            
    outdict['crs']['properties']['latest_update'] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    
    with open(os.path.join(args.input_dir, 'combined_plume_metadata.json'), 'w') as fout:
        fout.write(json.dumps(outdict, cls=SerialEncoder, indent=2, sort_keys=True)) 

 
if __name__ == '__main__':
    main()

