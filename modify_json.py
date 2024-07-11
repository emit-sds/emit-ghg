import json
import numpy
import os

def np_encoder(object):
    if isinstance(object, numpy.generic):
        return object.item()

output_folder = "json_files"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
num=10
z_ind=0
for z in numpy.linspace(120,180,num):
  z_ind=z_ind+1
  g_ind=0
  for g in numpy.linspace(0,3,num):
   g_ind=g_ind+1
   w_ind=0
   for w in numpy.linspace(0,6,num):
    w_ind=w_ind+1
    c_ind=0 
    for c in numpy.linspace(0,120,num):
        c_ind=c_ind+1
        s_ind=0
        for s in numpy.linspace(0,60,num):
            s_ind=s_ind+1
        
            with open("phil_original.json", "r") as jsonFile:
                data = json.load(jsonFile)
            data["MODTRAN"][0]["MODTRANINPUT"]["GEOMETRY"]["H1ALT"] = 400
            
            data["MODTRAN"][0]["MODTRANINPUT"]["GEOMETRY"]["OBSZEN"] = z
            data["MODTRAN"][0]["MODTRANINPUT"]["SURFACE"]["GNDALT"] = g
            data["MODTRAN"][0]["MODTRANINPUT"]["ATMOSPHERE"]["H2OSTR"] = w
            data["MODTRAN"][0]["MODTRANINPUT"]["GEOMETRY"]["PARM2"] = s
            
            dist=data["MODTRAN"][0]["MODTRANINPUT"]["ATMOSPHERE"]["PROFILES"][0]["PROFILE"]
            dist[0]=g
            dist[1]=g+0.5
            dist[2]=g+0.5001
            
            new_elements = []
            new_inds = []
    
            for index, element in enumerate(dist[3:]):
              if element > dist[2]:
                new_elements.append(element)
                new_inds.append(index + 3)  
    
            new_dist = dist[:3] + new_elements
            data["MODTRAN"][0]["MODTRANINPUT"]["ATMOSPHERE"]["PROFILES"][0]["PROFILE"]=new_dist
            data["MODTRAN"][0]["MODTRANINPUT"]["ATMOSPHERE"]["NLAYERS"]=len(new_dist)
            
            conc=data["MODTRAN"][0]["MODTRANINPUT"]["ATMOSPHERE"]["PROFILES"][1]["PROFILE"]
            indices_of_new_dist= list(range(3))+new_inds
            conc[0:2]=[c,c]
            new_conc = [conc[index] for index in indices_of_new_dist]
            data["MODTRAN"][0]["MODTRANINPUT"]["ATMOSPHERE"]["PROFILES"][1]["PROFILE"]=new_conc
            
            
            base_name = 'z' + str(z_ind) + '_g' + str(g_ind) + '_w' + str(w_ind) + '_s' + str(s_ind) + '_c' + str(c_ind)
            fn=base_name+'.json'
            filename = os.path.join(output_folder, fn)
            data["MODTRAN"][0]["MODTRANINPUT"]["NAME"]=base_name
    
            with open(filename, "w") as jsonFile:
                json.dump(data, jsonFile, default=np_encoder,indent=4)
        
        
             
