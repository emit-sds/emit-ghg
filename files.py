import os



class Filenames():

    def __init__(self, output_base):

        self.target_file = f'{output_base}_target' #target
        self.mf_file = f'{output_base}_mf' #MF
        self.mf_uncert_file = f'{output_base}_mf_uncert' #Uncertainty
        self.mf_sens_file = f'{output_base}_sens' #Sensitivity
        self.flare_file = f'{output_base}_flares.json' #Flares
        self.mf_ort_file = f'{output_base}_mf_ort' #MF - ORT
        self.mf_ort_cog = f'{output_base}_mf_ort.tif' #MF - ORT
        self.mf_ort_ql = f'{output_base}_mf_ort.png' #MF - ORT
        self.mf_scaled_color_ort_file = f'{output_base}_mf_scaled_color_ort.tif' #MF - ORT - Scaled Color
        self.sens_ort_file = f'{output_base}_sens_ort' #Sensitivity ort
        self.sens_ort_cog = f'{output_base}_sens_ort.tif' #Sensitivity ort
        self.sens_scaled_color_ort_file = f'{output_base}_sens_scaled_color_ort.tif' #Sensitivity -  ORT - Scaled Color
        self.uncert_ort_file = f'{output_base}_uncert_ort' #Uncertainty ort
        self.uncert_ort_cog = f'{output_base}_uncert_ort.tif' #Uncertainty ort
        self.uncert_scaled_color_ort_file = f'{output_base}_uncert_scaled_color_ort.tif' #Uncertainty -  ORT - Scaled Color

class EMIT_DAAC_Filenames():

    def __init__(self, output_path, radiance_filename, gas_in, version = '002'):

        toks = os.path.splitext(os.path.basename(radiance_filename))[0].split('_')
        datetime, orbit, scene_number = toks[4:]
        fid = 'emit' + datetime

        output_base = os.path.join(output_path, fid)

        gas = gas_in.upper()

        self.target_file = f'{output_base}_target' #target
        self.mf_file = f'{output_base}_mf' #MF
        self.mf_uncert_file = f'{output_base}_mf_uncert' #Uncertainty
        self.mf_sens_file = f'{output_base}_sens' #Sensitivity
        self.flare_file = f'{output_base}_flares.json' #Flares

        self.mf_ort_file = f'{output_path}/EMIT_L2B_{gas}ENH_{version}_{datetime}_{orbit}_{scene_number}.tif' #MF - ORT

        self.mf_ort_cog = f'{output_base}_mf_ort.tif' #MF - ORT
        self.mf_ort_ql = f'{output_base}_mf_ort.png' #MF - ORT
        self.mf_scaled_color_ort_file = f'{output_base}_mf_scaled_color_ort.tif' #MF - ORT - Scaled Color

        self.sens_ort_file = f'{output_path}/EMIT_L2B_{gas}SENS_{version}_{datetime}_{orbit}_{scene_number}.tif' #SENS - ORT
        self.sens_ort_cog = f'{output_base}_sens_ort.tif' #Sensitivity ort
        self.sens_scaled_color_ort_file = f'{output_base}_sens_scaled_color_ort.tif' #Sensitivity -  ORT - Scaled Color

        self.uncert_ort_file = f'{output_path}/EMIT_L2B_{gas}UNCERT_{version}_{datetime}_{orbit}_{scene_number}.tif' #UNCERT - ORT
        self.uncert_ort_cog = f'{output_base}_uncert_ort.tif' #Uncertainty ort
        self.uncert_scaled_color_ort_file = f'{output_base}_uncert_scaled_color_ort.tif' #Uncertainty -  ORT - Scaled Color

class AV3_DAAC_Filenames():

    def __init__(self, output_path, radiance_filename, gas_in, hash='00000000'):

        toks = os.path.splitext(os.path.basename(radiance_filename))[0].split('_')
        fid = toks[0]
        scene_number = toks[1]

        output_base = os.path.join(output_path, fid)

        gas = gas_in.upper()

        self.target_file = f'{output_base}_target' #target
        self.mf_file = f'{output_base}_mf' #MF
        self.mf_uncert_file = f'{output_base}_mf_uncert' #Uncertainty
        self.mf_sens_file = f'{output_base}_sens' #Sensitivity
        self.flare_file = f'{output_base}_flares.json' #Flares

        self.mf_ort_file = f'{output_path}/{fid}_{scene_number}_L2B_GHG_{hash}_{gas}_ORT.tif' #MF - ORT

        self.mf_ort_cog = f'{output_base}_mf_ort.tif' #MF - ORT
        self.mf_ort_ql = f'{output_base}_mf_ort.png' #MF - ORT
        self.mf_scaled_color_ort_file = f'{output_base}_mf_scaled_color_ort.tif' #MF - ORT - Scaled Color

        self.sens_ort_file = f'{output_path}/{fid}_{scene_number}_L2B_GHG_{hash}_{gas}_SNS_ORT.tif' # SNS ORT
        self.sens_ort_cog = f'{output_base}_sens_ort.tif' #Sensitivity ort
        self.sens_scaled_color_ort_file = f'{output_base}_sens_scaled_color_ort.tif' #Sensitivity -  ORT - Scaled Color

        self.uncert_ort_file = f'{output_path}/{fid}_{scene_number}_L2B_GHG_{hash}_{gas}_UNC_ORT.tif' #UNC - ORT
        self.uncert_ort_cog = f'{output_base}_uncert_ort.tif' #Uncertainty ort
        self.uncert_scaled_color_ort_file = f'{output_base}_uncert_scaled_color_ort.tif' #Uncertainty -  ORT - Scaled Color