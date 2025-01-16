



class Filenames():

    def __init__(self, output_base):

        self.target_file = f'{output_base}_target', #target
        self.mf_file = f'{output_base}_mf', #MF
        self.mf_uncert_file = f'{output_base}_mf_uncert', #Uncertainty
        self.mf_sens_file = f'{output_base}_sens', #Sensitivity
        self.flare_file = f'{output_base}_flares.json', #Flares
        self.mf_ort_file = f'{output_base}_mf_ort', #MF - ORT
        self.mf_ort_cog = f'{output_base}_mf_ort.tif', #MF - ORT
        self.mf_scaled_color_ort_file = f'{output_base}_mf_scaled_color_ort.tif', #MF - ORT - Scaled Color
        self.sens_ort_file = f'{output_base}_sens_ort', #Sensitivity ort
        self.sens_ort_cog = f'{output_base}_sens_ort.tif', #Sensitivity ort
        self.sens_scaled_color_ort_file = f'{output_base}_sens_scaled_color_ort.tif', #Sensitivity -  ORT - Scaled Color
        self.uncert_ort_file = f'{output_base}_uncert_ort', #Uncertainty ort
        self.uncert_ort_cog = f'{output_base}_uncert_ort.tif', #Uncertainty ort
        self.uncert_scaled_color_ort_file = f'{output_base}_uncert_scaled_color_ort.tif' #Uncertainty -  ORT - Scaled Color