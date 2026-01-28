# David R Thompson

import numpy as np

def calc_plume_velocity(wind_vector_pixels, separation_s):
    aircraft_altitude_m = 5000
    aircraft_speed_mps = 100
    aircraft_distance_m = separation_s * aircraft_speed_mps 
    plume_height_m = 41.8
    layover_distance_m = plume_height_m * aircraft_distance_m / aircraft_altitude_m
    # layover is in first dimension
    apparent_projection_mps = np.array([layover_distance_m / separation_s, 0.0])
    pixel_size_m = np.array([2.5, 2.5])
    plume_velocity_mps = pixel_size_m * wind_vector_pixels  / separation_s - apparent_projection_mps
    scalar_plume_velocity_mps = np.sqrt(sum(pow(plume_velocity_mps,2)))
    return scalar_plume_velocity_mps

# Calculate estimate and bootstrap confidence intervals for pair 1
pair1_separation_s = 15 - 0
pair1_pixel_vector = np.array([12.1,-25.9])

pair1_velocity = calc_plume_velocity(pair1_pixel_vector, pair1_separation_s)

pair1_pixel_vector_samples = np.loadtxt('output/AV320250126t183602_frame_1_boot.csv', delimiter=',')
pair1_velocity_samples = np.array([calc_plume_velocity(q, pair1_separation_s) \
        for q in pair1_pixel_vector_samples])
pair1_pct2p5 = np.percentile(pair1_velocity_samples,2.5)
pair1_pct97p5 = np.percentile(pair1_velocity_samples,97.5)
print("Frames 1->2: %f meters per second (95%% confidence: %f to %f)"%(pair1_velocity, pair1_pct2p5, pair1_pct97p5))


# Calculate estimate and bootstrap confidence intervals for pair 2
pair2_separation_s = 27 - 15
pair2_pixel_vector = np.array([10.7,-15.2])

pair2_velocity = calc_plume_velocity(pair2_pixel_vector, pair2_separation_s)

pair2_pixel_vector_samples = np.loadtxt('output/AV320250126t183602_frame_2_boot.csv', delimiter=',')
pair2_velocity_samples = np.array([calc_plume_velocity(q, pair2_separation_s) \
        for q in pair2_pixel_vector_samples])
pair2_pct2p5 = np.percentile(pair2_velocity_samples,2.5)
pair2_pct97p5 = np.percentile(pair2_velocity_samples,97.5)
print("Frames 2->3: %f meters per second (95%% confidence: %f to %f)"%(pair2_velocity, pair2_pct2p5, pair2_pct97p5))

print("Average velocity: ",(pair1_velocity+pair2_velocity)/2.0)
