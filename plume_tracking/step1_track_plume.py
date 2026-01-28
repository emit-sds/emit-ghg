# David R Thompson david.r.thompson@jpl.nasa.gov
# Optical plume tracking

import cv2
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


# Function to preprocess the channels as frames
def split_channels_as_frames(image_path):
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not read the image at {image_path}")

    # Split the image into R, G, and B channels
    b, g, r = cv2.split(image) 
    return [r,g,b]


separations = [0,15,12] # seconds between frames
GSD = 2.5
ycutoff = [30,550] # part of the image to analyze
source = [556,141] # source pixel location 
manual_mask = np.any(cv2.imread('data/manual_mask.png'),axis=2)


# Function to track features across frames using Optical Flow
def track_features_with_optical_flow(frames, prefix,thresh=150):
    
    # Convert the first frame to grayscale
    prev_gray = frames[0]
    output_frame = cv2.cvtColor(np.array(prev_gray/2+100,dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.imwrite("output/%s_frame_0.png"%(prefix),prev_gray)#output_frame)
    cv2.imwrite("output/%s_frame_0_gray.png"%(prefix),prev_gray)

    # Detect good features to track in the first frame
    features = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=5)

    for i in range(1, len(frames)):

        stationary_threshold = separations[i]/GSD*1.0 # 1 mps min speed for time separation 

        gpxk = ConstantKernel(1.0) * RBF(50.0)
        gpyk = ConstantKernel(1.0) * RBF(50.0)
        gpx = GaussianProcessRegressor(kernel=gpxk, alpha=10,normalize_y = True)
        gpy = GaussianProcessRegressor(kernel=gpyk, alpha=10,normalize_y = True)
        dists_to_source = []
        curr_gray = frames[i]

        # Calculate optical flow
        next_features, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features, 
                                                          None, winSize=(31, 31), maxLevel=5, 
                                                          criteria=(cv2.TERM_CRITERIA_EPS, 100, 0.01))

        # Select the points that have been tracked
        good_prev = features[st == 1]
        good_next = next_features[st == 1]

        # Calculate motion vectors and average velocity
        motion_vectors = good_next - good_prev
        
        blur_prev = cv2.blur(prev_gray, (5, 5))
        blur_next = cv2.blur(curr_gray, (5, 5))
        
        use = []
        for j in range(good_next.shape[0]):
            
            good = good_next[j]
            goodp = good_prev[j]
            try:
                if blur_next[int(good[1]),int(good[0])] > thresh and \
                    blur_prev[int(goodp[1]),int(goodp[0])] > thresh and \
                    np.sqrt(sum(pow(motion_vectors[j,:],2))) > stationary_threshold:
                    use.append(j)
                
            except IndexError:
                continue
        
        use = np.array(use,dtype=int)
        motion_vectors = motion_vectors[use,:]

        # Our estimate is the median
        median_velocity = np.median(motion_vectors, axis=0)

        # Bootstrap uncertainties
        bootstrap_medians = []
        nboot = 10000
        for iteration in range(nboot):
            samples = np.random.choice(np.arange(motion_vectors.shape[0]), 
                size=motion_vectors.shape[0])
            my_median = np.median(motion_vectors[samples,:], axis=0)
            bootstrap_medians.append(my_median)
        bootstrap_medians = np.array(bootstrap_medians) 
        np.savetxt('output/%s_frame_%i_boot.csv'%(prefix,i),bootstrap_medians,delimiter=',',fmt='%10.8f')
       
        print(f"Frame {i}: median velocity = {median_velocity} pixels/frame")
  
        gpx.fit(good_next, good_prev[:,0])
        gpy.fit(good_next, good_prev[:,1])
        gridx, gridy = np.meshgrid(np.arange(0,curr_gray.shape[1],15),
                               np.arange(0,curr_gray.shape[0],15))
        gridx = gridx.flatten()
        gridy = gridy.flatten()
        xpred = gpx.predict(np.c_[gridx,gridy])
        ypred = gpy.predict(np.c_[gridx,gridy])
                
        # Visualize the tracked points and motion vectors with GP
        resize_factor = 5
        large = cv2.resize(prev_gray,np.array([prev_gray.shape[1],prev_gray.shape[0]])*resize_factor)
        output_frame = cv2.cvtColor(np.array(155+large/3,dtype=np.uint8), cv2.COLOR_GRAY2BGR)
        for ny, nx, py, px in zip(ypred,xpred,gridy.flatten(), gridx.flatten()):
            
            dist_to_source = np.sqrt(pow(ny-source[1],2)+pow(nx-source[0],2))
            dists_to_source.append(dist_to_source)
            
            if blur_prev[int(py),int(px)] > thresh and \
                blur_next[int(ny),int(nx)] > thresh and py>ycutoff[0] and py<ycutoff[1] and \
                manual_mask[py,px]<1:
                cv2.arrowedLine(output_frame, (int(nx*resize_factor), int(ny*resize_factor)), 
                        (int(px*resize_factor), int(py*resize_factor)), (40,40,192), 12)
 
        cv2.imwrite("output/%s_frame_%i.png"%(prefix,i),output_frame)
        cv2.imwrite("output/%s_frame_%i_gray.png"%(prefix,i),curr_gray)
        
        # Now plot Lucas Kanade track
        output_frame = cv2.cvtColor(np.array(125+large/2,dtype=np.uint8), cv2.COLOR_GRAY2BGR)
        for p1, p2 in zip(good_prev[use], good_next[use]):
            x1, y1 = p1.ravel()
            x2, y2 = p2.ravel()
            cv2.arrowedLine(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), (192, 0, 0), 2)
            
        cv2.imwrite("output/%s_frame_%i_lk.png"%(prefix,i),output_frame)
        
        # Update the previous frame and features
        prev_gray = curr_gray.copy()
        features = good_next.reshape(-1, 1, 2)

        cv2.destroyAllWindows()
        
        print('Avg. distance to source:',np.mean(np.array(dists_to_source)) * GSD)
        
        
image_path = "data/AV320250126t183602_66989_CH4_stack_rgb.png"

# Load frames
frames = split_channels_as_frames(image_path)

# Track objects across frames
prefix = image_path.split('/')[-1].split('_')[0]
track_features_with_optical_flow(frames, prefix)
