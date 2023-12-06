# Planar object pose estimation

<img src='images/result_readme.gif'></img>

The task is to detect and estimate the pose of the planar object on an image relative to the camera. Your goal is to estimate the 3D pose of the planar object in space and validate it with visualization of an imaginary coordinate system. 

## Description of solution

In this project, I used standard computer vision methods. I used SIFT to find key points and their descriptions in the images. SIFT (Scale-Invariant Feature Transform) allows you to detect features that are resistant to scale changes, rotations, and lighting. Next, I used BFM (Brute Force Matcher), an algorithm to find all possible pairs of keypoints between two sets. It is used to compare keypoint descriptors in two images. Next, we take the good matches and use them to find the homography matrix of our planar object. We do this using RANSAC. 

RANSAC (Random Sample Consensus) is an algorithm used to detect and extract outliers in a data set that may contain erroneous or random values. Its main goal is to select the set of points that best defines the model and exclude those points that fall outside the model as outliers.

Next, to estimate a planar object in 3D space, we use the SolvePnP function. This function solves the PnP (positioning and orientation) problem to estimate the position and orientation of an object in three dimensions. Also, for this function, we needed the camera matrix and distortion, which we got during the calibration of our camera, the project has the corresponding files