**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car]: ./output_images/processed_car.png
[car_hog]: ./output_images/processed_car_hog.png
[notcat]: ./output_images/processed_not_car.png
[notcat_hog]: ./output_images/processed_not_car_hog.png
[slide_multiple_windows]: ./output_images/processed_slide_multiple_windows.png
[processed_result]: ./output_images/processed_test1.png
[processed_result_heatmap]: ./output_imagesprocessed_test1_heatmap.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

All lesson functions with little changes are in `features_classifier.py` file.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car][car]
![car_hog][car_hog]
![notcat][notcat]
![notcat_hog][notcat_hog]

Then I explored `slide_window` and `search_windows` functions from `features_classifier` module with parameters xy_window and overlap

![slide_multiple_windows][slide_multiple_windows]

####2. Explain how you settled on your final choice of HOG parameters.

I played with HOG parameters a lot and ended up with following:

* color_space = 'YCrCb' #RGB, HSV, LUV, HLS, YUV, YCrCb
* orient = 9
* pixels_per_cell = 8
* cells_per_block = 2
* hog_channel = "ALL" #0, 1, 2, "ALL"
* spatial_size = (32, 32)
* histogram_bins = 32
* y_range = [400, 656]
* spatial_features = True
* hisogram_features = True
* hog_features = True
* svc = LinearSVC()
* heatmap_threshold = 1.5
* video_resolution = (720, 1280)
* heatmaps_to_change_average = 15

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using in `Train model` section in Jupyter notebook

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![slide_multiple_windows][slide_multiple_windows]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To optimize performance of clacifier, I'm processing image and then work with only regions that I'm interested in, instead of getting features of each region individually

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![processed_result][processed_result]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_processed.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Also I found that boxes are jumping sometimes and sometimes dissapears for a second. Then I decided to take on a count `heatmap_sum` and take average heatmap if I found more than a threshold in variable `heatmaps_to_change_average` also removing the oldest one to not track regions where car is no longer drives.

### Here are six frames and their corresponding heatmaps:

![processed_result][processed_result]
![processed_result_heatmap][processed_result_heatmap]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My model in general gives around 99.2% accuracy, but sometime it drops to 98.9, which I think is ok for now

I faced problems with RGB, so I converted to `RGB2YCrCb`. After that I found that if I resize image with `scale` = 1.5 I receive much better detection. 
Also I found that processing average heatmaps gives much more stable results.
Sometimes I detect cars on the driving oposite direction, I thought about limiting image processing on x, but then I decided to leave it as is, cause it might be useful later.
I still have some problems on dark regions, I think I can improve it later.

