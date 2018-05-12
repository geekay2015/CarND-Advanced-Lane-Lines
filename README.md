## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, my goal was to write a software pipeline to identify the lane boundaries in a video, with a detailed writeup of the project. Please Check out the- 

**Project Writeup**
[MyProjectWritup](CarND-Advanced-Lane-Lines/AdvancedLaneFinding.md) for this project

**Project Output Video**
https://youtu.be/EOi2E8rn2NQ

<a href="https://www.youtube.com/watch?v=EOi2E8rn2NQ&feature=youtu.be
" target="_blank"><img src="http://img.youtube.com/vi/5ZKbpNY-rok/0.jpg" 
alt="YouTube" width="240" height="180" border="10" /></a>


The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  
