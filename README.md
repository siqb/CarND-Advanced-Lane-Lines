## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## How to Use

My implementaion requires PyGame to be installed. It can be installed as follows:

```pip install pygame```

After this, simply execute the file:

```python project.py```

A MoviePy window should pop up which looks like this:

[image7]: ./pics/launch.png "MoviePy"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `project.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

```python
    def undistort(self, img, mtx, dist):
        # Undistorting a test image:
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        return undist
```



#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a mask on my undistorted image to create a thresholded binary image. The idea here is that the HSV color space is really good at extracting yellow lines whereas the RGB color space is really good at extracting white lines. The algorithm is as follows:

1. Define the best thresholds for all three HSV channels for extracting yellow lane lines
2. Define the best thresholds for all three RGB channels for extracting white lane lines
3. Extract the thresholded yellow pixels from the HSV version of the image
4. Extract the thresholded white pixels from the RGB version of the image
5. Combine the detected pixels into a new RGB image
6. Convert the new RGB image to grayscale
7. Threshold the new grayscale image such that it contains only 1's and 0's

This code can be found in the ```mask()``` function in ```project.py```. Exact line numbers are subject to change but it can currently be found at line 497. The code is also reproduced here below:

```python
def mask(self,img):
    #Thresholds
    yellow_lower_filter = np.array([0, 100, 100])
    yellow_upper_filter = np.array([80, 255, 255])

    white_lower_filter = np.array([200, 200, 200])
    white_upper_filter = np.array([255, 255, 255])

    #yellow masking
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    yellow_mask = cv2.inRange(hsv, yellow_lower_filter, yellow_upper_filter)
    yellow_mask = cv2.bitwise_and(img, img, mask=yellow_mask)

    #white masking
    rgb = img
    white_mask = cv2.inRange(rgb, white_lower_filter, white_upper_filter)
    white_mask = cv2.bitwise_and(img, img, mask=white_mask)

    #combined masks
    combined_mask = cv2.addWeighted(white_mask, 1., yellow_mask, 1., 0.)

    #convert to binary image
    # Just convert to grayscale and then set a threshold for anything greater then 0 for that gray 
    gray_mask = cv2.cvtColor(combined_mask, cv2.COLOR_RGB2GRAY)
    binary = np.zeros_like(gray_mask)
    binary[(gray_mask > 0)] = 1
    return binary
```


Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.



The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
def transform(self,img):
    print("Perfoming perspective transform on image")
    w,h = 1280,720
    x,y = 0.5*w, 0.8*h
    src = np.float32([[200./1280*w,720./720*h],
                  [453./1280*w,547./720*h],
                  [835./1280*w,547./720*h],
                  [1100./1280*w,720./720*h]])
    dst = np.float32([[(w-x)/2.,h],
                  [(w-x)/2.,0.82*h],
                  [(w+x)/2.,0.82*h],
                  [(w+x)/2.,h]])    

    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    #Compute the inverse perspective transform:
    M_inv = cv2.getPerspectiveTransform(dst, src)
    #Warp an image using the perspective transform, M:
    warped = cv2.warpPerspective(img, M, img_size)
    #warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 320, 720      | 
| 453, 547      | 320, 590.4    |
| 835, 547      | 960, 590.4    |
| 1100, 720     | 960, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Here is the general concept to identifying lane lines:

1. Generate an undistorted, transformed, binary image
2. Use a sliding window approach to identify a best fit line to fit the lane line. Do this on the first frame only!
   1. Take a histogram of the bottom half of the image because it is closest to the car
   2. Find the peak of the histogram because this represents the lane line
   3. Apply a slide window updwards from this location and search for more non-zero pixels within the window
   4. Gather all of the detected line pixels into an array
   5. Fit a second order polynomial to these points - this polynomial represents your lane line!
3. For all subsequent frames, do not use the sliding window approach. Since we already have a general idea of where the lane should appear based on the last frame, search the new frame within a small margin on the previous best fit line. We could use the sliding window approach on each frame throughout the entire video but seearching within a margin is more computationally efficient. 
    1. Take the best fit line from the previous fram
    2. Find non-zero pixels in the vicinity of this previous best fit line
    3. Gather all of the detected line pixels into an array
    4. Fit a second order polynomial to these points - this polynomial represents your lane line!

This is the code in ```pipeline_main``` for calling the functions that identify lane lines. As you can see, it calls the less efficent ```find_lines``` only for the first frame. After that, it always calls ```find_lines_in_margin```.

```python
# Use sliding windows approach for first frame only
if self.frame_counter <= 0:
    img_left = self.left_line.find_lines(img_tx)
    img_right = self.right_line.find_lines(img_tx)
# After first frame, search within a margin of the previous best fit line
else:
    img_left = self.left_line.find_lines_in_margin(img_tx,self.left_line.recent_fitted[-1])
    img_right = self.right_line.find_lines_in_margin(img_tx,self.right_line.recent_fitted[-1])
self.frame_counter += 1
```
I will not reproduce the code for ```find_lines``` and ```find_lines_in_margin``` since it mostly came from the course lecture material with only minor modifications.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated radius of curvature of the lane in a function called ```get_curvature``` starting at line 266 (number subject to change) in `project.py`. This function is part of the```Line()``` class and is called on a per line basis. Code is reproduced here:

```python
def get_curvature(self, img):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image    

    # Choose Y of radius of curvature
    # Max Y value (bottom of the image)
    y_eval = np.max(ploty)

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Again, extract line pixel positions
    linex = nonzerox[self.lane_inds[-1]]
    liney = nonzeroy[self.lane_inds[-1]] 

    # Fit new polynomials to x,y in world space
    line_fit_cr = np.polyfit(liney*ym_per_pix, linex*xm_per_pix, 2)
    # Calculate the new radii of curvature
    line_curverad = ((1 + (2*line_fit_cr[0]*y_eval*ym_per_pix + line_fit_cr[1])**2)**1.5) / np.absolute(2*line_fit_cr[0])

    # Now our radius of curvature is in meters
    # Return the curvature values in metres
    self.radius_of_curvature.append(line_curverad)
    self.avg_curvature = sum(self.radius_of_curvature)/len(self.radius_of_curvature)
    return self.avg_curvature
```
The general idea here is:
1. Generate a new best fit line through the lane line pixels using a pixels to meters conversion
2. Calculate the radius of the curvature based on this formula:
3. Calculate a running average of the radius of curvature over a fixed number of frames so it isn't so jittery

I calculated the vehicle offset from the center of the lane in a function called ```get_vehicle_position``` starting at line 573 (number subject to change) in `project.py`. This function is part of the```MyVideoProcessor()``` class. Code is reproduced here:

```python
def get_vehicle_position(self, image):

    # Center col of image gives position of camera (and hence the car).
    camera_position = image.shape[1]/2
    xm_per_pix = 3.7/700

    # Center of lane is diff between predicted lane lines at a position closest to the car. 
    # Image height is 720 pixels so pixel 720 is closet to the car
    lane_center = (self.right_line.recent_xfitted[-1][719] + self.left_line.recent_xfitted[-1][719])/2

    # Offset of car from the lane’s center
    center_offset_pixels = (camera_position - lane_center)*xm_per_pix
    self.vehicle_pos.append(center_offset_pixels)
    self.avg_vehicle_pos = sum(self.vehicle_pos)/len(self.vehicle_pos)
    return center_offset_pixels
```
The general idea here is:

1. Assume that the center column of the image represents the camera position and hence that of the vehicle
2. Take the average of the left and right lane fit lines at the position closest to the car
3. Subtract the center of the lane from the position of the camera and convert from pixels to meters
4. Calculate a running average of the vehicle position over a fixed number of frames so it isn't so jittery

I got a consistent negative vehicle offset throughout the duration of the entire video clip. This indicates that the car was driving left of center for the entire duration. I visited the Udacity forums and learned that this is indeed the case! The magnitude of the offset is larger around curves than on straight portions of road which intuitively makes sense because drivers naturally tend to drift to left around curves - at least in the USA where the driver always sits on the left side of the vehicle. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The method I used to highlight the lane was:

1. Take the warped image which already has lane fit lines drawn onto it on frame by frame basis
2. Break the best fit lines for each lane into 20 segments and save the delineating pixel locations as vertices
3. Construct a rectangular polygon out of these vertices on its own image
4. Unwarp the polygon image
5. Combine the unwarped polygon image with the original image
6. Repeat for every frame

The code is in ```project.py``` from lines 402 - 417 (line numbers subject to change!) but is reproduced here:

```python
# Create a polygon to highlight the ego-lane
left_vertices = []
for i in range(0,20):
    left_vertices.append([int(round(self.left_line.recent_xfitted[-1][i*719//20])),i*719//20])

right_vertices = []
for i in range(0,20):
    right_vertices.append([int(round(self.right_line.recent_xfitted[-1][i*719//20])),i*719//20])
vertices = np.array([left_vertices + right_vertices[::-1]])
poly_img = np.zeros_like(img)
poly_img = cv2.flip(poly_img,1)
cv2.fillPoly(poly_img, [vertices], [0,255, 0])

# Unwarp polygon and project back onto original image
poly_img = self.transform(poly_img, operation="unwarp")
img = cv2.addWeighted(img, 1, poly_img, 0.3, 0)
```

The result is a very smooth green polygon which perfectly highlights the ego-lane throughout the entire video clip. It does not jitter and perfectly hugs the left and right lane lines in every single frame.

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

##### Fragile Color Mask

As with several of the other projects, my implementation makes huge assumptions about its ability to identify the lane lines based on color and lighting conditions. This will certainly break down under the right conditions. An improvement could be to dynamically change the color mask values based on some external indicator of road conditions. For example, in overcast conditions, the low threshold for detecting a given color may need to be lowered.

##### Urban Environments

I suspect that this kind of pipeline would also not work well in urban enviornments. As with other projects, the test video is a long stretch of relatively empty highway and therefore we are able to smooth out irregularities by averaging certain measurements across several frames. It is probably not reasonable to expect to be able to average over so many frames in a busy city atomosphere where so much can happen in the span of just a couple of frames. For example, a single sudden sharp right turn at an intersection would ruin the run averages collected over several frames. Of course there are ways to mitigate this (mapping, path planning, etc.) but the pipeline as it stands would not be sufficient.

##### Use Deep Learning

An improvement would be to implement a neural network just for the purpose of detecting lane lines. As we have seen from previous projects, neural networks are only as good as the data set you feed them but in my opinion, its still a less fragile approach than attempting to manually extract color information from lines on the fly. For example, I have personally seen lane lines where the paint is faded away or painted over for several continuous meters. This kind of pipeline would fail as soon as the lane line disappears but a deep learning approach could infer that the two non-continuous line segments are part of the same lane line.  



