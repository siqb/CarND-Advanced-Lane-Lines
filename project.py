import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import collections

def finding_corners(fname):

    """
    
    Finding corners
    
    """

    # prepare object points
    nx = 8#TODO: enter the number of inside corners in x
    ny = 6#TODO: enter the number of inside corners in y
    
    # Make a list of calibration images
    fname = 'calibration_test.png'
    img = cv2.imread(fname)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        plt.imshow(img)
    return ret, corners

def cal_undistort(img, objpoints, imgpoints):

    """
    
    Correcting for distortion
    
    """

    print("Undistorting image")
    # Use cv2.calibrateCamera() and cv2.undistort()
    # Camera calibration, given object points, image points, and the shape of the grayscale image:
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # Undistorting a test image:
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    #undist = np.copy(img)  # Delete this line
    #plt.imshow(img)
    #plt.imshow(undist)
    return undist

def undistort(img, mtx, dist):
    # Undistorting a test image:
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


 # Define a function that takes an image, gradient orientation,
 # and threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):

    """
    
    Applying sobel
    
    """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def corners_unwarp(img, nx, ny, mtx, dist):

    """
    
    Undistort and transform
    
    Define a function that takes an image, number of x and y points, 
    camera matrix and distortion coefficients
    
    """

    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100 # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                    [img_size[0]-offset, img_size[1]-offset], 
                                    [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

        # Return the resulting image and matrix
        return warped, M

def calibrate_cameras():
    print("Calibrating cameras")
    objpoints = []
    imgpoints = [] 
    undistortable = []
    objp = np.zeros((6*9,3),np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    images = glob.glob('camera_cal/calibration*.jpg')
    for image in images:
        print("Calibrating image", image)
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)
            #undistortable.append(img)
            #cv2.drawChessboardCorners(img, (8, 6), corners, ret)
            #plt.imshow(img)
            #plt.show()
            
        # Use cv2.calibrateCamera() and cv2.undistort()
        # Camera calibration, given object points, image points, and the shape of the grayscale image:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        # Undistorting a test image:
        undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    
    #for image in undistortable:
    #    print("undist")
    #    plt.imshow(cal_undistort(image, objpoints,imgpoints))
    #    plt.show()
    #
    #images = glob.glob('test_images/*.jpg')
    #for image in images:
    #    print("undist")
    #    img = cv2.imread(image)
    #    plt.imshow(cal_undistort(img, objpoints,imgpoints))
    #    plt.show()


    return mtx, dist 

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, line_side):
        self.line_side = line_side
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        #self.recent_xfitted = [] 
        self.recent_xfitted = collections.deque(maxlen=10)
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        #collections.deque(maxlen=10)None 
        self.radius_of_curvature = collections.deque(maxlen=10)
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        #self.allx = None  
        self.allx = collections.deque(maxlen=10)  
        #y values for detected line pixels
        #self.ally = None
        self.ally = collections.deque(maxlen=10)
        self.N_SAMPLES = 10

        self.average_xfitted = 0
        self.lane_inds = collections.deque(maxlen=30)
        self.avg_lane_inds = None
        #self.avg_allx = collections.deque(maxlen=10)
        #self.avg_ally = collections.deque(maxlen=10)
        self.avg_allx = None 
        self.avg_ally = None 


        self.avg_curvature = None

    def calculate_average_fit(self):
        # Average fit of left line
        sum_fits = np.zeros_like(self.recent_xfitted[-1])
        for fit in self.recent_xfitted:
            sum_fits += fit
        self.bestx = sum_left_fits/float(len(self.recent_xfitted))

    def calculate_average_fit2(self):
        # Average fit of left line
        self.average_xfitted = sum(self.recent_xfitted)//len(self.recent_xfitted)
        #self.avg_allx = sum(self.allx)//len(self.allx)
        #self.avg_ally = sum(self.allx)//len(self.ally)
        #print("lane inds", self.lane_inds)
        #print("sum lane inds", sum(self.lane_inds))
        #self.avg_lane_inds = sum(self.lane_inds)//len(self.lane_inds)

    
    def save_current_fit(self,fit,fitx):

        self.current_fit = fit

        if len(self.recent_xfitted) < self.N_SAMPLES: 
            self.recent_xfitted.append(fit)
        else:
            self.recent_xfitted = [fit]

    #def get_curvature(self, img, left_lane_inds, right_lane_inds):
    #def get_curvature(self, img, left_lane_inds):
    def get_curvature(self, img):
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    
        ploty = np.linspace(0, 719, num=720)# to cover same y-range as image    
        
        # Choose Y of radius of curvature
        # Max Y value (bottom of the image)
        y_eval = np.max(ploty)
        
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Again, extract left and right line pixel positions
        #left_lane_inds = self.allx 
        #left_lane_inds = self.average_xfitted 
        #left_lane_inds = self.avg_allx 
        #left_lane_inds = self.avg_lane_inds 

        leftx = nonzerox[self.lane_inds[-1]]
        lefty = nonzeroy[self.lane_inds[-1]] 

        #leftx = self.avg_allx
        #lefty = self.avg_ally
        
        #leftx = np.array(self.allx)
        #lefty = np.array(self.ally)

        print("my leftx", leftx)

        #rightx = nonzerox[right_lane_inds]
        #righty = nonzeroy[right_lane_inds]
        
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        #right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        

        #left_curve_fitx = (left_fit_cr[0]*y_eval*ym_per_pix)**2 + left_fit_cr[1]*yeval*

        #right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        #rturn the curvature values in metres
        #return left_curverad, right_curverad
        self.radius_of_curvature.append(left_curverad)
        self.avg_curvature = sum(self.radius_of_curvature)/len(self.radius_of_curvature)




        return self.avg_curvature

        """
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        print(left_curverad, 'm', right_curverad, 'm')
        # Example values: 632.1 m    626.2 m
        """
    
    def find_lines(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        if self.line_side == "left":
            leftx_base = np.argmax(histogram[:midpoint])
        elif self.line_side == "right":
            leftx_base = np.argmax(histogram[midpoint:]) + midpoint
            #rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        #rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        #right_lane_inds = []
        
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            #win_xright_low = rightx_current - margin
            #win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2) 
            #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            #(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            #good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            #(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            #right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            #if len(good_right_inds) > minpix:        
            #    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        #right_lane_inds = np.concatenate(right_lane_inds)
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]

        print("your leftx", leftx)


        lefty = nonzeroy[left_lane_inds] 
        #rightx = nonzerox[right_lane_inds]
        #righty = nonzeroy[right_lane_inds] 
        
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        #right_fit = np.polyfit(righty, rightx, 2)
    
        """ 
        
        VISUALIZATIONS
        
        """
    
    
        # Generate x and y values for plotting
        # numpy.linspace(start, stop, num = 50, endpoint = True, retstep = False, dtype = None)
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Draw the fit lines
        for i in range(0,720):
            out_img[i][int(round(left_fitx[i]))]=[100,100,100]
        #plt.imshow(out_img)
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        #plt.xlim(0, 1280)
        #plt.ylim(720, 0)
    
        #self.allx = nonzerox[left_lane_inds]
        #self.ally = nonzeroy[left_lane_inds]

        self.allx.append(nonzerox[left_lane_inds].astype(np.float))
        self.ally.append(nonzeroy[left_lane_inds].astype(np.float))

        self.lane_inds.append(left_lane_inds)

        self.recent_xfitted.append(left_fitx)
        #return left_fit, right_fit, out_img

        return left_fit, out_img
    
    #def find_lines_in_margin(self, binary_warped, left_fit, right_fit):
    def find_lines_in_margin(self, binary_warped, left_fit):
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
        left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        
        #right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
        #right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
        #right_fit[1]*nonzeroy + right_fit[2] + margin)))  
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        #rightx = nonzerox[right_lane_inds]
        #righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        #right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                      ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        #right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        #right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
        #                              ploty])))])
        #right_line_pts = np.hstack((right_line_window1, right_line_window2))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        for i in range(0,720):
            result[i][int(round(left_fitx[i]))]=[100,100,100]
        #plt.imshow(result)
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        #plt.xlim(0, 1280)
        #plt.ylim(720, 0)

        self.allx = leftx
        self.ally = lefty

        #self.save_current_fit(left_fit, left_fitx)
        #self.calculate_average_fit()
        
        return result


def transform(img):
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

def unwarp(img):
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
    unwarped = cv2.warpPerspective(img, M_inv, img_size)
    #warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return unwarped

def mask(img):
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


def create_color_binary(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    print("Creating color binary of image")
    img = np.copy(img)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    # L channel (luminosity) will help avoid shadows and portions with dark pixels on the road.
    l_channel = hls[:,:,1]
    # S channel (saturation) of HLS will help detect white & yellow lines
    s_channel = hls[:,:,2]
    # Thresholding on R & G channels will help detect yellow lanes (yellow is a mix of red and green)

    #R & G channel [200:255] respectively for yellow lanes
    #S channel [140:255] for white lanes
    #L channel [140:255] for avoid shadows

    # Sobel x
    # Good explanation - https://www.tutorialspoint.com/dip/sobel_operator.htm
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx)) # scale sobel output so that irrespective of image format (i.e. jpg, png, etc.), we get the same output.
        
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))*255
    
    return color_binary

class MyVideoProcessor(object):

    # constructor function
    def __init__(self):
        # values of the last n fits of the line
        self.past_frames_left = []
        self.past_frames_right = []  
        #polynomial coefficients averaged over the last n iterations
        self.best_fit_left = None 
        self.best_fit_right = None 
        # Lane line instantiations
        self.left_line = Line("left")
        self.right_line = Line("right")
        # Misc global variables
        self.frame_counter = 0
        self.LIMIT = 60


        self.vehicle_pos = collections.deque(maxlen=50)
        self.avg_vehicle_pos = None

    def get_vehicle_position(self, image):
        
        # Center col of image gives position of camera (and hence the car).
        camera_position = image.shape[1]/2
 
        xm_per_pix = 3.7/700
    
        # Center of lane is diff between predicted lane lines at a position closest to the car. 
        # Image height is 720 pixels so pixel 720 is closet to the car
        #lane_center = (right_x_predictions[719] + left_x_predictions[719])/2
        #lane_center = (self.right_line.allx[-1][719] + self.left_line.allx[-1][719])/2
        lane_center = (self.right_line.recent_xfitted[-1][719] + self.left_line.recent_xfitted[-1][719])/2
    
        print("lane centrer", lane_center)
        print("camera_position", camera_position)
        # Offset of car from the lane’s center
        center_offset_pixels = (camera_position - lane_center)*xm_per_pix
        self.vehicle_pos.append(center_offset_pixels)
        self.avg_vehicle_pos = sum(self.vehicle_pos)/len(self.vehicle_pos)
        return center_offset_pixels

    def pipeline_function(self, img):
        print("#####Entering main pipeline for frame#####")
        img = undistort(img, g_mtx, g_dist)
        img_binary = mask(img)
        img_tx = transform(img_binary)
        
        #if self.frame_counter <= self.LIMIT:
        if True:
            left_fit, img_left = self.left_line.find_lines(img_tx)
            right_fit, img_right = self.right_line.find_lines(img_tx)
            #img_left = self.left_line.find_lines(img_tx)
            #img_right = self.right_line.find_lines(img_tx)
        else:
            #img = line.find_lines_in_margin(img_tx,self.past_frames_left[-1],self.past_frames_right[-1])
            img_left = self.left_line.find_lines_in_margin(img_tx,self.past_frames_left[-1])
            img_right = self.right_line.find_lines_in_margin(img_tx,self.past_frames_right[-1])

        self.left_line.calculate_average_fit2()
        self.right_line.calculate_average_fit2()

        img_debug = img_left + img_right
        self.frame_counter += 1

        #self.get_average_fit() 
        
        #left_curvature, right_curvature = get_curvature(img_tx, left_lane_inds, right_lane_inds)
        left_curvature = self.left_line.get_curvature(img_tx)
        right_curvature = self.right_line.get_curvature(img_tx)

        vehicle_pos = self.get_vehicle_position(img_tx)


        #top_left = [0, int(round(self.left_line.recent_xfitted[-1][0]))]
        #bottom_left = [719, int(round(self.left_line.recent_xfitted[-1][719]))]
        #top_right = [0, int(round(self.right_line.recent_xfitted[-1][0]))]
        #bottom_right = [719, int(round(self.right_line.recent_xfitted[-1][719]))]

        left_vertices = []
        for i in range(0,20):
            left_vertices.append([int(round(self.left_line.recent_xfitted[-1][i*719//20])),i*719//20])

        right_vertices = []
        for i in range(0,20):
            right_vertices.append([int(round(self.right_line.recent_xfitted[-1][i*719//20])),i*719//20])

        vertices = np.array([left_vertices + right_vertices[::-1]])
            
        #top_left = [int(round(self.left_line.recent_xfitted[-1][0])),0]
        #middle_left = [int(round(self.left_line.recent_xfitted[-1][0])),719//2]
        #bottom_left = [int(round(self.left_line.recent_xfitted[-1][719])),719]

        #top_right = [int(round(self.right_line.recent_xfitted[-1][0])),0]
        #middle_right = [int(round(self.right_line.recent_xfitted[-1][0])),719//2]
        #bottom_right = [int(round(self.right_line.recent_xfitted[-1][719])),719]
        
        #vertices = np.array([[top_left, top_right, middle_right, bottom_right, bottom_left, middle_left]])

        poly_img = np.zeros_like(img)
        poly_img = cv2.flip(poly_img,1)

        cv2.fillPoly(poly_img, [vertices], [0,255, 0])
        
        #top_left = (0, int(round(self.left_line.recent_xfitted[-1][0])))
        #bottom_right = (719, int(round(self.right_line.recent_xfitted[-1][719])))
        #cv2.rectangle(img,top_left,bottom_right,(0,255,0), 2)


        poly_img = unwarp(poly_img)
        img = cv2.addWeighted(img, 1, poly_img, 0.3, 0)
        
        left_curve_text = "Left Curvature = " + str(round(left_curvature,2)) + "m"
        cv2.putText(img, left_curve_text, (100,100), cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),lineType=cv2.LINE_AA)
        right_curve_text = "Right Curvature = " + str(round(right_curvature,2)) + "m"
        cv2.putText(img, right_curve_text, (100,150), cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),lineType=cv2.LINE_AA)
    
        offset_text = "Offset = " + str(round(vehicle_pos,2)) + "m"
        cv2.putText(img, offset_text, (100,200), cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),lineType=cv2.LINE_AA)

        print("#####Done processing frame#####")
        return img

if __name__ == "__main__":
   

    vid_processor_obj = MyVideoProcessor()

    #g_mtx, g_dist = calibrate_cameras()
    g_mtx = np.array([
             [  1.15396093e+03,   0.00000000e+00,   6.69705357e+02],
             [  0.00000000e+00,   1.14802496e+03,   3.85656234e+02],
             [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]
            ])
    g_dist = np.array([[ -2.41017956e-01,  -5.30721173e-02,  -1.15810355e-03,  -1.28318856e-04, 2.67125290e-02]])

    print("Camera calibration complete!")
    print("mtx:",str(g_mtx))
    print("dist:",str(g_dist))

    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip
    from IPython.display import HTML
    
    mode = "test" 
    if mode == "test":
        clip1 = VideoFileClip("project_video.mp4")
        output_clip = clip1.fl_image(vid_processor_obj.pipeline_function)
        output_clip.preview()
    
    elif mode != "test":
        video_output = 'output.mp4'
        clip1 = VideoFileClip("project_video.mp4")
        output_clip = clip1.fl_image(vid_processor_obj.pipeline_function)
        output_clip.write_videofile(video_output, audio=False)



