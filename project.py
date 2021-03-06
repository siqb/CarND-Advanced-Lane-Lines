import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import collections
import pickle

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, line_side):
        self.line_side = line_side
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        #self.recent_xfitted = [] 
        self.recent_xfitted = collections.deque(maxlen=10)
        self.recent_fitted = collections.deque(maxlen=10)
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
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

        self.lane_inds = collections.deque(maxlen=30)
        self.avg_lane_inds = None
        self.avg_curvature = None

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
    
    def find_lines(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the line and right halves of the histogram
        # These will be the starting point for the lane lines
        midpoint = np.int(histogram.shape[0]//2)
        if self.line_side == "left":
            linex_base = np.argmax(histogram[:midpoint])
        elif self.line_side == "right":
            linex_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        linex_current = linex_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive lane pixel indices
        line_lane_inds = []
        
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xline_low = linex_current - margin
            win_xline_high = linex_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xline_low,win_y_low),(win_xline_high,win_y_high),
            (0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_line_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xline_low) &  (nonzerox < win_xline_high)).nonzero()[0]
            # Append these indices to the lists
            line_lane_inds.append(good_line_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_line_inds) > minpix:
                linex_current = np.int(np.mean(nonzerox[good_line_inds]))
        
        # Concatenate the arrays of indices
        line_lane_inds = np.concatenate(line_lane_inds)
        
        # Extract line pixel positions
        linex = nonzerox[line_lane_inds]
        liney = nonzeroy[line_lane_inds] 
        
        # Fit a second order polynomial to each
        line_fit = np.polyfit(liney, linex, 2)
    
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        line_fitx = line_fit[0]*ploty**2 + line_fit[1]*ploty + line_fit[2]
        
        out_img[nonzeroy[line_lane_inds], nonzerox[line_lane_inds]] = [255, 0, 0]

        # Draw the fit lines
        for i in range(0,720):
            out_img[i][int(round(line_fitx[i]))]=[100,100,100]

        # Uncomment to generate a static image
        #plt.imshow(out_img)
        #plt.plot(line_fitx, ploty, color='yellow')
        #plt.xlim(0, 1280)
        #plt.ylim(720, 0)
    
        self.lane_inds.append(line_lane_inds)
        self.recent_xfitted.append(line_fitx)
        self.recent_fitted.append(line_fit)

        return out_img
    
    def find_lines_in_margin(self, binary_warped, line_fit):
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        line_lane_inds = ((nonzerox > (line_fit[0]*(nonzeroy**2) + line_fit[1]*nonzeroy + 
        line_fit[2] - margin)) & (nonzerox < (line_fit[0]*(nonzeroy**2) + 
        line_fit[1]*nonzeroy + line_fit[2] + margin))) 
        
        # Again, extract lane line pixel positions
        linex = nonzerox[line_lane_inds]
        liney = nonzeroy[line_lane_inds] 

        # Fit a second order polynomial to each
        line_fit = np.polyfit(liney, linex, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        line_fitx = line_fit[0]*ploty**2 + line_fit[1]*ploty + line_fit[2]
        
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in lane line pixels
        out_img[nonzeroy[line_lane_inds], nonzerox[line_lane_inds]] = [255, 0, 0]
        
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        line_line_window1 = np.array([np.transpose(np.vstack([line_fitx-margin, ploty]))])
        line_line_window2 = np.array([np.flipud(np.transpose(np.vstack([line_fitx+margin, 
                                      ploty])))])
        line_line_pts = np.hstack((line_line_window1, line_line_window2))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([line_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        for i in range(0,720):
            result[i][int(round(line_fitx[i]))]=[100,100,100]

        # Uncomment to generate a static image
        #plt.imshow(result)
        #plt.plot(line_fitx, ploty, color='yellow')
        #plt.xlim(0, 1280)
        #plt.ylim(720, 0)

        self.lane_inds.append(line_lane_inds)
        self.recent_xfitted.append(line_fitx)
        self.recent_fitted.append(line_fit)
        return line_fit, out_img

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

    def calibrate_cameras(self):
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

        with open('camera_calibration_pickle.p','wb') as file_pi:
            pickle.dump((mtx, dist), file_pi)

    def undistort(self, img, mtx, dist):
        # Undistorting a test image:
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        return undist

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

    def transform(self,img,operation):
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
        if operation == "warp":
            M = cv2.getPerspectiveTransform(src, dst)
            transformed = cv2.warpPerspective(img, M, img_size)
        elif operation == "unwarp":
            #Compute the inverse perspective transform:
            M_inv = cv2.getPerspectiveTransform(dst, src)
            #Warp an image using the perspective transform, M:
            transformed = cv2.warpPerspective(img, M_inv, img_size)
        return transformed 

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

    def pipeline_function(self, img):

        print("#####Entering main pipeline for frame#####")

        # Undistort, create binary, and perspective transform image
        img = self.undistort(img, g_mtx, g_dist)
        img_binary = self.mask(img)
        img_tx = self.transform(img_binary, operation = "warp")
        
        # Use sliding windows approach for first frame only
        if self.frame_counter <= 0:
            img_left = self.left_line.find_lines(img_tx)
            img_right = self.right_line.find_lines(img_tx)
        # After first frame, search within a margin of the previous best fit line
        else:
            img_left = self.left_line.find_lines_in_margin(img_tx,self.left_line.recent_fitted[-1])
            img_right = self.right_line.find_lines_in_margin(img_tx,self.right_line.recent_fitted[-1])
        self.frame_counter += 1

        self.left_line.calculate_average_fit2()
        self.right_line.calculate_average_fit2()
        img_debug = img_left + img_right

        # Collect KPIs
        left_curvature = self.left_line.get_curvature(img_tx)
        right_curvature = self.right_line.get_curvature(img_tx)
        vehicle_pos = self.get_vehicle_position(img_tx)
        
        # Create a polygon to highlight the ego-lane
        left_vertices = []
        for i in range(0,20):
            left_vertices.append([int(round(self.left_line.recent_xfitted[-1][i*719//20])),i*719//20])

        right_vertices = []
        for i in range(0,20):
            right_vertices.append([int(round(self.right_line.recent_xfitted[-1][i*719//20])),i*719//20])
        vertices = np.array([left_vertices + right_vertices[::-1]])
        img_poly = np.zeros_like(img)
        img_poly = cv2.flip(img_poly,1)
        cv2.fillPoly(img_poly, [vertices], [0,255, 0])
        
        # Unwarp polygon and project back onto original image
        img_poly = self.transform(img_poly, operation="unwarp")
        img_final = np.copy(img)
        img_final = cv2.addWeighted(img, 1, img_poly, 0.3, 0)
        
        # Print KPIs on screen
        left_curve_text = "Left Curvature = " + str(round(left_curvature,2)) + "m"
        cv2.putText(img_final, left_curve_text, (100,100), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,(255,255,255),
                    lineType=cv2.LINE_AA)
        right_curve_text = "Right Curvature = " + str(round(right_curvature,2)) + "m"
        cv2.putText(img_final, right_curve_text, (100,150), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,(255,255,255),
                    lineType=cv2.LINE_AA)
        offset_text = "Offset = " + str(round(vehicle_pos,2)) + "m"
        cv2.putText(img_final, offset_text, (100,200), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,(255,255,255),lineType=cv2.LINE_AA)

        print("#####Done processing frame#####")

        # Uncomment to generate a static image
        #plt.imshow(img)
        #plt.imshow(img_binary)
        #plt.imshow(img_tx)
        #plt.imshow(img_debug)
        #plt.imshow(img_poly)
        #plt.imshow(img_final)
        
        return img_final

if __name__ == "__main__":
   
    vid_processor_obj = MyVideoProcessor()
    
    try:
        with open('camera_calibration_pickle.p','rb') as file_pi:
            g_mtx, g_dist = pickle.load(file_pi)
            print("Loaded camera calibration data!")
    except:
        print("Could not find camera calibration data!")
        vid_processor_obj.calibrate_cameras()
        with open('camera_calibration_pickle.p','rb') as file_pi:
            g_mtx, g_dist = pickle.load(file_pi)

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



