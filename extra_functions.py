def finding_corners(fname):

    """

    Finding corners
    
    UNUSED FUNCTION

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
    
    UNUSED FUNCTION

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


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):

    """
    
    Applying sobel

    Define a function that takes an image, gradient orientation,
    and threshold min / max values.
    
    UNUSED FUNCTION

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

def hls_select(img, thresh=(0, 255)):

    """
    
    Select S-channel
    
    Define a function that thresholds the S-channel of HLS
    
    UNUSED FUNCTION

    """

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
    
    UNUSED FUNCTION
    
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

def create_color_binary(img, s_thresh=(170, 255), sx_thresh=(20, 100)):

    """
    
    Color binary

    UNUSED FUNCTION
    
    """
    
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
