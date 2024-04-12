import math
import cv2 as cv
import numpy as np
import matlab.engine

from model import BoundingBox, ObjectDimensions


eng = matlab.engine.start_matlab()

def stereo_fusion(left_image, right_image):
    left_gray = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY)
    right_gray = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)

    stereo = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*4,  # Adjust as needed
        blockSize=5,
        P1=8*3*15**2,  # Adjust as needed
        P2=32*3*15**2,  # Adjust as needed
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2
    )


    disparity = stereo.compute(left_gray, right_gray).astype(np.float32)/16

    # Normalize and apply a color map
    disparity = cv.normalize(src=disparity, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    disparity = cv.applyColorMap(disparity, cv.COLORMAP_JET)

    # Normalize the image for representation
    # min = disparity.min()
    # max = disparity.max()
    # disparity = np.uint8(255 * (disparity - min) / (max - min))
    # disparity = cv.normalize(disparity_map, disparity_map, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    return disparity


async def testStereo():
    # Camera parameters to undistort and rectify images
    cv_file = cv.FileStorage()
    cv_file.open('stereoMap.xml', cv.FileStorage_READ)
    left_maps_x = cv_file.getNode('stereoMapL_x').mat()
    left_maps_y = cv_file.getNode('stereoMapL_y').mat()
    left_maps_x = cv_file.getNode('stereoMapR_x').mat()
    left_maps_y = cv_file.getNode('stereoMapR_y').mat()

    # Load stereo images
    img_left = cv.imread("images/coffeeL.png")
    img_right = cv.imread("images/coffeeR.png")

    # Rectify stereo images using the rectification maps
    rectified_left = cv.remap(img_left, left_maps_x, left_maps_y, cv.INTER_LINEAR)
    rectified_right = cv.remap(img_right, left_maps_x, left_maps_y, cv.INTER_LINEAR)

    cv.imwrite("images/rectified_left3.png", rectified_left)
    cv.imwrite("images/rectified_right3.png", rectified_right)

    left_image = cv.imread('images/coffeeL.png')
    right_image = cv.imread('images/coffeeR.png')
    result_image1 = stereo_fusion(left_image, right_image)

    left_image = cv.imread('images/rectified_left3.png')
    right_image = cv.imread('images/rectified_right3.png')
    result_image2 = stereo_fusion(left_image, right_image)

    cv.imwrite("images/disparity_map_before.png", result_image1)
    cv.imwrite("images/disparity_map_after.png", result_image2)

def testMatlab():
    eng.eval('load("stereoParams_10_04.mat")', nargout=0)
    eng.workspace['I1'] = eng.imread("images/nr_left.jpg");
    eng.workspace['I2'] = eng.imread("images/nr_right.jpg");
    t = eng.eval('rectifyStereoImages(I1, I2, stereoParams_10_04)', nargout=3)
    eng.imwrite(t[0], "images/rect_left.png", nargout=0);
    eng.imwrite(t[1], "images/rect_right.png", nargout=0);   

def rectImage(left, right):
    cv.imwrite("images/nonrect_left.png", left)
    cv.imwrite("images/nonrect_right.png", right)
    #eng = matlab.engine.start_matlab()
    eng.eval('load("stereoParams_11_04.mat")', nargout=0)
    eng.workspace['I1'] = eng.imread("images/nonrect_left.png");
    eng.workspace['I2'] = eng.imread("images/nonrect_right.png");
    t = eng.eval('rectifyStereoImages(I1, I2, stereoParams_11_04)', nargout=3)
    eng.imwrite(t[0], "images/rect_left.png", nargout=0);
    eng.imwrite(t[1], "images/rect_right.png", nargout=0);   


def testWLSStereo():
    left = cv.imread('images/rect_left.png')
    right = cv.imread('images/rect_right.png')
    # left = cv.imread('images/left_test.png')
    # right = cv.imread('images/right_test.png')
    wlsImage, disp = stereoWLS(left, right)
    # not needed
    cv.imwrite("images/wls.png", wlsImage)
    return wlsImage, disp, left, right

def stereoWLS(left_rect, right_rect):
    left_image = cv.cvtColor(left_rect, cv.COLOR_BGR2GRAY)
    right_image = cv.cvtColor(right_rect, cv.COLOR_BGR2GRAY)

    left_image = cv.medianBlur(left_image, 11)
    right_image = cv.medianBlur(right_image, 11)

    window_size = 1
    min_disp = 0
    nDispFactor = 5 #
    num_disp = 16*nDispFactor - min_disp
    left_matcher = cv.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
    lmbda = 8000 #
    sigma = 2.0 #
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher);
    left_disp = left_matcher.compute(left_image, right_image);
    right_disp = right_matcher.compute(right_image,left_image);

    # Now create DisparityWLSFilter
    wls_filter = cv.ximgproc.createDisparityWLSFilter(left_matcher);
    wls_filter.setLambda(lmbda);
    wls_filter.setSigmaColor(sigma);
    filtered_disp = wls_filter.filter(left_disp, left_image, disparity_map_right=right_disp);



    # Normalize and apply a color map
    filteredImg = cv.normalize(src=filtered_disp, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    filteredImg_colour = cv.applyColorMap(filteredImg, cv.COLORMAP_JET)
    return filteredImg_colour, filteredImg

def stereoFuse(left, right):
    rectImage(left, right)
    return testWLSStereo()

def getDimensionsBounding(left, right, bounding_box: BoundingBox):
    disparity_map_rgb, disparity_map = stereoWLS(left, right)
    baseline = 0.055  # meters
    Focal_Lengths = [614.0822, 614.1323]
    avgFocalLength = (Focal_Lengths[0] + Focal_Lengths[1]) / 2
    start_point = (bounding_box.x1, bounding_box.y1)
    end_point = (bounding_box.x2, bounding_box.y2)

    img = left.copy()

    # Draw the bounding box
    cv.rectangle(img, start_point, end_point, (0, 255, 0), 2)

    # Create a mask from the bounding box
    mask = np.zeros_like(disparity_map)
    cv.rectangle(mask, start_point, end_point, 255, -1)

    # Apply mask to disparity map
    masked_disparity = apply_mask(disparity_map, mask)

    # Store coordinates of bounding box in format (top-left, top-right, bottom-right, bottom-left)
    x_tl, y_tl = start_point
    x_br, y_br = end_point
    box_coordinates = [(x_tl, y_tl), (x_br, y_tl), (x_br, y_br), (x_tl, y_br)]

    non_zero_elements = masked_disparity[masked_disparity != 0]  # Extract non-zero elements
    #non_zero_elements = disparity_map
    disparity = non_zero_elements.mean()
    print(masked_disparity)
    disparity_min = non_zero_elements.min()
    disparity_max = non_zero_elements.max()
    disparity_diff = disparity_max - disparity_min
        
    # Condition for line being perpedincular to the camera or not
    if disparity_diff < 15:     
        distance = getDistance(disparity, avgFocalLength, baseline)
        width = getWidth(box_coordinates, distance, avgFocalLength)
        height = getHeight(box_coordinates, distance, avgFocalLength, disparity_diff)
        length_est = math.sqrt(width**2 + height**2)
            
    else:
            horizontal_sections = np.linspace(start_point[0], end_point[0], 6).astype(int)
            vertical_sections = np.linspace(start_point[1], end_point[1], 6).astype(int)

            section_widths = []
            section_heights = []
            total_width = 0
            total_height = 0

            for i in range(5):
                for j in range(5):
                    section_start_x = horizontal_sections[i]
                    section_end_x = horizontal_sections[i+1]
                    
                    section_start_y = vertical_sections[j]
                    section_end_y = vertical_sections[j+1]

                    # Define section bounding box
                    section_bbox = [(section_start_x, section_start_y), (section_end_x, section_end_y)]

                    # Apply mask to disparity map for section
                    section_mask = np.zeros_like(disparity_map)
                    cv.rectangle(section_mask, section_bbox[0], section_bbox[1], 255, -1)
                    section_masked_disparity = apply_mask(disparity_map, section_mask)

                    # Calculate average disparity for section
                    section_non_zero_elements = section_masked_disparity[section_masked_disparity != 0]
                    section_disparity = section_non_zero_elements.mean()

                    # Calculate distance for section
                    section_distance = getDistance(section_disparity, avgFocalLength, baseline)

                    # Calculate width for section
                    section_width = getWidth(section_bbox, section_distance,avgFocalLength)
                    section_widths.append(section_width)

                    total_width += section_width

                    # Calculate height for section
                    section_height = getHeight(section_bbox, section_distance, avgFocalLength, disparity_diff)
                    section_heights.append(section_height)

                    total_height += section_height
                    
                    width = total_width/5
                    height = total_height/5
                    length_est = (math.sqrt(width**2 + height**2))

                    # Display bounding box for section
                    cv.rectangle(img, section_bbox[0], section_bbox[1], (255, 0, 0), 2)
                    
                    distance = 0
        # distance_max = getDistance(disparity_min, avgFocalLength, baseline)
        # distance_min = getDistance(disparity_max, avgFocalLength, baseline)
        # distance_diff = distance_max - distance_min
        # width = getWidth(box_coordinates, distance_max, avgFocalLength)
        # length_est = math.sqrt(width**2 + distance_diff**2)
        # distance = 0
        # height = getHeight(box_coordinates, distance_max, avgFocalLength)
    
    print(f"distance {distance} distance_max width {width*100:.2f} height {height} length {length_est}")
    return ObjectDimensions(
        distance = distance*100,
        distance_max = -1,
        distance_min = -1,
        distance_diff = -1,
        width = width*100,
        height = height*100,
        length = length_est*100,
        disparity_diff = disparity_diff
        )
     
     

def getDistance(disparity, avgFocalLength, baseline):
    distance = (avgFocalLength) * (baseline / (disparity / 4.0))
    return distance


def getWidth(boxPoints, distance, avgFocalLength):
    pixelWidth = boxPoints[1][0] - boxPoints[0][0]
    return (distance * pixelWidth) / avgFocalLength


def getHeight(boxPoints, distance, avgFocalLength, disparity_diff):
    if disparity_diff < 10: 
        pixelHeight = boxPoints[2][1] - boxPoints[1][1]
    else:
        pixelHeight = boxPoints[1][1] - boxPoints[0][1]
    return abs((distance * pixelHeight) / avgFocalLength)
   # pixelHeight = boxPoints[2][1] - boxPoints[1][1]
    #return abs((distance * pixelHeight) / avgFocalLength)

def apply_mask(disparity_map, mask):
    masked_disparity = cv.bitwise_and(disparity_map, disparity_map, mask=mask)
    return masked_disparity
