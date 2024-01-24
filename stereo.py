import cv2 as cv
import numpy as np

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