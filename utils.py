import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import linear_model, datasets
import numpy as np
import cv2

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines_ori(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_lines_ransac(img, lines, color=[255, 0, 0], thickness=10):
    img2 = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    left_x = []
    left_y = []
    right_x = []
    right_y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            # dis = math.sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1))
            # if dis < 10:
            #     continue
            slope = math.atan2((y2 - y1), (x2 - x1)) * 57.3
            # if (abs(abs(slope) - 90) < 5):
            #     print('skip %f' % slope)
            #     continue
            # if (abs(slope) < 5):
            #     print('skip %f' % slope)
            #     continue
            if slope > 0:
                # right
                right_x.append([x1]),right_x.append([x2])
                right_y.append(y1),right_y.append(y2)
            elif slope < 0:
                # left
                left_x.append([x1]), left_x.append([x2])
                left_y.append(y1), left_y.append(y2)
    # Robustly fit linear model with RANSAC algorithm
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model_ransac.fit(left_x, left_y)
    left_inlier_mask = model_ransac.inlier_mask_
    left_outlier_mask = np.logical_not(left_inlier_mask)
    # find left far, near points in the inliner points
    left_far_near_x = np.array([-999999, 999999])
    for i in range(left_inlier_mask.size):
        if np.equal(left_inlier_mask[i],True):
            # far point x
            if left_x[i][0] > left_far_near_x[0]:
                left_far_near_x[0] = left_x[i][0]
            # near point x
            if left_x[i][0] < left_far_near_x[1]:
                left_far_near_x[1] = left_x[i][0]
    left_far_near_y = model_ransac.predict(left_far_near_x[:, np.newaxis])
    left_far_near_y = left_far_near_y.astype(np.int64)

    model_ransac2 = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model_ransac2.fit(right_x, right_y)
    right_inlier_mask = model_ransac2.inlier_mask_
    right_outlier_mask = np.logical_not(right_inlier_mask)
    right_far_near_x = np.array([999999, -999999])
    for i in range(right_inlier_mask.size):
        if np.equal(right_inlier_mask[i],True):
            if right_x[i][0] > right_far_near_x[1]:
                right_far_near_x[1] = right_x[i][0]
            if right_x[i][0] < right_far_near_x[0]:
                right_far_near_x[0] = right_x[i][0]
    right_far_near_y = model_ransac2.predict(right_far_near_x[:, np.newaxis])
    right_far_near_y = right_far_near_y.astype(np.int64)

    cv2.line(img2, (left_far_near_x[0], left_far_near_y[0]),
             (left_far_near_x[1], left_far_near_y[1]), color, thickness)
    cv2.line(img2, (right_far_near_x[0], right_far_near_y[0]),
             (right_far_near_x[1], right_far_near_y[1]), color, thickness)

    cv2.addWeighted(img2, 0.6, img, 0.4, 0., img)

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    img2 = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    print('slopes:')
    left_line_far = [9999,9999]
    left_line_near = [-9999,-9999]
    right_line_far = [9999,9999]
    right_line_near = [-9999,-9999]
    left_line_far_line = []
    left_line_near_line = []
    right_line_far_line = []
    right_line_near_line = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            # dis = math.sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1))
            # if dis < 10:
            #     continue
            slope = math.atan2((y2 - y1), (x2 - x1)) * 57.3
            # if (abs(abs(slope) - 90) < 5):
            #     print('skip %f' % slope)
            #     continue
            # if (abs(slope) < 5):
            #     print('skip %f' % slope)
            #     continue
            if slope > 0:
                # left
                if y1 < left_line_far[1]:
                    left_line_far = [x1,y1]
                    left_line_far_line = line[0]
                    left_line_far_line_slope = slope
                if y1 > left_line_near[1]:
                    left_line_near = [x1, y1]
                    left_line_near_line = line[0]
                    left_line_near_line_slope = slope
                if y2 < left_line_far[1]:
                    left_line_far = [x2,y2]
                    left_line_far_line = line[0]
                    left_line_far_line_slope = slope
                if y2 > left_line_near[1]:
                    left_line_near = [x2, y2]
                    left_line_near_line = line[0]
                    left_line_near_line_slope = slope
            elif slope < 0:
                # right
                if y1 < right_line_far[1]:
                    right_line_far = [x1,y1]
                    right_line_far_line = line[0]
                    right_line_far_line_slope = slope
                if y1 > right_line_near[1]:
                    right_line_near = [x1, y1]
                    right_line_near_line = line[0]
                    right_line_near_line_slope = slope
                if y2 < right_line_far[1]:
                    right_line_far = [x2,y2]
                    right_line_far_line = line[0]
                    right_line_far_line_slope = slope
                if y2 > right_line_near[1]:
                    right_line_near = [x2, y2]
                    right_line_near_line = line[0]
                    right_line_near_line_slope = slope
            print(slope)
    cv2.line(img2, (left_line_far[0], left_line_far[1]), (left_line_near[0], left_line_near[1]), color, thickness)
    cv2.line(img2, (right_line_far[0], right_line_far[1]), (right_line_near[0], right_line_near[1]), color, thickness)
    # draw lines
    cv2.line(img2, (left_line_near_line[0], left_line_near_line[1]), (left_line_near_line[2], left_line_near_line[3]), [0, 255, 0], 5)
    print('left near slop: %f' % left_line_near_line_slope)
    cv2.line(img2, (left_line_far_line[0], left_line_far_line[1]), (left_line_far_line[2], left_line_far_line[3]), [0,255,0], 5)
    print('left far slop: %f' % left_line_far_line_slope)
    cv2.line(img2, (right_line_near_line[0], right_line_near_line[1]), (right_line_near_line[2], right_line_near_line[3]), [0, 0, 255], 8)
    print('right near slop: %f' % right_line_near_line_slope)
    cv2.line(img2, (right_line_far_line[0], right_line_far_line[1]), (right_line_far_line[2], right_line_far_line[3]), [0,0,255], 8)
    print('right far slop: %f' % right_line_far_line_slope)

    cv2.addWeighted(img2, 0.6, img, 0.4, 0.,img)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    # draw_lines_ori(line_img, lines)
    draw_lines_ransac(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)