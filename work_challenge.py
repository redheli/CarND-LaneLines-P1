import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import sys
print(os.getcwd())
p = os.getcwd()
sys.path.append('/home/max/avmap/udacity/CarND-LaneLines-P1')
import utils
images = os.listdir("test_images_challenge/")
output_folder = 'test_images_output'
if os.path.isdir(output_folder) is False:
    os.mkdir(output_folder)

images = ['challenge.jpg']

for image_name in images:

    # Read in and grayscale the image
    # image_name = 'solidYellowCurve2.jpg'
    # image_name = 'solidWhiteCurve.jpg'
    image = mpimg.imread('./test_images_challenge/'+image_name)
    # gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    gray = utils.grayscale(image)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    # blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    blur_gray = utils.gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    # edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    edges = utils.canny(blur_gray, low_threshold, high_threshold)

    plt.subplot(221),plt.imshow(image,cmap = 'gray')
    plt.title('Original Image \n{}'.format(image_name) )
    plt.subplot(222),plt.imshow(edges,cmap = 'gray')
    plt.title('Canny Edges \n{}'.format(image_name) )

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    img_width = 720
    # left_bottom = [160, img_width - 60]
    # left_top = [500, 450]
    # right_top = [820, 450]
    # right_bottom = [1200, img_width - 60]
    left_bottom = [160, img_width - 60]
    left_top = [580, 420]
    right_top = [730, 420]
    right_bottom = [1200, img_width - 60]
    #vertices = np.array([[(0,imshape[0]),(0, 0), (imshape[1], 0), (imshape[1],imshape[0])]], dtype=np.int32)
    vertices = np.array([[(left_bottom[0], left_bottom[1] ),
                            (left_top[0], left_top[1]),
                            (right_top[0], right_top[1]),
                            (right_bottom[0],right_bottom[1])]], dtype=np.int32)

    # cv2.fillPoly(mask, vertices, ignore_mask_color)
    # masked_edges = cv2.bitwise_and(edges, mask)
    masked_edges = utils.region_of_interest(edges, vertices)
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 0.5 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 1     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 16 #minimum number of pixels making up a line
    max_line_gap = 6    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    # lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                # min_line_length, max_line_gap)


    # # Iterate over the output "lines" and draw lines on a blank image
    # for line in lines:
    #     for x1,y1,x2,y2 in line:
    #         cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

    line_image = utils.hough_lines(masked_edges, rho, theta, threshold,
                                min_line_length, max_line_gap)

    plt.subplot(224), plt.imshow(line_image,cmap = 'gray')
    plt.title('Line Image')
    plt.imshow(line_image)


    # Create a "color" binary image to combine with line image
    # color_edges = np.dstack((edges, edges, edges))

    # Draw the lines on the edge image
    # lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

    alpha = 0.8
    beta = 1.0
    remda = 0.

    final_image = utils.weighted_img(line_image, image, alpha, beta, remda)

    x = [left_bottom[0], left_top[0], right_top[0], right_bottom[0], left_bottom[0]]
    y = [left_bottom[1], left_top[1], right_top[1], right_bottom[1], left_bottom[1]]
    plt.subplot(223),plt.plot(x, y, 'b--', lw=4)

    plt.imshow(final_image)
    plt.title('Final Lines')
    plt.show()

    RGB_img = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_folder+'/'+image_name, RGB_img)