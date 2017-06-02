import os
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import math

INPUT_DIR = 'test_images/'
OUTPUT_DIR = 'test_images_output/'

GAUSSIAN_KERNEL_SIZE = 5

CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

HOUGH_RHO = 2
HOUGH_THETA = (np.pi / 180)
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LEN = 50
HOUGH_MAX_LINE_GAP = 150

# weight of current frame when averaging with previous frame
CURR_WEIGHT = 0.6
PREV_WEIGHT = 1 - CURR_WEIGHT

# from (0..1) - used to throw out consecutive values differing by > this theshold * 100
SMOOTHING_THRESHOLD = 0.30

# cache of previous frame lines to help with video smoothing
l_cache = None
r_cache = None


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
    """
    global l_cache, r_cache  # opt-in to using the global cache variables

    all_slopes_and_intercepts = np.zeros((len(lines), 2))
    all_ys = []

    # save all non-infinite (slope, intercept) combos into all_slopes_and_intercepts
    # also grab all corresponding y values
    for i, line in enumerate(lines):
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if not np.isinf(slope) and not np.isinf(intercept):
                all_slopes_and_intercepts[i] = [slope, intercept]
                all_ys.append(y1)
                all_ys.append(y2)

    # max and min slopes across all lines
    max_slope = all_slopes_and_intercepts[all_slopes_and_intercepts.argmax(axis=0)[0]][0]
    min_slope = all_slopes_and_intercepts[all_slopes_and_intercepts.argmin(axis=0)[0]][0]
    min_y = np.min(all_ys)  # min y value across all lines

    # slopes and intercepts for all valid left lines
    l_ms = []
    l_bs = []
    # slopes and intercepts for all valid right lines
    r_ms = []
    r_bs = []

    for [m, b] in all_slopes_and_intercepts:
        if valid_for_left_line(m, min_slope):
            l_ms.append(m)
            l_bs.append(b)
        elif valid_for_right_line(m, max_slope):
            r_ms.append(m)
            r_bs.append(b)

    # calculate each lane line, using the previous frame's cached line if this frame cannot be calculated
    l_line, r_line = calculate_lines(img, l_ms, r_ms, l_bs, r_bs, min_y, l_cache, r_cache)

    # draw and cache the lines
    cv2.line(img, (l_line[0], l_line[1]), (l_line[2], l_line[3]), color, thickness)
    cv2.line(img, (r_line[0], r_line[1]), (r_line[2], r_line[3]), color, thickness)
    l_cache = l_line
    r_cache = r_line


def valid_for_left_line(slope, min_slope):
    # true if slope is negative and slope is reasonably close to the global minimum slope
    return slope < 0 and values_are_within_range(slope, min_slope)


def valid_for_right_line(slope, max_slope):
    # true if slope is positive and slope is reasonably close to the global maximum slope
    return slope > 0 and values_are_within_range(slope, max_slope)


def values_are_within_range(value, max_value):
    return abs(value - max_value) / max_value <= SMOOTHING_THRESHOLD


def calculate_lines(img, l_slopes, r_slopes, l_intercepts, r_intercepts, min_y, left_cache, right_cache):
    # if we have the previous frame's cached lines return a weighted average with the new ones
    # otherwise, just return the current frame's newly calculated lines
    l_avg_slope = np.average(l_slopes)
    l_avg_intercept = np.average(l_intercepts)
    l_y1 = int(img.shape[0])
    l_x1 = int((l_y1 - l_avg_intercept) / l_avg_slope)
    l_y2 = int(min_y)
    l_x2 = int((l_y2 - l_avg_intercept) / l_avg_slope)

    l_new_line = np.array([l_x1, l_y1, l_x2, l_y2], dtype='float32')
    l_new_line = l_new_line if left_cache is None else (CURR_WEIGHT * l_new_line) + (PREV_WEIGHT * left_cache)

    r_avg_slope = np.average(r_slopes)
    r_avg_intercept = np.average(r_intercepts)
    r_y1 = int(img.shape[0])
    r_x1 = int((r_y1 - r_avg_intercept) / r_avg_slope)
    r_y2 = int(min_y)
    r_x2 = int((r_y2 - r_avg_intercept) / r_avg_slope)

    r_new_line = np.array([r_x1, r_y1, r_x2, r_y2], dtype='float32')
    r_new_line = r_new_line if right_cache is None else (CURR_WEIGHT * r_new_line) + (PREV_WEIGHT * right_cache)

    # prevent lines from crossing
    if r_x2 < l_x2:
        avg_x = (r_x2 + l_x2) / 2
        new_l_y2 = l_avg_slope * avg_x + l_avg_intercept
        new_r_y2 = r_avg_slope * avg_x + r_avg_intercept
        l_new_line[2] = avg_x
        r_new_line[2] = avg_x
        l_new_line[3] = new_l_y2
        r_new_line[3] = new_r_y2

    return l_new_line, r_new_line


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


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


def process_image(image):
    # convert to grayscale
    gray_img = grayscale(image)

    # increase contrast by darkening
    dark_img = increase_contrast(gray_img)

    # convert to HSV
    hsv_img = to_hsv(image)

    # yellow and white color mask
    color_img = color_mask(hsv_img)

    # combine color mask and dark image
    contrast_color_img = cv2.bitwise_or(color_img, dark_img)

    # apply gaussian blur
    blur_img = gaussian_blur(contrast_color_img, GAUSSIAN_KERNEL_SIZE)

    # detect canny edges
    canny_img = canny(blur_img, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

    # apply region of interest mask
    region_img = region_mask(canny_img)

    # hough transform
    hough_img = hough_lines(region_img,
                            HOUGH_RHO,
                            HOUGH_THETA,
                            HOUGH_THRESHOLD,
                            HOUGH_MIN_LINE_LEN,
                            HOUGH_MAX_LINE_GAP)

    result = weighted_img(hough_img, image)
    return result


def video_stuff():
    clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4")
    # clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile('test_videos_output/solidYellowLeft.mp4', audio=False)


def run_pipeline():
    """
    Applie  s the lane detection pipeline and saves final images
    This is the main function for running against the images
    """
    for file_name in os.listdir(INPUT_DIR):
        original_img = mpimg.imread(INPUT_DIR + file_name)

        finished = process_image(original_img)

        save_image(OUTPUT_DIR + file_name, finished)
        show_image(finished)


def increase_contrast(img):
    return np.array(img / 2, np.uint8)


def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def color_mask(img):
    # HSV threshold range for yellow colors
    yellow_min = np.array([20, 100, 100])
    yellow_max = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(img, yellow_min, yellow_max)

    # HSV threshold range for white colors
    white_max = np.array([255, 255, 255])
    white_min = np.array([0, 0, 235])
    white_mask = cv2.inRange(img, white_min, white_max)

    # combine color masks
    return cv2.bitwise_or(yellow_mask, white_mask)


def region_mask(img):
    bottom_left = (0, img.shape[0])
    top_left = (450, 310)
    top_right = (490, 310)
    bottom_right = (img.shape[1], img.shape[0])

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return region_of_interest(img, vertices)


def save_image(file_path, img):
    colored_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, colored_image)


def show_image(img):
    plt.imshow(img)
    plt.show()


# run_pipeline()
video_stuff()
