# **Finding Lane Lines on the Road**

## Pipeline Description:

	My implemented pipeline has the following steps:

    1. Convert the image to grayscale
    2. Increase the image contrast
        * I found this to help with area selection accuracy
    3. Convert the original image to HSV and use it to apply a color mask for yellows and whites
        * After much research, I found that HSV images make it easier to isolate yellows.
        * It took a long time wrestling with the HSV values
    4. Combine the color mask with the contrast-boosted image
    5. Apply a Gaussian blur
    6. Detect Canny edges
    7. Apply a simple region of interest
        * I used a polygon with the two lower corners and two hard-coded top corner values.
        * This could use some tinkering to reduce such blatant assumptions
    8. Hough transform which also calls the improved draw_lines method
        * See code comments for more implementation details.
        * I iterate through the lines and classify them as "belonging" to the right or left.
            * Lines with non-infinite negative slopes belong to the left
            * Lines with non-infinite positive slopes belong to the right
        * I then use average values to form a single left and single right line,
        	* Also, extrapolate the length based on the y-axis maximum value
        * To help with smoothing, throw out outliers deviating from reasonable slopes
        * I also attempt to cache the lines for previous frames, using them for a weighted average
        * The two lines are also trimmed if they cross each other.
    9. Superimpose final hough lines onto original image

## Potential Shortcomings:

	My implementation, while slightly improved by leveraging HSV conversion, is extremely naive.

	It makes several assumptions around the region of interest.
	If the camera was removed within the vehicle's cabin, this could cause problems.

	More importantly, poor weather, large hills, tight curves, low light, and lane obstructions could all affect the output negatively.


## Possible Improvements:

	* Improvements are easy if we combine this approach with other sensor data, e.g. Lidar
	* Run various combinations of Hough and Canny parameters to truly dial in on optimal values
	* Consider a region of interest formed by non-linear lines to help with curves and hills
	* Improve overall smoothing and value averaging