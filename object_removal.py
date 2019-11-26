# Program Description: Ball Removal without Machine Learning
# Author: Bryan Flood

# Running the Code:
# Simply change the file path to point to the location of your image
# This program outputs both the before and after of the image

### Research:
# To research this problem, I looked up the most effective methods of object removal
# in Image Processing. 

# The most popular solutions are either pattern based or use Generative Adversarial Networks.
# Photo editing software like Photoshop and GIMP provide their own solutions.
# Adobe provides Fill based object removal and Spot Healing.
# Resynthesizer is the most popular solution for GIMP.

# The use of GANs is off the table as it is machine learning.
# Pattern Based is too complex to implement for this kind of assignment as it 
# is usually used in conjunction with quilting and subsampling.

# Both the Resynthesizer thesis and the Pattern Matching provided by Adobe inspired 
# me to come up with this solution


### Coming up with the algorithm:
# The ball removal was easy. Circle Hough is the best solution for detecting balls and circles.

# To isolate the white, I first tried to use the HSV color space which is notoriously good at isolating colors.
# I found this method produced quite nosy results and focused too much on intensity of the white.
# I ended up just converting the bgr image to gray scale and thresholding to retain only the high values.

# To maximize performance, I created a smaller cropped image based on where white was located in the image.
# When cropping I provide additional padding around the masked area to account for any shadows cast
# and I kept both sides equal so that the algorithm would be just as effective no matter how the image is orientated.

# This means that Circle Hough only must be run on a small subset of the image minimizing runtime and false positives.
# From the dimensions that Circle Hough provides I can easily remove the ball from the image.

# To replace the background based on the research I had done inpainting seemed to be the obvious choice.
# I learned quickly that it was usually only effective when used remove thin lines not large holes.
# In most case producing a distinct and unnatural radial pattern.


# My solution was to mask the top half of the outline and use inpaint to blend it better into the background.
# And for the cases whereby a sample was not available or not of high enough quality 
# I made it so the code used the pure inpaint solution.

### Improvements
# There are many improvements that could be made to this algorithm. An algorithm I implemented before this
# Sampled the whole image with multiple smaller samples used to cover the hole.
# This implemented with some basic pattern matching and rough line based stitching using Sobel would produce far better results.
# I moved to a far simpler method due to the fact it was easier to predict when the algorithm would fall back to inpaint
# and because it produced better results with my main images.
# Realistically some form of machine learning should be used for a practical application for example
# A Generative Adversarial Network trained on grass etc.

# A great paper to look at is Image Texture Tools by Paul Francis Harrison
# http://www.logarithmic.net/pfh/thesis 
# This is probably the best non-machine learning based solution to this problem
# Well Worth a Read


### Performance: 
# My focus with this solution was quality but it still ended up being rather performant.
# The main performance optimisation was running the circle hough on the subset of the image.
# When timed on macOS execution took only a few milliseconds.
# This code was able to run on a live video feed.

# From my testing my solution is quite resistant to false positives.
# When a sample is valid the resulting image is very high quality.



### Error Checking:
# I ensured that the code worked on all images required.
# I preform various error checking within this code:
#       -> I ensure that the image is not None
#       -> I ensure that a ball is detected
#       -> I ensure that sample falls within the bounds of the image
#       -> I ensure that the sample is of high enough quality to be used


### Algorithm:
#  1. Crop image to location of the white ball
#  2. Detect Ball within cropped image using Circle Hough and get its dimensions
#  3. Convert cropped coordinates to lie within the full image
#  4. Get a Sample area of the image to cover the hole
#  5. Convert sample to HSV to measure Standard deviation
#  6. Check if sample is valid based on Standard deviation and whether it lies within the image
#  7. If sample is valid, blend sample in with background using inpainting
#  8. If not valid, use inpainting to fill in hole
#  9. Display before and after images



import cv2 
import numpy as np

# Crop to size of mask
# This is not a tight crop 
# Also ensures both sides are equal
def crop_image(image, mask):
    y_index, x_index = np.where(mask != 0)
    # Both arrays are sorted based on the contents of y_index
    # So x_index needs to be resorted based on its own contents
    x_index.sort()
    first_x = x_index[0]
    first_y = y_index[0]
    last_x = x_index[len(x_index)-1]
    last_y = y_index[len(y_index)-1]

    # find max width so crop will be square
    x_diff = last_x - first_x
    y_diff = last_y - first_y
    max_diff = x_diff if x_diff > y_diff else y_diff 

    # Increase size of cropped area
    min_y = int(first_y*.8)
    max_y = first_y + int(max_diff * 1.5)
    min_x = int(first_x*.8)
    max_x = first_x + int(max_diff * 1.5)
    return image[min_y:max_y, min_x:max_x], min_x, min_y


# Crop to area of white
# This is an optimization that means circle hough
# Doesn't have to run on the whole image 
def crop_to_ball(image):
    # Threshold for white
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask[mask < 220] = 0

    # Remove noise and convert to binary image
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask,kernel,iterations = 2)
    mask = cv2.bilateralFilter(mask,9,75,75)
    mask = cv2.dilate(mask,kernel,iterations = 5)
    mask = cv2.bilateralFilter(mask,9,75,75)
    mask[mask != 0] = 255

    return crop_image(image, mask)

# Use circle hough to check if a ball is present and return its dimensions
def check_for_ball(image):
    # Convert to grey and blur to prepare for CircleHough
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray,(5,5))
    height, _, _ = image.shape

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=130, param2=30, minRadius=int(height*.15), maxRadius=int(height*.38))
    if circles is not None:
        for x, y, r in circles[0]:
            return int(x), int(y), int(r), height
    return False

    

# Sample part of the image to be used to cover the removed object
def sample_image(image, x, y, r, height):
    # Take into consideration the width/height of the crop
    offset_multiplier = height * .60

    # Copy sample area of image to cover hole 
    sample = image[y-r-5:y+r+5, int(x-r-5-offset_multiplier):int(x+r+5-offset_multiplier)]
    image[y-r-5:y+r+5, x-r-5:x+r+5] = sample
    return image, sample

# Use inpainting to blend the sample with the original image
def inpaint_sample(image, x, y, r):
    # Copy image as cv2.line alters any image passed to it
    line = image.copy()
    line_size = 5

    # Create white lines to mask the top of the sample
    cv2.line(line, (x-r-line_size,y-r-line_size), (x+r+line_size, y-r-line_size), (255, 255, 255), line_size)
    cv2.line(line, (x-r-line_size,y-r-line_size), (x-r-line_size, y), (255, 255, 255), line_size)
    cv2.line(line, (x+r+line_size,y-r-line_size), (x+r+line_size, y), (255, 255, 255), line_size)
    
    # Convert image to grayscale then binary
    line = cv2.cvtColor(line,cv2.COLOR_BGR2GRAY)
    line[line != 255] = 0

    # Use mask to blend sample with the background
    sample = cv2.inpaint(image,line,5,cv2.INPAINT_NS)
    return sample

# If no samples are availible inpaint the whole hole
def radial_fill(image, x, y, r):
    # Copy image as cv2.circle alters any image passed to it
    original = image.copy()
    cv2.circle(image, (x, y+15), r+20, (255, 255, 255), -1)

    # Covert to gray and then binary
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image[image != 255] = 0

    radial = cv2.inpaint(original,image,3,cv2.INPAINT_NS)
    return radial


# This removes the ball from the scene and replaces 
# The hole it leaves with grass
def remove_ball(image):
    if(image is None):
        print("Image provided was empty")
        return False

    # Copy images to maintain original
    radial_src = image.copy()
    image_sample = image.copy()

    # Crop to white ball ROI
    cropped_image, min_x, min_y = crop_to_ball(image)

    # Check if a ball was detected
    try:
        x, y, r, width = check_for_ball(cropped_image)
    except TypeError:
        print("No balls were found")
        return False

    # Hough is calculated on cropped image
    # Cordinates need to be updated to match 
    # whole image
    x += min_x
    y += min_y

    sample_valid = True
    
    # Find out if a sample is availible
    try:
        # Get a sample from the image
        image_sample, sample = sample_image(image_sample, x, y, r, width)

        # Get a HSV copy to calculate Standard Deviation
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)

        # Use standard deviation to determine 
        # if a sample area is good enough to be sampled
        _, std = cv2.meanStdDev(hsv)
        h_std, _,  _ =  std
        sample_valid = h_std < 5

    except ValueError:
        print("Sample out of bounds and invalid\nRadial inpaint will be used instead")
        sample_valid = False

    print("Ball has been replaced")

    if(sample_valid):
        return inpaint_sample(image_sample, x, y, r)
    else:
        return radial_fill(radial_src, x, y, r)



# Main function that displays images
def main(file_path = "./images/golf.jpg"):
    original_image = cv2.imread(file_path, 1)
    cv2.imshow("Original Image", original_image)

    cv2.imshow("Final Image", remove_ball(original_image))

    # Ensure window doesn't close
    cv2.waitKey(0) 
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()