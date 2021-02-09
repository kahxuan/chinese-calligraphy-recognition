import numpy as np
import cv2

# returns an image with white character and black background
def segment_image(image):
    
    assert len(image.shape) == 2, 'image must be in grayscale'
    
    # use k-means to segment pixels
    samples = np.float32(image.reshape((-1, 1)))
    k = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.95) 
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(samples, k, None, criteria, 10, flags) 

    # black for background, white for char
    # assume background take up more pixels
    white_region = np.bincount(labels.flatten()).argmax()
    if white_region:
        tmp = [255, 0]
    else:
        tmp = [0, 255]
    centers = np.array(tmp, dtype=np.uint8).reshape((2, 1))

    segmented = centers[labels.flatten()] 
    segmented = segmented.reshape((image.shape)) 

    return segmented