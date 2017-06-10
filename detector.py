import numpy as np
import os
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from utils import *

'''
RUN_DETECTOR Given an image, runs the SVM detector and outputs bounding
boxes and scores

Arguments:
    im - the image matrix

    clf - the sklearn SVM object. You will probably use the 
        decision_function() method to determine whether the object is 
        a face or not.
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    window_size - an array which contains the height and width of the sliding
    	window

    cell_size - each cell will be of size (cell_size, cell_size) pixels

    block_size - each block will be of size (block_size, block_size) cells

    nbins - number of histogram bins

Returns:
    bboxes - D x 4 bounding boxes that tell [xmin ymin width height] per bounding
    	box

    scores - the SVM scores associated with each bounding box in bboxes

You can compute the HoG features using the compute_hog_features() method
that you implemented in PS3. We have provided an implementation in utils.py,
but feel free to use your own implementation. You will use the HoG features
in a sliding window based detection approach.

Recall that using a sliding window is to take a certain section (called the 
window) of the image and compute a score for it. This window then "slides"
across the image, shifting by either n pixels up or down (where n is called 
the window's stride). 

Using a sliding window approach (with stride of block_size * cell_size / 2),
compute the SVM score for that window. If it's greater than 1 (the SVM decision
boundary), add it to the bounding box list. At the very end, after implementing 
nonmaximal suppression, you will filter the nonmaximal bounding boxes out.
'''
def run_detector(im, clf, window_size, cell_size, block_size, nbins, thresh=1):
    window_height, window_width = window_size[0], window_size[1]
    image_height, image_width = im.shape[0], im.shape[1]
    # Number of pixels we move the sliding window to compute the features for a new window
    stride = block_size * cell_size / 2
    
    num_windows_across = ((image_width - window_width)/stride) + 1
    num_windows_down = ((image_height - window_height)/stride) + 1   
    bboxes = []
    scores = []
    # Iterate over the image sliding the window by stride pixels each time
    for i in range(num_windows_down):
        for j in range(num_windows_across):
            window_x_min = j * stride
            window_y_min = i * stride
            
            window = im[window_y_min:window_y_min + window_height, window_x_min:window_x_min + window_width]
            
            # For each window we calculate the HoG features and determine whether the window contains a face or not
            features = compute_hog_features(window, cell_size, block_size, nbins).flatten().reshape((1, -1))
            distance = clf.decision_function(features)
            
            # Append window parameters to the list of bounding boxes and the distance to the list of scores if the distance
            # to the separator exceeds threshold thresh
            if distance >= thresh:
                bboxes.append([window_x_min, window_y_min, window_width, window_height])
                scores.append(distance)
    
    bboxes = np.array(bboxes).reshape((-1, 4))
    scores = np.array(scores).reshape((-1, 1))
    return bboxes, scores


'''
NON_MAX_SUPPRESSION Given a list of bounding boxes, returns a subset that
uses high confidence detections to suppresses other overlapping
detections. Detections can partially overlap, but the
center of one detection can not be within another detection.

Arguments:
    bboxes - ndarray of size (N,4) where N is the number of detections,
        and each row is [x_min, y_min, width, height]
    
    confidences - ndarray of size (N, 1) of the SVM confidence of each bounding
    	box.

    img_size - [height,width] dimensions of the image.

Returns:
    nms_bboxes -  ndarray of size (N, 4) where N is the number of non-overlapping
        detections, and each row is [x_min, y_min, width, height]. Each bounding box
        should not be overlapping significantly with any other bounding box.

In order to get the list of maximal bounding boxes, first sort bboxes by 
confidences. Then go through each of the bboxes in order, adding them to
the list if they do not significantly overlap with any already in the list. 
A significant overlap is if the center of one bbox is in the other bbox.
'''
def non_max_suppression(bboxes, confidences):
    num_initial_boxes = bboxes.shape[0]
    
    # First sort all the initial bounding boxes in descending order by their corresponding confidences
    combined = np.concatenate((bboxes, confidences), axis=1)
    sorted_indices = np.argsort(combined[:,4])[::-1]
    sorted_combined = combined[sorted_indices]
    
    # Construct a filtered list of bounding boxes where boxes with a higher confidence are added first to the final list,
    # but only if the center of the bounding box is not within any bounding box already in the list
    nms_bboxes = np.empty((0, 4), dtype=int)
    
    # Define a function that determines if a row vector corresponding to a bounding box overlaps significantly with any other
    # bounding box
    def overlaps(current_box, final_list):
        current_box = current_box.flatten()
        current_center_x = current_box[0] + current_box[2]/2
        current_center_y = current_box[1] + current_box[3]/2
        
        # Iterate through all other bounding boxes, determining if the center of the current box falls within their boundaries
        num_final_boxes = final_list.shape[0]
        for i in range(num_final_boxes):
            other_box = final_list[i]
            other_min_x, other_min_y = other_box[0], other_box[1]
            other_max_x, other_max_y = other_min_x + other_box[2], other_min_y + other_box[3]
            
            # An overlap occurs if both the current box's center's x-coordinate falls within the other box's range 
            # of x-coordinates and the current box's center's y-coordinate falls within the other box's range of y-coordinates
            if (other_min_x <= current_center_x <= other_max_x) and (other_min_y <= current_center_y <= other_max_y):
                return True
            
        # If no overlap is detected after iterating through every other member of the final list of bounding boxes, 
        # then return False
        return False
    
    for i in range(num_initial_boxes):
        current_box = sorted_combined[i:i + 1, 0:4]
        if not overlaps(current_box, nms_bboxes):
            nms_bboxes = np.concatenate((nms_bboxes, current_box), axis=0)
    
    return nms_bboxes


if __name__ == '__main__':
    block_size = 2
    cell_size = 6
    nbins = 9
    window_size = np.array([36, 36])

    # compute or load features for training
    if not (os.path.exists('data/features_pos.npy') and os.path.exists('data/features_neg.npy')):
        features_pos = get_positive_features('data/caltech_faces/Caltech_CropFaces', cell_size, window_size, block_size, nbins)
        num_negative_examples = 10000
        features_neg = get_random_negative_features('data/train_non_face_scenes', cell_size, window_size, block_size, nbins, num_negative_examples)
        np.save('data/features_pos.npy', features_pos)
        np.save('data/features_neg.npy', features_neg)
    else:
        features_pos = np.load('data/features_pos.npy')
        features_neg = np.load('data/features_neg.npy')

    X = np.vstack((features_pos, features_neg))
    Y = np.hstack((np.ones(len(features_pos)), np.zeros(len(features_neg))))

    # Train the SVM
    clf = LinearSVC(C=1, tol=1e-6, max_iter=10000, fit_intercept=True, loss='hinge')
    clf.fit(X, Y)
    score = clf.score(X, Y)

    # Part A: Sliding window detector
    im = imread('data/people.jpg', 'L').astype(np.uint8)
    bboxes, scores = run_detector(im, clf, window_size, cell_size, block_size, nbins)
    plot_img_with_bbox(im, bboxes, 'Without nonmaximal suppresion')
    plt.show()

    # Part B: Nonmaximal suppression
    bboxes = non_max_suppression(bboxes, scores)
    plot_img_with_bbox(im, bboxes, 'With nonmaximal suppresion')
    plt.show()
