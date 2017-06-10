import re
from scipy import fftpack, ndimage
import scipy.misc
import cv2
import numpy as np
from matplotlib import pyplot as plt

import os
from sklearn.svm import LinearSVC
from scipy.ndimage import imread
from utils import *

# Implement the Tsai and Hsieh algorithm that applies a Discrete Fourier transform and Hough transform, then reverses the
# Fourier transform to filter out less related regions in the original image.  ## TODO - Work in progress
def apply_fft():
    img = cv2.imread('data/other/paper_example.png', 0)
    f = np.fft.fft2(img)    
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
#     magnitude_spectrum = np.log(1 + np.abs(fshift))

    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    
#     scipy.misc.imsave("magnitude_1.jpg", magnitude_spectrum)
    scipy.misc.imsave("magnitude_1.jpg", magnitude_spectrum)
    
    img = cv2.imread("magnitude_1.jpg", 0)
    img = np.uint8(img)
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,200)

#     shy = np.abs(fshift)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(fshift,(x1,y1),(x2,y2),(0,0,255),8)

#     cv2.imwrite('hough_1-2.jpg',img)
    
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    
#     img_back = np.abs(img_back)
    img_back = 20 * np.log(np.abs(img_back))

    cv2.imwrite('filtered_image.jpg', img_back)
    print "img_back"
    print img_back
    plt.subplot(131),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(img_back)
    plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    
if __name__ == '__main__':
    apply_fft()
