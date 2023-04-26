import numpy as np
import cv2
import matplotlib.pyplot as plt 
from scipy import signal as sig
from scipy import ndimage

def grad_x(gray_image):
    
    kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype = "int32")
    return sig.convolve2d(gray_image, kernel_x, mode='same')

def grad_y(gray_image):
    
    kernel_y = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]], dtype = "int32")
    return sig.convolve2d(gray_image, kernel_y, mode='same')

def harris_corner(gray_image, k, thres):
        
    # solbel_x and Sobel_y
    I_X = grad_x(gray_image)
    I_Y = grad_y(gray_image)
    
    I_XX = ndimage.gaussian_filter(I_X**2, sigma=1)
    I_XY = ndimage.gaussian_filter(I_Y*I_X, sigma=1)
    I_YY = ndimage.gaussian_filter(I_Y**2, sigma=1)    

    detMat = I_XX * I_YY - I_XY**2
    traceMat = I_XX + I_YY
    R = detMat - k * traceMat**2
    cv2.normalize(R, R, 0, 1, cv2.NORM_MINMAX)
    
    return np.where(R >= thres)

def k_points_extraction(corners):
    k_pointss = []
    
    for point in zip(*corners[::-1]):
        point1 = point[0]
        point2 = point[1]
        
        k_pointss.append(tuple([int(round(point1)),int(round(point2))])) 
    return k_pointss, len(k_pointss)


if __name__ == "__main__":
    
    #setting the thres and k value
    k = 0.06
    thres = 0.8
    
    # reading the image 
    image = cv2.imread("/home/sailesh/sensor/image/img_1.jpg", cv2.COLOR_RGB2BGR)
    #convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #getting corners using the harris_corner
    corners = harris_corner(gray_image, k, thres)
    
    #getting key points   
    k_pointss, len = k_points_extraction(corners)
    
    rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for pt in k_pointss:
        cv2.circle(rgbImage, pt, 3, (255, 0, 0), -1) 
    
    fig, ax = plt.subplots()
    corners = np.array(corners)
    ax.imshow(rgbImage, interpolation='nearest', cmap=plt.cm.gray)
    ax.plot(corners[:, 1], corners[:, 0], '.r', markersize=3)
    plt.show()    
    
    
