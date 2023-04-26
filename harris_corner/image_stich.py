import haris_corner as hcd
import skimg

def stiched_img(img1, img2, key1, key2, des1, des2, threshold):
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
    search_params = dict(checks = 5000)
    flann = hcd.cv2.FlannBasedMatcher(index_params,search_params)

    features = flann.knnMatch(des1,des2, k=15)

    ratio_pass = []
    for f in features:
        if f[0].distance < threshold *f[1].distance:
            ratio_pass.append(f)
            features = hcd.np.asarray(ratio_pass)

    features = hcd.np.array(features)
    if len(features[:,0]) >= 2:
        goal = hcd.np.float32([ key1[f.queryIdx].pt for f in features[:,0] ]).reshape(-1,1,2)
        init = hcd.np.float32([ key2[f.trainIdx].pt for f in features[:,0] ]).reshape(-1,1,2)
        H, _ = hcd.cv2.findHomography(init, goal, hcd.cv2.RANSAC, 2.0)
        print('Found match')
    else:
        print("Not able to find enough k_points")

    final = hcd.cv2.warpPerspective(img1,H,(img1.shape[1] + img2.shape[1], img2.shape[0]))
    final[0:img1.shape[0], 0:img1.shape[1]] = img1

    rows, cols = hcd.np.where(final[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final = final[min_row:max_row, min_col:max_col, :]
    return final

if __name__ == "__main__":
    
    # reading the 4 imgs     
    
    img1 = skimg.io.imread("/home/sailesh/sensor/img/img_1.jpg")
    img_gray_scale1 = skimg.color.rgb2gray(img1)
    img_sift1 = hcd.cv2.imread("/home/sailesh/sensor/img/img_1.jpg", hcd.cv2.COLOR_RGB2BGR)
    gray_sift1 = hcd.cv2.cvtColor(img_sift1, hcd.cv2.COLOR_RGB2GRAY)

    img2 = skimg.io.imread("/home/sailesh/sensor/img/img_2.jpg")
    img_gray_scale2 = skimg.color.rgb2gray(img2)
    img_sift2 = hcd.cv2.imread("/home/sailesh/sensor/img/img_2.jpg", hcd.cv2.COLOR_RGB2BGR)
    gray_sift2 = hcd.cv2.cvtColor(img_sift2, hcd.cv2.COLOR_RGB2GRAY)

    img3 = skimg.io.imread("/home/sailesh/sensor/img/img_3.jpg")
    img_gray_scale3 = skimg.color.rgb2gray(img3)
    img_sift3 = hcd.cv2.imread("/home/sailesh/sensor/img/img_3.jpg", hcd.cv2.COLOR_RGB2BGR)
    gray_sift3 = hcd.cv2.cvtColor(img_sift3, hcd.cv2.COLOR_RGB2GRAY)

    img4 = skimg.io.imread("/home/sailesh/sensor/img/img_4.jpg")
    img_gray_scale4 = skimg.color.rgb2gray(img4)
    img_sift4 = hcd.cv2.imread("/home/sailesh/sensor/img/img_4.jpg", hcd.cv2.COLOR_RGB2BGR)
    gray_sift4 = hcd.cv2.cvtColor(img_sift4, hcd.cv2.COLOR_RGB2GRAY)
    
    # detecting the corners using the hcd
    corners1 = hcd.harris_corner_det(img_gray_scale1, 0.0496, 0.365)
    corners2 = hcd.harris_corner_det(img_gray_scale2, 0.05, 0.55)
    corners3 = hcd.harris_corner_det(img_gray_scale3, 0.05, 0.383)
    corners4 = hcd.harris_corner_det(img_gray_scale4, 0.05, 0.621)
    
    # extracting key points
    k_points1, len1 = hcd.keypoint_extraction(corners1)
    k_points2, len2 = hcd.keypoint_extraction(corners2)
    k_points3 ,len3= hcd.keypoint_extraction(corners3)
    k_points4 ,len4= hcd.keypoint_extraction(corners4)
    print(len1)
    print(len2)
    print(len3)
    print(len4)
    
    # using SIFT to create correspondence
    sift_detector = hcd.cv2.SIFT.create()
    
    key1, des1 = sift_detector.compute(gray_sift1, hcd.cv2.KeyPoint.convert(k_points1))
    key2, des2 = sift_detector.compute(gray_sift2, hcd.cv2.KeyPoint.convert(k_points2))
    key3, des3 = sift_detector.compute(gray_sift3, hcd.cv2.KeyPoint.convert(k_points3))
    key4, des4 = sift_detector.compute(gray_sift4, hcd.cv2.KeyPoint.convert(k_points4))
    
    # using the keys and des to stich the img
    stich1 = stiched_img(img1, img2, key1, key2, des1, des2, 0.25)
      
    hcd.plt.imshow(stich1)
    hcd.plt.show()
    
    stich2 = stiched_img(img3, img4, key3, key4, des3, des4, 0.25)
    
    hcd.plt.imshow(stich2)
    hcd.plt.show()
    
    # #stichting the already stiched imgs 
    stich_grey = hcd.cv2.cvtColor(stich1, hcd.cv2.COLOR_RGB2GRAY)
    stich_corners1 = hcd.harris_corner_det(stich_grey, 0.05, 0.973)
    k_pointsStich1, len5 = hcd.keypoint_extraction(stich_corners1)
    key5, descriptions5 = sift_detector.compute(stich_grey, hcd.cv2.KeyPoint.convert(k_pointsStich1))
    
    stich2Gray = hcd.cv2.cvtColor(stich2, hcd.cv2.COLOR_RGB2GRAY)
    stich_corners2 = hcd.harris_corner_det(stich2Gray, 0.05, 0.98)
    k_pointsStich2, len6 = hcd.keypoint_extraction(stich_corners1)
    key6, descriptions6 = sift_detector.compute(stich2Gray, hcd.cv2.KeyPoint.convert(k_pointsStich2))
    
    print(len5)
    print(len6)

    
    stichWhole = stiched_img(stich1, stich2, key5, key6, descriptions5, descriptions6,0.2)
    
    hcd.plt.imshow(stichWhole)
    hcd.plt.show()