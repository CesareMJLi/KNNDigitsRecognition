
"""
Created on Mar 29 2018 

@author: Mingju Li
"""

#----------------------Imports---------------------------

import numpy as np
import cv2

from scipy.misc.pilutil import imresize
# imresize(arr, size[, interp, mode]) 
# resize the pictures

from skimage.feature import hog
# extract Histogram of Oriented Gradients (HOG) for a given image.

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
# Split arrays or matrices into random train and test subsets

from sklearn.metrics import accuracy_score

from sklearn.utils import shuffle
# shuffle the samples we are going to use


#----------------------Constants delaration---------------------------

INPUT_IMAGE = "input_image.png"
INPUT_IMAGE_AFTERLOAD = "input_image_overlay.png"

OUTPUT_IMAGE = "output_image.png"

TRAIN_IMAGE_AFTERLOAD = "train_image_overlay.png"
TRAIN_IMAGE = "train_image.jpg"

IMG_HEIGHT = 28
IMG_WIDTH = 28

#----------------------Functions of training---------------------------

# this function aims to processes a custom training image
# this function could create a new image in the current folder, with all the numbers in a rectangle
def load_digits_from_img(img_file):
    # in this example, the input_image.jpg has a 10*10 matrix, from 1 to 9
    print("START TO PROCESSING TRAINING IMAGE:\n")
    img = cv2.imread(img_file)
    train_set = []
    target_set = []
    start = 1

    # processing the input img to get rid of the noise and make the pattern more clear
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # information about cv2.cvtColor https://blog.csdn.net/jningwei/article/details/77725559
    # this function could convert the color of original image into a new color space
    plt.imshow(img_gray)
    kernel = np.ones((5,5),dtype=int)
    #return a 5*5 matrix filled with 1
    ret, thresh = cv2.threshold(img_gray,127,255,0)
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    thresh = cv2.dilate(thresh,kernel,iterations = 1)
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    # repeat doing erode and dilate to make the img more clear
    # remember even if we use thresh here, it does not mean a number
    # it is a image!

    # find the contours
    _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = rect_from_contoursHierarchy(contours,hierarchy) 
    # https://blog.csdn.net/sunny2038/article/details/12889059
    # https://blog.csdn.net/jfuck/article/details/9620889 

    for i,r in enumerate(rectangles):
        # https://blog.csdn.net/hjxu2016/article/details/77833984
        # >>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        # >>> list(enumerate(seasons))
        # [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
        x,y,w,h = r
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # draw a rectangle

        # obetain a digit matrix from a image
        img_digit = img_gray[y:y+h,x:x+w]
        img_digit = (255-img_digit)
        img_digit = imresize(img_digit,(IMG_WIDTH, IMG_HEIGHT))
        # from our preset width and height we derive the size of 28*28 matrix

        # if i==0:
        #     print("***Here is the digit matrix for the first sample:***\n")
        #     print(img_digit)
        #     print("*"*30)

        train_set.append(img_digit)
        target_set.append(start%10)

        if i>0 and (i+1)%10 == 0:
            start +=1
    
    cv2.imwrite(TRAIN_IMAGE_AFTERLOAD,img)
    return np.array(train_set), np.array(target_set)
        
# given contours and hierarchy, return a list of final bounding rectanges with respond to the numbers
def rect_from_contoursHierarchy(contours,hierarchy):
    hierarchy = hierarchy[0]
    # select the root hierarchy

    boundary = [cv2.boundingRect(c) for c in contours]
    # here each boundary is a x,y,w,h

    final_boundary = []
    # we have to find the most common heirarchy
    # since the finding contours is done in a tree-level method
    # so this is necessary

    u, indices = np.unique(hierarchy[:,-1], return_inverse=True)
    # u are the unique element values while indces are the mapping to u
    # >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    # >>> u, indices = np.unique(a, return_inverse=True)
    # >>> u
    # array([1, 2, 3, 4, 6])
    # >>> indices
    # array([0, 1, 4, 3, 1, 2, 1])
    # >>> u[indices]
    # array([1, 2, 6, 4, 2, 3, 2])

    most_common_heirarchy = u[np.argmax(np.bincount(indices))]

    for r,hr in zip(boundary, hierarchy):
        # each hr is an array, noting its parents/sons/... 
        # zip is a wrapping func
        # here r is the rectangle and hr is its relative hierarchy
        x,y,w,h = r

        # here we set this condition to make sure we only use the samples we want our model to use
        # this help to avoid mark some small pattern and decrease the noise
        #  https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html

        if ((w*h)>250) and (10 <= w <= 200) and (10 <= h <= 200) and hr[3] == most_common_heirarchy: 
            # hr[3] is the parent contours
            # here I think the most common heirarchy should be -1
            final_boundary.append(r)    

    return final_boundary

# turen the pixels into hog
def pixels_to_hog(img_array):
    # the input a an array of 100 matrix
    hog_featuresData = []
    for i in img_array:
        fd = hog(i, 
                 orientations=10, 
                 pixels_per_cell=(5,5),
                 cells_per_block=(1,1), 
                 visualise=False)
        hog_featuresData.append(fd)
    hog_features = np.array(hog_featuresData, 'float64')
    return np.float32(hog_features)
#----------------------Functions of testing---------------------------

def KNN_MachineLearning(img_file, model):
    print("START TO PROCESSING INPUT IMAGE:\n")
    img = cv2.imread(img_file)

    blank_image = np.zeros((img.shape[0],img.shape[1],3), dtype=int)
    blank_image.fill(255)
    # 0 is color black and 255 is color white
    # here we create a new image of the input size

    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    plt.imshow(img_gray)
    kernel = np.ones((5,5),dtype=int)

    # pre-processing the input images
    ret,thresh = cv2.threshold(img_gray,127,255,0)   
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    thresh = cv2.dilate(thresh,kernel,iterations = 1)
    thresh = cv2.erode(thresh,kernel,iterations = 1)

    _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = rect_from_contoursHierarchy(contours,hierarchy)  
    #rectangles of bounding the digits in user image

    for r in rectangles:
        x,y,w,h = r
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.rectangle(img,(380,0),(511,111),(255,0,0),3)
        # the parameters are two points of the rectangle color and the type of the line
        
        img_digit = img_gray[y:y+h,x:x+w]
        img_digit = (255-img_digit)
        img_digit = imresize(img_digit,(IMG_WIDTH ,IMG_HEIGHT))

        hog_img_data = pixels_to_hog([img_digit])  

        # for each rectangle area we obtained, we use our trained model to do the prediction
        pred = model.predict(hog_img_data)
        
        # put the result into the orginal image
        cv2.putText(img, str(int(pred[0])), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        
        # put the result into the new blank image
        # cv2.putText(blank_image, str(int(pred[0])), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

    plt.imshow(img)
    cv2.imwrite(INPUT_IMAGE_AFTERLOAD,img) 
    # cv2.imwrite(OUTPUT_IMAGE,blank_image) 
    cv2.destroyAllWindows()


#----------------------Class---------------------------

# define a custom model in a similar class wrapper with train and predict methods
# here we adapt the KNN method to solve this problem. so we define a class with attibutes k and models
# and 2 methods of train and predict

class KNN_MODEL():
    def __init__(self, k = 3):
        self.k = k
        self.model = cv2.ml.KNearest_create()
        # further information of KNN cv2 could be found here https://docs.opencv.org/3.2.0/d5/d26/tutorial_py_knn_understanding.html

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.findNearest(samples, self.k)
        return results.ravel()

#----------------------Training---------------------------

TRAIN_USER_IMG = 'train_image.jpg'
TEST_USER_IMG = 'input_image.png'

digits, labels = load_digits_from_img(TRAIN_USER_IMG)
# Here the digits is the training input and the labels are the numbers 
# the digits is an array of size 100, each element is a 28*28 matrix
# while the labels is an array of size 100, each element is the corresponding label of one element

print('The digits of the training model is ',digits.shape)
print('The labels of the training model is ',labels.shape)

digits, labels = shuffle(digits, labels, random_state=256)
# >>> arr = np.arange(10)
# >>> np.random.shuffle(arr)
# >>> arr
# [1 7 5 2 9 4 3 6 0 8]
# it could shuffle several arrays in the very same random way

train_digits_data = pixels_to_hog(digits)

x_train, x_test, y_train, y_test = train_test_split(train_digits_data, labels, test_size=0.33, random_state=42)

model = KNN_MODEL(k = 3)
model.train(x_train, y_train)
preds = model.predict(x_test)
print('The Accuracy of this model is : ',accuracy_score(y_test, preds))

#----------------------Testing---------------------------

model = KNN_MODEL(k = 5)
model.train(train_digits_data, labels)
# here we use all of our training data to train the model to get the best accuracy
KNN_MachineLearning(INPUT_IMAGE, model)