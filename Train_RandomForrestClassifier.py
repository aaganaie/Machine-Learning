import numpy as np
import skimage
import cv2
import matplotlib.pyplot as plt
import os
from skimage import measure
import csv

##ASK  IMAGE DATA FROM ME , TO EXECUTE THIS MODULE

#extract features from preproccesed image by drawing contours and regionprops
def feature_extraction(ppi,OI,GI):
    
    _,contours,_=cv2.findContours(ppi,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    labeled_image=cv2.drawContours(GI, contours, -1, (0,255,0), 3)
    features=measure.regionprops(labeled_image,GI,cache=True)
    return features,labeled_image


#pre processing the image 
def pre_process(img):
    
    #Otsu thresholding with binary inversion after applying Gussian Blur filter
    blur = cv2.medianBlur(img,5)
    ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    #erosion and dilation , they are optional , may be used to decrease the unwanted object detection in image 
    #if we know the kind of image we are dealing 
    #for generalised purpose we may not use them as sometimes might give less accuracy
    kernel = np.ones((5,5),np.uint8)
    #th = cv2.dilate(th,kernel,iterations = 1)
    #th = cv2.erode(th,kernel,iterations = 3)
    return th


features=[]
target=[]

path=r'Image_data\Non Hindi'

# go through each image of non hindi and find features and append to feature list 
for root,dir_name,filename in os.walk(path):
    for name in filename:
        path = os.path.join(root,name)
        
        Image=cv2.imread(path)
        #plt.imshow(Image)
        #plt.show()
        Gray_Image=cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        pre_processed_img=pre_process(Gray_Image)
        Region_Prop,Labeled_Image=feature_extraction(pre_processed_img,Image,Gray_Image)
        
        for i in Region_Prop:
            #print("here")
            Convex_area=i.convex_area
            Eccentricity=i.eccentricity
            Euler_number=i.euler_number
            Inertia_tensor_eigval=list(i.inertia_tensor_eigvals)
            Moments_hu=list(i.moments_hu)
            Solidity=i.solidity
            Weighted_moments_hu=list(i.weighted_moments_hu)
            features.append([Convex_area,Eccentricity,Euler_number,Inertia_tensor_eigval[0],Inertia_tensor_eigval[1],Moments_hu[0],Moments_hu[1],Moments_hu[2],Solidity, Weighted_moments_hu[0], Weighted_moments_hu[1], Weighted_moments_hu[2],0])
            

path=r'Image_data\Hindi'


# go through each image of hindi and find features and append to feature list 
for root,dir_name,filename in os.walk(path):
    for name in filename:
        path = os.path.join(root,name)
        
        Image=cv2.imread(path)
        #plt.imshow(Image)
        #plt.show()
        Gray_Image=cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        pre_processed_img=pre_process(Gray_Image)
        Region_Prop,Labeled_Image=feature_extraction(pre_processed_img,Image,Gray_Image)
        
        for i in Region_Prop:
            #print("here")
            Convex_area=i.convex_area
            Eccentricity=i.eccentricity
            Euler_number=i.euler_number
            Inertia_tensor_eigval=list(i.inertia_tensor_eigvals)
            Moments_hu=list(i.moments_hu)
            Solidity=i.solidity
            Weighted_moments_hu=list(i.weighted_moments_hu)
            features.append([Convex_area,Eccentricity,Euler_number,Inertia_tensor_eigval[0],Inertia_tensor_eigval[1],Moments_hu[0],Moments_hu[1],Moments_hu[2],Solidity, Weighted_moments_hu[0], Weighted_moments_hu[1], Weighted_moments_hu[2],1])
            

#create a csv file for using calculated features
#may take several minutes /hours depending on amount of data stored

with open ("Image_Data.csv",'w',newline='')as file:
    writer=csv.writer(file,delimiter=',')
    for data in features :
        writer.writerow(data)
        
    
