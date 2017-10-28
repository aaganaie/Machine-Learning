# Machine-Learning

This is a Hindi/Devnagri Language detection program , made in jupyter and python idle.

A large Dataset of images were used . 
csv of this Dataset is on -: https://drive.google.com/open?id=0B4LbH2aC3oMeb3JMNFdMVmRHZ28
Download the above File and keep in the same folder with other file on reprository for program to work .
(LIMIT OF GITHUB IS 25 MB and file has 94MB size)

IF someone wants to get the original images of Hindi and Non hindi images contact me on -: arshadfriendly.ganaie@gmail.com

First the input image is preprocessed by using otsu threshholding. erosion and dilation can also be used.
Next we used contours of cv2 and Regionprops of skimage to extract features of image detected region.
RandomForestClassifier is used to classify the text in images. It is trained with features and targets,

