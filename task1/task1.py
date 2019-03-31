
import numpy as np
import cv2 
import math
img = np.array(cv2.imread(r"C:\Users\aditya vikram\Desktop\CVIP_PROJECT3\original_imgs\original_imgs\noise.jpg",0))
out_img = np.zeros((img.shape))
b = np.array([[1,1,1],
             [1,1,1],
             [1,1,1]])

k_h = b.shape[0]
k_w = b.shape[1]
h= b.shape[0]//2
w= b.shape[1]//2
def erosion(sample):
    img = sample
    output_img = np.zeros((sample.shape))
    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            cnt = 255
            for x in range(i-h,i+h+1):
                for y in range(j-w,j+w+1):
                    if(x>=0 and x<sample.shape[0] and y>=0 and y<sample.shape[1]) :
                        if(img[x][y]<cnt):
                            cnt = img[x][y]
            output_img[i][j]=cnt                          
    return output_img   

def dilation(sample):
    img = sample
    out_img = np.zeros((sample.shape))
    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            cnt = 0
            for x in range(i-h,i+h+1):
                for y in range(j-w,j+w+1):
                    if( x>=0 and  x<sample.shape[0] and y>0 and y<sample.shape[1]) :
                        if(img[x][y]>cnt):
                            cnt = img[x][y]
            out_img[i][j]=cnt                        
    return out_img    
#opening	
e1 = erosion(img)
d1 = dilation(e1)
#closing
d2 = dilation(d1)
e2 = erosion(d2)
cv2.imwrite('res_noise1.png',e2)
cv2.imshow("open_close",e2)
cv2.waitKey(0)
cv2.destroyAllWindows()
#closing
d3 = dilation(img)
e3 = erosion(d3)
#opening
e4 = erosion(e3)
d4 = dilation(e4)
cv2.imwrite('res_noise2.png',d4)
cv2.imshow("close_open",d4)
cv2.waitKey(0)
cv2.destroyAllWindows()

def boundary_extraction(image1,image2):
    erode_1 = erosion(image1)
    op_1 = np.subtract(image1,erode_1)
    cv2.imshow("first",op_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
	cv2.imwrite('res_bound1.png',op_1)
    erode_2 = erosion(image2)
    op_2 = np.subtract(image2,erode_2)
    cv2.imshow('second',op_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
	cv2.imwrite('res_bound2.png',op_2)
#boundary extraction
boundary_extraction(e2,d4)

