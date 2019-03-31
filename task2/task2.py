import numpy as np
import cv2 
img = np.array(cv2.imread(r"C:\Users\aditya vikram\Desktop\CVIP_PROJECT3\original_imgs\original_imgs\turbine.jpg",0))
b = np.array([[-1,-1,-1,-1,-1],
             [-1,-1,-1,-1,-1],
             [-1,-1,24,-1,-1],
             [-1,-1,-1,-1,-1],
             [-1,-1,-1,-1,-1]])
k_h = b.shape[0]
k_w = b.shape[1]
h= b.shape[0]//2
w= b.shape[1]//2
output_img = np.zeros((img.shape))
def conv(image,b):
    for i in range(h,image.shape[0]-h):
        for j in range(w,image.shape[1]-w):
            s = 0
            for x in range(k_w):
                for y in range(k_h):
                    s = s + (b[x][y]*img[i-h+x][j-w+y])
                    output_img[i][j]=s                          
    return output_img
output_img = np.abs(conv(img,b))
threshold= int(np.max(output_img)) 
threshold= (0.90 * threshold)
cord = []
for i in range(output_img.shape[0]):
    for j in range(output_img.shape[1]):
        if(output_img[i][j]<threshold):
            output_img[i][j]= 0
        else:
            cord.append((i,j))
cv2.putText(output_img," (249,445)",(cord[0][1],cord[0][0]),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255))
cv2.imshow("detected_image",output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("res_point.jpg",output_img)
print("location of detected point",cord)
print("START OF TASK 2b")
import matplotlib.pyplot as plt
image = cv2.imread(r"C:\Users\aditya vikram\Desktop\CVIP_PROJECT3\original_imgs\original_imgs\segment.jpg",0)
image_bbox = cv2.imread(r"C:\Users\aditya vikram\Desktop\proj3_cvip_outputs\task2\res_segment.png")
out_img = image
fl = image.flatten()
nonzero = np.nonzero(fl)
from collections import Counter
d = Counter(fl[nonzero])
lists = sorted(d.items())
x,y = zip(*lists)
plt.plot(x,y)
plt.show()

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if(image[i][j]<208):
            image[i][j] =0 
        else:
            image[i][j]=255
cv2.imshow("detect",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("res_segment.jpg",image)
cv2.imshow("rectangle",image_bbox)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("res_boundingbox.jpg",image_bbox)
