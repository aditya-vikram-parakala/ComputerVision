import numpy as np
import cv2 
img = np.array(cv2.imread(r"C:\Users\aditya vikram\Desktop\CVIP_PROJECT3\original_imgs\original_imgs\hough.jpg",0))
b = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
bt = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
img_original = np.array(cv2.imread(r"C:\Users\aditya vikram\Desktop\CVIP_PROJECT3\original_imgs\original_imgs\hough.jpg"))
k_h = b.shape[0]
k_w = b.shape[1]
h= b.shape[0]//2
w= b.shape[1]//2
output_img= np.zeros((img.shape))
shapes_blurred = cv2.GaussianBlur(img, (5, 5), 1.5)
def conv(image,b):
    for i in range(h,image.shape[0]-h):
        for j in range(w,image.shape[1]-w):
            s = 0
            for x in range(k_w):
                for y in range(k_h):
                    s = s + (b[x][y]*img[i-h+x][j-w+y])
                    output_img[i][j]=s                          
    return output_img
c_x = conv(shapes_blurred,b)
c_y = conv(shapes_blurred,bt)
edge_magnitude = np.sqrt(c_x ** 2 + c_y ** 2)
new_img = edge_magnitude
for i in range(edge_magnitude.shape[0]):
    for j in range(edge_magnitude.shape[1]):
        if(edge_magnitude[i][j]<100):
            new_img[i][j] = 0    
edges = new_img 
edges_2 = new_img
edge = cv2.Canny(shapes_blurred,100,200)
edge_2 = cv2.Canny(shapes_blurred,100,200)
#Accumulator for the image
def accumulator_matrix(img,a1,a2, rho_steps=1, theta_steps=0.5):
    ht,wd=img.shape 
    # maximum rhos value i.e. diagonal of the image
    img_diagonal = np.ceil(np.sqrt(ht**2 + wd**2)) 
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_steps)
    angles = np.deg2rad(np.arange(a1, a2, theta_steps))
    #hough space of dimensions rhosxtheta
    H = np.zeros((len(rhos), len(angles)))
    #indices of pixels without clutter(noise)
    yindex, xindex = np.nonzero(img)
    for i in range(len(xindex)): # looping through edge points
        x = xindex[i]
        y = yindex[i]
        #calculating the rho values for each theta we have (x,y) from above step
        for j in range(len(angles)):
            p = (x * np.cos(angles[j]) + y * np.sin(angles[j]))
            rho=int(p+img_diagonal)
            H[rho, j]=H[rho, j]+1
    return H, rhos, angles
def hough_peaks(H,num_peaks,neighborhood_size=3):
    #to find local maxima in the hough space  
    indicies = []
    H_new = np.copy(H)
    for i in range(num_peaks):
        index_x = np.argmax(H_new) # find argmax in flattened array
        H1_idx = np.unravel_index(index_x,H_new.shape) 
        indicies.append(H1_idx)
        idx_y, idx_x = H1_idx 
        if (idx_x - (neighborhood_size/2)) < 0: 
            min_x = 0
        else: 
            min_x = idx_x - (neighborhood_size/2)
        if ((idx_x + (neighborhood_size/2) + 1) > H.shape[1]): 
            max_x = H.shape[1]
        else: 
            max_x = idx_x + (neighborhood_size/2) + 1
        if (idx_y - (neighborhood_size/2)) < 0: 
            min_y = 0
        else: 
            min_y = idx_y - (neighborhood_size/2)
        if ((idx_y + (neighborhood_size/2) + 1) > H.shape[0]): 
            max_y = H.shape[0]
        else: 
            max_y = idx_y + (neighborhood_size/2) + 1
        min_x = int(min_x)
        min_y = int(min_y)
        max_x = int(max_x)
        max_y = int(max_y)
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                #remove neighborhosods in H1
                H_new[y, x] = 0
                #mark peaks in original H
                if (x == min_x or x == (max_x - 1)):
                    H[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y, x] = 255
    return indicies, H

def draw_lines(img, indicies, rhos, angles):
    s_factor=1000 # scaling factor to extend the line drawn     
    for i in range(len(indicies)):
        rho = rhos[indicies[i][0]]
        theta = angles[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x_a = int(x0 - s_factor*b)
        y_a = int(y0 + s_factor*a)
        x_b = int(x0 + s_factor*b)
        y_b = int(y0 - s_factor*a)
        cv2.line(img,(x_a, y_a),(x_b, y_b),(0, 0, 255),3)
H,rhos,angles = accumulator_matrix(edge,-10,20)
indicies,H = hough_peaks(H, 10, neighborhood_size=7) 
draw_lines(img_original, indicies, rhos, angles)
cv2.imshow('Vertical Lines',img_original)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("red_lines.jpg",img_original)
img_color = cv2.imread(r'C:\Users\aditya vikram\Desktop\CVIP_PROJECT3\original_imgs\original_imgs\hough.jpg')
H, rhos, angles = accumulator_matrix(edge_2,-40,-20)
indicies, H = hough_peaks(H,15,11)
draw_lines(img_color, indicies, rhos, angles)
cv2.imshow('Diagonal Lines',img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("blue_lines.jpg",img_color)
indices= []
image_original = cv2.imread(r'C:\Users\aditya vikram\Desktop\CVIP_PROJECT3\original_imgs\original_imgs\hough.jpg')
circle = cv2.Canny(image_original,100,200)

def hough_circle(img,angle1,angle2,step_angles):
    radius = 22
    theta = np.arange(angle1,angle2,step_angles)
    theta = np.deg2rad(theta)
    hspace = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i][j]>190):
                for z in range(len(theta)):        
                    a = int((i - radius*np.cos(theta[z])))
                    b = int((j- radius*np.sin(theta[z])))
                    try:
                        hspace[a,b] = hspace[a,b]+1
                    except:
                        pass
    cv2.imwrite("cir_accum.jpg",hspace)
    return hspace                    
hough = hough_circle(circle,0,360,1)
cor = []
for i in range(hough.shape[0]):
    for j in range(hough.shape[1]):
        if(hough[i][j]>150):
            cor.append((j,i))
for k in range(len(cor)):
    cv2.circle(image_original,cor[k],22,(0,0,255),2)
cv2.imshow("circles",image_original)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("coin.jpg",image_original)
#source: refered stackoverflow, github repo, pythoncreek for understanding the concept and a part of implementation. 