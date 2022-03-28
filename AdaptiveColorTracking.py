import cv2
import pandas as pd
import colorsys
import numpy as np
from numpy.linalg import norm


#img_path = r'C:\Users\Balaji\Downloads\color detection\colorpic.jpg'
#img = cv2.imread('lenna.png')
cap = cv2.VideoCapture(0)

# declaring global variables (are used later on)
clicked = False
first= False
auto=False
r = g = b = x_pos = y_pos = 0

# function to get x,y coordinates of mouse double click
def draw_function(event, x, y, flags, param):
    if (event == cv2.EVENT_LBUTTONDBLCLK or auto==True):
        global b, g, r, x_pos, y_pos, clicked
        clicked = True
        x_pos = x
        y_pos = y
        b, g, r = frameG[y, x]
        b = int(b)
        g = int(g)
        r = int(r)

def brightness(img):
    if len(img.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        return np.average(norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(img)


cv2.namedWindow('frame')
cv2.setMouseCallback('frame', draw_function)
text2='a'
kernel = np.ones((5,5), np.uint8) 

lower_blue = np.array([30, 30, 30])
upper_blue = np.array([40, 40, 40])
text = ' R=' 
br=0
br0=[]
i=0

br0test=[]
brtest=[]
global xs, ys

while True:
    _, imageFrame = cap.read()
    frameG = cv2.GaussianBlur(imageFrame, (7,7), 0)#cv2.medianBlur(frame, 21)#
    frameG = cv2.erode(frameG, kernel, iterations = 2)
    frameG = cv2.dilate(frameG, kernel, iterations = 1)
    
    #i+=1
    #if(i%2==0):
    br0.append(br)
    br=brightness(imageFrame)
    dbr=np.abs(br-br0[-1])
    del br0[:-1]
    

    if(first==True and dbr>2):
        auto=True
        event=1
        flags=1
        param=1
        x=int(xs)
        y=int(ys)
        draw_function(event, x, y, flags, param)
        auto=False

    
    if clicked:
        first=True
        # cv2.rectangle(image, start point, endpoint, color, thickness)-1 fills entire rectangle
        cv2.rectangle(imageFrame, (20, 20), (750, 60), (b, g, r), -1)

        # Creating text string to display( Color name and RGB values )
        text = ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)
        
        hsv = cv2.cvtColor(frameG, cv2.COLOR_BGR2HSV)
        h,s,v=colorsys.rgb_to_hsv(r/255, g/255, b/255)
        h=h*360

        text2 = ' H=' + str(round(h,2)) + ' S=' + str(round(s,2)) + ' V=' + str(round(v,2))

        lower_blue = np.array([(h-11)/360*179, (s-0.3)*255, (v-0.2)*255])
        upper_blue = np.array([(h+11)/360*179, (s+0.3)*255, (v+0.2)*255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue) 

        # cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType )
        cv2.putText(imageFrame, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # For very light colours we will display text in black colour
        if r + g + b >= 600:
            cv2.putText(imageFrame, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(imageFrame, text2, (50, 80), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA) 

        clicked = False
        
    hsv = cv2.cvtColor(frameG, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue) 
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            x, y, width, height = cv2.boundingRect(cnt)

            imageFrame=cv2.rectangle(imageFrame, (x , y), (x + width, y + height),(0, 0, 255), 2)
            
            xs=int(x+width/2)
            ys=int(y+height/2)
            
            cv2.rectangle(imageFrame,(xs,ys),(xs+2,ys+2),(255,0,255),3)
            
    cv2.rectangle(imageFrame, (20, 20), (750, 60), (b, g, r), -1)
    cv2.putText(imageFrame, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)    
    cv2.putText(imageFrame, text2, (50, 80), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)    


    cv2.imshow("frame",imageFrame)
    cv2.imshow("mask",mask)

    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
