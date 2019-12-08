import cv2
import numpy as np

def nothing(x):
    pass
# Create a black image, a window
img = cv2.imread('anh.jpg')
# img=np.uint8(img)
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)
cv2.createTrackbar('S','image',0,255,nothing)
cv2.createTrackbar('V','image',0,255,nothing)

# ảnh background
image_bg = cv2.imread('bg.jpg')
# ảnh chroma key
img_source = cv2.imread('number9.jpg')
x, y, z = img_source.shape
# ảnh dùng làm nền đã được cv2.resize
image_bg = cv2.resize(image_bg, (y, x))
# ảnh gốc được convert sang HSV
img_1 = cv2.cvtColor(img_source, cv2.COLOR_BGR2HSV)
temp=0
while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    
    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')
    s = cv2.getTrackbarPos('S','image')
    v = cv2.getTrackbarPos('V','image')
    value_color = np.uint8([[[b,g,r]]])
    hsvGreen = cv2.cvtColor(value_color, cv2.COLOR_BGR2HSV)
    low=hsvGreen[0,0,0]-20,s,v
    height=hsvGreen[0,0,0]+20,255,255
    lower=np.array(low)
    upper=np.array(height)

    mask=cv2.inRange(img_1,lower,upper)
    # cv2.imshow('mask_pure', mask)

    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    # mask = cv2.erode(mask, np.ones((2,2), np.uint8), iterations=1)

    mask_not=cv2.bitwise_not(mask)
    final=cv2.bitwise_and(img_source, img_source, mask = mask_not)
    final_not=cv2.bitwise_and(image_bg, image_bg, mask = mask)

    # out=final+final_not
    # out=cv2.GaussianBlur(out,(3,3),1)
    final_output = final_not+final
    final_output=cv2.GaussianBlur(final_output,(3,3),1)
    temp=final_output
    cv2.imshow('mask.png', final_output)
    # cv2.imshow('mask.png', mask)

cv2.destroyAllWindows()
cv2.imwrite('number9.png',temp)