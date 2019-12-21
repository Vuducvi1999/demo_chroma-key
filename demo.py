import cv2
import numpy as np


def nothing(x):
    pass


# tạo thanh trackbar
cv2.namedWindow('trackbar')
cv2.createTrackbar('R', 'trackbar', 0, 255, nothing)
cv2.createTrackbar('G', 'trackbar', 0, 255, nothing)
cv2.createTrackbar('B', 'trackbar', 0, 255, nothing)
cv2.createTrackbar('S', 'trackbar', 0, 255, nothing)
cv2.createTrackbar('V', 'trackbar', 0, 255, nothing)

# ảnh background
image_bg = cv2.imread('background image/bg4.jpg')
# ảnh chroma key
img_source = cv2.imread('input/number7.jpg')

# ảnh background đã được resize = kích cỡ ảnh chroma
x, y, z = img_source.shape
image_bg = cv2.resize(image_bg, (y, x))
# ảnh gốc được convert sang HSV
img_converted = cv2.cvtColor(img_source, cv2.COLOR_BGR2HSV)
# ảnh lưu kết quả chỉnh sửa cuối cùng
temp = None
while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get thông tin từ trackbar
    r = cv2.getTrackbarPos('R', 'trackbar')
    g = cv2.getTrackbarPos('G', 'trackbar')
    b = cv2.getTrackbarPos('B', 'trackbar')
    s = cv2.getTrackbarPos('S', 'trackbar')
    v = cv2.getTrackbarPos('V', 'trackbar')

    # get thông tin màu background hệ RGB cần detect
    value_color = np.uint8([[[b, g, r]]])
    # chuyển đổi màu background sang hệ màu HSV
    hsvGreen = cv2.cvtColor(value_color, cv2.COLOR_BGR2HSV)
    # ngưỡng dùng để detect background
    low = hsvGreen[0, 0, 0]-20, s, v
    height = hsvGreen[0, 0, 0]+20, 255, 255
    print(low, height)
    # trường hợp background màu đỏ
    if low[0] < 0 or height[0] > 180:
        # Range for lower red
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(img_converted, lower_red, upper_red)
        # Range for upper range
        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask = cv2.inRange(img_converted, lower_red, upper_red)
    else:
        lower = np.array(low)
        upper = np.array(height)
        # bắt đầu detect background
        mask = cv2.inRange(img_converted, lower, upper)

    # morphology erode => xói mòn background => mở rộng viền object
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones((2, 2), np.uint8))
    cv2.imshow('background_detect', mask)

    # detect object
    mask_not = cv2.bitwise_not(mask)
    cv2.imshow('object_detect', mask_not)

    # hiển thị object
    final = cv2.bitwise_and(img_source, img_source, mask=mask_not)
    cv2.imshow('show_object', final)

    # hiển thị background
    final_not = cv2.bitwise_and(image_bg, image_bg, mask=mask)
    cv2.imshow('show_background', final_not)

    # tạo ảnh đã remove background
    final_output = final_not+final
    # smooth image
    final_output = cv2.GaussianBlur(final_output, (3, 3), 1)
    temp = final_output
    # Show ảnh
    cv2.imshow('final_image', final_output)

cv2.destroyAllWindows()
cv2.imwrite('number9.png', temp)
