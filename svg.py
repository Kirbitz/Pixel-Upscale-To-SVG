import cv2
import numpy as np

def convert_to_svg(img):
    height, width = img.shape[:2]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("test", img_hsv)
    blue_mask = cv2.inRange(img_hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))[1]
    contours = cv2.findContours(blue_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)[0]

    svg_file = open('svg/test.svg', 'w+')
    svg_file.write(f"<svg width={str(width)} height={str(height)} xmlns=\"http://www.w3.org/2000/svg\">")

    for contour in contours:
        svg_file.write("<path d=\"M")
        for i in range(len(contour)):
            x, y = contour[i][0]
            svg_file.write(f"{str(x)} {str(y)} ")
        svg_file.write('" fill="blue"/>')

    svg_file.write("</svg>")
    svg_file.close()