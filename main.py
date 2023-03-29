import cv2
import numpy as np
import os
from up_scale_algs import scale_2x

def main():
    # for filename in os.listdir(os.path.join(os.getcwd(), 'pixel_art')):
    #     img = cv2.imread(os.path.join(os.getcwd(), 'pixel_art', filename), cv2.IMREAD_COLOR)
    #     img = EPX(img)
    #     cv2.imshow(filename, img)
    
    img = cv2.imread("pixel_art/enter_the_gungeon_guns.png", cv2.IMREAD_COLOR)
    img = scale_2x(img, Iterations=2)
    cv2.imshow("Read and Show", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()