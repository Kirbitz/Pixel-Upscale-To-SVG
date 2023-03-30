import cv2
import numpy as np
import os
import up_scale_algs as usa

def main():
    # for filename in os.listdir(os.path.join(os.getcwd(), 'pixel_art')):
    #     img = cv2.imread(os.path.join(os.getcwd(), 'pixel_art', filename), cv2.IMREAD_COLOR)
    #     img = usa.scale_2x(img)
    #     cv2.imshow(filename, img)
    
    img = cv2.imread("pixel_art/super_mario_3_mario.png", cv2.IMREAD_COLOR)
    img = usa.scale_2x(img, Iterations=1)
    cv2.imshow("Read and Show", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()