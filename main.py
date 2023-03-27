import cv2
import numpy as np
import os

def main():
    for filename in os.listdir(os.path.join(os.getcwd(), 'pixel_art')):
        img = cv2.imread(os.path.join(os.getcwd(), 'pixel_art', filename), cv2.IMREAD_COLOR)

        cv2.imshow(filename, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()