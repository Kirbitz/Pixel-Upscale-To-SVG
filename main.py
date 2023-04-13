import cv2
import numpy as np
import up_scale_algs as usa

gui_name = "Image Window"

ANIMATED = True
file_path = "pixel_art\wikipedia_sample_img.png"
SPRITE_WIDTH, SPRITE_HEIGHT = 16, 16
FPS = 5
FRAME_TIME = 1000 // FPS # milliseconds per frame of animation
currentFrame = -1

img = cv2.imread(file_path, cv2.IMREAD_COLOR)
full_img = img.copy()

sprites_per_row = img.shape[1] // SPRITE_WIDTH
sprites_per_column = img.shape[0] // SPRITE_HEIGHT

scale = 4
scale_type = 0

def update_scale_type(x):
    global scale_type
    scale_type = x
    update_image_upscale()

def update_image_frame(frame_increment = 1):
    global full_img, img, currentFrame
    currentFrame += frame_increment
    currentFrame = currentFrame % (sprites_per_row * sprites_per_column)

    img = full_img[(currentFrame // sprites_per_row) * SPRITE_HEIGHT:((currentFrame // sprites_per_row) + 1) * SPRITE_HEIGHT, 
                   (currentFrame % sprites_per_row) * SPRITE_WIDTH:((currentFrame % sprites_per_row) + 1) * SPRITE_WIDTH]

    update_image_upscale()

def update_image_upscale():
    if scale_type == 0:
        img_show = usa.nearest_neighbor(img, 2 ** scale)
    elif scale_type == 1:
        img_show = usa.EPX(img, Iterations=scale)
    elif scale_type == 2:
        img_show = usa.scale_2x(img, Iterations=scale)
    elif scale_type == 3:
        img_show = usa.eagle_2x(img, Iterations=scale)
    elif scale_type == 4:
        img_show = usa.bilinear(img, 2 ** scale)
    
    cv2.imshow(gui_name, img_show)

def main():
    cv2.namedWindow(gui_name)
    global img
    
    if not ANIMATED:
        update_image_upscale()
        cv2.createTrackbar('Scale Type', gui_name, 0, 4, update_scale_type) # create the trackbar after showing the image to ensure trackbar fits within window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    
    update_image_frame(0)
    cv2.createTrackbar('Scale Type', gui_name, 0, 4, update_scale_type)

    # animation loop (runs while main window is open)
    while (cv2.getWindowProperty(gui_name, cv2.WND_PROP_VISIBLE) == 1):
        update_image_frame()
        cv2.waitKey(FRAME_TIME)
        


if __name__ == '__main__':
    main()