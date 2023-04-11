import cv2
import up_scale_algs as usa
import svg as svg

gui_name = "Image Window"
img = cv2.imread("pixel_art/wikipedia_sample_img.png", cv2.IMREAD_COLOR)
scale = 4
scale_type = 0

def image_upscale(x):
    global scale_type
    scale_type = x

    if scale_type == 0:
        img_show = usa.nearest_neighbor(img, 2 ** scale)
    elif scale_type == 1:
        img_show = usa.EPX(img, Iterations=scale)
        svg.convert_to_svg(img_show)
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
    cv2.createTrackbar('Scale Type', gui_name, 0, 4, image_upscale)
    image_upscale(0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()