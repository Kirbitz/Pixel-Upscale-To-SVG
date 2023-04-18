import cv2
import numpy as np

def convert_to_svg(img, path):
    height, width = img.shape[:2]
    most_pixels = ""
    max_list = 0
    colors = {}

    for i in range(height):
        for j in range(width):
            key = f"#{img[i, j, 2]:02x}{img[i, j, 1]:02x}{img[i, j, 0]:02x}"
            if key in colors.keys():
                colors[key].append([i, j])
            else:
                colors[key] = [[i, j]]

    for key in colors.keys():
        if max_list < len(colors[key]):
            max_list = len(colors[key])
            most_pixels = key

    svg_file = open(path, 'w+')
    svg_file.write(f'<html><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -0.5 {str(width)} {str(height)}" shape-rending="crispEdges">')
    
    for color in colors.keys():
        if most_pixels == color:
            continue
        svg_file.write(f'<path stroke="{color}" d="')
        current = colors[color][0]
        w = 0
        for x_y in colors[color]:
            if current[1] == x_y[1] and x_y[0] == (current[0] + w):
                w += 1
            else:
                svg_file.write(f"M{str(current[1])} {str(current[0])}h{str(w)}")
                w = 1
                current = x_y
        svg_file.write('" />')

    svg_file.write("</svg></html>")
    svg_file.close()