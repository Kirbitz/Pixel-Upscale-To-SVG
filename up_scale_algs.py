import numpy as np


def three_or_more_equal(a, b, c, d):
    return (np.array_equal(a, b) and (np.array_equal(a, c) or np.array_equal(a, d))) or (np.array_equal(a, c) and np.array_equal(a, d)) or (np.array_equal(b, c) and np.array_equal(b, d))

def nearest_neighbor(img, Iterations=1):
    for k in range(Iterations):
        img_scaled = []
        for i in range(0, len(img)):
            img_row_top = []
            img_row_bottom = []
            for j in range(0, len(img[i])):
                p = np.array([img[i][j], img[i][j], img[i][j], img[i][j]])

                img_row_top.append(p[0])
                img_row_top.append(p[1])
                img_row_bottom.append(p[2])
                img_row_bottom.append(p[3])
            
            img_scaled.append(img_row_top)
            img_scaled.append(img_row_bottom)
        
        img = np.array(img_scaled, dtype=np.uint8)
    return img

def EPX(img, Iterations=1):
    for k in range(Iterations):
        img_scaled = []
        for i in range(0, len(img)):
            img_row_top = []
            img_row_bottom = []
            for j in range(0, len(img[i])):
                p = np.array([img[i][j], img[i][j], img[i][j], img[i][j]])

                if not (i == 0 or j == 0 or i == len(img) - 1 or j == len(img[i]) - 1):
                    a, b, c, d = img[i-1][j], img[i][j+1], img[i][j-1], img[i+1][j]

                    if not three_or_more_equal(a, b, c, d):
                        if np.array_equal(c, a):
                            p[0] = a
                        if np.array_equal(a, b):
                            p[1] = b
                        if np.array_equal(d, c):
                            p[2] = c
                        if np.array_equal(b, d):
                            p[3] = d

                img_row_top.append(p[0])
                img_row_top.append(p[1])
                img_row_bottom.append(p[2])
                img_row_bottom.append(p[3])
            
            img_scaled.append(img_row_top)
            img_scaled.append(img_row_bottom)
        
        img = np.array(img_scaled, dtype=np.uint8)
    return img

def scale_2x(img, Iterations=1):
    for k in range(Iterations):
        img_scaled = []
        for i in range(0, len(img)):
            img_row_top = []
            img_row_bottom = []
            for j in range(0, len(img[i])):
                p = np.array([img[i][j], img[i][j], img[i][j], img[i][j]])

                if not (i == 0 or j == 0 or i == len(img) - 1 or j == len(img[i]) - 1):
                    a, b, c, d = img[i-1][j], img[i][j+1], img[i][j-1], img[i+1][j]

                    if np.array_equal(c, a) and not np.array_equal(c, d) and not np.array_equal(a, b):
                        p[0] = a
                    if np.array_equal(a, b) and not np.array_equal(a, c) and not np.array_equal(b, d):
                        p[1] = b
                    if np.array_equal(d, c) and not np.array_equal(d, b) and not np.array_equal(c, a):
                        p[2] = c
                    if np.array_equal(b, d) and not np.array_equal(b, a) and not np.array_equal(d, c):
                        p[3] = d

                img_row_top.append(p[0])
                img_row_top.append(p[1])
                img_row_bottom.append(p[2])
                img_row_bottom.append(p[3])
            
            img_scaled.append(img_row_top)
            img_scaled.append(img_row_bottom)
        
        img = np.array(img_scaled, dtype=np.uint8)
    return img

def eagle_2x(img, Iterations=1):
    for k in range(Iterations):
        imgScaled = []
        for i in range(0, len(img)):
            imgTopRow = []
            imgBottomRow = []
            for j in range(0, len(img[1])):
                p = np.full((4,3),img[i,j])

                if not(i == 0 or j == 0 or i == len(img) - 1 or j == len(img[1]) - 1):
                    if np.array_equal(img[i,j-1],img[i-1,j-1]) and np.array_equal(img[i,j-1], img[i-1,j]):
                        p[0] = img[i-1,j-1]
                    if np.array_equal(img[i-1,j], img[i-1,j+1]) and np.array_equal(img[i-1,j],img[i,j+1]):
                        p[1] = img[i-1,j+1]
                    if np.array_equal(img[i,j-1],img[i+1,j-1]) and np.array_equal(img[i,j-1], img[i+1,j]):
                        p[2] = img[i+1,j-1]
                    if np.array_equal(img[i,j+1],img[i+1,j+1]) and np.array_equal(img[i,j+1], img[i+1,j]):
                        p[3] = img[i+1,j+1]

                imgTopRow.append(p[0])
                imgTopRow.append(p[1])
                imgBottomRow.append(p[2])
                imgBottomRow.append(p[3])
            imgScaled.append(imgTopRow)
            imgScaled.append(imgBottomRow)
        img = np.array(imgScaled, dtype=np.uint8)
    return img

