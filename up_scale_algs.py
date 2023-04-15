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

def bilinear(img,scaleFactor): #There is a vectorized version of this that runs faster
    
    height = len(img) * scaleFactor
    width = len(img[1]) * scaleFactor
    imgScaled = np.zeros((height, width, 3))
    xRatio = float(len(img[1]) - 1) / (width - 1)
    yRatio = float(len(img)-1) / (height - 1)

    for i in range(height):
        for j in range(width):
            x1 = np.uint32(np.floor(xRatio * j ))
            y1 = np.uint32(np.floor(yRatio * i))
            xh = np.uint32(np.ceil(xRatio * j))
            yh = np.uint32(np.ceil(yRatio * i))
            xWeight = (xRatio * j) - x1
            yWeight = (yRatio * i) - y1
            a = img[y1, x1]
            b = img[y1, xh]
            c = img[yh, x1]
            d = img[yh, xh]
            pixel = a * (1 - xWeight) * (1 - yWeight) + b * xWeight * (1 - yWeight) + c * yWeight * (1-xWeight) + d * xWeight * yWeight
            imgScaled[i,j] = pixel
    img = np.array(imgScaled, dtype=np.uint8)
    return img

# Interpolation kernel
def _u(s, a):
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0
  
  
# Padding
def _padding_(img, H, W, C):
    zimg = np.zeros((H+4, W+4, C))
    zimg[2:H+2, 2:W+2, :C] = img
      
    # Pad the first/last two col and row
    zimg[2:H+2, 0:2, :C] = img[:, 0:1, :C]
    zimg[H+2:H+4, 2:W+2, :] = img[H-1:H, :, :]
    zimg[2:H+2, W+2:W+4, :] = img[:, W-1:W, :]
    zimg[0:2, 2:W+2, :C] = img[0:1, :, :C]
      
    # Pad the missing eight points
    zimg[0:2, 0:2, :C] = img[0, 0, :C]
    zimg[H+2:H+4, 0:2, :C] = img[H-1, 0, :C]
    zimg[H+2:H+4, W+2:W+4, :C] = img[H-1, W-1, :C]
    zimg[0:2, W+2:W+4, :C] = img[0, W-1, :C]
    return zimg
# Bicubic operation, the math for this operation can be found here:https://link.springer.com/article/10.1007/s11554-022-01254-8
def bicubic(img, ratio):
    a= -.5
    # Get image size
    H, W, C = img.shape
    # Here H = Height, W = weight,
    # C = Number of channels if the 
    # image is coloured.
    img = _padding_(img, H, W, C) / 255.0
      
    # Create new image
    dH = np.uint32(np.floor(H*ratio))
    dW = np.uint32(np.floor(W*ratio))
    dst = np.zeros((dH, dW, 3))  
    h = 1/ratio
    for c in range(C):
        for j in range(dH):
            for i in range(dW):
                # Getting the coordinates of the
                # nearby values
                x, y = i * h + 2, j * h + 2
                x1 = 1 + x - np.floor(x)
                x2 = x - np.floor(x)
                x3 = np.floor(x) + 1 - x
                x4 = np.floor(x) + 2 - x
  
                y1 = 1 + y - np.floor(y)
                y2 = y - np.floor(y)
                y3 = np.floor(y) + 1 - y
                y4 = np.floor(y) + 2 - y
                  
                # make the x matrix, the pixelMatrix, and the yMatrix here. in the paper they are denoted F(u), C and F(v)
                #For some reason, changing the variable names breaks line 209.
                mat_l = np.matrix([[_u(x1, a), _u(x2, a), _u(x3, a), _u(x4, a)]])
                mat_m = np.matrix([[img[int(y-y1), int(x-x1), c],img[int(y-y2), int(x-x1), c],img[int(y+y3), int(x-x1), c],img[int(y+y4), int(x-x1), c]],
                                   [img[int(y-y1), int(x-x2), c],img[int(y-y2), int(x-x2), c],img[int(y+y3), int(x-x2), c],img[int(y+y4), int(x-x2), c]],
                                   [img[int(y-y1), int(x+x3), c],img[int(y-y2), int(x+x3), c],img[int(y+y3), int(x+x3), c],img[int(y+y4), int(x+x3), c]],
                                   [img[int(y-y1), int(x+x4), c],img[int(y-y2), int(x+x4), c],img[int(y+y3), int(x+x4), c],img[int(y+y4), int(x+x4), c]]])
                mat_r = np.matrix([[_u(y1, a)], [_u(y2, a)], [_u(y3, a)], [_u(y4, a)]])
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)
    return dst
  

#_wd calculates the distance between the colors of two pixels.
def _wd_(p1, p2):
    b,g,r = abs(p1[0] - p2[0]), abs(p1[1] - p2[1]), abs(p1[2] - p2[2]) 
    y = abs(r*0.299 + g*0.587 + b*0.144)
    u = abs(r*-0.169 - g*0.331 + b*0.500)
    v = abs(r*0.500 - g*0.419 - b*0.081)
    return (48*y + 7*u + 6*v)

def testXbr(img):
    print(_wd_(img[2,2],img[3,3]))
def _xbrInterp_(e,f,h):
    newColor = f if(_wd_(e,f) <= _wd_(e,h)) else h
    return .5*e + .5*newColor
'''
description of the algorithm can be found here: https://forums.libretro.com/t/xbr-algorithm-tutorial/123 
Pixels are represented in the following format
       |A1|B1|C1|
    |A0|A |B |C |C4|
    |D0|D |E |F |F4|
    |G0|G |H |I |I4|
       |G5|H5|I5|  
Consider E as the central pixel.'''
def xBR(img, Iterations=1):
    for k in range(Iterations):
        padded = np.zeros((len(img) + 6, len(img[1]) + 6,3), dtype=np.uint8)
        padded[3:-3,3:-3]= img
        imgScaled = np.zeros((len(img) *2, len(img[1]) * 2,3), dtype=np.uint8)
        img = padded
        
        ratio = 1/2
        for row in range(len(imgScaled)):
            for col in range(len(imgScaled[1])):
                x,y = np.uint32(np.floor(col * ratio) + 3), np.uint32(np.floor(row * ratio) + 3)
                a1, b1, c1 = img[y-2,x-1], img[y-2,x], img[y-2,x+1]
                a0, a, b, c, c4 = img[y-1,x-2], img[y-1,x-1], img[y-1,x], img[y-1,x+1], img[y-1,x+2]
                d0, d, e, f ,f4 = img[y,x-2], img[y,x-1], img[y,x], img[y,x+1], img[y,x+2]
                g0, g, h, i, i4 = img[y+1,x-2], img[y+1,x-1], img[y+1,x], img[y+1, x+1], img[y+1,x+2]
                g5, h5, i5 = img[y+2,x-1], img[y+2, x], img[y+2,x+1] 
                edge1 = _wd_(e,c) + _wd_(e,g) + _wd_(i,f4) + _wd_(i,h5) + 4*_wd_(h,f)
                edge2 = _wd_(h,d) + _wd_(h,i5) + _wd_(f,i4) + _wd_(f,b) + 4*_wd_(e,i)
                
                if(edge1 < edge2):
                    imgScaled[row,col] = _xbrInterp_(e,f,h)
                else:
                    imgScaled[row,col] = e
        img = np.array(imgScaled, dtype = np.uint8)
    return img