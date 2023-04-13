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

'''
def _computeSpline_(p0,p1,p2,p3,x):
    result = p1 + 0.5 * x*(p2 - p0 + x*(2.0*p0 - 5.0*p1 + 4.0*p2 - p3 + x*(3.0*(p1 - p2) + p3 - p0)))
    print(result)
    return result
    
def bicubic(img, scaleFactor):
    baseHeight = len(img)
    baseWidth = len(img[1])
    padded = np.zeros((baseHeight+4, baseWidth+4, 3),dtype=np.uint8)
    padded[2:-2, 2:-2] = img
    img = padded 
    
    newHeight, newWidth = baseHeight * scaleFactor, baseWidth * scaleFactor
    imgScaled = np.zeros((newHeight,newWidth,3), dtype=np.uint8)
    xRatio = float(baseWidth - 1) / (newWidth - 1)
    yRatio = float(baseHeight - 1) / (newHeight - 1)
    for i in range(newHeight):
        for j in range(newWidth):
            x0 = np.uint32(np.floor(xRatio * j)) + 1 if j > 0 else 0
            x1 = np.uint32(np.floor(xRatio * j)) + 2
            x2 = np.uint32(np.ceil(xRatio * j)) + 2
            x3 = np.uint32(np.ceil(xRatio * j)) + 3
            y0 = np.uint32(np.floor(yRatio * i)) + 1 if i > 0 else 0
            y1 = np.uint32(np.floor(yRatio * i)) + 2
            y2 = np.uint32(np.ceil(yRatio * i)) + 2
            y3 = np.uint32(np.ceil(yRatio * i)) + 3
            #print(x0,x1,x2,x3)
            #print(y0,y1,y2,y3)
            #print()
            p00,p10,p20,p30 = img[y0,x0], img[y0,x1], img[y0,x2], img[y0,x3]
            p01,p11,p21,p31 = img[y1,x0], img[y1,x1], img[y1,x2], img[y1,x3]
            p02,p12,p22,p32 = img[y2,x0], img[y2,x1], img[y2,x2], img[y2,x3]
            p03,p13,p23,p33 = img[y3,x0], img[y3,x1], img[y3,x2], img[y3,x3]
            
            deltaX = x2 - x1
            #deltaX = ((xRatio * j) + 2) - x1
            deltaY = y2 - y1
            #deltaY = ((yRatio * i) + 2) - y1
            #print(p00,p10,p20,p30)
            #print()
            #imgScaled[i,j] = _computeSpline_(_computeSpline_(p00,p10,p20,p30, deltaX), _computeSpline_(p01,p11,p21,p31,deltaX),_computeSpline_(p02,p12,p22,p32,deltaX),_computeSpline_(p03,p13,p23,p33,deltaX),deltaY)
            imgScaled[i,j] = _computeSpline_(_computeSpline_(p00,p01,p02,p03,deltaY),_computeSpline_(p10,p11,p12,p13,deltaY),_computeSpline_(p20,p21,p22,p23,deltaY),\
                                             _computeSpline_(p30,p31,p32,p33,),deltaX)
            print()
    img = np.array(imgScaled, dtype=np.uint8)
    return img
'''
# Interpolation kernel
def _u(s, a):
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0
  
  
# Padding
def _padding(img, H, W, C):
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
    img = _padding(img, H, W, C) / 255.0
      
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
                #For some reason, changing the variable names breaks line 265.
                mat_l = np.matrix([[_u(x1, a), _u(x2, a), _u(x3, a), _u(x4, a)]])
                mat_m = np.matrix([[img[int(y-y1), int(x-x1), c],img[int(y-y2), int(x-x1), c],img[int(y+y3), int(x-x1), c],img[int(y+y4), int(x-x1), c]],
                                   [img[int(y-y1), int(x-x2), c],img[int(y-y2), int(x-x2), c],img[int(y+y3), int(x-x2), c],img[int(y+y4), int(x-x2), c]],
                                   [img[int(y-y1), int(x+x3), c],img[int(y-y2), int(x+x3), c],img[int(y+y3), int(x+x3), c],img[int(y+y4), int(x+x3), c]],
                                   [img[int(y-y1), int(x+x4), c],img[int(y-y2), int(x+x4), c],img[int(y+y3), int(x+x4), c],img[int(y+y4), int(x+x4), c]]])
                mat_r = np.matrix([[_u(y1, a)], [_u(y2, a)], [_u(y3, a)], [_u(y4, a)]])
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)
    return dst
  