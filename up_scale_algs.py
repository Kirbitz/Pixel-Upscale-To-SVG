import numpy as np
import cv2
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
    height, width = img.shape[:2]
    height_scaled = height * scaleFactor
    width_scaled = width * scaleFactor
    img_scaled = np.zeros((height_scaled, width_scaled, 3))
    x_ratio = float(width - 1) / (width_scaled - 1)
    y_ratio = float(height-1) / (height_scaled - 1)

    x1 = np.array([np.uint32(np.floor(x_ratio * np.arange(width_scaled)))]).repeat(height_scaled, axis=0)
    y1 = np.uint32(np.floor(y_ratio * np.arange(height_scaled)[:, np.newaxis])).repeat(width_scaled, axis=1)
    xh = np.array([np.uint32(np.ceil(x_ratio * np.arange(width_scaled)))]).repeat(height_scaled, axis=0)
    yh = np.uint32(np.ceil(y_ratio * np.arange(height_scaled)[:, np.newaxis])).repeat(width_scaled, axis=1)

    a = img[np.ix_(y1[:, 0], x1[0])]
    b = img[np.ix_(y1[:, 0], xh[0])]
    c = img[np.ix_(yh[:, 0], x1[0])]
    d = img[np.ix_(yh[:, 0], xh[0])]

    x_weight = np.array([(x_ratio * np.arange(width_scaled))]).repeat(height_scaled, axis=0) - x1
    y_weight = (y_ratio * np.arange(height_scaled)[:, np.newaxis]).repeat(width_scaled, axis=1) - y1

    img_scaled[:, :, 0] = a[:, :, 0] * (1-x_weight) * (1-y_weight) + b[:, :, 0] * x_weight * (1-y_weight) + c[:, :, 0] * y_weight * (1-x_weight) + d[:, :, 0] * x_weight * y_weight
    img_scaled[:, :, 1] = a[:, :, 1] * (1-x_weight) * (1-y_weight) + b[:, :, 1] * x_weight * (1-y_weight) + c[:, :, 1] * y_weight * (1-x_weight) + d[:, :, 1] * x_weight * y_weight
    img_scaled[:, :, 2] = a[:, :, 2] * (1-x_weight) * (1-y_weight) + b[:, :, 2] * x_weight * (1-y_weight) + c[:, :, 2] * y_weight * (1-x_weight) + d[:, :, 2] * x_weight * y_weight

    img = np.array(img_scaled, dtype=np.uint8)
    return img

# Interpolation kernel
def _u(s, a):
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0
  
  
# Padding
def _padding_(img, H, W, C, add):
    zimg = np.zeros((H+(2*add), W+(2*add), C))
    zimg[add:H+add, add:W+add, :C] = img
      
    # Pad the first/last two col and row
    zimg[add:H+add, 0:2, :C] = img[:, 0:1, :C]
    zimg[H+add:H+(2*add), add:W+add, :] = img[H-1:H, :, :]
    zimg[add:H+add, W+add:W+(2*add), :] = img[:, W-1:W, :]
    zimg[0:add, add:W+add, :C] = img[0:1, :, :C]
      
    # Pad the missing eight points
    zimg[0:add, 0:add, :C] = img[0, 0, :C]
    zimg[H+add:H+(2*add), 0:add, :C] = img[H-1, 0, :C]
    zimg[H+add:H+(2*add), W+add:W+(2*add), :C] = img[H-1, W-1, :C]
    zimg[0:add, W+add:W+(2*add), :C] = img[0, W-1, :C]
    return zimg
# Bicubic operation, the math for this operation can be found here:https://link.springer.com/article/10.1007/s11554-022-01254-8
def bicubic(img, ratio):
    a= -.5
    # Get image size
    H, W, C = img.shape
    # Here H = Height, W = weight,
    # C = Number of channels if the 
    # image is coloured.
    img = _padding_(img, H, W, C,2) / 255.0
      
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
    y,u,v = np.abs(np.subtract(p1[:,:,0],p2[:,:,0])), np.abs(np.subtract(p1[:,:,1],p2[:,:,1])),\
          np.abs(np.subtract(p1[:,:,2],p2[:,:,2]))
    return np.add(np.multiply(48,y),np.multiply(7,u), np.multiply(6,v))

def _xbrInterp_(e,f,h):
    FMask = _wd_(e,f) <= _wd_(e,h)
    newColor = np.where(FMask, f,h)
    return np.add(np.multiply(.5,e),np.multiply(.5,newColor))
'''
description of the algorithm can be found here: https://forums.libretro.com/t/xbr-algorithm-tutorial/123 
Pixels are represented in the following format
       |A1|B1|C1|
    |A0|A |B |C |C4|
    |D0|D |E |F |F4|
    |G0|G |H |I |I4|
       |G5|H5|I5|  
Consider E as the central pixel.'''
def xBRvec(img, Iterations = 1):
    cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    for k in range(Iterations):
        img = nearest_neighbor(img)
        img = _padding_(img,len(img),len(img[1]), 3, 3)
        a1,b1,c1 = img[:-4, 1:-3], img[:-4, 2:-2], img[:-4, 3:-1]
        a0,a,b,c,c4 = img[1:-3, :-4], img[1:-3, 1:-3], img[1:-3, 2:-2], img[1:-3, 3:-1], img[1:-3, 4:]
        d0,d,e,f,f4 = img[2:-2, :-4], img[2:-2, 1:-3], img[2:-2, 2:-2], img[2:-2, 3:-1], img[2:-2, 4:]
        g0,g,h,i,i4 = img[3:-1, :-4], img[3:-1, 1:-3], img[3:-1, 2:-2], img[3:-1, 3:-1], img[3:-1, 4:]
        g5,h5,i5 = img[4:,1:-3], img[4:,2:-2],img[4:,3:-1]

        ec, eg, if4, ih5, hf = _wd_(e,c), _wd_(e,g), _wd_(i,f4), _wd_(i,h5), _wd_(h,f)
        hd, hi5, fi4, fb, ei = _wd_(h,d), _wd_(h,i5), _wd_(f,i4), _wd_(f,b), _wd_(e,i)
        #Bottom Right Edge Detection Rule
        edge = ec + eg + if4 + ih5 + (4 * hf)
        opposite = hd  + hi5 + fi4 + fb + (4 * ei)
        points = edge < opposite
        fMask = _wd_(e,f) <= _wd_(e,h)
        spots = np.logical_and(points, fMask)
        notSpots = np.logical_not(spots)
        e[spots] = np.add(np.multiply(.5,e[spots]),np.multiply(.5,f[spots]))
        e[notSpots] = np.add(np.multiply(.5,e[notSpots]), np.multiply(.5,h[notSpots]))
        
        ea,ei, gd0, gh5 = _wd_(e,a), _wd_(e,i), _wd_(g,d0), _wd_(g,h5)
        bd, dg0, hg5, eg = _wd_(b,d), _wd_(d,g0), _wd_(h,g5), _wd_(e,g)
        #Bottom Left Edge Detection Rule
        edge = ea + ei + gd0 + gh5 + (4*hd)
        opposite = bd + dg0 + hf + hg5 + (4*eg)
        points = edge<opposite
        dMask = _wd_(e,d) <= _wd_(e,h)
        spots = np.logical_and(points,dMask)
        notSpots = np.logical_not(spots)
        e[spots] = np.add(np.multiply(.5,e[spots]),np.multiply(.5,d[spots]))
        e[notSpots] = np.add(np.multiply(.5,e[notSpots]), np.multiply(.5,h[notSpots]))

        ec, eg, d0a, ab1 = _wd_(e,c), _wd_(e,g), _wd_(d0,a), _wd_(a,b1)
        a0d, a1b, ea = _wd_(a0,d), _wd_(a1,b), _wd_(e,a)
        #Top Left Edge Detection Rule
        edge = ec + eg + d0a + ab1 + (4 * bd)
        opposite = hd + fb + a0d + a1b + (4*ea)
        points = edge<opposite
        dMask = _wd_(e,d) <= _wd_(e,b)
        spots = np.logical_and(points, dMask)
        notSpots = np.logical_not(spots)
        e[spots] = np.add(np.multiply(.5,e[spots]),np.multiply(.5,d[spots]))
        e[notSpots] = np.add(np.multiply(.5,e[notSpots]), np.multiply(.5,b[notSpots]))

        ei, ea, b1c, cf4 = _wd_(e,i), _wd_(e,a), _wd_(b1,c), _wd_(c,f4)
        bc1, fc4, ec = _wd_(b,c1), _wd_(f,c4), _wd_(e,c)
        #Top Right Edge Detection Rule
        edge = ei + ea + b1c + cf4 + (4 * fb)
        opposite = bd + bc1 + hf + fc4 + (4*ec)
        points = edge<opposite
        bMask = _wd_(e,b) <= _wd_(e,f)
        spots = np.logical_and(points, bMask)
        notSpots = np.logical_not(spots)
        e[spots] = np.add(np.multiply(.5,e[spots]), np.multiply(.5,b[spots]))
        e[notSpots] = np.add(np.multiply(.5,e[notSpots]), np.multiply(.5,f[notSpots]))

        img = np.array(e, dtype = np.uint8)
    cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    return img
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
    cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    for k in range(Iterations):
        #padded = np.zeros((len(img) + 6, len(img[1]) + 6,3), dtype=np.uint8)
        #padded[3:-3,3:-3]= img
        imgScaled = np.zeros((len(img) *2, len(img[1]) * 2,3), dtype=np.uint8)
        img = _padding_(img, len(img), len(img[1]), 3, 3)
        ratio = 1/2
        for row in range(len(imgScaled)):
            for col in range(len(imgScaled[1])):
                x,y = np.uint32(np.floor(col * ratio) + 3), np.uint32(np.floor(row * ratio) + 3)
                a1, b1, c1 = img[y-2,x-1], img[y-2,x], img[y-2,x+1]
                a0, a, b, c, c4 = img[y-1,x-2], img[y-1,x-1], img[y-1,x], img[y-1,x+1], img[y-1,x+2]
                d0, d, e, f ,f4 = img[y,x-2], img[y,x-1], img[y,x], img[y,x+1], img[y,x+2]
                g0, g, h, i, i4 = img[y+1,x-2], img[y+1,x-1], img[y+1,x], img[y+1, x+1], img[y+1,x+2]
                g5, h5, i5 = img[y+2,x-1], img[y+2, x], img[y+2,x+1]
                
                #Setting all weights
                ec, eg, if4, ih5, hf = _wd_(e,c), _wd_(e,g), _wd_(i,f4), _wd_(i,h5), _wd_(h,f)
                hd, hi5, fi4, fb, ei = _wd_(h,d), _wd_(h,i5), _wd_(f,i4), _wd_(f,b), _wd_(e,i)
                ea, gd0, gh5 = _wd_(e,a), _wd_(g,d0), _wd_(g,h5)
                bd, dg0, hg5 = _wd_(b,d), _wd_(d,g0), _wd_(h,g5)
                d0a, ab1 = _wd_(d0,a), _wd_(a,b1)
                a0d, a1b = _wd_(a0,d), _wd_(a1,b)
                b1c, cf4 = _wd_(b1,c), _wd_(c,f4)
                bc1, fc4 = _wd_(b,c1), _wd_(f,c4)

                #Bottom Right Edge Detection Rule
                edge = ec + eg + if4 + ih5 + (4 * hf)
                opposite = hd  + hi5 + fi4 + fb + (4 * ei)
                if(edge < opposite):e = _xbrInterp_(e,f,h)

                #Bottom Left Edge Detection Rule
                edge = ea + ei + gd0 + gh5 + (4*hd)
                opposite = bd + dg0 + hf + hg5 + (4*eg)
                if(edge < opposite): e = _xbrInterp_(e,d,h)

                #Top Left Edge Detection Rule
                edge = ec + eg + d0a + ab1 + (4 * bd)
                opposite = hd + fb + a0d + a1b + (4*ea)
                if(edge < opposite): e = _xbrInterp_(e,d,b)
            
                #Top Right Edge Detection Rule
                edge = ei + ea + b1c + cf4 + (4 * fb)
                opposite = bd + bc1 + hf + fc4 + (4*ec)
                if(edge < opposite): e = _xbrInterp_(e,b,f)
                
                imgScaled[row,col] = e

        img = np.array(imgScaled, dtype = np.uint8)
    cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    return img
