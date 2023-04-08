import numpy as np

def three_or_more_equal(a, b, c, d):
    return (np.array_equal(a, b) and (np.array_equal(a, c) or np.array_equal(a, d))) or (np.array_equal(a, c) and np.array_equal(a, d)) or (np.array_equal(b, c) and np.array_equal(b, d))

def nearest_neighbor(img, scale_factor):
    img = img.repeat(scale_factor, 1).repeat(scale_factor, 0)
    return img

def EPX(img, Iterations=1):
    for k in range(Iterations):
        img_scaled = img.repeat(2, 1).repeat(2, 0)

        for i in range(1, len(img) - 1):
            for j in range(1, len(img[0]) - 1):
                a, b, c, d = img[i-1][j], img[i][j+1], img[i][j-1], img[i+1][j]

                if not three_or_more_equal(a, b, c, d):
                    if np.array_equal(c, a):
                        img_scaled[i*2][j*2] = a
                    if np.array_equal(a, b):
                        img_scaled[i*2][j*2+1] = b
                    if np.array_equal(d, c):
                        img_scaled[i*2+1][j*2] = c
                    if np.array_equal(b, d):
                        img_scaled[i*2+1][j*2+1] = d
        
        img = np.array(img_scaled, dtype=np.uint8)
    return img

def scale_2x(img, Iterations=1):
    for k in range(Iterations):
        img_scaled = img.repeat(2, 1).repeat(2, 0)

        for i in range(1, len(img) - 1):
            for j in range(1, len(img[0]) - 1):
                a, b, c, d = img[i-1][j], img[i][j+1], img[i][j-1], img[i+1][j]

                if np.array_equal(c, a) and not np.array_equal(c, d) and not np.array_equal(a, b):
                    img_scaled[i*2][j*2] = a
                if np.array_equal(a, b) and not np.array_equal(a, c) and not np.array_equal(b, d):
                    img_scaled[i*2][j*2+1] = b
                if np.array_equal(d, c) and not np.array_equal(d, b) and not np.array_equal(c, a):
                    img_scaled[i*2+1][j*2] = c
                if np.array_equal(b, d) and not np.array_equal(b, a) and not np.array_equal(d, c):
                    img_scaled[i*2+1][j*2+1] = d
        
        img = np.array(img_scaled, dtype=np.uint8)
    return img

def eagle_2x(img, Iterations=1):
    for k in range(Iterations):
        img_scaled = img.repeat(2, 1).repeat(2, 0)
        
        for i in range(1, len(img) - 1):
            for j in range(1, len(img[0]) - 1):
                p = np.full((4,3),img[i,j])

                if not(i == 0 or j == 0 or i == len(img) - 1 or j == len(img[1]) - 1):
                    if np.array_equal(img[i,j-1],img[i-1,j-1]) and np.array_equal(img[i,j-1], img[i-1,j]):
                        img_scaled[i*2][j*2] = img[i-1,j-1]
                    if np.array_equal(img[i-1,j], img[i-1,j+1]) and np.array_equal(img[i-1,j],img[i,j+1]):
                        img_scaled[i*2][j*2+1] = img[i-1,j+1]
                    if np.array_equal(img[i,j-1],img[i+1,j-1]) and np.array_equal(img[i,j-1], img[i+1,j]):
                        img_scaled[i*2+1][j*2] = img[i+1,j-1]
                    if np.array_equal(img[i,j+1],img[i+1,j+1]) and np.array_equal(img[i,j+1], img[i+1,j]):
                        img_scaled[i*2+1][j*2+1] = img[i+1,j+1]

        img = np.array(img_scaled, dtype=np.uint8)
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


