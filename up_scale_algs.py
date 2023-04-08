import numpy as np

def three_or_more_equal(a, b, c, d):
    return np.logical_or(np.logical_and(np.all(a == b, axis=2), np.logical_or(np.all(a == c, axis=2), np.all(a == d, axis=2))), np.logical_and(np.all(a == c, axis=2), np.all(a == d, axis=2)), np.logical_and(np.all(b == c, axis=2), np.all(b == d, axis=2)))

def nearest_neighbor(img, scale_factor):
    img = img.repeat(scale_factor, 1).repeat(scale_factor, 0)
    return img

def EPX(img, Iterations=1):
    for k in range(Iterations):
        img_scaled = img.repeat(2, 1).repeat(2, 0)

        a = img[0:-2, 1:-1]
        b = img[1:-1, 2:]
        c = img[1:-1, 0:-2]
        d = img[2:, 1:-1]

        three_or_more_mask = np.logical_not(three_or_more_equal(a, b, c, d))
        a_mask = np.logical_and(three_or_more_mask, np.all(a == c, axis=2))
        b_mask = np.logical_and(three_or_more_mask, np.all(a == b, axis=2))
        c_mask = np.logical_and(three_or_more_mask, np.all(d == c, axis=2))
        d_mask = np.logical_and(three_or_more_mask, np.all(b == d, axis=2))

        img_scaled[2:-2:2, 2:-2:2][a_mask] = a[a_mask]
        img_scaled[2:-2:2, 3:-2:2][b_mask] = b[b_mask]
        img_scaled[3:-2:2, 2:-2:2][c_mask] = c[c_mask]
        img_scaled[3:-2:2, 3:-2:2][d_mask] = d[d_mask]
        
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


