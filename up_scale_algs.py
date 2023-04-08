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

        a = img[0:-2, 1:-1]
        b = img[1:-1, 2:]
        c = img[1:-1, 0:-2]
        d = img[2:, 1:-1]

        a_mask = np.logical_and(np.all(a == c, axis=2), np.logical_not(np.all(c == d, axis=2)), np.logical_not(np.all(a == b, axis=2)))
        b_mask = np.logical_and(np.all(a == b, axis=2), np.logical_not(np.all(a == c, axis=2)), np.logical_not(np.all(b == d, axis=2)))
        c_mask = np.logical_and(np.all(d == c, axis=2), np.logical_not(np.all(d == b, axis=2)), np.logical_not(np.all(a == c, axis=2)))
        d_mask = np.logical_and(np.all(b == d, axis=2), np.logical_not(np.all(b == a, axis=2)), np.logical_not(np.all(d == c, axis=2)))

        img_scaled[2:-2:2, 2:-2:2][a_mask] = a[a_mask]
        img_scaled[2:-2:2, 3:-2:2][b_mask] = b[b_mask]
        img_scaled[3:-2:2, 2:-2:2][c_mask] = c[c_mask]
        img_scaled[3:-2:2, 3:-2:2][d_mask] = d[d_mask]
        
        img = np.array(img_scaled, dtype=np.uint8)
    return img

def eagle_2x(img, Iterations=1):
    for k in range(Iterations):
        img_scaled = img.repeat(2, 1).repeat(2, 0)

        s = img[:-2, :-2]
        t = img[0:-2, 1:-1]
        u = img[:-2, 2:]
        v = img[1:-1, 0:-2]
        w = img[1:-1, 2:]
        x = img[2:, :-2]
        y = img[2:, 1:-1]
        z = img[2:, 2:]

        top_left_mask = np.logical_and(np.all(v == s, axis=2), np.all(s == t, axis=2))
        top_right_mask = np.logical_and(np.all(t == u, axis=2), np.all(u == w, axis=2))
        bottom_left_mask = np.logical_and(np.all(v == x, axis=2), np.all(x == y, axis=2))
        bottom_right_mask = np.logical_and(np.all(w == z, axis=2), np.all(z == y, axis=2))

        img_scaled[2:-2:2, 2:-2:2][top_left_mask] = s[top_left_mask]
        img_scaled[2:-2:2, 3:-2:2][top_right_mask] = u[top_right_mask]
        img_scaled[3:-2:2, 2:-2:2][bottom_left_mask] = x[bottom_left_mask]
        img_scaled[3:-2:2, 3:-2:2][bottom_right_mask] = z[bottom_right_mask]

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


