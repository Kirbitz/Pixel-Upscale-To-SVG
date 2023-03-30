import numpy as np

def three_or_more_equal(a, b, c, d):
    return (np.array_equal(a, b) and (np.array_equal(a, c) or np.array_equal(a, d))) or (np.array_equal(a, c) and np.array_equal(a, d)) or (np.array_equal(b, c) and np.array_equal(b, d))

def nearest_neighbor(img, Iterations=1):
    for k in range(Iterations):
        img = np.repeat(img, 2, axis=1)
        img = np.repeat(img, 2, axis=0)
    return img

def EPX(img, Iterations=1):
    for k in range(Iterations):
        img_scaled = np.repeat(img, 2, axis=1)
        img_scaled = np.repeat(img_scaled, 2, axis=0)

        for i in range(1, len(img) - 1):
            for j in range(1, len(img[i] - 1)):
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
        img_scaled = np.repeat(img, 2, axis=1)
        img_scaled = np.repeat(img_scaled, 2, axis=0)
        for i in range(1, len(img) - 1):
            for j in range(1, len(img[i]) - 1):
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