import numpy as np
import cv2


def three_or_more_equal(a, b, c, d):
    return np.logical_or(
        np.logical_and(
            np.all(a == b, axis=2),
            np.logical_or(np.all(a == c, axis=2), np.all(a == d, axis=2)),
        ),
        np.logical_and(np.all(a == c, axis=2), np.all(a == d, axis=2)),
        np.logical_and(np.all(b == c, axis=2), np.all(b == d, axis=2)),
    )


def nearest_neighbor(img, scale_factor):
    img = img.repeat(scale_factor, 1).repeat(scale_factor, 0)
    return img


def EPX(img, Iterations=1):
    for _ in range(Iterations):
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
    for _ in range(Iterations):
        img_scaled = img.repeat(2, 1).repeat(2, 0)

        a = img[0:-2, 1:-1]
        b = img[1:-1, 2:]
        c = img[1:-1, 0:-2]
        d = img[2:, 1:-1]

        a_mask = np.logical_and(
            np.all(a == c, axis=2),
            np.logical_not(np.all(c == d, axis=2)),
            np.logical_not(np.all(a == b, axis=2)),
        )
        b_mask = np.logical_and(
            np.all(a == b, axis=2),
            np.logical_not(np.all(a == c, axis=2)),
            np.logical_not(np.all(b == d, axis=2)),
        )
        c_mask = np.logical_and(
            np.all(d == c, axis=2),
            np.logical_not(np.all(d == b, axis=2)),
            np.logical_not(np.all(a == c, axis=2)),
        )
        d_mask = np.logical_and(
            np.all(b == d, axis=2),
            np.logical_not(np.all(b == a, axis=2)),
            np.logical_not(np.all(d == c, axis=2)),
        )

        img_scaled[2:-2:2, 2:-2:2][a_mask] = a[a_mask]
        img_scaled[2:-2:2, 3:-2:2][b_mask] = b[b_mask]
        img_scaled[3:-2:2, 2:-2:2][c_mask] = c[c_mask]
        img_scaled[3:-2:2, 3:-2:2][d_mask] = d[d_mask]

        img = np.array(img_scaled, dtype=np.uint8)
    return img


def eagle_2x(img, Iterations=1):
    for _ in range(Iterations):
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
        bottom_left_mask = np.logical_and(
            np.all(v == x, axis=2), np.all(x == y, axis=2)
        )
        bottom_right_mask = np.logical_and(
            np.all(w == z, axis=2), np.all(z == y, axis=2)
        )

        img_scaled[2:-2:2, 2:-2:2][top_left_mask] = s[top_left_mask]
        img_scaled[2:-2:2, 3:-2:2][top_right_mask] = u[top_right_mask]
        img_scaled[3:-2:2, 2:-2:2][bottom_left_mask] = x[bottom_left_mask]
        img_scaled[3:-2:2, 3:-2:2][bottom_right_mask] = z[bottom_right_mask]

        img = np.array(img_scaled, dtype=np.uint8)
    return img


def bilinear(
    img, scaleFactor
):  # There is a vectorized version of this that runs faster
    height, width = img.shape[:2]
    height_scaled = height * scaleFactor
    width_scaled = width * scaleFactor
    img_scaled = np.zeros((height_scaled, width_scaled, 3))
    x_ratio = float(width - 1) / (width_scaled - 1)
    y_ratio = float(height - 1) / (height_scaled - 1)

    x1 = np.array([np.uint32(np.floor(x_ratio * np.arange(width_scaled)))]).repeat(
        height_scaled, axis=0
    )
    y1 = np.uint32(np.floor(y_ratio * np.arange(height_scaled)[:, np.newaxis])).repeat(
        width_scaled, axis=1
    )
    xh = np.array([np.uint32(np.ceil(x_ratio * np.arange(width_scaled)))]).repeat(
        height_scaled, axis=0
    )
    yh = np.uint32(np.ceil(y_ratio * np.arange(height_scaled)[:, np.newaxis])).repeat(
        width_scaled, axis=1
    )

    a = img[np.ix_(y1[:, 0], x1[0])]
    b = img[np.ix_(y1[:, 0], xh[0])]
    c = img[np.ix_(yh[:, 0], x1[0])]
    d = img[np.ix_(yh[:, 0], xh[0])]

    x_weight = (
        np.array([(x_ratio * np.arange(width_scaled))]).repeat(height_scaled, axis=0)
        - x1
    )
    y_weight = (y_ratio * np.arange(height_scaled)[:, np.newaxis]).repeat(
        width_scaled, axis=1
    ) - y1

    img_scaled[:, :, 0] = (
        a[:, :, 0] * (1 - x_weight) * (1 - y_weight)
        + b[:, :, 0] * x_weight * (1 - y_weight)
        + c[:, :, 0] * y_weight * (1 - x_weight)
        + d[:, :, 0] * x_weight * y_weight
    )
    img_scaled[:, :, 1] = (
        a[:, :, 1] * (1 - x_weight) * (1 - y_weight)
        + b[:, :, 1] * x_weight * (1 - y_weight)
        + c[:, :, 1] * y_weight * (1 - x_weight)
        + d[:, :, 1] * x_weight * y_weight
    )
    img_scaled[:, :, 2] = (
        a[:, :, 2] * (1 - x_weight) * (1 - y_weight)
        + b[:, :, 2] * x_weight * (1 - y_weight)
        + c[:, :, 2] * y_weight * (1 - x_weight)
        + d[:, :, 2] * x_weight * y_weight
    )

    img = np.array(img_scaled, dtype=np.uint8)
    return img


# Interpolation kernel
def _u(s, a):
    test = np.abs(s)
    temp = np.zeros(s.shape, dtype=np.float32)
    mod1 = s[np.logical_and(np.abs(s) >= 0, np.abs(s) <= 1)]
    temp[np.logical_and(np.abs(s) >= 0, np.abs(s) <= 1)] = (
        (a + 2) * (np.abs(mod1) ** 3) - (a + 3) * (np.abs(mod1) ** 2) + 1
    )
    mod2 = s[np.logical_and(np.abs(s) > 1, np.abs(s) <= 2)]
    temp[np.logical_and(np.abs(s) > 1, np.abs(s) <= 2)] = (
        a * (np.abs(mod2) ** 3)
        - (5 * a) * (np.abs(mod2) ** 2)
        + (8 * a) * np.abs(mod2)
        - 4 * a
    )
    return temp
    """ if (np.abs(s) >= 0) & (np.abs(s) <= 1):
        return (a + 2) * (abs(s) ** 3) - (a + 3) * (abs(s) ** 2) + 1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a * (abs(s) ** 3) - (5 * a) * (abs(s) ** 2) + (8 * a) * abs(s) - 4 * a
    return 0 """


# Padding
def __padding(img, H, W, C, add):
    zimg = np.zeros((H + (2 * add), W + (2 * add), C))
    zimg[add : H + add, add : W + add, :C] = img

    # Pad the first/last two col and row
    zimg[add : H + add, 0:2, :C] = img[:, 0:1, :C]
    zimg[H + add : H + (2 * add), add : W + add, :] = img[H - 1 : H, :, :]
    zimg[add : H + add, W + add : W + (2 * add), :] = img[:, W - 1 : W, :]
    zimg[0:add, add : W + add, :C] = img[0:1, :, :C]

    # Pad the missing eight points
    zimg[0:add, 0:add, :C] = img[0, 0, :C]
    zimg[H + add : H + (2 * add), 0:add, :C] = img[H - 1, 0, :C]
    zimg[H + add : H + (2 * add), W + add : W + (2 * add), :C] = img[H - 1, W - 1, :C]
    zimg[0:add, W + add : W + (2 * add), :C] = img[0, W - 1, :C]
    return zimg


# Bicubic operation, the math for this operation can be found here:https://link.springer.com/article/10.1007/s11554-022-01254-8
def bicubic(img, ratio):
    a = -0.5
    # Get image size
    H, W, C = img.shape
    # Here H = Height, W = weight,
    # C = Number of channels if the
    # image is coloured.
    img = __padding(img, H, W, C, 2) / 255.0

    # Create new image
    dH = np.uint32(np.floor(H * ratio))
    dW = np.uint32(np.floor(W * ratio))
    dst = np.zeros((dH, dW, 3))
    h = 1 / ratio

    x, y = (
        np.array([np.arange(dW)]).repeat(dH, axis=0) * h + 2,
        np.arange(dH)[:, np.newaxis].repeat(dW, axis=1) * h + 2,
    )
    x1 = 1 + x - np.floor(x)
    x2 = x - np.floor(x)
    x3 = np.floor(x) + 1 - x
    x4 = np.floor(x) + 2 - x

    y1 = 1 + y - np.floor(y)
    y2 = y - np.floor(y)
    y3 = np.floor(y) + 1 - y
    y4 = np.floor(y) + 2 - y

    manip_x1 = _u(x1, a)
    manip_x2 = _u(x2, a)
    manip_x3 = _u(x3, a)
    manip_x4 = _u(x4, a)

    manip_y1 = _u(y1, a)
    manip_y2 = _u(y2, a)
    manip_y3 = _u(y3, a)
    manip_y4 = _u(y4, a)

    pixel_pos1 = img[np.ix_((y - y1).astype(int)[:, 0], (x - x1).astype(int)[0])]
    pixel_pos2 = img[np.ix_((y - y2).astype(int)[:, 0], (x - x1).astype(int)[0])]
    pixel_pos3 = img[np.ix_((y + y3).astype(int)[:, 0], (x - x1).astype(int)[0])]
    pixel_pos4 = img[np.ix_((y + y4).astype(int)[:, 0], (x - x1).astype(int)[0])]

    pixel_pos5 = img[np.ix_((y - y1).astype(int)[:, 0], (x - x2).astype(int)[0])]
    pixel_pos6 = img[np.ix_((y - y2).astype(int)[:, 0], (x - x2).astype(int)[0])]
    pixel_pos7 = img[np.ix_((y + y3).astype(int)[:, 0], (x - x2).astype(int)[0])]
    pixel_pos8 = img[np.ix_((y + y4).astype(int)[:, 0], (x - x2).astype(int)[0])]

    pixel_pos9 = img[np.ix_((y - y1).astype(int)[:, 0], (x + x3).astype(int)[0])]
    pixel_pos10 = img[np.ix_((y - y2).astype(int)[:, 0], (x + x3).astype(int)[0])]
    pixel_pos11 = img[np.ix_((y + y3).astype(int)[:, 0], (x + x3).astype(int)[0])]
    pixel_pos12 = img[np.ix_((y + y4).astype(int)[:, 0], (x + x3).astype(int)[0])]

    pixel_pos13 = img[np.ix_((y - y1).astype(int)[:, 0], (x + x4).astype(int)[0])]
    pixel_pos14 = img[np.ix_((y - y2).astype(int)[:, 0], (x + x4).astype(int)[0])]
    pixel_pos15 = img[np.ix_((y + y3).astype(int)[:, 0], (x + x4).astype(int)[0])]
    pixel_pos16 = img[np.ix_((y + y4).astype(int)[:, 0], (x + x4).astype(int)[0])]

    for c in range(C):
        dst[:, :, c] = (
            (
                manip_x1 * pixel_pos1[:, :, c]
                + manip_x2 * pixel_pos2[:, :, c]
                + manip_x3 * pixel_pos3[:, :, c]
                + manip_x4 * pixel_pos4[:, :, c]
            )
            * manip_y1
            + (
                manip_x1 * pixel_pos5[:, :, c]
                + manip_x2 * pixel_pos6[:, :, c]
                + manip_x3 * pixel_pos7[:, :, c]
                + manip_x4 * pixel_pos8[:, :, c]
            )
            * manip_y2
            + (
                manip_x1 * pixel_pos9[:, :, c]
                + manip_x2 * pixel_pos10[:, :, c]
                + manip_x3 * pixel_pos11[:, :, c]
                + manip_x4 * pixel_pos12[:, :, c]
            )
            * manip_y3
            + (
                manip_x1 * pixel_pos13[:, :, c]
                + manip_x2 * pixel_pos14[:, :, c]
                + manip_x3 * pixel_pos15[:, :, c]
                + manip_x4 * pixel_pos16[:, :, c]
            )
            * manip_y4
        )

    return dst


# __wd calculates the distance between the colors of two pixels.
def __wd(p1, p2):
    y, u, v = (
        np.abs(np.subtract(p1[:, :, 0], p2[:, :, 0])),
        np.abs(np.subtract(p1[:, :, 1], p2[:, :, 1])),
        np.abs(np.subtract(p1[:, :, 2], p2[:, :, 2])),
    )
    return np.add(np.multiply(48, y), np.multiply(7, u), np.multiply(6, v))


# Color Interpolation for Vectorized xBR
def __blendColors(edge, opposite, e, c1, c2):
    edr = edge < opposite
    mask = __wd(e, c1) <= __wd(e, c2)
    spots = np.logical_and(edr, mask)
    inverse = np.logical_and(edr, np.logical_not(spots))
    e[spots] = np.add(np.multiply(0.5, e[spots]), np.multiply(0.5, c1[spots]))
    e[inverse] = np.add(np.multiply(0.5, e[inverse]), np.multiply(0.5, c2[inverse]))
    # No Blend (BROKEN)
    # e[spots] = c1[spots]
    # e[inverse] = c2[inverse]
    return e


"""
description of the algorithm can be found here: https://forums.libretro.com/t/xbr-algorithm-tutorial/123 
Pixels are represented in the following format
       |A1|B1|C1|
    |A0|A |B |C |C4|
    |D0|D |E |F |F4|
    |G0|G |H |I |I4|
       |G5|H5|I5|  
Consider E as the central pixel."""


def xBR(img, Iterations=1):
    cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    for k in range(Iterations):
        imgScaled = np.zeros((len(img) * 2, len(img[1]) * 2, 3), dtype=np.uint8)
        img = __padding(img, len(img), len(img[1]), 3, 2)
        a1, b1, c1 = img[:-4, 1:-3], img[:-4, 2:-2], img[:-4, 3:-1]
        a0, a, b, c, c4 = (
            img[1:-3, :-4],
            img[1:-3, 1:-3],
            img[1:-3, 2:-2],
            img[1:-3, 3:-1],
            img[1:-3, 4:],
        )
        d0, d, e, f, f4 = (
            img[2:-2, :-4],
            img[2:-2, 1:-3],
            img[2:-2, 2:-2],
            img[2:-2, 3:-1],
            img[2:-2, 4:],
        )
        g0, g, h, i, i4 = (
            img[3:-1, :-4],
            img[3:-1, 1:-3],
            img[3:-1, 2:-2],
            img[3:-1, 3:-1],
            img[3:-1, 4:],
        )
        g5, h5, i5 = img[4:, 1:-3], img[4:, 2:-2], img[4:, 3:-1]

        # setting the weighted distances
        ec, eg, if4, ih5, hf = (
            __wd(e, c),
            __wd(e, g),
            __wd(i, f4),
            __wd(i, h5),
            __wd(h, f),
        )
        hd, hi5, fi4, fb, ei = (
            __wd(h, d),
            __wd(h, i5),
            __wd(f, i4),
            __wd(f, b),
            __wd(e, i),
        )
        ea, gd0, gh5 = __wd(e, a), __wd(g, d0), __wd(g, h5)
        bd, dg0, hg5 = __wd(b, d), __wd(d, g0), __wd(h, g5)
        d0a, ab1 = __wd(d0, a), __wd(a, b1)
        a0d, a1b = __wd(a0, d), __wd(a1, b)
        b1c, cf4 = __wd(b1, c), __wd(c, f4)
        bc1, fc4 = __wd(b, c1), __wd(f, c4)

        # Top Right Edge Detection Rule
        edge = ei + ea + b1c + cf4 + (4 * fb)
        opposite = bd + bc1 + hf + fc4 + (4 * ec)
        e = __blendColors(edge, opposite, e, b, f)

        # Top Left Edge Detection Rule
        edge = ec + eg + d0a + ab1 + (4 * bd)
        opposite = hd + fb + a0d + a1b + (4 * ea)
        e = __blendColors(edge, opposite, e, d, b)

        # Bottom Left Edge Detection Rule
        edge = ea + ei + gd0 + gh5 + (4 * hd)
        opposite = bd + dg0 + hf + hg5 + (4 * eg)
        e = __blendColors(edge, opposite, e, d, h)

        # Bottom Right Edge Detection Rule
        edge = ec + eg + if4 + ih5 + (4 * hf)
        opposite = hd + hi5 + fi4 + fb + (4 * ei)
        e = __blendColors(edge, opposite, e, f, h)

        imgScaled[1::2, 1::2] = e
        imgScaled[:-1:2, 1::2] = e
        imgScaled[1::2, :-1:2] = e
        imgScaled[:-1:2, :-1:2] = e

        img = np.array(imgScaled, dtype=np.uint8)
    cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    return img
