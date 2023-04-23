import numpy as np


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
