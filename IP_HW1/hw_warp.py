import cv2
import numpy as np

# read image
room = cv2.imread("board.jpg")
img = cv2.imread("image.jpg")

h_img, w_img, c_img = img.shape
h_room, w_room, c_room = room.shape

img_pts = np.array([
    [0, 0],
    [w_img - 1, 0],
    [w_img - 1, h_img - 1],
    [0, h_img - 1]
], dtype=np.int32)

tv_coord = [(253, 241), (413, 215), (413, 387), (253, 375)]
dst_pts = np.array(tv_coord, dtype=np.int32)

# homography
""" 
1. 去解Ah = b，把4個點img的(x, y) -> dst的(X, Y)
2. h11x + h12y + h13 - h31xX - h32yX = X
   h21x + h22y + h23 - h31xY - h32yY = Y
3. 
"""


def homography(img_pts, dst_pts):
    A = []
    b = []
    # 4 個點
    for i in range(4):
        x = img_pts[i][0]
        y = img_pts[i][1]
        X = dst_pts[i][0]
        Y = dst_pts[i][1]

        A.append([x, y, 1, 0, 0, 0, -X*x, -X*y])
        b.append(X)
        A.append([0, 0, 0, x, y, 1, -Y*x, -Y*y])
        b.append(Y)

    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    h = np.linalg.solve(A, b)
    # 把h33設為1
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0]
    ], dtype=np.float64)
    return H

H = homography(img_pts, dst_pts)
H_inv = np.linalg.inv(H)



def clamp(v, low, high):
    return max(low, min(high, v))


def nearest(img, x, y):
    x_n = int(round(x))
    y_n = int(round(y))
    x_n = clamp(x_n, 0, img.shape[1]-1)
    y_n = clamp(y_n, 0, img.shape[0]-1)
    return img[y_n, x_n].astype(np.float32)

def bilinear(img, x, y):
    x1 = int(np.floor(x))
    y1 = int(np.floor(y))
    x2 = x1 + 1
    y2 = y1 + 1

    dx = x - x1
    dy = y - y1

    x1 = clamp(x1, 0, img.shape[1]-1)
    x2 = clamp(x2, 0, img.shape[1]-1)
    y1 = clamp(y1, 0, img.shape[0]-1)
    y2 = clamp(y2, 0, img.shape[0]-1)

    top = (1 - dx) * img[y1, x1].astype(np.float32) + dx * img[y1, x2].astype(np.float32)
    bottom = (1 - dx) * img[y2, x1].astype(np.float32) + dx * img[y2, x2].astype(np.float32)
    val = (1 - dy) * top + dy * bottom
    return val

def cubic_interpolate(p0, p1, p2, p3, x):
    a = (-0.5 * p0) + (1.5 * p1) - (1.5 * p2) + (0.5 * p3)
    b = (p0) - (2.5 * p1) + (2.0 * p2) - (0.5 * p3)
    c = (-0.5 * p0) + (0.5 * p2)
    d = p1
    return a*(x**3) + b*(x**2) + c*x + d

def bicubic(img, x, y):
    x1 = int(np.floor(x))
    y1 = int(np.floor(y))
    fx = x - x1
    fy = y - y1

    # 取 16 個像素
    pixels = np.zeros((4, 4, img.shape[2]), dtype=np.float32)
    for m in range(-1, 3):
        for n in range(-1, 3):
            px = clamp(x1 + n, 0, img.shape[1]-1)
            py = clamp(y1 + m, 0, img.shape[0]-1)
            pixels[m+1, n+1] = img[py, px].astype(np.float32)

    # 先對 x 方向做三次插值
    col_values = np.zeros((4, img.shape[2]), dtype=np.float32)
    for row in range(4):
        col_values[row] = cubic_interpolate(
            pixels[row, 0],
            pixels[row, 1],
            pixels[row, 2],
            pixels[row, 3],
            fx
        )
    # 再對 y 方向做三次插值
    val = cubic_interpolate(
        col_values[0],
        col_values[1],
        col_values[2],
        col_values[3],
        fy
    )
    return val

def get_pixel_value(img, x, y, method="nearest"):
    if method == "nearest":
        return nearest(img, x, y)
    elif method == "bilinear":
        return bilinear(img, x, y)
    elif method == "bicubic":
        return bicubic(img, x, y)

# warp
warp_nearest  = room.copy()
warp_bilinear = room.copy()
warp_bicubic  = room.copy()

# tv coordiantes [(253, 241), (413, 215), (413, 387), (253, 375)]
x0 = 253
x1 = 413
y0 = 215
y1 = 387

for y_new in range(y0, y1+1):
    for x_new in range(x0, x1+1):
        vec = np.array([x_new, y_new, 1.0], dtype=np.float64)
        src_vec = H_inv.dot(vec)
        x_old = src_vec[0] / src_vec[2]
        y_old = src_vec[1] / src_vec[2]

        if 0 <= x_old < w_img and 0 <= y_old < h_img:
            # nearest
            color_n = get_pixel_value(img, x_old, y_old, method="nearest")
            # clip + uint8
            color_n = np.clip(color_n, 0, 255).astype(np.uint8)
            warp_nearest[y_new, x_new] = color_n

            # bilinear
            color_b = get_pixel_value(img, x_old, y_old, method="bilinear")
            color_b = np.clip(color_b, 0, 255).astype(np.uint8)
            warp_bilinear[y_new, x_new] = color_b

            # bicubic
            color_c = get_pixel_value(img, x_old, y_old, method="bicubic")
            color_c = np.clip(color_c, 0, 255).astype(np.uint8)
            warp_bicubic[y_new, x_new] = color_c


#show image
cv2.imshow("warp_nearest", warp_nearest)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("warp_bilinear", warp_bilinear)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("warp_bicubic", warp_bicubic)

cv2.waitKey(0)
cv2.destroyAllWindows()




# save
cv2.imwrite("warp_nearest.jpg", warp_nearest)
print("saved warp_nearest.jpg")

cv2.imwrite("warp_bilinear.jpg", warp_bilinear)
print("saved warp_bilinear.jpg")

cv2.imwrite("warp_bicubic.jpg", warp_bicubic)
print("saved warp_bicubic.jpg")
