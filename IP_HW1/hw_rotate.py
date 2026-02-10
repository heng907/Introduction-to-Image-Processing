import cv2
import numpy as np

# read image
image = cv2.imread("image.jpg")  # 替換成你的影像路徑
h, w, c = image.shape

# rotate 30 degrees
angle = np.radians(30)
cos_a = np.cos(angle)
sin_a = np.sin(angle)

# center of image
center_x = w // 2 
center_y = h // 2

# 開黑圖
rot_nearest   = np.zeros((h, w, c), np.uint8)
rot_bilinear  = np.zeros((h, w, c), np.uint8)
rot_bicubic   = np.zeros((h, w, c), np.uint8)

# Bicubic Interpolation
def cubic_interpolate(p0, p1, p2, p3, x):
    a = (-0.5 * p0) + (1.5 * p1) - (1.5 * p2) + (0.5 * p3)
    b = (p0) - (2.5 * p1) + (2 * p2) - (0.5 * p3)
    c = (-0.5 * p0) + (0.5 * p2)
    d = p1
    return a * (x ** 3) + b * (x ** 2) + c * x + d

def clamp(val, low, high):
    return max(low, min(high, val))

def get_pixel_value(img, x, y):
    x = clamp(x, 0, w - 1)
    y = clamp(y, 0, h - 1)
    return img[int(y), int(x)].astype(np.float32)


for y_new in range(h):
    for x_new in range(w):
        # 用旋轉矩陣求原圖座標
        x_old = (x_new - center_x) * cos_a + (y_new - center_y) * sin_a + center_x
        y_old = -(x_new - center_x) * sin_a + (y_new - center_y) * cos_a + center_y

        # Nearest Neighbor
        x_nn, y_nn = int(round(x_old)), int(round(y_old))
        if 0 <= x_nn < w and 0 <= y_nn < h:
            rot_nearest[y_new, x_new] = image[y_nn, x_nn]

        # Bilinear Interpolation
        x1, y1 = int(np.floor(x_old)), int(np.floor(y_old))
        x2, y2 = x1 + 1, y1 + 1
        dx, dy = (x_old - x1), (y_old - y1)

        if 0 <= x1 < w - 1 and 0 <= y1 < h - 1:
            top    = (1 - dx) * get_pixel_value(image, x1, y1) + dx * get_pixel_value(image, x2, y1)
            bottom = (1 - dx) * get_pixel_value(image, x1, y2) + dx * get_pixel_value(image, x2, y2)
            val_bilinear = (1 - dy) * top + dy * bottom
            # clip + 轉為 uint8
            val_bilinear = np.clip(val_bilinear, 0, 255).astype(np.uint8)
            rot_bilinear[y_new, x_new] = val_bilinear

        # Bicubic Interpolation
        if 1 <= x1 < w - 2 and 1 <= y1 < h - 2:
            fx, fy = x_old - x1, y_old - y1

            # 取 16 個像素
            pixels = np.zeros((4, 4, c), dtype=np.float32)
            for i in range(-1, 3):
                for j in range(-1, 3):
                    px = x1 + j
                    py = y1 + i
                    # 邊界處理
                    px = clamp(px, 0, w - 1)
                    py = clamp(py, 0, h - 1)
                    pixels[i + 1, j + 1] = image[py, px].astype(np.float32)

            # 先對 x 方向做三次插值 (對每一 row)
            col_values = np.zeros((4, c), dtype=np.float32)
            for row in range(4):
                col_values[row] = cubic_interpolate(
                    pixels[row, 0],
                    pixels[row, 1],
                    pixels[row, 2],
                    pixels[row, 3],
                    fx
                )

            # 再對 y 方向做三次插值
            val_bicubic = cubic_interpolate(
                col_values[0],
                col_values[1],
                col_values[2],
                col_values[3],
                fy
            )
            # clip + 轉為 uint8
            val_bicubic = np.clip(val_bicubic, 0, 255).astype(np.uint8)
            rot_bicubic[y_new, x_new] = val_bicubic

#show image
cv2.imshow("rotated_nearest", rot_nearest)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("rotated_bilinear", rot_bilinear)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("rotated_bicubic", rot_bicubic)

cv2.waitKey(0)
cv2.destroyAllWindows()


# save
cv2.imwrite("rotated_nearest.jpg", rot_nearest)
print("saved rotated_nearest.jpg")

cv2.imwrite("rotated_bilinear.jpg", rot_bilinear)
print("saved rotated_bilinear.jpg")

cv2.imwrite("rotated_bicubic.jpg", rot_bicubic)
print("saved rotated_bicubic.jpg")
