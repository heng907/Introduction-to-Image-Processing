import cv2
import numpy as np

# histogram equalization
def equalization(img):
    H = img.shape[0]
    W = img.shape[1]

    pixels = H * W

    # compute p_r(r_k) = n_k / MN
    n_k = np.zeros(256)
    for i in range(H):
        for j in range(W):
            n_k[img[i][j]] += 1

    for i in range(256):
        n_k[i] /= pixels

    # compute s_k = T(r_k)

    s_k = np.zeros(256)
    s_k = n_k.cumsum()
    for i in range(256):
        s_k[i] = s_k[i] * (256-1)

    # equalize
    for i in range(H):
        for j in range(W):
            img[i][j] = round(s_k[img[i][j]])
    return img

# histogram specification
def specification(src, ref):
    
    # find the height & width
    src_H = src.shape[0]
    src_W = src.shape[1]
    ref_H = ref.shape[0]
    ref_W = ref.shape[1]

    src_pixels = src_H * src_W
    ref_pixels = ref_H * ref_W
 
    # source and reference p_r(r_k) = n_k/MN
    src_n_k = np.zeros(256)
    for i in range(src_H):
        for j in range(src_W):
            src_n_k[src[i][j]] += 1

    for i in range(256):
        src_n_k[i] /= src_pixels

    ref_n_k = np.zeros(256)
    for i in range(ref_H):
        for j in range(ref_W):
            ref_n_k[ref[i][j]] += 1

    for i in range(256):
        ref_n_k[i] /= ref_pixels

    # compute source and reference s_k = T(r_k)

    src_s_k = np.zeros(256)
    src_s_k = src_n_k.cumsum()
    for i in range(256):
        src_s_k[i] = src_s_k[i] * (256-1)


    ref_s_k = np.zeros(256)
    ref_s_k = ref_n_k.cumsum()
    for i in range(256):
        ref_s_k[i] = ref_s_k[i] * (256-1)

    # inverse transform

    mapping = np.zeros(256)

    for i in range(256):
        diff = np.abs(ref_s_k - src_s_k[i])
        mapping[i] = np.argmin(diff)

    result = mapping[src]
    return result

eq_source_img = cv2.imread("Q1.jpeg", cv2.IMREAD_GRAYSCALE)

spec_source_img = cv2.imread("Q2_source.jpg", cv2.IMREAD_GRAYSCALE)
spec_reference_img = cv2.imread("Q2_reference.jpg", cv2.IMREAD_GRAYSCALE)

equalized = equalization(eq_source_img)
matched = specification(spec_source_img, spec_reference_img)

cv2.imwrite("Q1_equalization.jpg", equalized)
cv2.imwrite("Q2_specification.jpg", matched)
