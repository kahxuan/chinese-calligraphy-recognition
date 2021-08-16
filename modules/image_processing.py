import cv2 
import random
import numpy as np 

# returns an image with white character and black background
def segment_image(image):
    
    samples = np.float32(image.reshape((-1, 3)))
    k = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.99) 
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(samples, k, None, criteria, 10, flags)

    segmented = labels.reshape(image.shape[:2])

    # assume background take up more pixels
    white_region = np.bincount(labels.flatten()).argmax()
    if white_region or segmented[:, 5].mean() > 0.9:
        segmented = 1 - segmented
    
    return segmented * 255


def get_boundary(img, axis):
    emp = np.where(np.sum(img, axis=axis) == 0)[0]
    diff = (emp - np.roll(emp, 1))[1:]
    lo = emp[np.where(diff != 1)[0][0]]
    hi = emp[np.where(diff != 1)[0][-1] + 1]
    return lo, hi


def resize_by_scale(img, max_hw = 108):
    h, w = img.shape
    new_h, new_w = max_hw, max_hw
    if h > w:
        new_w = int(w * new_h / h)
    else:
        new_h = int(h * new_w / w)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return img_resized


def pad_to_square(img, n):
    h, w = img.shape
    top = (n - h) // 2
    bottom = (n - h + 1) // 2
    left = (n - w) // 2
    right = (n - w + 1) // 2
    img_square = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return img_square


def norm_char_size(img):

    # crop character
    img_padded = np.pad(img, (1, 1), 'constant', constant_values=(0, 0))
    top, bottom = get_boundary(img_padded, 0)
    left, right = get_boundary(img_padded, 1)
    img_padded = img_padded[left:right + 1, top:bottom + 1]

    img_resized = resize_by_scale(img_padded, 108)
    img_square = pad_to_square(img_resized, 128)
    
    return img_square



def augment_affine(img):
    h, w = img.shape

    rands = [(random.random() * 3 - 1.5) / 10 for _ in range(3)]
    stretch_coefs = [1, random.random() * 0.2 + 0.8]
    random.shuffle(stretch_coefs)

    src_tri = [
        [0, 0], 
        [w, 0], 
        [0, h]
    ]

    dest_tri = [
        [0, h*rands[0]], 
        [w*stretch_coefs[0], h*rands[1]], 
        [w*rands[2], h*stretch_coefs[1]]
    ]

    src_tri = np.array(src_tri).astype(np.float32)
    dest_tri = np.array(dest_tri).astype(np.float32)
    warp_mat = cv2.getAffineTransform(src_tri, dest_tri)
    res = cv2.warpAffine(img, warp_mat, (h, w))

    return res


def augment_rotate(img):
    center = (img.shape[1]//2, img.shape[0]//2)
    angle = random.random() * 10 - 6
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    res = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    return res


def augment_morph(img):
    max_coef = int(128 * 0.03)
    coef = random.randint(0, max_coef)
    if coef == 0:
        return img

    se = np.ones((coef, coef))
    if random.random() > 0.5:
        return cv2.erode(img, se)
    return cv2.dilate(img, se)


def augment_image(img):
    h, w = img.shape
    res = np.pad(img, (int(h*0.2), int(w * 0.2)), 'constant', constant_values=(0, 0))
    res = augment_affine(res)
    res = augment_rotate(res)
    res = augment_morph(res)
    res = norm_char_size(res)
    return res