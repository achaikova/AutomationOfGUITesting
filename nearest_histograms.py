import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np
from scipy import spatial

def prepare_mask(image, x, y):
    # create a mask with the size 100x100 at the given coordinates
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[x:x + 100, y:y + 100] = 255
    masked_img = np.array(cv2.bitwise_and(image, image, mask=mask))
    return masked_img


def kdtree_search(histograms):
    # find the nearest neighbors by euclidean and manhattan distance
    tree = spatial.KDTree(histograms)
    res_manh, res_manh_ind = tree.query(histograms, p=1, k=2)
    res_eucl, res_eucl_ind = tree.query(histograms, p=2, k=2)
    manh_ind_min = np.argmin(res_manh[:, 1])
    eucl_ind_min = np.argmin(res_eucl[:, 1])
    return res_eucl_ind[eucl_ind_min][0], res_eucl_ind[eucl_ind_min][1], res_manh_ind[manh_ind_min][0], \
           res_manh_ind[manh_ind_min][1]


def get_dim(arr):
    # round image size to a divisor of 10
    return arr[0] // 10 * 10, arr[1] // 10 * 10


def get_coordinates(hist):
    # get coordinates of histogram
    y = hist // ((j_r - 90) // 10)
    x = hist - y * ((j_r - 90) // 10)
    return x * 10, y * 10


def plot_histogram(image, hist_indices, plot_ind):
    plt.subplot(plot_ind)
    for hist_ind in hist_indices[:2]:
        y, x = get_coordinates(hist_ind)
        masked_img = prepare_mask(image, x, y)
        hist_mask = cv2.calcHist([img], [0], masked_img, [256], [0, 256])
        plt.plot(hist_mask)


def plot_result(image, hist_indices):
    plot_ind = 231
    for hist_ind in hist_indices:
        y, x = get_coordinates(hist_ind)
        masked_img = prepare_mask(image, x, y)
        fig, ax = plt.subplot(plot_ind), plt.imshow(masked_img, 'gray')
        fig.title.set_text('Coordinates:x={}, y={}'.format(y, x))
        plot_ind += 1
        if plot_ind == 233:
            plot_ind += 1
    plot_histogram(image, hist_indices[:2], 233)
    plot_histogram(image, hist_indices[2:], 236)
    plt.xlim([0, 256])
    plt.show()


if __name__ == "__main__":
    img = cv2.imread(sys.argv[1], 0)

    if img is None:
        sys.exit("Could not read the image")

    hists = []
    i_r, j_r = get_dim(img.shape)
    for i in range(0, i_r - 90, 10):
        for j in range(0, j_r - 90, 10):
            msk = prepare_mask(img, i, j)
            hist_mask = cv2.calcHist([img], [0], msk, [256], [0, 256])
            hists.append(hist_mask)
    hists_shape = np.array(hists).shape
    hists = np.reshape(np.array(hists), (hists_shape[0], hists_shape[1]))
    res_hist_indices = kdtree_search(hists)
    plot_result(img, res_hist_indices)
