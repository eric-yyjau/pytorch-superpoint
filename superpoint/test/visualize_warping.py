"""Testing file (not sorted yet)

"""

import torch
import numpy as np


from utils.utils import inv_warp_image_batch
from numpy.linalg import inv
import cv2
import matplotlib.pyplot as plt
from utils.draw import plot_imgs

from utils.utils import pltImshow
path = '/home/yoyee/Documents/deepSfm/logs/superpoint_hpatches_pretrained/predictions/'
for i in range(10):
    data = np.load(path + str(i) + '.npz')
    # p1 = '/home/yoyee/Documents/deepSfm/datasets/HPatches/v_abstract/1.ppm'
    # p2 = '/home/yoyee/Documents/deepSfm/datasets/HPatches/v_abstract/2.ppm'
    # H = '/home/yoyee/Documents/deepSfm/datasets/HPatches/v_abstract/H_1_2'
    # img = np.load(p1)
    # warped_img = np.load(p2)

    H = data['homography']
    img1 = data['image'][:,:,np.newaxis]
    img2 = data['warped_image'][:,:,np.newaxis]
    # warped_img_H = inv_warp_image_batch(torch.tensor(img), torch.tensor(inv(H)))
    warped_img1 = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))


    # img_cat = np.concatenate((img, warped_img, warped_img_H), axis=1)
    # pltImshow(img_cat)

    # from numpy.linalg import inv
    # warped_img1 = cv2.warpPerspective(img1, inv(H), (img2.shape[1], img2.shape[0]))
    img1 = np.concatenate([img1, img1, img1], axis=2)
    warped_img1 = np.stack([warped_img1, warped_img1, warped_img1], axis=2)
    img2 = np.concatenate([img2, img2, img2], axis=2)
    plot_imgs([img1, img2, warped_img1], titles=['img1', 'img2', 'warped_img1'], dpi=200)
    plt.savefig( 'test' + str(i) + '.png')