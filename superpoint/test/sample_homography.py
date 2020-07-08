"""Testing file for homography (not sorted yet)
"""

import numpy as np
import tensorflow as tf
import torch
# corner_img = np.array([(0, 0), (img_w, 0), (0, img_h), (img_w, img_h)])
import cv2
from utils.utils import warp_points_np


def sample_homography(inv_scale=3):
  corner_img = np.array([(-1, -1), (-1, 1), (1, -1), (1, 1)])
  offset_r = 1 - 1/inv_scale
  img_offset = np.array([(-1, -1), (-1, offset_r), (offset_r, -1), (offset_r, offset_r)])
  corner_map = np.random.rand(4,2)/inv_scale + img_offset
  matrix = cv2.getPerspectiveTransform(np.float32(corner_img), np.float32(corner_map))
  return matrix



import matplotlib.pyplot as plt

def plot_points(matrix, ls='--', lw=1.2, colors=None):
  x_points, y_points = matrix[:,0], matrix[:,1]
  size = len(x_points)
  colors = ['red', 'blue', 'orange', 'green'] if not None else colors
  for i in range(size):
    plt.plot(x_points[i], y_points[i], color=colors[i], marker='o')
    # plt.plot(x_points[i], x_points[(i+1) % size],color=colors[i],linestyle=ls, linewidth=lw)
    # plt.plot(x_points[i], x_points[(i + 1) % size], linestyle=ls, linewidth=lw)
    # [y_points[i], y_points[(i+1) % size]],
             # color=colors[i],
             # linestyle=ls, linewidth=lw)

def printCorners(corner_img, mat_homographies):
  points = warp_points_np(corner_img, mat_homographies)
  # plot
  plot_points(corner_img)
  for i in range(points.shape[0]):
    plot_points(points[i,:,:])
  plt.show()

def test_sample_homography():
  batch_size = 30
  filename = '../configs/superpoint_coco_train.yaml'
  import yaml
  with open(filename, 'r') as f:
    config = yaml.load(f)
  test_tf = False
  test_corner_def = True

  if test_tf == True:
    from utils.homographies import sample_homography as sample_homography
    boundary = 1
    # from utils.homographies import sample_homography_np as sample_homography
    # mat_homographies = matrix[np.newaxis, :,:]
    # mat_homographies = [sample_homography(tf.constant([boundary,boundary]),
    mat_homographies = [sample_homography(np.array([boundary,boundary]),
                                          **config['data']['warped_pair']['params']) for i in range(batch_size)]
    mat_homographies = np.stack(mat_homographies, axis=0)
    corner_img = np.array([[0., 0.], [0., boundary], [boundary, boundary], [boundary, 0.]])
    printCorners(corner_img, mat_homographies)

  if test_corner_def:
    corner_img = np.array([(-1, -1), (-1, 1), (1, 1), (1, -1)])
    from utils.homographies import sample_homography_np as sample_homography
    boundary = 2
    mat_homographies = [sample_homography(np.array([boundary,boundary]), shift=-1,
                                          **config['data']['warped_pair']['params']) for i in range(batch_size)]
    mat_homographies = np.stack(mat_homographies, axis=0)
    printCorners(corner_img, mat_homographies)


  else:
    from utils.utils import sample_homography
    mat_homographies = [sample_homography(1) for i in range(batch_size)]

  # sess = tf.Session()
  # with sess.as_default():
  #   m = mat_homographies[0].eval()

  print("end")

def test_valid_mask():
  from utils.utils import pltImshow
  batch_size = 1
  mat_homographies = [sample_homography(3) for i in range(batch_size)]
  mat_H = np.stack(mat_homographies, axis=0)


  corner_img = np.array([(-1, -1), (-1, 1), (1, -1), (1, 1)])
  # printCorners(corner_img, mat_H)
  # points = warp_points_np(corner_img, mat_homographies)

  mat_H = torch.tensor(mat_H, dtype=torch.float32)
  mat_H_inv = torch.stack([torch.inverse(mat_H[i, :, :]) for i in range(batch_size)])
  from utils.utils import compute_valid_mask, labels2Dto3D
  device = 'cpu'
  shape = torch.tensor([240, 320])
  for i in range(1):
    r = 3
    mask_valid = compute_valid_mask(shape, inv_homography=mat_H_inv, device=device, erosion_radius=r)
    pltImshow(mask_valid[0,:,:])
    cell_size = 8
    mask_valid = labels2Dto3D(mask_valid.view(batch_size, 1, mask_valid.shape[1], mask_valid.shape[2]), cell_size=cell_size)
    mask_valid = torch.prod(mask_valid[:,:cell_size*cell_size,:,:], dim=1)
    pltImshow(mask_valid[0,:,:].cpu().numpy())

  mask = {}
  mask.update({'homographies': mat_H, 'masks': mask_valid})
  np.savez_compressed('h2.npz', **mask)
  print("finish testing valid mask")

if __name__ == '__main__':
    # test_sample_homography()
    test_valid_mask()


'''
x_points = np.array([0, 0, 20, 20])
y_points = np.array([0, 20, 20, 0])
matrix = np.array([x_points, y_points])
# colors = ['red', 'blue', 'magenta', 'green']
colors = ['r', 'b', 'm', 'g']
size = len(x_points)
plot_points(matrix, colors)
plt.ylim([-5,25])
plt.xlim([-5,25])
plt.axes().set_aspect('equal')
plt.show()
'''