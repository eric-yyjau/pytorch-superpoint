#!/usr/bin/env python
# coding: utf-8

# # Non-correspondences

# In[1]:


import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
print(module_path)
# get_ipython().run_line_magic('matplotlib', 'inline')
# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))


# In[2]:


# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# import dense_correspondence_manipulation.utils.utils as utils
# utils.add_dense_correspondence_to_python_path()
import utils.correspondence_tools.correspondence_plotter as correspondence_plotter
import utils.correspondence_tools.correspondence_finder as correspondence_finder
# from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
import os
import numpy as np
import torch
import time


# In[3]:


def uvto1d(points, H):
    # assert points.dim == 2
#     print("points: ", points[0])
#     print("H: ", H)
    return points[0]*H + points[1]

H, W = 30, 40
uv_a_matches = (torch.tensor(np.array([1]), dtype=torch.float32), 
                torch.tensor(np.array([1]), dtype=torch.float32))
matches_a = uv_a_matches[0]*H + uv_a_matches[1]
matches_a = uvto1d(uv_a_matches, H)
print("uv_a_matches: ", uv_a_matches)
print("matches_a: ", matches_a)

uv_b_matches = (torch.tensor(np.array([1]), dtype=torch.float32), 
                torch.tensor(np.array([1]), dtype=torch.float32))
matches_b = uvto1d(uv_b_matches, H)

img_b_shape = (H, W)
img_a_shape = img_b_shape

# image_a_pred


# In[4]:


# num_attempts = 5

# img_a_index = dataset.get_random_image_index(scene)
# img_a_rgb, img_a_depth, _, img_a_pose = dataset.get_rgbd_mask_pose(scene, img_a_index)

# img_b_index = dataset.get_img_idx_with_different_pose(scene, img_a_pose, num_attempts=50)
# img_b_rgb, img_b_depth, _, img_b_pose = dataset.get_rgbd_mask_pose(scene, img_b_index)

# img_a_depth_numpy = np.asarray(img_a_depth)
# img_b_depth_numpy = np.asarray(img_b_depth)

# start = time.time()
# uv_a, uv_b = correspondence_finder.batch_find_pixel_correspondences(img_a_depth_numpy, img_a_pose, 
#                                                                     img_b_depth_numpy, img_b_pose,
#                                                                     num_attempts=num_attempts,
#                                                                     device='CPU')


start = time.time()
# uv_b_non_matches = correspondence_finder.create_non_correspondences(uv_b, img_a_depth_numpy.shape, num_non_matches_per_match=10)
uv_b_non_matches = correspondence_finder.create_non_correspondences(uv_b_matches, img_b_shape, num_non_matches_per_match=10, img_b_mask=None)
print  (time.time() - start, "seconds for non-matches")
if uv_b_non_matches is not None:
    print (uv_b_non_matches[0].shape)

    import torch
    # This just checks to make sure nothing is out of bounds
    print (torch.min(uv_b_non_matches[0]))
    print (torch.min(uv_b_non_matches[1]))
    print (torch.max(uv_b_non_matches[0]))
    print (torch.max(uv_b_non_matches[1]))
    
#     fig, axes = correspondence_plotter.plot_correspondences_direct(img_a_rgb, img_a_depth_numpy, img_b_rgb, img_b_depth_numpy, uv_a, uv_b, show=False)
#     uv_a_long = (torch.t(uv_a[0].repeat(3, 1)).contiguous().view(-1,1), torch.t(uv_a[1].repeat(3, 1)).contiguous().view(-1,1))
#     uv_b_non_matches_long = (uv_b_non_matches[0].view(-1,1), uv_b_non_matches[1].view(-1,1) )
#     correspondence_plotter.plot_correspondences_direct(img_a_rgb, img_a_depth_numpy, img_b_rgb, img_b_depth_numpy, uv_a_long, uv_b_non_matches_long, use_previous_plot=(fig,axes),
#                                                   circ_color='r')


# In[5]:


print("uv_a_matches ", uv_a_matches)
print("img_a_shape ", img_a_shape)
uv_a_non_matches = correspondence_finder.create_non_correspondences(uv_a_matches, img_a_shape, num_non_matches_per_match=10, img_b_mask=None)
uv_b_non_matches = correspondence_finder.create_non_correspondences(uv_b_matches, img_b_shape, num_non_matches_per_match=10, img_b_mask=None)

non_matches_a = uvto1d(uv_a_non_matches, H)
non_matches_b = uvto1d(uv_b_non_matches, H)

print("non_matches_a: ", non_matches_a)

# uv_b_non_matches # (u, v)


# In[6]:


torch.tensor(np.array([1]), dtype=torch.float32)


# In[7]:


from utils.loss_functions.pixelwise_contrastive_loss import PixelwiseContrastiveLoss


# In[8]:


def pltImshow(img):
    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.show()


# In[9]:


D = 3
# image_a = torch.tensor(np.random.rand(H, W), dtype=torch.float32)
# image_b = torch.tensor(np.random.rand(H, W), dtype=torch.float32)

image_a_pred = torch.tensor(np.random.rand(1, H*W, D), dtype=torch.float32)
image_b_pred = torch.tensor(np.random.rand(1, H*W, D), dtype=torch.float32)

# pltImshow(image_a.numpy())
# pltImshow(image_b.numpy())
# print("image_a_pred: ", ))


# In[10]:


alpha = 0.5

match_loss, matches_a_descriptors, matches_b_descriptors =     PixelwiseContrastiveLoss.match_loss(image_a_pred, image_b_pred, matches_a.long(), matches_b.long())
non_match_loss, num_hard_negatives, non_matches_a_descriptors, non_matches_b_descriptors =     PixelwiseContrastiveLoss.non_match_descriptor_loss(image_a_pred, image_b_pred, 
                                                   non_matches_a.long().squeeze(), non_matches_b.long().squeeze())


print("match_loss: ", match_loss)
print("matches_a_descriptors: ", matches_a_descriptors.shape)
print("matches_b_descriptors: ", matches_b_descriptors.shape)

print("non_match_loss: ", non_match_loss)
print("num_hard_negatives: ", num_hard_negatives)
print("non_matches_a_descriptors: ", non_matches_a_descriptors.shape)

# loss, match_loss, non_match_loss = \
#     PixelwiseContrastiveLoss.get_triplet_loss(image_a_pred,
#                                         image_b_pred,
#                                         matches_a.long(),
#                                         matches_b.long(),
#                                         non_matches_a.long(),
#                                         non_matches_b.long(), 
#                                         alpha = alpha)


# In[11]:


def scale_homography(H, shape, shift=(-1,-1)):
    height, width = shape[0], shape[1]
    trans = np.array([[2./width, 0., shift[0]], [0., 2./height, shift[1]], [0., 0., 1.]])
    H_tf = np.linalg.inv(trans) @ H @ trans
    return H_tf

def scale_homography_torch(H, shape, shift=(-1,-1), dtype=torch.float32):
    height, width = shape[0], shape[1]
    trans = torch.tensor([[2./width, 0., shift[0]], [0., 2./height, shift[1]], [0., 0., 1.]], dtype=dtype)
    print("torch.inverse(trans) ", torch.inverse(trans))
    print("H: ", H)
    H_tf = torch.inverse(trans) @ H @ trans
    return H_tf

homographies = np.identity(3)[np.newaxis,:,:]
height, width = 240, 320
image_shape = np.array([height, width])

homographies_H = np.stack([scale_homography(homographies, image_shape, shift=(-1,-1)) for H in homographies])
homographies_H = np.stack([scale_homography_torch(torch.tensor(homographies, dtype=torch.float32), 
                                                  image_shape, shift=(-1,-1)) for H in homographies])

print("homographies: ", homographies)
print("homographies_H: ", homographies_H)


# In[12]:


H, W = 240, 320
cell_size = 8
Hc, Wc = H//cell_size, W//cell_size

device = 'cpu'

def get_coor_cells(Hc, Wc, cell_size, device='cpu', uv=False):
    coor_cells = torch.stack(torch.meshgrid(torch.arange(Hc), torch.arange(Wc)), dim=2)
    coor_cells = coor_cells.type(torch.FloatTensor).to(device)
    coor_cells = coor_cells.view(-1, 2)
    # change vu to uv
    if uv:
        coor_cells = torch.stack((coor_cells[:,1], coor_cells[:,0]), dim=1) # (y, x) to (x, y)

    return coor_cells

coor_cells = get_coor_cells(Hc, Wc, cell_size=cell_size, device=device, uv=True)
print("coor_cells: ", coor_cells)
print("coor_cells: ", coor_cells.shape)

from utils.utils import filter_points
filtered_points, mask = filter_points(coor_cells, torch.tensor([Wc, Hc]), return_mask=True)


def warp_coor_cells_with_homographies(coor_cells, homographies, uv=False):
    from utils.utils import warp_points
    # warped_coor_cells = warp_points(coor_cells.view([-1, 2]), homographies, device)
#     warped_coor_cells = normPts(coor_cells.view([-1, 2]), shape)
    warped_coor_cells = coor_cells
    if uv == False:
        warped_coor_cells = torch.stack((warped_coor_cells[:,1], warped_coor_cells[:,0]), dim=1) # (y, x) to (x, y)

    warped_coor_cells = warp_points(warped_coor_cells, homographies, device)

    if uv == False:
        warped_coor_cells = torch.stack((warped_coor_cells[:, :, 1], warped_coor_cells[:, :, 0]), dim=2)  # (batch, x, y) to (batch, y, x)

    # shape_cell = torch.tensor([H//cell_size, W//cell_size]).type(torch.FloatTensor).to(device)
    # warped_coor_mask = denormPts(warped_coor_cells, shape_cell)

    return warped_coor_cells

matches_a = uvto1d(coor_cells, Hc)
print("matches_a: ", matches_a.shape)

matches_b = matches_a
print("matches_b: ", matches_b.shape)


# In[13]:


match_loss, matches_a_descriptors, matches_b_descriptors =     PixelwiseContrastiveLoss.match_loss(image_a_pred, image_b_pred, matches_a.long(), matches_b.long())
print("match_loss: ", match_loss)
print("matches_a_descriptors: ", matches_a_descriptors.shape)
print("matches_b_descriptors: ", matches_b_descriptors.shape)


# In[14]:


def create_non_matches(uv_a, uv_b_non_matches, multiplier):
    """
    Simple wrapper for repeated code
    :param uv_a:
    :type uv_a:
    :param uv_b_non_matches:
    :type uv_b_non_matches:
    :param multiplier:
    :type multiplier:
    :return:
    :rtype:
    """
    uv_a_long = (torch.t(uv_a[0].repeat(multiplier, 1)).contiguous().view(-1, 1),
                 torch.t(uv_a[1].repeat(multiplier, 1)).contiguous().view(-1, 1))

    uv_b_non_matches_long = (uv_b_non_matches[0].view(-1, 1), uv_b_non_matches[1].view(-1, 1))

    return uv_a_long, uv_b_non_matches_long


# In[20]:


def descriptor_loss_sparse(descriptors, descriptors_warped, homographies, mask_valid=None,
                           cell_size=8, device='cpu', descriptor_dist=4, lamda_d=250,
                           num_matching_attempts=1000, num_masked_non_matches_per_match=10, **config):
    """
    consider batches of descriptors
    :param descriptors:
        Output from descriptor head
        tensor [descriptors, Hc, Wc]
    :param descriptors_warped:
        Output from descriptor head of warped image
        tensor [descriptors, Hc, Wc]
    """

    def uv_to_tuple(uv):
        return (uv[:, 0], uv[:, 1])

    def tuple_to_uv(uv_tuple):
        return torch.stack([uv_tuple[:, 0], uv_tuple[:, 1]])

    def tuple_to_1d(uv_tuple, H):
        return uv_tuple[0] * H + uv_tuple[1]

    def uv_to_1d(points, H):
        # assert points.dim == 2
    #     print("points: ", points[0])
    #     print("H: ", H)
        return points[...,0]*H + points[...,1]

    ## calculate matches loss
    def get_match_loss(image_a_pred, image_b_pred, matches_a, matches_b):
        match_loss, matches_a_descriptors, matches_b_descriptors = \
            PixelwiseContrastiveLoss.match_loss(image_a_pred, image_b_pred, matches_a.long(), matches_b.long())
        return match_loss

    def get_non_matches_corr(img_b_shape, uv_a, uv_b_matches, num_masked_non_matches_per_match=10):
        ## sample non matches
        uv_b_matches = uv_b_matches.squeeze()
        uv_b_matches_tuple = uv_to_tuple(uv_b_matches)
        uv_b_non_matches_tuple = correspondence_finder.create_non_correspondences(uv_b_matches_tuple,
                        img_b_shape, num_non_matches_per_match=num_masked_non_matches_per_match, img_b_mask=None)

        ## create_non_correspondences
        #     print("img_b_shape ", img_b_shape)
        #     print("uv_b_matches ", uv_b_matches.shape)
        # print("uv_a: ", uv_to_tuple(uv_a))
        # print("uv_b_non_matches: ", uv_b_non_matches)
    #     print("uv_b_non_matches: ", tensorUv2tuple(uv_b_non_matches))
        uv_a_tuple, uv_b_non_matches_tuple = \
            create_non_matches(uv_to_tuple(uv_a), uv_b_non_matches_tuple, num_masked_non_matches_per_match)
        return uv_a_tuple, uv_b_non_matches_tuple

    def get_non_match_loss(image_a_pred, image_b_pred, non_matches_a, non_matches_b):
        ## non matches loss
        non_match_loss, num_hard_negatives, non_matches_a_descriptors, non_matches_b_descriptors = \
                        PixelwiseContrastiveLoss.non_match_descriptor_loss(image_a_pred, image_b_pred,
                                                   non_matches_a.long().squeeze(), non_matches_b.long().squeeze())
        return non_match_loss

    from utils.utils import filter_points
    from utils.utils import crop_or_pad_choice

    Hc, Wc = descriptors.shape[1], descriptors.shape[2]
    
    image_a_pred = descriptors.view(1, -1, Hc*Wc).transpose(1,2)  # torch [batch_size, H*W, D]
    # print("image_a_pred: ", image_a_pred.shape)
    image_b_pred = descriptors_warped.view(1, -1, Hc*Wc).transpose(1,2)  # torch [batch_size, H*W, D]

    # matches
    uv_a = get_coor_cells(Hc, Wc, cell_size, uv=True)
    # print("uv_a: ", uv_a.shape)

    homographies_H = scale_homography_torch(homographies, image_shape, shift=(-1,-1))
    
    uv_b_matches = warp_coor_cells_with_homographies(uv_a, homographies_H, uv=True)
    uv_b_matches = uv_b_matches.squeeze(0)
    # print("uv_b_matches: ", uv_b_matches.shape)

    # filtering out of range points
    # choice = crop_or_pad_choice(x_all.shape[0], self.sift_num, shuffle=True)

    uv_b_matches, mask = filter_points(uv_b_matches, torch.tensor([Wc, Hc]), return_mask=True)
    uv_a = uv_a[mask]

    # crop to the same length
    choice = crop_or_pad_choice(uv_b_matches.shape[0], num_matching_attempts, shuffle=True)
    choice = torch.tensor(choice)
    uv_a =         uv_a[choice]
    uv_b_matches = uv_b_matches[choice]

    matches_a = uv_to_1d(uv_a, Hc)
    matches_b = uv_to_1d(uv_b_matches, Hc)

    # print("matches_a: ", matches_a.shape)
    # print("matches_b: ", matches_b.shape)
    # print("matches_b max: ", matches_b.max())

    match_loss = get_match_loss(image_a_pred, image_b_pred, matches_a, matches_b)

    # non matches
    img_b_shape = (Hc, Wc)

    # get non matches correspondence
    uv_a_tuple, uv_b_non_matches_tuple = get_non_matches_corr(img_b_shape,
                                     uv_a, uv_b_matches,
                                     num_masked_non_matches_per_match=num_masked_non_matches_per_match)

    non_matches_a = tuple_to_1d(uv_a_tuple, Hc)
    non_matches_b = tuple_to_1d(uv_b_non_matches_tuple, Hc)

    # print("non_matches_a: ", non_matches_a)
    # print("non_matches_b: ", non_matches_b)

    non_match_loss = get_non_match_loss(image_a_pred, image_b_pred, non_matches_a, non_matches_a)
    non_match_loss = non_match_loss.mean()

    loss = lamda_d*match_loss + non_match_loss
    return loss
    pass



batch_size = 2
descriptors = torch.tensor(np.random.rand(2, D, Hc, Wc), dtype=torch.float32)
homographies = np.tile(homographies, [batch_size, 1, 1])
print("descriptors: ", descriptors.shape)
descriptors_warped = torch.tensor(np.random.rand(2, D, Hc, Wc), dtype=torch.float32)
descriptor_loss = descriptor_loss_sparse(descriptors[0], descriptors_warped[0], torch.tensor(homographies[0], dtype=torch.float32))
print("descriptor_loss: ", descriptor_loss)


def batch_descriptor_loss_sparse(descriptors, descriptors_warped, homographies):
    loss = []
    batch_size = descriptors.shape[0]
    for i in range(batch_size):
        loss.append(descriptor_loss_sparse(descriptors[i], descriptors_warped[i],
                                             torch.tensor(homographies[i], dtype=torch.float32)) )
    loss = torch.stack(loss)
    return loss.mean()


# batch_size = descriptors.shape[0]
# batch_loss = torch.stack([descriptor_loss_sparse(descriptors[i, ...], descriptors_warped[i, ...],
#               torch.tensor(homographies[i, ...], dtype=torch.float32)) \
#               for i in range(batch_size)])

loss = batch_descriptor_loss_sparse(descriptors, descriptors_warped, torch.tensor(homographies, dtype=torch.float32))
print("batch descriptor_loss: ", loss)


# def batch_loss(func):
#     def inner(*args, **kwargs):
#         return func(*args, **kwargs)
#     return inner

# In[ ]:


# print("uv_b_matches: ", uv_b_matches)
if uv_b_matches != None:
    print("none")


# In[ ]:


print("homographies: ", homographies.shape)


# In[ ]:


# uv_a_masked_long, uv_b_masked_non_matches_long =         self.create_non_matches(matches_1, matches_2_masked_non_matches, self.num_masked_non_matches_per_match)


# In[ ]:


matches_a.long().type()


# In[ ]:


def compute_loss_on_dataset(dcn, data_loader, loss_config, num_iterations=500,):
    """

    Computes the loss for the given number of iterations

    :param dcn:
    :type dcn:
    :param data_loader:
    :type data_loader:
    :param num_iterations:
    :type num_iterations:
    :return:
    :rtype:
    """
    dcn.eval()

    # loss_vec = np.zeros(num_iterations)
    loss_vec = []
    match_loss_vec = []
    non_match_loss_vec = []
    counter = 0
    pixelwise_contrastive_loss = PixelwiseContrastiveLoss(dcn.image_shape, config=loss_config)

    batch_size = 1

    for i, data in enumerate(data_loader, 0):

        # get the inputs
        data_type, img_a, img_b, matches_a, matches_b, non_matches_a, non_matches_b, metadata = data
        data_type = data_type[0]

        if len(matches_a[0]) == 0:
            print ("didn't have any matches, continuing")
            continue

        img_a = Variable(img_a.cuda(), requires_grad=False)
        img_b = Variable(img_b.cuda(), requires_grad=False)

        if data_type == "matches":
            matches_a = Variable(matches_a.cuda().squeeze(0), requires_grad=False)
            matches_b = Variable(matches_b.cuda().squeeze(0), requires_grad=False)
            non_matches_a = Variable(non_matches_a.cuda().squeeze(0), requires_grad=False)
            non_matches_b = Variable(non_matches_b.cuda().squeeze(0), requires_grad=False)

        # run both images through the network
        image_a_pred = dcn.forward(img_a)
        image_a_pred = dcn.process_network_output(image_a_pred, batch_size)

        image_b_pred = dcn.forward(img_b)
        image_b_pred = dcn.process_network_output(image_b_pred, batch_size)

        # get loss
        if data_type == "matches":
            loss, match_loss, non_match_loss =                 pixelwise_contrastive_loss.get_loss(image_a_pred,
                                                    image_b_pred,
                                                    matches_a,
                                                    matches_b,
                                                    non_matches_a,
                                                    non_matches_b)



            loss_vec.append(loss.data[0])
            non_match_loss_vec.append(non_match_loss.data[0])
            match_loss_vec.append(match_loss.data[0])


        if i > num_iterations:
            break

    loss_vec = np.array(loss_vec)
    match_loss_vec = np.array(match_loss_vec)
    non_match_loss_vec = np.array(non_match_loss_vec)

    loss = np.average(loss_vec)
    match_loss = np.average(match_loss_vec)
    non_match_loss = np.average(non_match_loss_vec)

    return loss, match_loss, non_match_loss

