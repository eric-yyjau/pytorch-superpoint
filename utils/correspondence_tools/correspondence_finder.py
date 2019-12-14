# torch
import torch

# math
import numpy as numpy
import numpy as np
import math
from numpy.linalg import inv
import random

# io
from PIL import Image

# torchvision
import sys
sys.path.insert(0, '../pytorch-segmentation-detection/vision/') # from subrepo
from torchvision import transforms


# from dense_correspondence_manipulation.utils.constants import *

# turns out to be faster to do this match generation on the CPU
# for the general size of params we expect
# also this will help by not taking up GPU memory, 
# allowing batch sizes to stay large
dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

def pytorch_rand_select_pixel(width,height,num_samples=1):
    two_rand_numbers = torch.rand(2,num_samples)
    two_rand_numbers[0,:] = two_rand_numbers[0,:]*width
    two_rand_numbers[1,:] = two_rand_numbers[1,:]*height
    two_rand_ints    = torch.floor(two_rand_numbers).type(dtype_long)
    return (two_rand_ints[0], two_rand_ints[1])

def get_default_K_matrix():
    K = numpy.zeros((3,3))
    K[0,0] = 533.6422696034836 # focal x
    K[1,1] = 534.7824445233571 # focal y
    K[0,2] = 319.4091030774892 # principal point x
    K[1,2] = 236.4374299691866 # principal point y
    K[2,2] = 1.0
    return K

def get_body_to_rdf():
    body_to_rdf = numpy.zeros((3,3))
    body_to_rdf[0,1] = -1.0
    body_to_rdf[1,2] = -1.0
    body_to_rdf[2,0] = 1.0
    return body_to_rdf

def invert_transform(transform4):
    transform4_copy = numpy.copy(transform4)
    R = transform4_copy[0:3,0:3]
    R = numpy.transpose(R)
    transform4_copy[0:3,0:3] = R
    t = transform4_copy[0:3,3]
    inv_t = -1.0 * numpy.transpose(R).dot(t)
    transform4_copy[0:3,3] = inv_t
    return transform4_copy

def apply_transform_torch(vec3, transform4):
    ones_row = torch.ones_like(vec3[0,:]).type(dtype_float).unsqueeze(0)
    vec4 = torch.cat((vec3,ones_row),0)
    vec4 = transform4.mm(vec4)
    return vec4[0:3]

def random_sample_from_masked_image(img_mask, num_samples):
    """
    Samples num_samples (row, column) convention pixel locations from the masked image
    Note this is not in (u,v) format, but in same format as img_mask
    :param img_mask: numpy.ndarray
        - masked image, we will select from the non-zero entries
        - shape is H x W
    :param num_samples: int
        - number of random indices to return
    :return: List of np.array
    """
    idx_tuple = img_mask.nonzero()
    num_nonzero = len(idx_tuple[0])
    if num_nonzero == 0:
        empty_list = []
        return empty_list
    rand_inds = random.sample(range(0,num_nonzero), num_samples)

    sampled_idx_list = []
    for i, idx in enumerate(idx_tuple):
        sampled_idx_list.append(idx[rand_inds])

    return sampled_idx_list

def random_sample_from_masked_image_torch(img_mask, num_samples):
    """

    :param img_mask: Numpy array [H,W] or torch.Tensor with shape [H,W]
    :type img_mask:
    :param num_samples: an integer
    :type num_samples:
    :return: tuple of torch.LongTensor in (u,v) format. Each torch.LongTensor has shape
    [num_samples]
    :rtype:
    """

    image_height, image_width = img_mask.shape

    if isinstance(img_mask, np.ndarray):
        img_mask_torch = torch.from_numpy(img_mask).float()
    else:
        img_mask_torch = img_mask

    # This code would randomly subsample from the mask
    mask = img_mask_torch.view(image_width*image_height,1).squeeze(1)
    mask_indices_flat = torch.nonzero(mask)
    if len(mask_indices_flat) == 0:
        return (None, None)

    rand_numbers = torch.rand(num_samples)*len(mask_indices_flat)
    rand_indices = torch.floor(rand_numbers).long()
    uv_vec_flattened = torch.index_select(mask_indices_flat, 0, rand_indices).squeeze(1)
    uv_vec = utils.flattened_pixel_locations_to_u_v(uv_vec_flattened, image_width)
    return uv_vec

def pinhole_projection_image_to_world(uv, z, K):
    """
    Takes a (u,v) pixel location to it's 3D location in camera frame.
    See https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html for a detailed explanation.

    :param uv: pixel location in image
    :type uv:
    :param z: depth, in camera frame
    :type z: float
    :param K: 3 x 3 camera intrinsics matrix
    :type K: numpy.ndarray
    :return: (x,y,z) in camera frame
    :rtype: numpy.array size (3,)
    """

    u_v_1 = np.array([uv[0], uv[1], 1])
    pos = z * np.matmul(inv(K),u_v_1)
    return pos

def pinhole_projection_world_to_image(world_pos, K, camera_to_world=None):
    """
    Projects from world position to camera coordinates
    See https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    :param world_pos:
    :type world_pos:
    :param K:
    :type K:
    :return:
    :rtype:
    """

    world_pos_vec = np.append(world_pos, 1)

    # transform to camera frame if camera_to_world is not None
    if camera_to_world is not None:
        world_pos_vec = np.dot(np.linalg.inv(camera_to_world), world_pos_vec)

    # scaled position is [X/Z, Y/Z, 1] where X,Y,Z is the position in camera frame
    scaled_pos = np.array([world_pos_vec[0]/world_pos_vec[2], world_pos_vec[1]/world_pos_vec[2], 1])
    uv = np.dot(K, scaled_pos)[:2]
    return uv



# in torch 0.3 we don't yet have torch.where(), although this
# is there in 0.4 (not yet stable release)
# for more see: https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
def where(cond, x_1, x_2):
    """
    We follow the torch.where implemented in 0.4.
    See http://pytorch.org/docs/master/torch.html?highlight=where#torch.where

    For more discussion see https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8


    Return a tensor of elements selected from either x_1 or x_2, depending on condition.
    :param cond: cond should be tensor with entries [0,1]
    :type cond:
    :param x_1: torch.Tensor
    :type x_1:
    :param x_2: torch.Tensor
    :type x_2:
    :return:
    :rtype:
    """
    cond = cond.type(dtype_float)    
    return (cond * x_1) + ((1-cond) * x_2)

def create_non_correspondences(uv_b_matches, img_b_shape, num_non_matches_per_match=100, img_b_mask=None):
    """
    Takes in pixel matches (uv_b_matches) that correspond to matches in another image, and generates non-matches by just sampling in image space.

    Optionally, the non-matches can be sampled from a mask for image b.

    Returns non-matches as pixel positions in image b.

    Please see 'coordinate_conventions.md' documentation for an explanation of pixel coordinate conventions.

    ## Note that arg uv_b_matches are the outputs of batch_find_pixel_correspondences()

    :param uv_b_matches: tuple of torch.FloatTensors, where each FloatTensor is length n, i.e.:
        (torch.FloatTensor, torch.FloatTensor)

    :param img_b_shape: tuple of (H,W) which is the shape of the image

    (optional)
    :param num_non_matches_per_match: int

    (optional)
    :param img_b_mask: torch.FloatTensor (can be cuda or not)
        - masked image, we will select from the non-zero entries
        - shape is H x W
     
    :return: tuple of torch.FloatTensors, i.e. (torch.FloatTensor, torch.FloatTensor).
        - The first element of the tuple is all "u" pixel positions, and the right element of the tuple is all "v" positions
        - Each torch.FloatTensor is of shape torch.Shape([num_matches, non_matches_per_match])
        - This shape makes it so that each row of the non-matches corresponds to the row for the match in uv_a
    """
    image_width  = img_b_shape[1]
    image_height = img_b_shape[0]
    # print("uv_b_matches: ", uv_b_matches)
    if uv_b_matches == None:
        return None

    num_matches = len(uv_b_matches[0])

    def get_random_uv_b_non_matches():
        return pytorch_rand_select_pixel(width=image_width,height=image_height, 
            num_samples=num_matches*num_non_matches_per_match)

    if img_b_mask is not None:
        img_b_mask_flat = img_b_mask.view(-1,1).squeeze(1)
        mask_b_indices_flat = torch.nonzero(img_b_mask_flat)
        if len(mask_b_indices_flat) == 0:
            print("warning, empty mask b")
            uv_b_non_matches = get_random_uv_b_non_matches()
        else:
            num_samples = num_matches*num_non_matches_per_match
            rand_numbers_b = torch.rand(num_samples)*len(mask_b_indices_flat)
            rand_indices_b = torch.floor(rand_numbers_b).long()
            randomized_mask_b_indices_flat = torch.index_select(mask_b_indices_flat, 0, rand_indices_b).squeeze(1)
            uv_b_non_matches = (randomized_mask_b_indices_flat%image_width, randomized_mask_b_indices_flat/image_width)
    else:
        uv_b_non_matches = get_random_uv_b_non_matches()
    
    # for each in uv_a, we want non-matches
    # first just randomly sample "non_matches"
    # we will later move random samples that were too close to being matches
    uv_b_non_matches = (uv_b_non_matches[0].view(num_matches,num_non_matches_per_match), uv_b_non_matches[1].view(num_matches,num_non_matches_per_match))

    # uv_b_matches can now be used to make sure no "non_matches" are too close
    # to preserve tensor size, rather than pruning, we can perturb these in pixel space
    copied_uv_b_matches_0 = torch.t(uv_b_matches[0].repeat(num_non_matches_per_match, 1))
    copied_uv_b_matches_1 = torch.t(uv_b_matches[1].repeat(num_non_matches_per_match, 1))

    diffs_0 = copied_uv_b_matches_0 - uv_b_non_matches[0].type(dtype_float)
    diffs_1 = copied_uv_b_matches_1 - uv_b_non_matches[1].type(dtype_float)

    diffs_0_flattened = diffs_0.view(-1,1)
    diffs_1_flattened = diffs_1.view(-1,1)

    diffs_0_flattened = torch.abs(diffs_0_flattened).squeeze(1)
    diffs_1_flattened = torch.abs(diffs_1_flattened).squeeze(1)


    need_to_be_perturbed = torch.zeros_like(diffs_0_flattened)
    ones = torch.zeros_like(diffs_0_flattened)
    num_pixels_too_close = 1.0
    threshold = torch.ones_like(diffs_0_flattened)*num_pixels_too_close

    # determine which pixels are too close to being matches
    need_to_be_perturbed = where(diffs_0_flattened < threshold, ones, need_to_be_perturbed)
    need_to_be_perturbed = where(diffs_1_flattened < threshold, ones, need_to_be_perturbed)

    minimal_perturb        = num_pixels_too_close/2
    minimal_perturb_vector = (torch.rand(len(need_to_be_perturbed))*2).floor()*(minimal_perturb*2)-minimal_perturb
    std_dev = 10
    random_vector = torch.randn(len(need_to_be_perturbed))*std_dev + minimal_perturb_vector
    perturb_vector = need_to_be_perturbed*random_vector

    uv_b_non_matches_0_flat = uv_b_non_matches[0].view(-1,1).type(dtype_float).squeeze(1)
    uv_b_non_matches_1_flat = uv_b_non_matches[1].view(-1,1).type(dtype_float).squeeze(1)

    uv_b_non_matches_0_flat = uv_b_non_matches_0_flat + perturb_vector
    uv_b_non_matches_1_flat = uv_b_non_matches_1_flat + perturb_vector

    # now just need to wrap around any that went out of bounds

    # handle wrapping in width
    lower_bound = 0.0
    upper_bound = image_width*1.0 - 1
    lower_bound_vec = torch.ones_like(uv_b_non_matches_0_flat) * lower_bound
    upper_bound_vec = torch.ones_like(uv_b_non_matches_0_flat) * upper_bound

    uv_b_non_matches_0_flat = where(uv_b_non_matches_0_flat > upper_bound_vec, 
        uv_b_non_matches_0_flat - upper_bound_vec, 
        uv_b_non_matches_0_flat)

    uv_b_non_matches_0_flat = where(uv_b_non_matches_0_flat < lower_bound_vec, 
        uv_b_non_matches_0_flat + upper_bound_vec, 
        uv_b_non_matches_0_flat)

    # handle wrapping in height
    lower_bound = 0.0
    upper_bound = image_height*1.0 - 1
    lower_bound_vec = torch.ones_like(uv_b_non_matches_1_flat) * lower_bound
    upper_bound_vec = torch.ones_like(uv_b_non_matches_1_flat) * upper_bound

    uv_b_non_matches_1_flat = where(uv_b_non_matches_1_flat > upper_bound_vec, 
        uv_b_non_matches_1_flat - upper_bound_vec, 
        uv_b_non_matches_1_flat)

    uv_b_non_matches_1_flat = where(uv_b_non_matches_1_flat < lower_bound_vec, 
        uv_b_non_matches_1_flat + upper_bound_vec, 
        uv_b_non_matches_1_flat)

    return (uv_b_non_matches_0_flat.view(num_matches, num_non_matches_per_match),
        uv_b_non_matches_1_flat.view(num_matches, num_non_matches_per_match))


# Optionally, uv_a specifies the pixels in img_a for which to find matches
# If uv_a is not set, then random correspondences are attempted to be found
def batch_find_pixel_correspondences(img_a_depth, img_a_pose, img_b_depth, img_b_pose, 
                                        uv_a=None, num_attempts=20, device='CPU', img_a_mask=None, K=None):
    """
    Computes pixel correspondences in batch

    :param img_a_depth: depth image for image a
    :type  img_a_depth: numpy 2d array (H x W) encoded as a uint16
    --
    :param img_a_pose:  pose for image a, in right-down-forward optical frame
    :type  img_a_pose:  numpy 2d array, 4 x 4 (homogeneous transform)
    --
    :param img_b_depth: depth image for image b
    :type  img_b_depth: numpy 2d array (H x W) encoded as a uint16
    -- 
    :param img_b_pose:  pose for image a, in right-down-forward optical frame
    :type  img_b_pose:  numpy 2d array, 4 x 4 (homogeneous transform)
    -- 
    :param uv_a:        optional arg, a tuple of (u,v) pixel positions for which to find matches
    :type  uv_a:        each element of tuple is either an int, or a list-like (castable to torch.LongTensor)
    --
    :param num_attempts: if random sampling, how many pixels will be _attempted_ to find matches for.  Note that
                            this is not the same as asking for a specific number of matches, since many attempted matches
                            will either be occluded or outside of field-of-view. 
    :type  num_attempts: int
    --
    :param device:      either 'CPU' or 'CPU'
    :type  device:      string
    --
    :param img_a_mask:  optional arg, an image where each nonzero pixel will be used as a mask
    :type  img_a_mask:  ndarray, of shape (H, W)
    --
    :param K:           optional arg, an image where each nonzero pixel will be used as a mask
    :type  K:           ndarray, of shape (H, W)
    --
    :return:            "Tuple of tuples", i.e. pixel position tuples for image a and image b (uv_a, uv_b). 
                        Each of these is a tuple of pixel positions
    :rtype:             Each of uv_a is a tuple of torch.FloatTensors
    """
    assert (img_a_depth.shape == img_b_depth.shape)
    image_width  = img_a_depth.shape[1]
    image_height = img_b_depth.shape[0]

    global dtype_float
    global dtype_long
    if device == 'CPU':
        dtype_float = torch.FloatTensor
        dtype_long = torch.LongTensor
    if device =='GPU':
        dtype_float = torch.cuda.FloatTensor
        dtype_long = torch.cuda.LongTensor

    if uv_a is None:
        uv_a = pytorch_rand_select_pixel(width=image_width,height=image_height, num_samples=num_attempts)
    else:
        uv_a = (torch.LongTensor([uv_a[0]]).type(dtype_long), torch.LongTensor([uv_a[1]]).type(dtype_long))
        num_attempts = 1

    if img_a_mask is None:
        uv_a_vec = (torch.ones(num_attempts).type(dtype_long)*uv_a[0],torch.ones(num_attempts).type(dtype_long)*uv_a[1])
        uv_a_vec_flattened = uv_a_vec[1]*image_width+uv_a_vec[0]
    else:
        img_a_mask = torch.from_numpy(img_a_mask).type(dtype_float)  
        
        # Option A: This next line samples from img mask
        uv_a_vec = random_sample_from_masked_image_torch(img_a_mask, num_samples=num_attempts)
        if uv_a_vec[0] is None:
            return (None, None)
        
        # Option B: These 4 lines grab ALL from img mask
        # mask_a = img_a_mask.squeeze(0)
        # mask_a = mask_a/torch.max(mask_a)
        # nonzero = (torch.nonzero(mask_a)).type(dtype_long)
        # uv_a_vec = (nonzero[:,1], nonzero[:,0])

        # Always use this line        
        uv_a_vec_flattened = uv_a_vec[1]*image_width+uv_a_vec[0]


    if K is None:
        K = get_default_K_matrix()

    K_inv = inv(K)
    body_to_rdf = get_body_to_rdf()
    rdf_to_body = inv(body_to_rdf)

    img_a_depth_torch = torch.from_numpy(img_a_depth).type(dtype_float)
    img_a_depth_torch = torch.squeeze(img_a_depth_torch, 0)
    img_a_depth_torch = img_a_depth_torch.view(-1,1)

    
    depth_vec = torch.index_select(img_a_depth_torch, 0, uv_a_vec_flattened)*1.0/DEPTH_IM_SCALE
    depth_vec = depth_vec.squeeze(1)
    
    # Prune based on
    # Case 1: depth is zero (for this data, this means no-return)
    nonzero_indices = torch.nonzero(depth_vec)
    if nonzero_indices.dim() == 0:
        return (None, None)
    nonzero_indices = nonzero_indices.squeeze(1)
    depth_vec = torch.index_select(depth_vec, 0, nonzero_indices)

    # prune u_vec and v_vec, then multiply by already pruned depth_vec
    u_a_pruned = torch.index_select(uv_a_vec[0], 0, nonzero_indices)
    u_vec = u_a_pruned.type(dtype_float)*depth_vec

    v_a_pruned = torch.index_select(uv_a_vec[1], 0, nonzero_indices)
    v_vec = v_a_pruned.type(dtype_float)*depth_vec

    z_vec = depth_vec

    full_vec = torch.stack((u_vec, v_vec, z_vec))

    K_inv_torch = torch.from_numpy(K_inv).type(dtype_float)
    point_camera_frame_rdf_vec = K_inv_torch.mm(full_vec)

    point_world_frame_rdf_vec = apply_transform_torch(point_camera_frame_rdf_vec, torch.from_numpy(img_a_pose).type(dtype_float))
    point_camera_2_frame_rdf_vec = apply_transform_torch(point_world_frame_rdf_vec, torch.from_numpy(invert_transform(img_b_pose)).type(dtype_float))

    K_torch = torch.from_numpy(K).type(dtype_float)
    vec2_vec = K_torch.mm(point_camera_2_frame_rdf_vec)

    u2_vec = vec2_vec[0]/vec2_vec[2]
    v2_vec = vec2_vec[1]/vec2_vec[2]

    maybe_z2_vec = point_camera_2_frame_rdf_vec[2]

    z2_vec = vec2_vec[2]

    # Prune based on
    # Case 2: the pixels projected into image b are outside FOV
    # u2_vec bounds should be: 0, image_width
    # v2_vec bounds should be: 0, image_height

    ## do u2-based pruning
    u2_vec_lower_bound = 0.0
    epsilon = 1e-3
    u2_vec_upper_bound = image_width*1.0 - epsilon  # careful, needs to be epsilon less!!
    lower_bound_vec = torch.ones_like(u2_vec) * u2_vec_lower_bound
    upper_bound_vec = torch.ones_like(u2_vec) * u2_vec_upper_bound
    zeros_vec       = torch.zeros_like(u2_vec)

    u2_vec = where(u2_vec < lower_bound_vec, zeros_vec, u2_vec)
    u2_vec = where(u2_vec > upper_bound_vec, zeros_vec, u2_vec)
    in_bound_indices = torch.nonzero(u2_vec)
    if in_bound_indices.dim() == 0:
        return (None, None)
    in_bound_indices = in_bound_indices.squeeze(1)

    # apply pruning
    u2_vec = torch.index_select(u2_vec, 0, in_bound_indices)
    v2_vec = torch.index_select(v2_vec, 0, in_bound_indices)
    z2_vec = torch.index_select(z2_vec, 0, in_bound_indices)
    u_a_pruned = torch.index_select(u_a_pruned, 0, in_bound_indices) # also prune from first list
    v_a_pruned = torch.index_select(v_a_pruned, 0, in_bound_indices) # also prune from first list

    ## do v2-based pruning
    v2_vec_lower_bound = 0.0
    v2_vec_upper_bound = image_height*1.0 - epsilon
    lower_bound_vec = torch.ones_like(v2_vec) * v2_vec_lower_bound
    upper_bound_vec = torch.ones_like(v2_vec) * v2_vec_upper_bound
    zeros_vec       = torch.zeros_like(v2_vec)    

    v2_vec = where(v2_vec < lower_bound_vec, zeros_vec, v2_vec)
    v2_vec = where(v2_vec > upper_bound_vec, zeros_vec, v2_vec)
    in_bound_indices = torch.nonzero(v2_vec)
    if in_bound_indices.dim() == 0:
        return (None, None)
    in_bound_indices = in_bound_indices.squeeze(1)

    # apply pruning
    u2_vec = torch.index_select(u2_vec, 0, in_bound_indices)
    v2_vec = torch.index_select(v2_vec, 0, in_bound_indices)
    z2_vec = torch.index_select(z2_vec, 0, in_bound_indices)
    u_a_pruned = torch.index_select(u_a_pruned, 0, in_bound_indices) # also prune from first list
    v_a_pruned = torch.index_select(v_a_pruned, 0, in_bound_indices) # also prune from first list

    # Prune based on
    # Case 3: the pixels in image b are occluded, OR there is no depth return in image b so we aren't sure

    img_b_depth_torch = torch.from_numpy(img_b_depth).type(dtype_float)
    img_b_depth_torch = torch.squeeze(img_b_depth_torch, 0)
    img_b_depth_torch = img_b_depth_torch.view(-1,1)

    uv_b_vec_flattened = (v2_vec.type(dtype_long)*image_width+u2_vec.type(dtype_long))  # simply round to int -- good enough 
                                                                       # occlusion check for smooth surfaces

    depth2_vec = torch.index_select(img_b_depth_torch, 0, uv_b_vec_flattened)*1.0/1000
    depth2_vec = depth2_vec.squeeze(1)

    # occlusion margin, in meters
    occlusion_margin = 0.003
    z2_vec = z2_vec - occlusion_margin
    zeros_vec = torch.zeros_like(depth2_vec)

    depth2_vec = where(depth2_vec < zeros_vec, zeros_vec, depth2_vec) # to be careful, prune any negative depths
    depth2_vec = where(depth2_vec < z2_vec, zeros_vec, depth2_vec)    # prune occlusions
    non_occluded_indices = torch.nonzero(depth2_vec)
    if non_occluded_indices.dim() == 0:
        return (None, None)
    non_occluded_indices = non_occluded_indices.squeeze(1)
    depth2_vec = torch.index_select(depth2_vec, 0, non_occluded_indices)

    # apply pruning
    u2_vec = torch.index_select(u2_vec, 0, non_occluded_indices)
    v2_vec = torch.index_select(v2_vec, 0, non_occluded_indices)
    u_a_pruned = torch.index_select(u_a_pruned, 0, non_occluded_indices) # also prune from first list
    v_a_pruned = torch.index_select(v_a_pruned, 0, non_occluded_indices) # also prune from first list

    uv_b_vec = (u2_vec, v2_vec)
    uv_a_vec = (u_a_pruned, v_a_pruned)
    return (uv_a_vec, uv_b_vec)
