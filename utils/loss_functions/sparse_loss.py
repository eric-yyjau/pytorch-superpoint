import utils.correspondence_tools.correspondence_finder as correspondence_finder
import numpy as np
import torch

from utils.homographies import scale_homography_torch
from utils.loss_functions.pixelwise_contrastive_loss import PixelwiseContrastiveLoss

def get_coor_cells(Hc, Wc, cell_size, device='cpu', uv=False):
    coor_cells = torch.stack(torch.meshgrid(torch.arange(Hc), torch.arange(Wc)), dim=2)
    coor_cells = coor_cells.type(torch.FloatTensor).to(device)
    coor_cells = coor_cells.view(-1, 2)
    # change vu to uv
    if uv:
        coor_cells = torch.stack((coor_cells[:,1], coor_cells[:,0]), dim=1) # (y, x) to (x, y)

    return coor_cells.to(device)

def warp_coor_cells_with_homographies(coor_cells, homographies, uv=False, device='cpu'):
    from utils.utils import warp_points
    # warped_coor_cells = warp_points(coor_cells.view([-1, 2]), homographies, device)
    # warped_coor_cells = normPts(coor_cells.view([-1, 2]), shape)
    warped_coor_cells = coor_cells
    if uv == False:
        warped_coor_cells = torch.stack((warped_coor_cells[:,1], warped_coor_cells[:,0]), dim=1) # (y, x) to (x, y)

    # print("homographies: ", homographies)
    warped_coor_cells = warp_points(warped_coor_cells, homographies, device)

    if uv == False:
        warped_coor_cells = torch.stack((warped_coor_cells[:, :, 1], warped_coor_cells[:, :, 0]), dim=2)  # (batch, x, y) to (batch, y, x)

    # shape_cell = torch.tensor([H//cell_size, W//cell_size]).type(torch.FloatTensor).to(device)
    # warped_coor_mask = denormPts(warped_coor_cells, shape_cell)

    return warped_coor_cells


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


def descriptor_loss_sparse(descriptors, descriptors_warped, homographies, mask_valid=None,
                           cell_size=8, device='cpu', descriptor_dist=4, lamda_d=250,
                           num_matching_attempts=1000, num_masked_non_matches_per_match=10, 
                           dist='cos', method='1d', **config):
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
        return torch.stack([uv_tuple[0], uv_tuple[1]])

    def tuple_to_1d(uv_tuple, W, uv=True):
        if uv:
            return uv_tuple[0] + uv_tuple[1]*W
        else:
            return uv_tuple[0]*W + uv_tuple[1]


    def uv_to_1d(points, W, uv=True):
        # assert points.dim == 2
        #     print("points: ", points[0])
        #     print("H: ", H)
        if uv:
            return points[..., 0] + points[..., 1]*W
        else:
            return points[..., 0]*W + points[..., 1]

    ## calculate matches loss
    def get_match_loss(image_a_pred, image_b_pred, matches_a, matches_b, dist='cos', method='1d'):
        match_loss, matches_a_descriptors, matches_b_descriptors = \
            PixelwiseContrastiveLoss.match_loss(image_a_pred, image_b_pred, 
                matches_a, matches_b, dist=dist, method=method)
        return match_loss

    def get_non_matches_corr(img_b_shape, uv_a, uv_b_matches, num_masked_non_matches_per_match=10, device='cpu'):
        ## sample non matches
        uv_b_matches = uv_b_matches.squeeze()
        uv_b_matches_tuple = uv_to_tuple(uv_b_matches)
        uv_b_non_matches_tuple = correspondence_finder.create_non_correspondences(uv_b_matches_tuple,
                                        img_b_shape, num_non_matches_per_match=num_masked_non_matches_per_match,
                                        img_b_mask=None)

        ## create_non_correspondences
        #     print("img_b_shape ", img_b_shape)
        #     print("uv_b_matches ", uv_b_matches.shape)
        # print("uv_a: ", uv_to_tuple(uv_a))
        # print("uv_b_non_matches: ", uv_b_non_matches)
        #     print("uv_b_non_matches: ", tensorUv2tuple(uv_b_non_matches))
        uv_a_tuple, uv_b_non_matches_tuple = \
            create_non_matches(uv_to_tuple(uv_a), uv_b_non_matches_tuple, num_masked_non_matches_per_match)
        return uv_a_tuple, uv_b_non_matches_tuple

    def get_non_match_loss(image_a_pred, image_b_pred, non_matches_a, non_matches_b, dist='cos'):
        ## non matches loss
        non_match_loss, num_hard_negatives, non_matches_a_descriptors, non_matches_b_descriptors = \
            PixelwiseContrastiveLoss.non_match_descriptor_loss(image_a_pred, image_b_pred,
                                                               non_matches_a.long().squeeze(),
                                                               non_matches_b.long().squeeze(),
                                                               M=0.2, invert=True, dist=dist)
        non_match_loss = non_match_loss.sum()/(num_hard_negatives + 1)
        return non_match_loss

    from utils.utils import filter_points
    from utils.utils import crop_or_pad_choice
    from utils.utils import normPts
    # ##### print configs
    # print("num_masked_non_matches_per_match: ", num_masked_non_matches_per_match)
    # print("num_matching_attempts: ", num_matching_attempts)
    # dist = 'cos'
    # print("method: ", method)

    Hc, Wc = descriptors.shape[1], descriptors.shape[2]
    img_shape = (Hc, Wc)
    # print("img_shape: ", img_shape)
    # img_shape_cpu = (Hc.to('cpu'), Wc.to('cpu'))

    # image_a_pred = descriptors.view(1, -1, Hc * Wc).transpose(1, 2)  # torch [batch_size, H*W, D]
    def descriptor_reshape(descriptors):
        descriptors = descriptors.view(-1, Hc * Wc).transpose(0, 1)  # torch [D, H, W] --> [H*W, d]
        descriptors = descriptors.unsqueeze(0)  # torch [1, H*W, D]
        return descriptors

    image_a_pred = descriptor_reshape(descriptors)  # torch [1, H*W, D]
    # print("image_a_pred: ", image_a_pred.shape)
    image_b_pred = descriptor_reshape(descriptors_warped)  # torch [batch_size, H*W, D]

    # matches
    uv_a = get_coor_cells(Hc, Wc, cell_size, uv=True, device='cpu')
    # print("uv_a: ", uv_a[0])

    homographies_H = scale_homography_torch(homographies, img_shape, shift=(-1, -1))

    # print("experiment inverse homographies")
    # homographies_H = torch.stack([torch.inverse(H) for H in homographies_H])
    # print("homographies_H: ", homographies_H.shape)
    # homographies_H = torch.inverse(homographies_H)


    uv_b_matches = warp_coor_cells_with_homographies(uv_a, homographies_H.to('cpu'), uv=True, device='cpu')
    # 
    # print("uv_b_matches before round: ", uv_b_matches[0])

    uv_b_matches.round_() 
    # print("uv_b_matches after round: ", uv_b_matches[0])
    uv_b_matches = uv_b_matches.squeeze(0)


    # filtering out of range points
    # choice = crop_or_pad_choice(x_all.shape[0], self.sift_num, shuffle=True)

    uv_b_matches, mask = filter_points(uv_b_matches, torch.tensor([Wc, Hc]).to(device='cpu'), return_mask=True)
    # print ("pos mask sum: ", mask.sum())
    uv_a = uv_a[mask]

    # crop to the same length
    shuffle = True
    if not shuffle: print("shuffle: ", shuffle)
    choice = crop_or_pad_choice(uv_b_matches.shape[0], num_matching_attempts, shuffle=shuffle)
    choice = torch.tensor(choice)
    uv_a = uv_a[choice]
    uv_b_matches = uv_b_matches[choice]

    if method == '2d':
        matches_a = normPts(uv_a, torch.tensor([Wc, Hc]).float()) # [u, v]
        matches_b = normPts(uv_b_matches, torch.tensor([Wc, Hc]).float())
    else:
        matches_a = uv_to_1d(uv_a, Wc)
        matches_b = uv_to_1d(uv_b_matches, Wc)

    # print("matches_a: ", matches_a.shape)
    # print("matches_b: ", matches_b.shape)
    # print("matches_b max: ", matches_b.max())

    if method == '2d':
        match_loss = get_match_loss(descriptors, descriptors_warped, matches_a.to(device), 
            matches_b.to(device), dist=dist, method='2d')
    else:
        match_loss = get_match_loss(image_a_pred, image_b_pred, 
            matches_a.long().to(device), matches_b.long().to(device), dist=dist)

    # non matches

    # get non matches correspondence
    uv_a_tuple, uv_b_non_matches_tuple = get_non_matches_corr(img_shape,
                                            uv_a, uv_b_matches,
                                            num_masked_non_matches_per_match=num_masked_non_matches_per_match)

    non_matches_a = tuple_to_1d(uv_a_tuple, Wc)
    non_matches_b = tuple_to_1d(uv_b_non_matches_tuple, Wc)

    # print("non_matches_a: ", non_matches_a)
    # print("non_matches_b: ", non_matches_b)

    non_match_loss = get_non_match_loss(image_a_pred, image_b_pred, non_matches_a.to(device),
                                        non_matches_b.to(device), dist=dist)
    # non_match_loss = non_match_loss.mean()

    loss = lamda_d * match_loss + non_match_loss
    return loss, lamda_d * match_loss, non_match_loss
    pass

"""
img[uv_b_matches.long()[:,1],uv_b_matches.long()[:,0]] = 1
from utils.utils import pltImshow
pltImshow(img.numpy())

"""

def batch_descriptor_loss_sparse(descriptors, descriptors_warped, homographies, **options):
    loss = []
    pos_loss = []
    neg_loss = []
    batch_size = descriptors.shape[0]
    for i in range(batch_size):
        losses = descriptor_loss_sparse(descriptors[i], descriptors_warped[i],
                    # torch.tensor(homographies[i], dtype=torch.float32), **options)
                    homographies[i].type(torch.float32), **options)
        loss.append(losses[0])
        pos_loss.append(losses[1])
        neg_loss.append(losses[2])
    loss, pos_loss, neg_loss = torch.stack(loss), torch.stack(pos_loss), torch.stack(neg_loss)
    return loss.mean(), None, pos_loss.mean(), neg_loss.mean()

if __name__ == '__main__':
    # config
    H, W = 240, 320
    cell_size = 8
    Hc, Wc = H // cell_size, W // cell_size

    D = 3
    torch.manual_seed(0)
    np.random.seed(0)

    batch_size = 2
    device = 'cpu'
    method = '2d'

    num_matching_attempts = 1000
    num_masked_non_matches_per_match = 200
    lamda_d = 1

    homographies = np.identity(3)[np.newaxis, :, :]
    homographies = np.tile(homographies, [batch_size, 1, 1])

    def randomDescriptor():
        descriptors = torch.tensor(np.random.rand(2, D, Hc, Wc)-0.5, dtype=torch.float32)
        dn = torch.norm(descriptors, p=2, dim=1)  # Compute the norm.
        descriptors = descriptors.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return descriptors

    # descriptors = torch.tensor(np.random.rand(2, D, Hc, Wc), dtype=torch.float32)
    # dn = torch.norm(descriptors, p=2, dim=1) # Compute the norm.
    # desc = descriptors.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    descriptors = randomDescriptor()
    print("descriptors: ", descriptors.shape)
    # descriptors_warped = torch.tensor(np.random.rand(2, D, Hc, Wc), dtype=torch.float32)
    descriptors_warped = randomDescriptor()
    descriptor_loss = descriptor_loss_sparse(descriptors[0], descriptors_warped[0],
                                             torch.tensor(homographies[0], dtype=torch.float32),
                                             method=method)
    print("descriptor_loss: ", descriptor_loss)

    # loss = batch_descriptor_loss_sparse(descriptors, descriptors_warped,
    #                                     torch.tensor(homographies, dtype=torch.float32),
    #                                     num_matching_attempts = num_matching_attempts,
    #                                     num_masked_non_matches_per_match = num_masked_non_matches_per_match,
    #                                     device=device,
    #                                     lamda_d = lamda_d, 
    #                                     method=method)
    # print("batch descriptor_loss: ", loss)

    loss = batch_descriptor_loss_sparse(descriptors, descriptors,
                                        torch.tensor(homographies, dtype=torch.float32),
                                        num_matching_attempts = num_matching_attempts,
                                        num_masked_non_matches_per_match = num_masked_non_matches_per_match,
                                        device=device,
                                        lamda_d = lamda_d,
                                        method=method)
    print("same descriptor_loss (pos should be 0): ", loss)

