"""losses
# losses for heatmap residule
# use it if you're computing residual loss. 
# current disable residual loss

"""
# losse
import torch

def print_var(points):
    print("points: ", points.shape)
    print("points: ", points)
    pass

# from utils.losses import pts_to_bbox
def pts_to_bbox(points, patch_size):  
    """
    input: 
        points: (y, x)
    output:
        bbox: (x1, y1, x2, y2)
    """

    shift_l = (patch_size+1) / 2
    shift_r = patch_size - shift_l
    pts_l = points-shift_l
    pts_r = points+shift_r+1
    bbox = torch.stack((pts_l[:,1], pts_l[:,0], pts_r[:,1], pts_r[:,0]), dim=1)
    return bbox
    pass

# roi pooling
# from utils.losses import _roi_pool
# def _roi_pool(pred_heatmap, rois, patch_size=8):
#     from utils.roi_pool import RoIPool  # noqa: E402
#     m = RoIPool(patch_size, 1.0)
#     patches = m(pred_heatmap, rois.float())
#     return patches

# torchvision roi pooling
def _roi_pool(pred_heatmap, rois, patch_size=8):
    from torchvision.ops import roi_pool
    patches = roi_pool(pred_heatmap, rois.float(), (patch_size, patch_size), spatial_scale=1.0)
    return patches
    pass

# from utils.losses import norm_patches
def norm_patches(patches):
    patch_size = patches.shape[-1]
    patches = patches.view(-1, 1, patch_size*patch_size)
    d = torch.sum(patches, dim=-1).unsqueeze(-1) + 1e-6
    patches = patches/d
    patches = patches.view(-1, 1, patch_size, patch_size)
    # print("patches: ", patches.shape)
    return patches

# from utils.losses import extract_patch_from_points
def extract_patch_from_points(heatmap, points, patch_size=5):
    """
    this function works in numpy
    """
    import numpy as np
    from utils.utils import toNumpy
    # numpy
    if type(heatmap) is torch.Tensor:
        heatmap = toNumpy(heatmap)
    heatmap = heatmap.squeeze()  # [H, W]
    # padding
    pad_size = int(patch_size/2)
    heatmap = np.pad(heatmap, pad_size, 'constant')
    # crop it
    patches = []
    ext = lambda img, pnt, wid: img[pnt[1]:pnt[1]+wid, pnt[0]:pnt[0]+wid]
    print("heatmap: ", heatmap.shape)
    for i in range(points.shape[0]):
        # print("point: ", points[i,:])
        patch = ext(heatmap, points[i,:].astype(int), patch_size)
        # print("patch: ", patch.shape)
        patches.append(patch)
        
        # if i > 10: break
    # extract points
    return patches

# from utils.losses import extract_patches
def extract_patches(label_idx, image, patch_size=7):
    """
    return:
        patches: tensor [N, 1, patch, patch]
    """
    rois = pts_to_bbox(label_idx[:,2:], patch_size).long()
    # filter out??
    rois = torch.cat((label_idx[:,:1], rois), dim=1)
    # print_var(rois)
    # print_var(image)
    patches = _roi_pool(image, rois, patch_size=patch_size)
    return patches

# from utils.losses import points_to_4d
def points_to_4d(points):
    """
    input: 
        points: tensor [N, 2] check(y, x)
    """
    num_of_points = points.shape[0]
    cols = torch.zeros(num_of_points, 1).float()
    points = torch.cat((cols, cols, points.float()), dim=1)
    return points

# from utils.losses import soft_argmax_2d
def soft_argmax_2d(patches, normalized_coordinates=True):
    """
    params:
        patches: (B, N, H, W)
    return:
        coor: (B, N, 2)  (x, y)

    """
    import torchgeometry as tgm
    m = tgm.contrib.SpatialSoftArgmax2d(normalized_coordinates=normalized_coordinates)
    coords = m(patches)  # 1x4x2
    return coords

## log on patches
# from utils.losses import do_log
def do_log(patches):
    patches[patches<0] = 1e-6
    patches_log = torch.log(patches)
    return patches_log

# from utils.losses import subpixel_loss
def subpixel_loss(labels_2D, labels_res, pred_heatmap, patch_size=7):
    """
    input:
        (tensor should be in GPU)
        labels_2D: tensor [batch, 1, H, W]
        labels_res: tensor [batch, 2, H, W]
        pred_heatmap: tensor [batch, 1, H, W]

    return:
        loss: sum of all losses
    """

    
    # soft argmax
    def _soft_argmax(patches):
        from models.SubpixelNet import SubpixelNet as subpixNet
        dxdy = subpixNet.soft_argmax_2d(patches) # tensor [B, N, patch, patch]
        dxdy = dxdy.squeeze(1) # tensor [N, 2]
        return dxdy
   
    points = labels_2D[...].nonzero()
    num_points = points.shape[0]
    if num_points == 0:
        return 0

    labels_res = labels_res.transpose(1,2).transpose(2,3).unsqueeze(1)
    rois = pts_to_bbox(points[:,2:], patch_size)
    # filter out??
    rois = torch.cat((points[:,:1], rois), dim=1)
    points_res = labels_res[points[:,0],points[:,1],points[:,2],points[:,3],:]  # tensor [N, 2]
    # print_var(rois)
    # print_var(labels_res)
    # print_var(points)
    # print("points max: ", points.max(dim=0))
    # print_var(labels_2D)
    # print_var(points_res)

    patches = _roi_pool(pred_heatmap, rois, patch_size=patch_size)
    # get argsoft max
    dxdy = _soft_argmax(patches)

    loss = (points_res - dxdy)
    loss = torch.norm(loss, p=2, dim=-1)
    loss = loss.sum()/num_points
    # print("loss: ", loss)
    return loss

def subpixel_loss_no_argmax(labels_2D, labels_res, pred_heatmap, **options):
    # extract points
    points = labels_2D[...].nonzero()
    num_points = points.shape[0]
    if num_points == 0:
        return 0

    def residual_from_points(labels_res, points):
        # extract residuals
        labels_res = labels_res.transpose(1,2).transpose(2,3).unsqueeze(1)
        points_res = labels_res[points[:,0],points[:,1],points[:,2],points[:,3],:]  # tensor [N, 2]
        return points_res

    points_res = residual_from_points(labels_res, points)
    # print_var(points_res)
    # extract predicted residuals
    pred_res = residual_from_points(pred_heatmap, points)
    # print_var(pred_res)

    # loss
    loss = (points_res - pred_res)
    loss = torch.norm(loss, p=2, dim=-1).mean()
    # loss = loss.sum()/num_points
    return loss
    pass

if __name__ == '__main__':

    ## example: 
    # device='cuda:0'
    # patches = subpixel_loss(warped_labels.to(device), labels_warped_res.to(device), warped_labels.to(device), 8, patch_size=11)

    pass