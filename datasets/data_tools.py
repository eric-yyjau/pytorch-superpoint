
import torch

quan = lambda x: x.round().long() 

def extrapolate_points(pnts):
    pnts_int = pnts.long().type(torch.FloatTensor)
    pnts_x, pnts_y = pnts_int[:,0], pnts_int[:,1]

    stack_1 = lambda x, y: torch.stack((x, y), dim=1)
    pnts_ext = torch.cat((pnts_int, stack_1(pnts_x, pnts_y+1),
        stack_1(pnts_x+1, pnts_y), pnts_int+1), dim=0)

    pnts_res = pnts - pnts_int # (x, y)
    x_res, y_res = pnts_res[:,0], pnts_res[:,1] # residuals
    res_ext = torch.cat(((1-x_res)*(1-y_res), (1-x_res)*y_res, 
            x_res*(1-y_res), x_res*y_res), dim=0)
    return pnts_ext, res_ext

def scatter_points(warped_pnts, H, W, res_ext = 1):
    warped_labels = torch.zeros(H, W)
    warped_labels[quan(warped_pnts)[:, 1], quan(warped_pnts)[:, 0]] = res_ext
    warped_labels = warped_labels.view(-1, H, W)
    return warped_labels

# from datasets.data_tools import get_labels_bi
def get_labels_bi(warped_pnts, H, W):
    from utils.utils import filter_points
    pnts_ext, res_ext = extrapolate_points(warped_pnts)
    # quan = lambda x: x.long()
    pnts_ext, mask = filter_points(pnts_ext, torch.tensor([W, H]), return_mask=True)
    res_ext = res_ext[mask]
    warped_labels_bi = scatter_points(pnts_ext, H, W, res_ext = res_ext)
    return warped_labels_bi

# from data_tools import warpLabels
def warpLabels(pnts, H, W, homography, bilinear = False):
    from utils.utils import homography_scaling_torch as homography_scaling
    from utils.utils import filter_points
    from utils.utils import warp_points
    if isinstance(pnts, torch.Tensor):
        pnts = pnts.long() 
    else:
        pnts = torch.tensor(pnts).long()
    warped_pnts = warp_points(torch.stack((pnts[:, 0], pnts[:, 1]), dim=1),
                                   homography_scaling(homography, H, W)) # check the (x, y)
    outs = {}
    # warped_pnts 
    # print("extrapolate_points!!")

    # ext_points = True
    if bilinear == True:
        warped_labels_bi = get_labels_bi(warped_pnts, H, W)
        outs['labels_bi'] = warped_labels_bi

    warped_pnts = filter_points(warped_pnts, torch.tensor([W, H]))
    warped_labels = scatter_points(warped_pnts, H, W, res_ext = 1)
    
    warped_labels_res = torch.zeros(H, W, 2)
    warped_labels_res[quan(warped_pnts)[:, 1], quan(warped_pnts)[:, 0], :] = warped_pnts - warped_pnts.round()
    # print("res sum: ", (warped_pnts - warped_pnts.round()).sum())
    outs.update({'labels': warped_labels, 'res': warped_labels_res, 'warped_pnts': warped_pnts})
    return outs


# from data_tools import np_to_tensor
def np_to_tensor(img, H, W):
    img = torch.tensor(img).type(torch.FloatTensor).view(-1, H, W)
    return img


if __name__ == '__main__':
    main()
