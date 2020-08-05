""" class to process superpoint net
# may be some duplication with model_wrap.py
"""

import torch
import numpy as np


class SuperPointNet_process(object):


    def __init__(self, **config):
        # N=500, patch_size=5, device='cuda:0'
        self.out_num_points = config.get('out_num_points', 500)
        self.patch_size = config.get('patch_size', 5)
        self.device = config.get('device', 'cuda:0')
        self.nms_dist = config.get('nms_dist', 4)
        self.conf_thresh = config.get('conf_thresh', 0.015)
        self.heatmap = None
        self.heatmap_nms_batch = None
        pass

    # @staticmethod
    def pred_soft_argmax(self, labels_2D, heatmap):
        """

        return:
            dict {'loss': mean of difference btw pred and res}
        """
        patch_size=self.patch_size
        device=self.device
        from utils.losses import norm_patches

        outs = {}
        # extract patches
        from utils.losses import extract_patches
        from utils.losses import soft_argmax_2d
        label_idx = labels_2D[...].nonzero()

        # patch_size = self.config['params']['patch_size']
        patches = extract_patches(label_idx.to(device), heatmap.to(device), 
            patch_size=patch_size)
        # norm patches
        # patches = norm_patches(patches)

        # predict offsets
        from utils.losses import do_log
        patches_log = do_log(patches)
        # soft_argmax
        dxdy = soft_argmax_2d(patches_log, normalized_coordinates=False) # tensor [B, N, patch, patch]
        dxdy = dxdy.squeeze(1) # tensor [N, 2]
        dxdy = dxdy-patch_size//2

        # loss
        outs['pred'] = dxdy
        # ls = lambda x, y: dxdy.cpu() - points_res.cpu()
        outs['patches'] = patches
        return outs

    # torch
    @staticmethod
    def sample_desc_from_points(coarse_desc, pts, cell_size=8):
        """
        inputs:
            coarse_desc: tensor [1, 256, Hc, Wc]
            pts: tensor [N, 2] (should be the same device as desc)
        return:
            desc: tensor [1, N, D]
        """
        # --- Process descriptor.
        samp_pts = pts.transpose(0,1)
        H, W = coarse_desc.shape[2]*cell_size, coarse_desc.shape[3]*cell_size
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            # desc = torch.zeros((D, 0))
            desc = torch.ones((1, 1, D))
        else:
            # Interpolate into descriptor map using 2D point locations.
            # samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            # samp_pts = samp_pts.to(self.device)
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True) # tensor [batch_size(1), D, 1, N]
            # desc = desc.data.cpu().numpy().reshape(D, -1)
            # desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
            desc = desc.squeeze().transpose(0,1).unsqueeze(0)
        return desc

    # extract residual
    @staticmethod
    def ext_from_points(labels_res, points):
        """
        input:
            labels_res: tensor [batch, channel, H, W]
            points: tensor [N, 4(pos0(batch), pos1(0), pos2(H), pos3(W) )]
        return:
            tensor [N, channel]
        """
        labels_res = labels_res.transpose(1,2).transpose(2,3).unsqueeze(1)
        points_res = labels_res[points[:,0],points[:,1],points[:,2],points[:,3],:]  # tensor [N, 2]
        return points_res

    # points_res = ext_from_points(labels_res, label_idx)

    @staticmethod
    def soft_argmax_2d(patches):
        """
        params:
          patches: (B, N, H, W)
        return:
          coor: (B, N, 2)  (x, y)

        """
        import torchgeometry as tgm
        m = tgm.contrib.SpatialSoftArgmax2d()
        coords = m(patches)  # 1x4x2
        return coords


    def heatmap_to_nms(self, heatmap, tensor=False, boxnms=False):
        """
        return: 
          heatmap_nms_batch: np [batch, 1, H, W]
        """
        to_floatTensor = lambda x: torch.from_numpy(x).type(torch.FloatTensor)
        from utils.var_dim import toNumpy
        heatmap_np = toNumpy(heatmap)
        ## heatmap_nms
        if boxnms:
            from utils.utils import box_nms
            heatmap_nms_batch = [box_nms(h.detach().squeeze(), self.nms_dist, min_prob=self.conf_thresh) \
                            for h in heatmap] # [batch, H, W]
            heatmap_nms_batch = torch.stack(heatmap_nms_batch, dim=0).unsqueeze(1)
            # print('heatmap_nms_batch: ', heatmap_nms_batch.shape)
        else:
            heatmap_nms_batch = [self.heatmap_nms(h, self.nms_dist, self.conf_thresh) \
                            for h in heatmap_np] # [batch, H, W]
            heatmap_nms_batch = np.stack(heatmap_nms_batch, axis=0)
            heatmap_nms_batch = heatmap_nms_batch[:,np.newaxis,...]
            if tensor:
                heatmap_nms_batch = to_floatTensor(heatmap_nms_batch)
                heatmap_nms_batch = heatmap_nms_batch.to(self.device)
        self.heatmap = heatmap
        self.heatmap_nms_batch = heatmap_nms_batch
        return heatmap_nms_batch
        pass


    @staticmethod
    def heatmap_nms(heatmap, nms_dist=4, conf_thresh=0.015):
        """
        input:
            heatmap: np [(1), H, W]
        """
        # nms_dist = self.config['model']['nms']
        # conf_thresh = self.config['model']['detection_threshold']
        heatmap = heatmap.squeeze()
        boxnms = False
        # print("heatmap: ", heatmap.shape)
        from utils.utils import getPtsFromHeatmap
        pts_nms = getPtsFromHeatmap(heatmap, conf_thresh, nms_dist)

        semi_thd_nms_sample = np.zeros_like(heatmap)
        semi_thd_nms_sample[pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)] = 1
        
        
        return semi_thd_nms_sample


    def batch_extract_features(self, desc, heatmap_nms_batch, residual):
        # extract pts, residuals for pts, descriptors
        """
        return: -- type: tensorFloat
          pts: tensor [batch, N, 2] (no grad)  (x, y)
          pts_offset: tensor [batch, N, 2] (grad) (x, y)
          pts_desc: tensor [batch, N, 256] (grad)
        """
        batch_size = heatmap_nms_batch.shape[0]
        
        pts_int, pts_offset, pts_desc = [], [], []
        pts_idx = heatmap_nms_batch[...].nonzero() # [N, 4(batch, 0, y, x)]
        for i in range(batch_size):
            mask_b = (pts_idx[:,0] == i) # first column == batch
            pts_int_b = pts_idx[mask_b][:,2:].float() # default floatTensor
            pts_int_b = pts_int_b[:, [1, 0]] # tensor [N, 2(x,y)]
            res_b = residual[mask_b]
            # print("res_b: ", res_b.shape)
            # print("pts_int_b: ", pts_int_b.shape)
            pts_b = pts_int_b + res_b # .no_grad()
            # extract desc
            pts_desc_b = self.sample_desc_from_points(desc[i].unsqueeze(0), pts_b).squeeze(0)
            # print("pts_desc_b: ", pts_desc_b.shape)
            # get random shuffle
            from utils.utils import crop_or_pad_choice
            choice = crop_or_pad_choice(pts_int_b.shape[0], out_num_points=self.out_num_points, shuffle=True)
            choice = torch.tensor(choice)
            pts_int.append(pts_int_b[choice])
            pts_offset.append(res_b[choice])
            pts_desc.append(pts_desc_b[choice])

        pts_int = torch.stack((pts_int), dim=0)
        pts_offset = torch.stack((pts_offset), dim=0)
        pts_desc = torch.stack((pts_desc), dim=0)
        return {'pts_int': pts_int, 'pts_offset': pts_offset, 'pts_desc': pts_desc}

