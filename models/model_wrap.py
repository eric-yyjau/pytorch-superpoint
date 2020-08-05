""" class to process superpoint net
# may be some duplication with model_wrap.py
# PointTracker is from Daniel's repo.
"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm


def labels2Dto3D(cell_size, labels):
    H, W = labels.shape[0], labels.shape[1]
    Hc, Wc = H // cell_size, W // cell_size
    labels = labels[:, np.newaxis, :, np.newaxis]
    labels = labels.reshape(Hc, cell_size, Wc, cell_size)
    labels = np.transpose(labels, [1, 3, 0, 2])
    labels = labels.reshape(1, cell_size ** 2, Hc, Wc)
    labels = labels.squeeze()
    dustbin = labels.sum(axis=0)
    dustbin = 1 - dustbin
    dustbin[dustbin < 0] = 0
    labels = np.concatenate((labels, dustbin[np.newaxis, :, :]), axis=0)
    return labels

def toNumpy(tensor):
    return tensor.detach().cpu().numpy()



class SuperPointFrontend_torch(object):
    """ Wrapper around pytorch net to help with pre and post image processing. """
    '''
    * SuperPointFrontend_torch:
    ** note: the input, output is different from that of SuperPointFrontend
    heatmap: torch (batch_size, H, W, 1)
    dense_desc: torch (batch_size, H, W, 256)
    pts: [batch_size, np (N, 3)]
    desc: [batch_size, np(256, N)]
    '''

    def __init__(self, config, weights_path, nms_dist, conf_thresh, nn_thresh,
                 cuda=False, trained=False, device='cpu',grad=False, load=True):
        self.config = config

        self.name = 'SuperPoint'
        self.cuda = cuda
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh  # L2 descriptor distance for good match.
        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.
        self.sparsemap = None
        self.heatmap = None
        self.pts = None
        self.pts_subpixel = None
        self.patches = None

        self.device=device
        self.subpixel = False
        if self.config['model']['subpixel']['enable']:
            self.subpixel = True

        if load:
            self.loadModel(weights_path)

    def loadModel(self, weights_path):
        # Load the network in inference mode.
        if weights_path[-4:] == '.tar':
            trained = True
        # if cuda:
        #     # Train on GPU, deploy on GPU.
        #     self.net.load_state_dict(torch.load(weights_path))

        # else:
            # Train on GPU, deploy on CPU.

            # trained = False
        if trained:
            # if self.subpixel:
            #     model = 'SubpixelNet'
            #     params = self.config['model']['subpixel']['params']
            # else:
            #     model = 'SuperPointNet'
            #     params = {}
            model = self.config['model']['name']
            params = self.config['model']['params']
            print("model: ", model)

            from utils.loader import modelLoader
            self.net = modelLoader(model=model, **params)
            # from models.SuperPointNet import SuperPointNet
            # self.net = SuperPointNet()
            checkpoint = torch.load(weights_path,
                                    map_location=lambda storage, loc: storage)
            self.net.load_state_dict(checkpoint['model_state_dict'])
        else:
            from models.SuperPointNet_pretrained import SuperPointNet
            self.net = SuperPointNet()
            self.net.load_state_dict(torch.load(weights_path,
                                                map_location=lambda storage, loc: storage))
        # if grad==False:
            # torch.no_grad(
        # self.net = self.net.cuda()
        self.net = self.net.to(self.device)
        # self.net.eval()

    def net_parallel(self):
        print("=== Let's use", torch.cuda.device_count(), "GPUs!")
        self.net = nn.DataParallel(self.net)

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def getSparsemap(self):
        return self.sparsemap

    @property
    def points(self):
        print("get pts")
        return self.pts

    @property
    def heatmap(self):
        # print("get heatmap")
        return self._heatmap

    @heatmap.setter
    def heatmap(self, heatmap):
        # print("set heatmap")
        self._heatmap = heatmap

    def soft_argmax_points(self, pts, patch_size=5):
        """
        input:
            pts: tensor [N x 2]
        """
        from utils.utils import toNumpy
        from utils.losses import extract_patch_from_points
        from utils.losses import soft_argmax_2d
        from utils.losses import norm_patches

        ##### check not take care of batch #####
        # print("not take care of batch! only take first element!")
        pts = pts[0].transpose().copy()
        patches = extract_patch_from_points(self.heatmap, pts, patch_size=patch_size)
        import torch
        patches = np.stack(patches)
        patches_torch = torch.tensor(patches, dtype=torch.float32).unsqueeze(0)

        # norm patches
        patches_torch = norm_patches(patches_torch)

        from utils.losses import do_log
        patches_torch = do_log(patches_torch)
        # patches_torch = do_log(patches_torch)
        # print("one tims of log!")
        # print("patches: ", patches_torch.shape)
        # print("pts: ", pts.shape)

        dxdy = soft_argmax_2d(patches_torch, normalized_coordinates=False)
        # print("dxdy: ", dxdy.shape)
        points = pts
        points[:,:2] = points[:,:2] + dxdy.numpy().squeeze() - patch_size//2
        self.patches = patches_torch.numpy().squeeze()
        self.pts_subpixel = [points.transpose().copy()]
        return self.pts_subpixel.copy()

    # @staticmethod
    def get_image_patches(self, pts, image, patch_size=5):
        """
        input:
            image: np [H, W]
        return:
            patches: np [N, patch, patch]

        """
        from utils.losses import extract_patch_from_points
        pts = pts[0].transpose().copy()
        patches = extract_patch_from_points(image, pts, patch_size=patch_size)
        patches = np.stack(patches)
        return patches


    def getPtsFromHeatmap(self, heatmap):
        '''
        :param self:
        :param heatmap:
            np (H, W)
        :return:
        '''
        heatmap = heatmap.squeeze()
        # print("heatmap sq:", heatmap.shape)
        H, W = heatmap.shape[0], heatmap.shape[1]
        xs, ys = np.where(heatmap >= self.conf_thresh)  # Confidence threshold.
        self.sparsemap = (heatmap >= self.conf_thresh)
        if len(xs) == 0:
            return np.zeros((3, 0))
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys # abuse of ys, xs
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]  # check the (x, y) here
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        return pts

    def sample_desc_from_points(self, coarse_desc, pts):
        # --- Process descriptor.
        H, W = coarse_desc.shape[2]*self.cell, coarse_desc.shape[3]*self.cell
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            samp_pts = samp_pts.to(self.device)
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return desc


    def subpixel_predict(self, pred_res, points, verbose=False):
        """
        input:
            labels_res: numpy [2, H, W]
            points: [3, N]
        return:
            subpixels: [3, N]
        """
        D = points.shape[0]
        if points.shape[1] == 0:
            pts_subpixel = np.zeros((D, 0))
        else:
            points_res = pred_res[:,points[1,:].astype(int), points[0,:].astype(int)]
            pts_subpixel = points.copy()
            if verbose: print("before: ", pts_subpixel[:,:5])
            pts_subpixel[:2,:] += points_res
            if verbose: print("after: ", pts_subpixel[:,:5])
        return pts_subpixel
        pass

    def run(self, inp, onlyHeatmap=False, train=True):
        """ Process a numpy image to extract points and descriptors.
        Input
          img - HxW tensor float32 input image in range [0,1].
        Output
          corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          desc - 256xN numpy array of corresponding unit normalized descriptors.
          heatmap - HxW numpy heatmap in range [0,1] of point confidences.
          """
        # assert img.ndim == 2, 'Image must be grayscale.'
        # assert img.dtype == np.float32, 'Image must be float32.'
        # H, W = img.shape[0], img.shape[1]
        # inp = img.copy()
        # inp = (inp.reshape(1, H, W))
        # inp = torch.from_numpy(inp)
        # inp = torch.autograd.Variable(inp).view(1, 1, H, W)
        # if self.cuda:
        inp = inp.to(self.device)
        batch_size, H, W = inp.shape[0], inp.shape[2], inp.shape[3]
        if train:
            # outs = self.net.forward(inp, subpixel=self.subpixel)
            outs = self.net.forward(inp)
            # semi, coarse_desc = outs[0], outs[1]
            semi, coarse_desc = outs['semi'], outs['desc']
        else:
            # Forward pass of network.
            with torch.no_grad():
                # outs = self.net.forward(inp, subpixel=self.subpixel)
                outs = self.net.forward(inp)
                # semi, coarse_desc = outs[0], outs[1]
                semi, coarse_desc = outs['semi'], outs['desc']

        # as tensor
        from utils.utils import labels2Dto3D, flattenDetection
        from utils.d2s import DepthToSpace
        # flatten detection
        heatmap = flattenDetection(semi, tensor=True)
        self.heatmap = heatmap
        # depth2space = DepthToSpace(8)
        # print(semi.shape)
        # heatmap = depth2space(semi[:,:-1,:,:]).squeeze(0)
        ## need to change for batches

        if onlyHeatmap:
            return heatmap

        # extract keypoints
        # pts = [self.getPtsFromHeatmap(heatmap[i,:,:,:].cpu().detach().numpy().squeeze()).transpose() for i in range(batch_size)]
        # pts = [self.getPtsFromHeatmap(heatmap[i,:,:,:].cpu().detach().numpy().squeeze()) for i in range(batch_size)]
        # print("heapmap shape: ", heatmap.shape)
        pts = [self.getPtsFromHeatmap(heatmap[i,:,:,:].cpu().detach().numpy()) for i in range(batch_size)]
        self.pts = pts
        


        if self.subpixel:
            labels_res = outs[2]
            self.pts_subpixel = [self.subpixel_predict(toNumpy(labels_res[i, ...]), pts[i]) for i in range(batch_size)]
        '''
        pts:
            list [batch_size, np(N_i, 3)] -- each point (x, y, probability)
        '''

        # interpolate description
        '''
        coarse_desc:
            tensor (Batch_size, 256, Hc, Wc)
        dense_desc:
            tensor (batch_size, 256, H, W)
        '''
        # m = nn.Upsample(scale_factor=(1, self.cell, self.cell), mode='bilinear')
        dense_desc = nn.functional.interpolate(coarse_desc, scale_factor=(self.cell, self.cell), mode='bilinear')
        # norm the descriptor
        def norm_desc(desc):
            dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
            desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
            return desc
        dense_desc = norm_desc(dense_desc)

        # extract descriptors
        dense_desc_cpu = dense_desc.cpu().detach().numpy()
        # pts_desc = [dense_desc_cpu[i, :, pts[i][:, 1].astype(int), pts[i][:, 0].astype(int)] for i in range(len(pts))]
        pts_desc = [dense_desc_cpu[i, :, pts[i][1,:].astype(int), pts[i][0, :].astype(int)].transpose() for i in range(len(pts))]

        if self.subpixel:
            return self.pts_subpixel, pts_desc, dense_desc, heatmap
        return pts, pts_desc, dense_desc, heatmap



class PointTracker(object):
    """ Class to manage a fixed memory of points and descriptors that enables
    sparse optical flow point tracking.

    Internally, the tracker stores a 'tracks' matrix sized M x (2+L), of M
    tracks with maximum length L, where each row corresponds to:
    row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m].
    """

    def __init__(self, max_length=2, nn_thresh=0.7):
        if max_length < 2:
            raise ValueError('max_length must be greater than or equal to 2.')
        self.maxl = max_length
        self.nn_thresh = nn_thresh
        self.all_pts = []
        for n in range(self.maxl):
            self.all_pts.append(np.zeros((2, 0)))
        self.last_desc = None
        self.tracks = np.zeros((0, self.maxl + 2))
        self.track_count = 0
        self.max_score = 9999
        self.matches = None
        self.last_pts = None
        self.mscores = None

    def nn_match_two_way(self, desc1, desc2, nn_thresh):
        """
        Performs two-way nearest neighbor matching of two sets of descriptors, such
        that the NN match from descriptor A->B must equal the NN match from B->A.

        Inputs:
          desc1 - MxN numpy matrix of N corresponding M-dimensional descriptors.
          desc2 - MxN numpy matrix of N corresponding M-dimensional descriptors.
          nn_thresh - Optional descriptor distance below which is a good match.

        Returns:
          matches - 3xL numpy array, of L matches, where L <= N and each column i is
                    a match of two descriptors, d_i in image 1 and d_j' in image 2:
                    [d_i index, d_j' index, match_score]^T
        """
        assert desc1.shape[0] == desc2.shape[0]
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.zeros((3, 0))
        if nn_thresh < 0.0:
            raise ValueError('\'nn_thresh\' should be non-negative')
        # Compute L2 distance. Easy since vectors are unit normalized.
        dmat = np.dot(desc1.T, desc2)
        dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
        # Get NN indices and scores.
        idx = np.argmin(dmat, axis=1)
        scores = dmat[np.arange(dmat.shape[0]), idx]
        # Threshold the NN matches.
        keep = scores < nn_thresh
        # Check if nearest neighbor goes both directions and keep those.
        idx2 = np.argmin(dmat, axis=0)
        keep_bi = np.arange(len(idx)) == idx2[idx]
        keep = np.logical_and(keep, keep_bi)
        idx = idx[keep]
        scores = scores[keep]
        # Get the surviving point indices.
        m_idx1 = np.arange(desc1.shape[1])[keep]
        m_idx2 = idx
        # Populate the final 3xN match data structure.
        matches = np.zeros((3, int(keep.sum())))
        matches[0, :] = m_idx1
        matches[1, :] = m_idx2
        matches[2, :] = scores
        self.mscores = matches
        return matches

    def get_offsets(self):
        """ Iterate through list of points and accumulate an offset value. Used to
        index the global point IDs into the list of points.

        Returns
          offsets - N length array with integer offset locations.
        """
        # Compute id offsets.
        offsets = []
        offsets.append(0)
        for i in range(len(self.all_pts) - 1):  # Skip last camera size, not needed.
            offsets.append(self.all_pts[i].shape[1])
        offsets = np.array(offsets)
        offsets = np.cumsum(offsets)
        return offsets

    def get_matches(self):
        return self.matches

    def get_mscores(self):
        return self.mscores

    def clear_desc(self):
        self.last_desc = None

    def update(self, pts, desc):
        """ Add a new set of point and descriptor observations to the tracker.

        Inputs
          pts - 3xN numpy array of 2D point observations.
          desc - DxN numpy array of corresponding D dimensional descriptors.
        """
        if pts is None or desc is None:
            print('PointTracker: Warning, no points were added to tracker.')
            return
        # pts = pts.transpose()
        # desc = desc.transpose()
        assert pts.shape[1] == desc.shape[1]
        # assert pts.shape[0] == desc.shape[0]
        # Initialize last_desc.
        if self.last_desc is None:
            self.last_desc = np.zeros((desc.shape[0], 0))
        # Remove oldest points, store its size to update ids later.
        remove_size = self.all_pts[0].shape[1]
        self.all_pts.pop(0)
        self.all_pts.append(pts)
        # Remove oldest point in track.
        self.tracks = np.delete(self.tracks, 2, axis=1)
        # Update track offsets.
        for i in range(2, self.tracks.shape[1]):
            self.tracks[:, i] -= remove_size
        self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
        offsets = self.get_offsets()
        # Add a new -1 column.
        self.tracks = np.hstack((self.tracks, -1 * np.ones((self.tracks.shape[0], 1))))
        # Try to append to existing tracks.
        matched = np.zeros((pts.shape[1])).astype(bool)
        matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
        self.matches = matches
        pts_id = pts[:2, :]
        if self.last_pts is not None:
            id1, id2 = self.last_pts[:, matches[0, :].astype(int)], pts_id[:, matches[1, :].astype(int)]

            self.matches = np.concatenate((id1, id2), axis=0)
        for match in matches.T:
            # Add a new point to it's matched track.
            id1 = int(match[0]) + offsets[-2]
            id2 = int(match[1]) + offsets[-1]
            found = np.argwhere(self.tracks[:, -2] == id1)
            if found.shape[0] > 0:
                matched[int(match[1])] = True
                row = int(found)
                self.tracks[row, -1] = id2
                if self.tracks[row, 1] == self.max_score:
                    # Initialize track score.
                    self.tracks[row, 1] = match[2]
                else:
                    # Update track score with running average.
                    # NOTE(dd): this running average can contain scores from old matches
                    #           not contained in last max_length track points.
                    track_len = (self.tracks[row, 2:] != -1).sum() - 1.
                    frac = 1. / float(track_len)
                    self.tracks[row, 1] = (1. - frac) * self.tracks[row, 1] + frac * match[2]
        # Add unmatched tracks.
        new_ids = np.arange(pts.shape[1]) + offsets[-1]
        new_ids = new_ids[~matched]
        new_tracks = -1 * np.ones((new_ids.shape[0], self.maxl + 2))
        new_tracks[:, -1] = new_ids
        new_num = new_ids.shape[0]
        new_trackids = self.track_count + np.arange(new_num)
        new_tracks[:, 0] = new_trackids
        new_tracks[:, 1] = self.max_score * np.ones(new_ids.shape[0])
        self.tracks = np.vstack((self.tracks, new_tracks))
        self.track_count += new_num  # Update the track count.
        # Remove empty tracks.
        keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
        self.tracks = self.tracks[keep_rows, :]
        # Store the last descriptors.
        self.last_desc = desc.copy()
        self.last_pts = pts[:2, :].copy()

        return

    def get_tracks(self, min_length):
        """ Retrieve point tracks of a given minimum length.
        Input
          min_length - integer >= 1 with minimum track length
        Output
          returned_tracks - M x (2+L) sized matrix storing track indices, where
            M is the number of tracks and L is the maximum track length.
        """
        if min_length < 1:
            raise ValueError('\'min_length\' too small.')
        valid = np.ones((self.tracks.shape[0])).astype(bool)
        good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
        # Remove tracks which do not have an observation in most recent frame.
        not_headless = (self.tracks[:, -1] != -1)
        keepers = np.logical_and.reduce((valid, good_len, not_headless))
        returned_tracks = self.tracks[keepers, :].copy()
        return returned_tracks

    def draw_tracks(self, out, tracks):
        """ Visualize tracks all overlayed on a single image.
        Inputs
          out - numpy uint8 image sized HxWx3 upon which tracks are overlayed.
          tracks - M x (2+L) sized matrix storing track info.
        """
        # Store the number of points per camera.
        pts_mem = self.all_pts
        N = len(pts_mem)  # Number of cameras/images.
        # Get offset ids needed to reference into pts_mem.
        offsets = self.get_offsets()
        # Width of track and point circles to be drawn.
        stroke = 1
        # Iterate through each track and draw it.
        for track in tracks:
            clr = myjet[int(np.clip(np.floor(track[1] * 10), 0, 9)), :] * 255
            for i in range(N - 1):
                if track[i + 2] == -1 or track[i + 3] == -1:
                    continue
                offset1 = offsets[i]
                offset2 = offsets[i + 1]
                idx1 = int(track[i + 2] - offset1)
                idx2 = int(track[i + 3] - offset2)
                pt1 = pts_mem[i][:2, idx1]
                pt2 = pts_mem[i + 1][:2, idx2]
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
                # Draw end points of each track.
                if i == N - 2:
                    clr2 = (255, 0, 0)
                    cv2.circle(out, p2, stroke, clr2, -1, lineType=16)


