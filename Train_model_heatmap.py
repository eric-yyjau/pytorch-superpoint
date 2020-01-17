"""This is the main training interface using heatmap trick

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

import numpy as np
import torch
# from torch.autograd import Variable
# import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
# from tqdm import tqdm
# from utils.loader import dataLoader, modelLoader, pretrainedLoader
import logging

from utils.tools import dict_update

# from utils.utils import labels2Dto3D, flattenDetection, labels2Dto3D_flattened
# from utils.utils import pltImshow, saveImg
from utils.utils import precisionRecall_torch
# from utils.utils import save_checkpoint

from pathlib import Path
from Train_model_frontend import Train_model_frontend


def thd_img(img, thd=0.015):
    img[img < thd] = 0
    img[img >= thd] = 1
    return img


def toNumpy(tensor):
    return tensor.detach().cpu().numpy()


def img_overlap(img_r, img_g, img_gray):  # img_b repeat
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img


class Train_model_heatmap(Train_model_frontend):
    """ Wrapper around pytorch net to help with pre and post image processing. """

    """
    * SuperPointFrontend_torch:
    ** note: the input, output is different from that of SuperPointFrontend
    heatmap: torch (batch_size, H, W, 1)
    dense_desc: torch (batch_size, H, W, 256)
    pts: [batch_size, np (N, 3)]
    desc: [batch_size, np(256, N)]
    """
    default_config = {
        "train_iter": 170000,
        "save_interval": 2000,
        "tensorboard_interval": 200,
        "model": {"subpixel": {"enable": False}},
        "data": {"gaussian_label": {"enable": False}},
    }

    def __init__(self, config, save_path=Path("."), device="cpu", verbose=False):
        # config
        # Update config
        print("Load Train_model_heatmap!!")

        self.config = self.default_config
        self.config = dict_update(self.config, config)
        print("check config!!", self.config)

        # init parameters
        self.device = device
        self.save_path = save_path
        self._train = True
        self._eval = True
        self.cell_size = 8
        self.subpixel = False

        self.max_iter = config["train_iter"]

        self.gaussian = False
        if self.config["data"]["gaussian_label"]["enable"]:
            self.gaussian = True

        if self.config["model"]["dense_loss"]["enable"]:
            print("use dense_loss!")
            from utils.utils import descriptor_loss
            self.desc_params = self.config["model"]["dense_loss"]["params"]
            self.descriptor_loss = descriptor_loss
            self.desc_loss_type = "dense"
        elif self.config["model"]["sparse_loss"]["enable"]:
            print("use sparse_loss!")
            self.desc_params = self.config["model"]["sparse_loss"]["params"]
            from utils.loss_functions.sparse_loss import batch_descriptor_loss_sparse

            self.descriptor_loss = batch_descriptor_loss_sparse
            self.desc_loss_type = "sparse"

        # load model
        # self.net = self.loadModel(*config['model'])
        self.printImportantConfig()
        pass

    ### loadModel inherited from Train_model_frontend
    # def loadModel(self):
    #     """
    #     load model from name and params
    #     init or load optimizer
    #     :return:
    #     """
    #     model = self.config["model"]["name"]
    #     params = self.config["model"]["params"]
    #     print("model: ", model)
    #     net = modelLoader(model=model, **params).to(self.device)
    #     logging.info("=> setting adam solver")
    #     optimizer = self.adamOptim(net, lr=self.config["model"]["learning_rate"])
    #
    #     n_iter = 0
    #     ## new model or load pretrained
    #     if self.config["retrain"] == True:
    #         logging.info("New model")
    #         pass
    #     else:
    #         path = self.config["pretrained"]
    #         mode = "" if path[:-3] == ".pth" else "full"
    #         logging.info("load pretrained model from: %s", path)
    #         net, optimizer, n_iter = pretrainedLoader(
    #             net, optimizer, n_iter, path, mode=mode, full_path=True
    #         )
    #         logging.info("successfully load pretrained model from: %s", path)
    #
    #     def setIter(n_iter):
    #         if self.config["reset_iter"]:
    #             logging.info("reset iterations to 0")
    #             n_iter = 0
    #         return n_iter
    #
    #     self.net = net
    #     self.optimizer = optimizer
    #     self.n_iter = setIter(n_iter)
    #     pass

    def detector_loss(self, input, target, mask=None, loss_type="softmax"):
        """
        # apply loss on detectors, default is softmax
        :param input: prediction
            tensor [batch_size, 65, Hc, Wc]
        :param target: constructed from labels
            tensor [batch_size, 65, Hc, Wc]
        :param mask: valid region in an image
            tensor [batch_size, 1, Hc, Wc]
        :param loss_type:
            str (l2 or softmax)
            softmax is used in original paper
        :return: normalized loss
            tensor
        """
        if loss_type == "l2":
            loss_func = nn.MSELoss(reduction="mean")
            loss = loss_func(input, target)
        elif loss_type == "softmax":
            loss_func_BCE = nn.BCELoss(reduction='none').cuda()
            loss = loss_func_BCE(nn.functional.softmax(input, dim=1), target)
            loss = (loss.sum(dim=1) * mask).sum()
            loss = loss / (mask.sum() + 1e-10)
        return loss

    def train_val_sample(self, sample, n_iter=0, train=False):
        """
        # key function
        :param sample:
        :param n_iter:
        :param train:
        :return:
        """
        to_floatTensor = lambda x: torch.tensor(x).type(torch.FloatTensor)

        task = "train" if train else "val"
        tb_interval = self.config["tensorboard_interval"]
        if_warp = self.config['data']['warped_pair']['enable']

        self.scalar_dict, self.images_dict, self.hist_dict = {}, {}, {}
        ## get the inputs
        # logging.info('get input img and label')
        img, labels_2D, mask_2D = (
            sample["image"],
            sample["labels_2D"],
            sample["valid_mask"],
        )
        # img, labels = img.to(self.device), labels_2D.to(self.device)

        # variables
        batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
        self.batch_size = batch_size
        det_loss_type = self.config["model"]["detector_loss"]["loss_type"]
        # print("batch_size: ", batch_size)
        Hc = H // self.cell_size
        Wc = W // self.cell_size

        # warped images
        # img_warp, labels_warp_2D, mask_warp_2D = sample['warped_img'].to(self.device), \
        #     sample['warped_labels'].to(self.device), \
        #     sample['warped_valid_mask'].to(self.device)
        if if_warp:
            img_warp, labels_warp_2D, mask_warp_2D = (
                sample["warped_img"],
                sample["warped_labels"],
                sample["warped_valid_mask"],
            )

        # homographies
        # mat_H, mat_H_inv = \
        # sample['homographies'].to(self.device), sample['inv_homographies'].to(self.device)
        if if_warp:
            mat_H, mat_H_inv = sample["homographies"], sample["inv_homographies"]

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        if train:
            # print("img: ", img.shape, ", img_warp: ", img_warp.shape)
            outs = self.net(img.to(self.device))
            semi, coarse_desc = outs["semi"], outs["desc"]
            if if_warp:
                outs_warp = self.net(img_warp.to(self.device))
                semi_warp, coarse_desc_warp = outs_warp["semi"], outs_warp["desc"]
        else:
            with torch.no_grad():
                outs = self.net(img.to(self.device))
                semi, coarse_desc = outs["semi"], outs["desc"]
                if if_warp:
                    outs_warp = self.net(img_warp.to(self.device))
                    semi_warp, coarse_desc_warp = outs_warp["semi"], outs_warp["desc"]
                pass

        # detector loss
        from utils.utils import labels2Dto3D

        if self.gaussian:
            labels_2D = sample["labels_2D_gaussian"]
            if if_warp:
                warped_labels = sample["warped_labels_gaussian"]
        else:
            labels_2D = sample["labels_2D"]
            if if_warp:
                warped_labels = sample["warped_labels"]

        add_dustbin = False
        if det_loss_type == "l2":
            add_dustbin = False
        elif det_loss_type == "softmax":
            add_dustbin = True

        labels_3D = labels2Dto3D(
            labels_2D.to(self.device), cell_size=self.cell_size, add_dustbin=add_dustbin
        ).float()
        mask_3D_flattened = self.getMasks(mask_2D, self.cell_size, device=self.device)
        loss_det = self.detector_loss(
            input=outs["semi"],
            target=labels_3D.to(self.device),
            mask=mask_3D_flattened,
            loss_type=det_loss_type,
        )
        # warp
        if if_warp:
            labels_3D = labels2Dto3D(
                warped_labels.to(self.device),
                cell_size=self.cell_size,
                add_dustbin=add_dustbin,
            ).float()
            mask_3D_flattened = self.getMasks(
                mask_warp_2D, self.cell_size, device=self.device
            )
            loss_det_warp = self.detector_loss(
                input=outs_warp["semi"],
                target=labels_3D.to(self.device),
                mask=mask_3D_flattened,
                loss_type=det_loss_type,
            )
        else:
            loss_det_warp = torch.tensor([0]).float().to(self.device)


        ## get labels, masks, loss for detection
        # labels3D_in_loss = self.getLabels(labels_2D, self.cell_size, device=self.device)
        # mask_3D_flattened = self.getMasks(mask_2D, self.cell_size, device=self.device)
        # loss_det = self.get_loss(semi, labels3D_in_loss, mask_3D_flattened, device=self.device)

        ## warping
        # labels3D_in_loss = self.getLabels(labels_warp_2D, self.cell_size, device=self.device)
        # mask_3D_flattened = self.getMasks(mask_warp_2D, self.cell_size, device=self.device)
        # loss_det_warp = self.get_loss(semi_warp, labels3D_in_loss, mask_3D_flattened, device=self.device)

        mask_desc = mask_3D_flattened.unsqueeze(1)
        lambda_loss = self.config["model"]["lambda_loss"]
        # print("mask_desc: ", mask_desc.shape)
        # print("mask_warp_2D: ", mask_warp_2D.shape)

        # descriptor loss
        if lambda_loss > 0:
            assert if_warp == True, "need a pair of images"
            loss_desc, mask, positive_dist, negative_dist = self.descriptor_loss(
                coarse_desc,
                coarse_desc_warp,
                mat_H,
                mask_valid=mask_desc,
                device=self.device,
                **self.desc_params
            )
        else:
            ze = torch.tensor([0]).to(self.device)
            loss_desc, positive_dist, negative_dist = ze, ze, ze

        loss = loss_det + loss_det_warp
        if lambda_loss > 0:
            loss += lambda_loss * loss_desc

        ##### try to minimize the error ######
        add_res_loss = False
        if add_res_loss and n_iter % 10 == 0:
            print("add_res_loss!!!")
            heatmap_org = self.get_heatmap(semi, det_loss_type)  # tensor []
            heatmap_org_nms_batch = self.heatmap_to_nms(
                self.images_dict, heatmap_org, name="heatmap_org"
            )
            if if_warp:
                heatmap_warp = self.get_heatmap(semi_warp, det_loss_type)
                heatmap_warp_nms_batch = self.heatmap_to_nms(
                    self.images_dict, heatmap_warp, name="heatmap_warp"
                )

            # original: pred
            ## check the loss on given labels!
            outs_res = self.get_residual_loss(
                sample["labels_2D"]
                * to_floatTensor(heatmap_org_nms_batch).unsqueeze(1),
                heatmap_org,
                sample["labels_res"],
                name="original_pred",
            )
            loss_res_ori = (outs_res["loss"] ** 2).mean()
            # warped: pred
            if if_warp:
                outs_res_warp = self.get_residual_loss(
                    sample["warped_labels"]
                    * to_floatTensor(heatmap_warp_nms_batch).unsqueeze(1),
                    heatmap_warp,
                    sample["warped_res"],
                    name="warped_pred",
                )
                loss_res_warp = (outs_res_warp["loss"] ** 2).mean()
            else:
                loss_res_warp = torch.tensor([0]).to(self.device)
            loss_res = loss_res_ori + loss_res_warp
            # print("loss_res requires_grad: ", loss_res.requires_grad)
            loss += loss_res
            self.scalar_dict.update(
                {"loss_res_ori": loss_res_ori, "loss_res_warp": loss_res_warp}
            )

        #######################################

        self.loss = loss

        self.scalar_dict.update(
            {
                "loss": loss,
                "loss_det": loss_det,
                "loss_det_warp": loss_det_warp,
                "positive_dist": positive_dist,
                "negative_dist": negative_dist,
            }
        )

        self.input_to_imgDict(sample, self.images_dict)

        if train:
            loss.backward()
            self.optimizer.step()

        if n_iter % tb_interval == 0 or task == "val":
            logging.info(
                "current iteration: %d, tensorboard_interval: %d", n_iter, tb_interval
            )

            # add clean map to tensorboard
            ## semi_warp: flatten, to_numpy

            heatmap_org = self.get_heatmap(semi, det_loss_type)  # tensor []
            heatmap_org_nms_batch = self.heatmap_to_nms(
                self.images_dict, heatmap_org, name="heatmap_org"
            )
            if if_warp:
                heatmap_warp = self.get_heatmap(semi_warp, det_loss_type)
                heatmap_warp_nms_batch = self.heatmap_to_nms(
                    self.images_dict, heatmap_warp, name="heatmap_warp"
                )


            def update_overlap(
                images_dict, labels_warp_2D, heatmap_nms_batch, img_warp, name
            ):
                # image overlap
                from utils.draw import img_overlap

                # result_overlap = img_overlap(img_r, img_g, img_gray)
                # overlap label, nms, img
                nms_overlap = [
                    img_overlap(
                        toNumpy(labels_warp_2D[i]),
                        heatmap_nms_batch[i],
                        toNumpy(img_warp[i]),
                    )
                    for i in range(heatmap_nms_batch.shape[0])
                ]
                nms_overlap = np.stack(nms_overlap, axis=0)
                images_dict.update({name + "_nms_overlap": nms_overlap})

            from utils.var_dim import toNumpy
            update_overlap(
                self.images_dict,
                labels_2D,
                heatmap_org_nms_batch[np.newaxis, ...],
                img,
                "original",
            )

            update_overlap(
                self.images_dict,
                labels_2D,
                toNumpy(heatmap_org),
                img,
                "original_heatmap",
            )
            if if_warp:
                update_overlap(
                    self.images_dict,
                    labels_warp_2D,
                    heatmap_warp_nms_batch[np.newaxis, ...],
                    img_warp,
                    "warped",
                )
                update_overlap(
                    self.images_dict,
                    labels_warp_2D,
                    toNumpy(heatmap_warp),
                    img_warp,
                    "warped_heatmap",
                )
            # residuals
            from utils.losses import do_log

            if self.gaussian:
                # original: gt
                self.get_residual_loss(
                    sample["labels_2D"],
                    sample["labels_2D_gaussian"],
                    sample["labels_res"],
                    name="original_gt",
                )
                if if_warp:
                    # warped: gt
                    self.get_residual_loss(
                        sample["warped_labels"],
                        sample["warped_labels_gaussian"],
                        sample["warped_res"],
                        name="warped_gt",
                    )

            # from utils.losses import do_log
            # patches_log = do_log(patches)

            # original: pred
            ## check the loss on given labels!
            # self.get_residual_loss(
            #     sample["labels_2D"]
            #     * to_floatTensor(heatmap_org_nms_batch).unsqueeze(1),
            #     heatmap_org,
            #     sample["labels_res"],
            #     name="original_pred",
            # )
            # print("heatmap_org_nms_batch: ", heatmap_org_nms_batch.shape)
            # get_residual_loss(to_floatTensor(heatmap_org_nms_batch).unsqueeze(1), heatmap_org,
            # sample['labels_res'], name='original_pred')
            # warped: pred
            # self.get_residual_loss(
            #     sample["warped_labels"]
            #     * to_floatTensor(heatmap_warp_nms_batch).unsqueeze(1),
            #     heatmap_warp,
            #     sample["warped_res"],
            #     name="warped_pred",
            # )
            # get_residual_loss(to_floatTensor(heatmap_warp_nms_batch).unsqueeze(1), heatmap_warp,
            # sample['warped_res'], name='warped_pred')

            # precision, recall
            # pr_mean = self.batch_precision_recall(
            #     to_floatTensor(heatmap_warp_nms_batch[:, np.newaxis, ...]),
            #     sample["warped_labels"],
            # )
            pr_mean = self.batch_precision_recall(
                to_floatTensor(heatmap_org_nms_batch[:, np.newaxis, ...]),
                sample["labels_2D"],
            )
            print("pr_mean")
            self.scalar_dict.update(pr_mean)

            self.printLosses(self.scalar_dict, task)
            self.tb_images_dict(task, self.images_dict, max_img=2)
            self.tb_hist_dict(task, self.hist_dict)

        self.tb_scalar_dict(self.scalar_dict, task)

        return loss.item()

    def heatmap_to_nms(self, images_dict, heatmap, name):
        """
        return: 
            heatmap_nms_batch: np [batch, H, W]
        """
        from utils.var_dim import toNumpy

        heatmap_np = toNumpy(heatmap)
        ## heatmap_nms
        heatmap_nms_batch = [self.heatmap_nms(h) for h in heatmap_np]  # [batch, H, W]
        heatmap_nms_batch = np.stack(heatmap_nms_batch, axis=0)
        # images_dict.update({name + '_nms_batch': heatmap_nms_batch})
        images_dict.update({name + "_nms_batch": heatmap_nms_batch[:, np.newaxis, ...]})
        return heatmap_nms_batch

    def get_residual_loss(self, labels_2D, heatmap, labels_res, name=""):
        if abs(labels_2D).sum() == 0:
            return
        outs_res = self.pred_soft_argmax(
            labels_2D, heatmap, labels_res, patch_size=5, device=self.device
        )
        self.hist_dict[name + "_resi_loss_x"] = outs_res["loss"][:, 0]
        self.hist_dict[name + "_resi_loss_y"] = outs_res["loss"][:, 1]
        err = abs(outs_res["loss"]).mean(dim=0)
        # print("err[0]: ", err[0])
        var = abs(outs_res["loss"]).std(dim=0)
        self.scalar_dict[name + "_resi_loss_x"] = err[0]
        self.scalar_dict[name + "_resi_loss_y"] = err[1]
        self.scalar_dict[name + "_resi_var_x"] = var[0]
        self.scalar_dict[name + "_resi_var_y"] = var[1]
        self.images_dict[name + "_patches"] = outs_res["patches"]
        return outs_res

    # tb_images_dict.update({'image': sample['image'], 'valid_mask': sample['valid_mask'],
    #     'labels_2D': sample['labels_2D'], 'warped_img': sample['warped_img'],
    #     'warped_valid_mask': sample['warped_valid_mask']})
    # if self.gaussian:
    #     tb_images_dict.update({'labels_2D_gaussian': sample['labels_2D_gaussian'],
    #     'labels_2D_gaussian': sample['labels_2D_gaussian']})

    ######## static methods ########
    @staticmethod
    def batch_precision_recall(batch_pred, batch_labels):
        precision_recall_list = []
        for i in range(batch_labels.shape[0]):
            precision_recall = precisionRecall_torch(batch_pred[i], batch_labels[i])
            precision_recall_list.append(precision_recall)
        precision = np.mean(
            [
                precision_recall["precision"]
                for precision_recall in precision_recall_list
            ]
        )
        recall = np.mean(
            [precision_recall["recall"] for precision_recall in precision_recall_list]
        )
        return {"precision": precision, "recall": recall}

    @staticmethod
    def pred_soft_argmax(labels_2D, heatmap, labels_res, patch_size=5, device="cuda"):
        """

        return:
            dict {'loss': mean of difference btw pred and res}
        """
        from utils.losses import norm_patches

        outs = {}
        # extract patches
        from utils.losses import extract_patches
        from utils.losses import soft_argmax_2d

        label_idx = labels_2D[...].nonzero().long()

        # patch_size = self.config['params']['patch_size']
        patches = extract_patches(
            label_idx.to(device), heatmap.to(device), patch_size=patch_size
        )
        # norm patches
        patches = norm_patches(patches)

        # predict offsets
        from utils.losses import do_log

        patches_log = do_log(patches)
        # soft_argmax
        dxdy = soft_argmax_2d(
            patches_log, normalized_coordinates=False
        )  # tensor [B, N, patch, patch]
        dxdy = dxdy.squeeze(1)  # tensor [N, 2]
        dxdy = dxdy - patch_size // 2

        # extract residual
        def ext_from_points(labels_res, points):
            """
            input:
                labels_res: tensor [batch, channel, H, W]
                points: tensor [N, 4(pos0(batch), pos1(0), pos2(H), pos3(W) )]
            return:
                tensor [N, channel]
            """
            labels_res = labels_res.transpose(1, 2).transpose(2, 3).unsqueeze(1)
            points_res = labels_res[
                points[:, 0], points[:, 1], points[:, 2], points[:, 3], :
            ]  # tensor [N, 2]
            return points_res

        points_res = ext_from_points(labels_res, label_idx)

        # loss
        outs["pred"] = dxdy
        outs["points_res"] = points_res
        # ls = lambda x, y: dxdy.cpu() - points_res.cpu()
        # outs['loss'] = dxdy.cpu() - points_res.cpu()
        outs["loss"] = dxdy.to(device) - points_res.to(device)
        outs["patches"] = patches
        return outs

    @staticmethod
    def flatten_64to1(semi, cell_size=8):
        """
        input: 
            semi: tensor[batch, cell_size*cell_size, Hc, Wc]
            (Hc = H/8)
        outpus:
            heatmap: tensor[batch, 1, H, W]
        """
        from utils.d2s import DepthToSpace

        depth2space = DepthToSpace(cell_size)
        heatmap = depth2space(semi)
        return heatmap

    @staticmethod
    def heatmap_nms(heatmap, nms_dist=4, conf_thresh=0.015):
        """
        input:
            heatmap: np [(1), H, W]
        """
        from utils.utils import getPtsFromHeatmap

        # nms_dist = self.config['model']['nms']
        # conf_thresh = self.config['model']['detection_threshold']
        heatmap = heatmap.squeeze()
        # print("heatmap: ", heatmap.shape)
        pts_nms = getPtsFromHeatmap(heatmap, conf_thresh, nms_dist)
        semi_thd_nms_sample = np.zeros_like(heatmap)
        semi_thd_nms_sample[
            pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)
        ] = 1
        return semi_thd_nms_sample


if __name__ == "__main__":
    # load config
    filename = "configs/superpoint_coco_train_heatmap.yaml"
    import yaml

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_default_tensor_type(torch.FloatTensor)
    with open(filename, "r") as f:
        config = yaml.load(f)

    from utils.loader import dataLoader as dataLoader

    # data = dataLoader(config, dataset='hpatches')
    task = config["data"]["dataset"]

    data = dataLoader(config, dataset=task, warp_input=True)
    # test_set, test_loader = data['test_set'], data['test_loader']
    train_loader, val_loader = data["train_loader"], data["val_loader"]

    # model_fe = Train_model_frontend(config)
    # print('==> Successfully loaded pre-trained network.')

    train_agent = Train_model_heatmap(config, device=device)

    train_agent.train_loader = train_loader
    # train_agent.val_loader = val_loader

    train_agent.loadModel()
    train_agent.dataParallel()
    train_agent.train()

    # epoch += 1
    try:
        model_fe.train()

    # catch exception
    except KeyboardInterrupt:
        logging.info("ctrl + c is pressed. save model")
    # is_best = True
