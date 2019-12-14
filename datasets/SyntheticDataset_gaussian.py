"""
Adapted from https://github.com/rpautrat/SuperPoint/blob/master/superpoint/datasets/synthetic_dataset.py

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

import torch.utils.data as data
import torch
import numpy as np
from imageio import imread

# from os import path as Path
import tensorflow as tf
from pathlib import Path
import tarfile

# import os
import random
import logging

from utils.tools import dict_update

from datasets import synthetic_dataset

# from models.homographies import sample_homography

from tqdm import tqdm
import cv2
import shutil
from settings import DEBUG as debug
from settings import DATA_PATH
from settings import SYN_TMPDIR

# DATA_PATH = '.'
import multiprocessing

TMPDIR = SYN_TMPDIR  # './datasets/' # you can define your tmp dir


def load_as_float(path):
    return imread(path).astype(np.float32) / 255


class SyntheticDataset_gaussian(data.Dataset):
    """
    """

    default_config = {
        "primitives": "all",
        "truncate": {},
        "validation_size": -1,
        "test_size": -1,
        "on-the-fly": False,
        "cache_in_memory": False,
        "suffix": None,
        "add_augmentation_to_test_set": False,
        "num_parallel_calls": 10,
        "generation": {
            "split_sizes": {"training": 10000, "validation": 200, "test": 500},
            "image_size": [960, 1280],
            "random_seed": 0,
            "params": {
                "generate_background": {
                    "min_kernel_size": 150,
                    "max_kernel_size": 500,
                    "min_rad_ratio": 0.02,
                    "max_rad_ratio": 0.031,
                },
                "draw_stripes": {"transform_params": (0.1, 0.1)},
                "draw_multiple_polygons": {"kernel_boundaries": (50, 100)},
            },
        },
        "preprocessing": {"resize": [240, 320], "blur_size": 11,},
        "augmentation": {
            "photometric": {
                "enable": False,
                "primitives": "all",
                "params": {},
                "random_order": True,
            },
            "homographic": {"enable": False, "params": {}, "valid_border_margin": 0,},
        },
    }

    # debug = True

    if debug == True:
        drawing_primitives = [
            "draw_checkerboard",
        ]
    else:
        drawing_primitives = [
            "draw_lines",
            "draw_polygon",
            "draw_multiple_polygons",
            "draw_ellipses",
            "draw_star",
            "draw_checkerboard",
            "draw_stripes",
            "draw_cube",
            "gaussian_noise",
        ]
    print(drawing_primitives)

    """
    def dump_primitive_data(self, primitive, tar_path, config):
        pass
    """

    def dump_primitive_data(self, primitive, tar_path, config):
        # temp_dir = Path(os.environ['TMPDIR'], primitive)
        temp_dir = Path(TMPDIR, primitive)

        tf.logging.info("Generating tarfile for primitive {}.".format(primitive))
        synthetic_dataset.set_random_state(
            np.random.RandomState(config["generation"]["random_seed"])
        )
        for split, size in self.config["generation"]["split_sizes"].items():
            im_dir, pts_dir = [Path(temp_dir, i, split) for i in ["images", "points"]]
            im_dir.mkdir(parents=True, exist_ok=True)
            pts_dir.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(size), desc=split, leave=False):
                image = synthetic_dataset.generate_background(
                    config["generation"]["image_size"],
                    **config["generation"]["params"]["generate_background"],
                )
                points = np.array(
                    getattr(synthetic_dataset, primitive)(
                        image, **config["generation"]["params"].get(primitive, {})
                    )
                )
                points = np.flip(points, 1)  # reverse convention with opencv

                b = config["preprocessing"]["blur_size"]
                image = cv2.GaussianBlur(image, (b, b), 0)
                points = (
                    points
                    * np.array(config["preprocessing"]["resize"], np.float)
                    / np.array(config["generation"]["image_size"], np.float)
                )
                image = cv2.resize(
                    image,
                    tuple(config["preprocessing"]["resize"][::-1]),
                    interpolation=cv2.INTER_LINEAR,
                )

                cv2.imwrite(str(Path(im_dir, "{}.png".format(i))), image)
                np.save(Path(pts_dir, "{}.npy".format(i)), points)

        # Pack into a tar file
        tar = tarfile.open(tar_path, mode="w:gz")
        tar.add(temp_dir, arcname=primitive)
        tar.close()
        shutil.rmtree(temp_dir)
        tf.logging.info("Tarfile dumped to {}.".format(tar_path))

    def parse_primitives(self, names, all_primitives):
        p = (
            all_primitives
            if (names == "all")
            else (names if isinstance(names, list) else [names])
        )
        assert set(p) <= set(all_primitives)
        return p

    def __init__(
        self,
        seed=None,
        task="train",
        sequence_length=3,
        transform=None,
        target_transform=None,
        getPts=False,
        warp_input=False,
        **config,
    ):
        from utils.homographies import sample_homography_np as sample_homography
        from utils.photometric import ImgAugTransform, customizedTransform
        from utils.utils import compute_valid_mask
        from utils.utils import inv_warp_image, warp_points

        torch.set_default_tensor_type(torch.FloatTensor)
        np.random.seed(seed)
        random.seed(seed)

        # Update config
        self.config = self.default_config
        self.config = dict_update(self.config, dict(config))

        self.transform = transform
        self.sample_homography = sample_homography
        self.compute_valid_mask = compute_valid_mask
        self.inv_warp_image = inv_warp_image
        self.warp_points = warp_points
        self.ImgAugTransform = ImgAugTransform
        self.customizedTransform = customizedTransform

        ######
        self.enable_photo_train = self.config["augmentation"]["photometric"]["enable"]
        self.enable_homo_train = self.config["augmentation"]["homographic"]["enable"]
        self.enable_homo_val = False
        self.enable_photo_val = False
        ######

        self.action = "training" if task == "train" else "validation"
        # self.warp_input = warp_input

        self.cell_size = 8
        self.getPts = getPts

        self.gaussian_label = False
        if self.config["gaussian_label"]["enable"]:
            # self.params_transform = {'crop_size_y': 120, 'crop_size_x': 160, 'stride': 1, 'sigma': self.config['gaussian_label']['sigma']}
            self.gaussian_label = True

        self.pool = multiprocessing.Pool(6)

        # Parse drawing primitives
        primitives = self.parse_primitives(
            config["primitives"], self.drawing_primitives
        )

        basepath = Path(
            DATA_PATH,
            "synthetic_shapes"
            + ("_{}".format(config["suffix"]) if config["suffix"] is not None else ""),
        )
        basepath.mkdir(parents=True, exist_ok=True)

        splits = {s: {"images": [], "points": []} for s in [self.action]}
        for primitive in primitives:
            tar_path = Path(basepath, "{}.tar.gz".format(primitive))
            if not tar_path.exists():
                self.dump_primitive_data(primitive, tar_path, self.config)

            # Untar locally
            logging.info("Extracting archive for primitive {}.".format(primitive))
            logging.info(f"tar_path: {tar_path}")
            tar = tarfile.open(tar_path)
            # temp_dir = Path(os.environ['TMPDIR'])
            temp_dir = Path(TMPDIR)
            tar.extractall(path=temp_dir)
            tar.close()

            # Gather filenames in all splits, optionally truncate
            truncate = self.config["truncate"].get(primitive, 1)
            path = Path(temp_dir, primitive)
            for s in splits:
                e = [str(p) for p in Path(path, "images", s).iterdir()]
                f = [p.replace("images", "points") for p in e]
                f = [p.replace(".png", ".npy") for p in f]
                splits[s]["images"].extend(e[: int(truncate * len(e))])
                splits[s]["points"].extend(f[: int(truncate * len(f))])

        # Shuffle
        for s in splits:
            perm = np.random.RandomState(0).permutation(len(splits[s]["images"]))
            for obj in ["images", "points"]:
                splits[s][obj] = np.array(splits[s][obj])[perm].tolist()

        self.crawl_folders(splits)

    def crawl_folders(self, splits):
        sequence_set = []
        for (img, pnts) in zip(
            splits[self.action]["images"], splits[self.action]["points"]
        ):
            sample = {"image": img, "points": pnts}
            sequence_set.append(sample)
        self.samples = sequence_set

    # def putGaussianMaps_par(self, center):
    #     crop_size_y = self.params_transform['crop_size_y']
    #     crop_size_x = self.params_transform['crop_size_x']
    #     stride = self.params_transform['stride']
    #     sigma = self.params_transform['sigma']

    #     grid_y = crop_size_y / stride
    #     grid_x = crop_size_x / stride
    #     start = stride / 2.0 - 0.5
    #     xx, yy = np.meshgrid(range(int(grid_x)), range(int(grid_y)))
    #     xx = xx * stride + start
    #     yy = yy * stride + start
    #     d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    #     exponent = d2 / 2.0 / sigma / sigma
    #     mask = exponent <= sigma
    #     cofid_map = np.exp(-exponent)
    #     cofid_map = np.multiply(mask, cofid_map)
    #     return cofid_map

    def putGaussianMaps(self, center, accumulate_confid_map):
        crop_size_y = self.params_transform["crop_size_y"]
        crop_size_x = self.params_transform["crop_size_x"]
        stride = self.params_transform["stride"]
        sigma = self.params_transform["sigma"]

        grid_y = crop_size_y / stride
        grid_x = crop_size_x / stride
        start = stride / 2.0 - 0.5
        xx, yy = np.meshgrid(range(int(grid_x)), range(int(grid_y)))
        xx = xx * stride + start
        yy = yy * stride + start
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        mask = exponent <= sigma
        cofid_map = np.exp(-exponent)
        cofid_map = np.multiply(mask, cofid_map)
        accumulate_confid_map += cofid_map
        accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0
        return accumulate_confid_map

    def __getitem__(self, index):
        """
        :param index:
        :return:
            labels_2D: tensor(1, H, W)
            image: tensor(1, H, W)
        """

        def checkSat(img, name=""):
            if img.max() > 1:
                print(name, img.max())
            elif img.min() < 0:
                print(name, img.min())

        def imgPhotometric(img):
            """

            :param img:
                numpy (H, W)
            :return:
            """
            augmentation = self.ImgAugTransform(**self.config["augmentation"])
            img = img[:, :, np.newaxis]
            img = augmentation(img)
            cusAug = self.customizedTransform()
            img = cusAug(img, **self.config["augmentation"])
            return img

        def get_labels(pnts, H, W):
            labels = torch.zeros(H, W)
            # print('--2', pnts, pnts.size())
            # pnts_int = torch.min(pnts.round().long(), torch.tensor([[H-1, W-1]]).long())
            pnts_int = torch.min(
                pnts.round().long(), torch.tensor([[W - 1, H - 1]]).long()
            )
            # print('--3', pnts_int, pnts_int.size())
            labels[pnts_int[:, 1], pnts_int[:, 0]] = 1
            return labels

        def get_label_res(H, W, pnts):
            quan = lambda x: x.round().long()
            labels_res = torch.zeros(H, W, 2)
            # pnts_int = torch.min(pnts.round().long(), torch.tensor([[H-1, W-1]]).long())

            labels_res[quan(pnts)[:, 1], quan(pnts)[:, 0], :] = pnts - pnts.round()
            # print("pnts max: ", quan(pnts).max(dim=0))
            # print("labels_res: ", labels_res.shape)
            labels_res = labels_res.transpose(1, 2).transpose(0, 1)
            return labels_res

        from datasets.data_tools import np_to_tensor
        from utils.utils import filter_points
        from utils.var_dim import squeezeToNumpy

        sample = self.samples[index]
        img = load_as_float(sample["image"])
        H, W = img.shape[0], img.shape[1]
        self.H = H
        self.W = W
        pnts = np.load(sample["points"])  # (y, x)
        pnts = torch.tensor(pnts).float()
        pnts = torch.stack((pnts[:, 1], pnts[:, 0]), dim=1)  # (x, y)
        pnts = filter_points(pnts, torch.tensor([W, H]))
        sample = {}

        # print('pnts: ', pnts[:5])
        # print('--1', pnts)
        labels_2D = get_labels(pnts, H, W)
        sample.update({"labels_2D": labels_2D.unsqueeze(0)})

        # assert Hc == round(Hc) and Wc == round(Wc), "Input image size not fit in the block size"
        if (
            self.config["augmentation"]["photometric"]["enable_train"]
            and self.action == "training"
        ) or (
            self.config["augmentation"]["photometric"]["enable_val"]
            and self.action == "validation"
        ):
            # print('>>> Photometric aug enabled for %s.'%self.action)
            # augmentation = self.ImgAugTransform(**self.config["augmentation"])
            img = imgPhotometric(img)
        else:
            # print('>>> Photometric aug disabled for %s.'%self.action)
            pass

        if not (
            (
                self.config["augmentation"]["homographic"]["enable_train"]
                and self.action == "training"
            )
            or (
                self.config["augmentation"]["homographic"]["enable_val"]
                and self.action == "validation"
            )
        ):
            # print('<<< Homograpy aug disabled for %s.'%self.action)
            img = img[:, :, np.newaxis]
            # labels = labels.view(-1,H,W)
            if self.transform is not None:
                img = self.transform(img)
            sample["image"] = img
            # sample = {'image': img, 'labels_2D': labels}
            valid_mask = self.compute_valid_mask(
                torch.tensor([H, W]), inv_homography=torch.eye(3)
            )
            sample.update({"valid_mask": valid_mask})
            labels_res = get_label_res(H, W, pnts)
            pnts_post = pnts
            # pnts_for_gaussian = pnts
        else:
            # print('>>> Homograpy aug enabled for %s.'%self.action)
            # img_warp = img
            from utils.utils import homography_scaling_torch as homography_scaling
            from numpy.linalg import inv

            homography = self.sample_homography(
                np.array([2, 2]),
                shift=-1,
                **self.config["augmentation"]["homographic"]["params"],
            )

            ##### use inverse from the sample homography
            homography = inv(homography)
            ######

            homography = torch.tensor(homography).float()
            inv_homography = homography.inverse()
            img = torch.from_numpy(img)
            warped_img = self.inv_warp_image(
                img.squeeze(), inv_homography, mode="bilinear"
            )
            warped_img = warped_img.squeeze().numpy()
            warped_img = warped_img[:, :, np.newaxis]

            # labels = torch.from_numpy(labels)
            # warped_labels = self.inv_warp_image(labels.squeeze(), inv_homography, mode='nearest').unsqueeze(0)
            warped_pnts = self.warp_points(pnts, homography_scaling(homography, H, W))
            warped_pnts = filter_points(warped_pnts, torch.tensor([W, H]))
            # pnts = warped_pnts[:, [1, 0]]
            # pnts_for_gaussian = warped_pnts
            # warped_labels = torch.zeros(H, W)
            # warped_labels[warped_pnts[:, 1], warped_pnts[:, 0]] = 1
            # warped_labels = warped_labels.view(-1, H, W)

            if self.transform is not None:
                warped_img = self.transform(warped_img)
            # sample = {'image': warped_img, 'labels_2D': warped_labels}
            sample["image"] = warped_img

            valid_mask = self.compute_valid_mask(
                torch.tensor([H, W]),
                inv_homography=inv_homography,
                erosion_radius=self.config["augmentation"]["homographic"][
                    "valid_border_margin"
                ],
            )  # can set to other value
            sample.update({"valid_mask": valid_mask})

            labels_2D = get_labels(warped_pnts, H, W)
            sample.update({"labels_2D": labels_2D.unsqueeze(0)})

            labels_res = get_label_res(H, W, warped_pnts)
            pnts_post = warped_pnts

        if self.gaussian_label:
            # warped_labels_gaussian = get_labels_gaussian(pnts)
            from datasets.data_tools import get_labels_bi

            labels_2D_bi = get_labels_bi(pnts_post, H, W)

            labels_gaussian = self.gaussian_blur(squeezeToNumpy(labels_2D_bi))
            labels_gaussian = np_to_tensor(labels_gaussian, H, W)
            sample["labels_2D_gaussian"] = labels_gaussian

            # add residua

        sample.update({"labels_res": labels_res})

        ### code for warped image
        if self.config["warped_pair"]["enable"]:
            from datasets.data_tools import warpLabels

            homography = self.sample_homography(
                np.array([2, 2]), shift=-1, **self.config["warped_pair"]["params"]
            )

            ##### use inverse from the sample homography
            homography = np.linalg.inv(homography)
            #####
            inv_homography = np.linalg.inv(homography)

            homography = torch.tensor(homography).type(torch.FloatTensor)
            inv_homography = torch.tensor(inv_homography).type(torch.FloatTensor)

            # photometric augmentation from original image

            # warp original image
            warped_img = img.type(torch.FloatTensor)
            warped_img = self.inv_warp_image(
                warped_img.squeeze(), inv_homography, mode="bilinear"
            ).unsqueeze(0)
            if (self.enable_photo_train == True and self.action == "train") or (
                self.enable_photo_val and self.action == "val"
            ):
                warped_img = imgPhotometric(
                    warped_img.numpy().squeeze()
                )  # numpy array (H, W, 1)
                warped_img = torch.tensor(warped_img, dtype=torch.float32)
                pass
            warped_img = warped_img.view(-1, H, W)

            # warped_labels = warpLabels(pnts, H, W, homography)
            warped_set = warpLabels(pnts, H, W, homography, bilinear=True)
            warped_labels = warped_set["labels"]
            warped_res = warped_set["res"]
            warped_res = warped_res.transpose(1, 2).transpose(0, 1)
            # print("warped_res: ", warped_res.shape)
            if self.gaussian_label:
                # print("do gaussian labels!")
                # warped_labels_gaussian = get_labels_gaussian(warped_set['warped_pnts'].numpy())
                # warped_labels_bi = self.inv_warp_image(labels_2D.squeeze(), inv_homography, mode='nearest').unsqueeze(0) # bilinear, nearest
                warped_labels_bi = warped_set["labels_bi"]
                warped_labels_gaussian = self.gaussian_blur(
                    squeezeToNumpy(warped_labels_bi)
                )
                warped_labels_gaussian = np_to_tensor(warped_labels_gaussian, H, W)
                sample["warped_labels_gaussian"] = warped_labels_gaussian
                sample.update({"warped_labels_bi": warped_labels_bi})

            sample.update(
                {
                    "warped_img": warped_img,
                    "warped_labels": warped_labels,
                    "warped_res": warped_res,
                }
            )

            # print('erosion_radius', self.config['warped_pair']['valid_border_margin'])
            valid_mask = self.compute_valid_mask(
                torch.tensor([H, W]),
                inv_homography=inv_homography,
                erosion_radius=self.config["warped_pair"]["valid_border_margin"],
            )  # can set to other value
            sample.update({"warped_valid_mask": valid_mask})
            sample.update(
                {"homographies": homography, "inv_homographies": inv_homography}
            )

        # labels = self.labels2Dto3D(self.cell_size, labels)
        # labels = torch.from_numpy(labels[np.newaxis,:,:])
        # input.update({'labels': labels})

        ### code for warped image

        # if self.config['gaussian_label']['enable']:
        #     heatmaps = np.zeros((H, W))
        #     # for center in pnts_int.numpy():
        #     for center in pnts[:, [1, 0]].numpy():
        #         # print("put points: ", center)
        #         heatmaps = self.putGaussianMaps(center, heatmaps)
        #     # import matplotlib.pyplot as plt
        #     # plt.figure(figsize=(5, 10))
        #     # plt.subplot(211)
        #     # plt.imshow(heatmaps)
        #     # plt.colorbar()
        #     # plt.subplot(212)
        #     # plt.imshow(np.squeeze(warped_labels.numpy()))
        #     # plt.show()
        #     # import time
        #     # time.sleep(500)
        #     # results = self.pool.map(self.putGaussianMaps_par, warped_pnts.numpy())

        #     warped_labels_gaussian = torch.from_numpy(heatmaps).view(-1, H, W)
        #     warped_labels_gaussian[warped_labels_gaussian>1.] = 1.

        #     sample['labels_2D_gaussian'] = warped_labels_gaussian

        if self.getPts:
            sample.update({"pts": pnts})

        return sample

    def __len__(self):
        return len(self.samples)

    ## util functions
    def gaussian_blur(self, image):
        """
        image: np [H, W]
        return:
            blurred_image: np [H, W]
        """
        aug_par = {"photometric": {}}
        aug_par["photometric"]["enable"] = True
        aug_par["photometric"]["params"] = self.config["gaussian_label"]["params"]
        augmentation = self.ImgAugTransform(**aug_par)
        # get label_2D
        # labels = points_to_2D(pnts, H, W)
        image = image[:, :, np.newaxis]
        heatmaps = augmentation(image)
        return heatmaps.squeeze()
