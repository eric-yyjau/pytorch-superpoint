import numpy as np
import tensorflow as tf
import torch

# from pathlib import Path
from pathlib import Path
import torch.utils.data as data
import random

# from .base_dataset import BaseDataset
from .utils import pipeline
from settings import DATA_PATH, EXPER_PATH, KITTI_DATA_PATH
from utils.tools import dict_update
import cv2
import logging
from numpy.linalg import inv

from utils.utils import homography_scaling_torch as homography_scaling
from utils.utils import filter_points


# from .utils import pipeline


class Kitti(data.Dataset):
    default_config = {
        "labels": None,
        "cache_in_memory": False,
        "validation_size": 100,
        "truncate": None,
        "preprocessing": {
            # 'resize': [375*0.5, 1242*0.5]
            "resize_ratio": 0.5
        },
        "num_parallel_calls": 10,
        "augmentation": {
            "photometric": {
                "enable": False,
                "primitives": "all",
                "params": {},
                "random_order": True,
            },
            "homographic": {"enable": False, "params": {}, "valid_border_margin": 0},
        },
        "warped_pair": {"enable": False, "params": {}, "valid_border_margin": 0},
        "homography_adaptation": {"enable": False},
    }
    from numpy.linalg import inv

    def __init__(
        self,
        export=False,
        transform=None,
        task="train",
        seed=0,
        sequence_length=1,
        **config
    ):

        # self.init_var()
        torch.set_default_tensor_type(torch.FloatTensor)
        np.random.seed(seed)
        random.seed(seed)
        # import functions
        from utils.homographies import sample_homography_np as sample_homography
        from utils.photometric import ImgAugTransform, customizedTransform
        from utils.utils import (
            inv_warp_image,
            inv_warp_image_batch,
            warp_points,
            compute_valid_mask,
        )

        self.sample_homography = sample_homography
        self.inv_warp_image = inv_warp_image
        self.inv_warp_image_batch = inv_warp_image_batch
        self.compute_valid_mask = compute_valid_mask
        self.ImgAugTransform = ImgAugTransform
        self.customizedTransform = customizedTransform
        self.warp_points = warp_points

        self.config = self.default_config
        self.config = dict_update(self.config, config)
        self.sequence_length = sequence_length
        self.transforms = transform
        self.labels = False
        self.cell_size = 8

        # self.root = Path(KITTI_DATA_PATH)
        self.root = Path(self.config["root"])
        scene_list_path = (
            self.root / "train.txt" if task == "train" else self.root / "val.txt"
        )
        self.scenes = [
            Path(self.root / folder[:-1]) for folder in open(scene_list_path)
        ]

        if self.config["preprocessing"]["resize_ratio"]:
            # self.sizer = [int(self.config['preprocessing']['resize_ratio'] * item) for item in [375, 1242]]
            self.sizer = np.array(self.config["preprocessing"]["resize"])
            logging.info("Resizing to [%d, %d]" % (self.sizer[0], self.sizer[1]))

        if self.config["labels"]:
            self.labels = True
            self.labels_path = Path(self.config["labels"], task)
            print("load labels from: ", self.config["labels"] + "/" + task)

        # if self.config['labels']:
        #     self.labels = True
        #     from models.model_wrap import labels2Dto3D
        #     self.labels2Dto3D = labels2Dto3D

        # sequence_set = []
        # if self.config['labels']:
        #     for (img, name) in zip(files['image_paths'], files['names']):
        #         p = Path(EXPER_PATH, self.config['labels'], task, '{}.npz'.format(name))
        #         if p.exists():
        #             sample = {'image': img, 'name': name, 'points': str(p)}
        #             sequence_set.append(sample)
        #     pass
        # else:
        #     for (img, name) in zip(files['image_paths'], files['names']):
        #         sample = {'image': img, 'name': name}
        #         sequence_set.append(sample)
        # self.samples = sequence_set

        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        # demi_length = (sequence_length-1)//2
        demi_length = sequence_length - 1
        # shifts = list(range(-demi_length, demi_length + 1))
        # shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = (
                np.genfromtxt(scene / "cam.txt").astype(np.float32).reshape((3, 3))
            )
            imu_pose_matrixs = (
                np.genfromtxt(scene / "imu_pose_matrixs.txt")
                .astype(np.float64)
                .reshape(-1, 4, 4)
            )
            # imgs = sorted(scene.files('*.jpg'))

            ##### test
            image_paths = list(scene.iterdir())
            names = [p.stem for p in image_paths]
            image_paths = [str(p) for p in image_paths]
            files = {"image_paths": image_paths, "names": names}
            #####

            imgs = sorted(scene.glob("*.jpg"))
            names = [p.stem for p in imgs]

            # X_files = sorted(scene.glob('*.npy'))
            if len(imgs) < sequence_length:
                continue
            # for i in range(demi_length, len(imgs)-demi_length):
            for i in range(0, len(imgs) - demi_length):
                sample = None
                # sample = {'intrinsics': intrinsics, 'imu_pose_matrixs': [imu_pose_matrixs[i]], 'imgs': [imgs[i]], 'Xs': [load_as_array(X_files[i])], 'scene_name': scene.name, 'frame_ids': [i]}
                if self.labels:
                    p = Path(
                        EXPER_PATH,
                        self.labels_path,
                        scene.name,
                        "{}.npz".format(names[i]),
                    )
                    if p.exists():
                        sample = {
                            "intrinsics": intrinsics,
                            "imu_pose_matrixs": [imu_pose_matrixs[i]],
                            "imgs": [imgs[i]],
                            "scene_name": scene.name,
                            "frame_ids": [i],
                        }
                        sample.update({"name": [names[i]], "points": [str(p)]})
                else:
                    sample = {
                        "intrinsics": intrinsics,
                        "imu_pose_matrixs": [imu_pose_matrixs[i]],
                        "imgs": [imgs[i]],
                        "scene_name": scene.name,
                        "frame_ids": [i],
                        "name": [names[i]],
                    }

                if sample is not None:
                    # for j in shifts:
                    for j in range(1, demi_length + 1):
                        sample["imgs"].append(imgs[i + j])
                        sample["imu_pose_matrixs"].append(imu_pose_matrixs[i + j])
                        # sample['Xs'].append(load_as_array(X_files[i])) # [3, N]
                        sample["frame_ids"].append(i + j)
                    sequence_set.append(sample)
                # print(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set
        logging.info("Finished crawl_folders for KITTI.")

    def __getitem__(self, index):
        def _read_image(path):
            cell = 8
            input_image = cv2.imread(str(path))
            # resize
            input_image = cv2.resize(
                input_image,
                (self.sizer[1], self.sizer[0]),
                interpolation=cv2.INTER_AREA,
            )

            """
            #  padding
            delta_w = desired_size - new_size[1]
            delta_h = desired_size - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)

            color = [0, 0, 0]
            new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=color)
            """
            #####

            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

            input_image = input_image.astype("float32") / 255.0
            return input_image

        def _preprocess(image):
            if self.transforms is not None:
                image = self.transforms(image)
            return image

        def getHomographies(homoAdapt_iter=1, params=None, add_identity=False):
            # homoAdapt_iter = self.config['homography_adaptation']['num']
            homographies = np.stack(
                [
                    self.sample_homography(np.array([2, 2]), shift=-1, **params)
                    for i in range(homoAdapt_iter)
                ]
            )
            ##### use inverse from the sample homography
            homographies = np.stack([inv(homography) for homography in homographies])
            if add_identity:
                homographies[0, :, :] = np.identity(3)
            ######

            homographies = torch.tensor(homographies, dtype=torch.float32)
            inv_homographies = torch.stack(
                [torch.inverse(homographies[i, :, :]) for i in range(homoAdapt_iter)]
            )
            # return {'homographies': homographies, 'inv_homographies': inv_homographies}
            return homographies, inv_homographies

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

        def getLabels(points_path, img_shape):
            pnts = np.load(points_path)["pts"]
            pnts = pnts.astype(int)
            labels = np.zeros((img_shape[0], img_shape[1]))
            labels[pnts[:, 1], pnts[:, 0]] = 1
            labels_2D = torch.tensor(labels[np.newaxis, :, :], dtype=torch.float32)
            return labels_2D, pnts

        def warpLabels(pnts, H, W):
            pnts = torch.tensor(pnts).long()
            warped_pnts = self.warp_points(
                torch.stack((pnts[:, 0], pnts[:, 1]), dim=1),
                homography_scaling(homography, H, W),
            )  # check the (x, y)
            warped_pnts = (
                filter_points(warped_pnts, torch.tensor([W, H])).round().long()
            )
            warped_labels = torch.zeros(H, W)
            warped_labels[warped_pnts[:, 1], warped_pnts[:, 0]] = 1
            warped_labels = warped_labels.view(-1, H, W)
            return warped_labels

        def get_img_from_sample(sample):
            return

        sample = self.samples[index]
        # imgs = [_preprocess(_read_image(img)[:, :, np.newaxis]) for img in sample['imgs']]
        imgs = [_read_image(img) for img in sample["imgs"]]
        assert len(imgs) == self.sequence_length

        intrinsics = np.copy(sample["intrinsics"])
        imu_pose_matrixs = sample["imu_pose_matrixs"]
        scene_name = sample["scene_name"]
        img_name = sample["name"][0]
        frame_ids = sample["frame_ids"]
        input = {}

        if self.sequence_length == 1:
            img_o = imgs[0]  # numpy array (H, W)
            H, W = img_o.shape[0], img_o.shape[1]
            # augmentation
            img_aug = img_o.copy()
            if self.config["augmentation"]["photometric"]["enable"] == True:
                img_aug = imgPhotometric(img_o)  # numpy array (H, W, 1)
                pass

            # img = _preprocess(img)
            img_aug = torch.tensor(img_aug, dtype=torch.float32)
            img_shape = img_aug.shape[:2]
            valid_mask = self.compute_valid_mask(
                torch.tensor([H, W]), inv_homography=torch.eye(3)
            )
            input.update({"valid_mask": valid_mask})

            if self.config["homography_adaptation"]["enable"]:
                homoAdapt_iter = self.config["homography_adaptation"]["num"]
                homographies, inv_homographies = getHomographies(
                    homoAdapt_iter,
                    params=self.config["homography_adaptation"]["homographies"][
                        "params"
                    ],
                )

                warped_img = self.inv_warp_image_batch(
                    img_aug.squeeze().repeat(homoAdapt_iter, 1, 1, 1),
                    inv_homographies,
                    mode="bilinear",
                ).unsqueeze(0)
                warped_img = warped_img.squeeze()
                # masks
                valid_mask = self.compute_valid_mask(
                    torch.tensor([H, W]),
                    inv_homography=inv_homographies,
                    erosion_radius=self.config["augmentation"]["homographic"][
                        "valid_border_margin"
                    ],
                )
                input.update(
                    {"image": warped_img, "valid_mask": valid_mask, "image_2D": img_aug}
                )
                input.update(
                    {"homographies": homographies, "inv_homographies": inv_homographies}
                )

            wapred_mode = "bilinear"  # 'bilinear'
            if self.labels:
                labels_2D, pnts = getLabels(sample["points"][0], img_shape)

                input.update({"labels_2D": labels_2D})
                # valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=torch.eye(3))
                # input.update({'valid_mask': valid_mask})

                if self.config["augmentation"]["homographic"]["enable"] == True:
                    homographies, inv_homographies = getHomographies(
                        homoAdapt_iter=1,
                        params=self.config["augmentation"]["homographic"]["params"],
                        add_identity=True,
                    )
                    homography, inv_homography = homographies[0], inv_homographies[0]

                    warped_img = self.inv_warp_image(
                        img_aug.squeeze(), inv_homography, mode=wapred_mode
                    ).unsqueeze(0)

                    warped_labels = warpLabels(pnts, H, W)

                    valid_mask = self.compute_valid_mask(
                        torch.tensor([H, W]),
                        inv_homography=inv_homography,
                        erosion_radius=self.config["augmentation"]["homographic"][
                            "valid_border_margin"
                        ],
                    )
                    input.update(
                        {
                            "image": warped_img,
                            "labels_2D": warped_labels,
                            "valid_mask": valid_mask,
                        }
                    )

                if self.config["warped_pair"]["enable"]:
                    homographies, inv_homographies = getHomographies(
                        homoAdapt_iter=1, params=self.config["warped_pair"]["params"]
                    )
                    homography, inv_homography = homographies[0], inv_homographies[0]
                    # print("homographies: ", homographies)
                    # print("params: ", self.config['warped_pair']['params'])
                    # images
                    warped_img = torch.tensor(img_o, dtype=torch.float32)
                    warped_img = self.inv_warp_image(
                        warped_img, inv_homography, mode=wapred_mode
                    ).unsqueeze(-1)

                    if self.config["augmentation"]["photometric"]["enable"] == True:
                        warped_img = imgPhotometric(
                            warped_img.numpy().squeeze()
                        )  # numpy array (H, W, 1)
                        warped_img = torch.tensor(warped_img, dtype=torch.float32)
                        pass

                    warped_img = warped_img.view(-1, H, W)

                    # print("warped: ", warped_img.shape)
                    # print("img: ", img_aug.shape)

                    warped_labels = warpLabels(pnts, H, W)

                    # warped_labels = self.inv_warp_image(labels_2D.squeeze(), inv_homography, mode='nearest').unsqueeze(0)

                    input.update(
                        {"warped_img": warped_img, "warped_labels": warped_labels}
                    )
                    valid_mask = self.compute_valid_mask(
                        torch.tensor([H, W]),
                        inv_homography=inv_homography,
                        erosion_radius=self.config["warped_pair"][
                            "valid_border_margin"
                        ],
                    )  # can set to other value
                    input.update({"warped_valid_mask": valid_mask})
                    input.update(
                        {"homographies": homography, "inv_homographies": inv_homography}
                    )

                    pass

            if "image" not in input:
                input.update({"image": img_aug.view(-1, H, W)})
            input.update(
                {"name": scene_name + "/" + img_name, "scene_name": scene_name}
            )
            return input
        else:
            raise ValueError(
                "sequence_length %d on KITTI has not been supported!"
                % self.sequence_length
            )

    def __len__(self):
        return len(self.samples)


def load_as_float(path):
    return np.array(imread(path)).astype(np.float32)


def load_as_array(path):
    return np.load(path)
