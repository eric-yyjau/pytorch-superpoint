"""dataloader for kitti raw dataset
inh --> (inherited from Coco)

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

import numpy as np
import torch

# from pathlib import Path
from pathlib import Path
import torch.utils.data as data
import random

# from .base_dataset import BaseDataset
from settings import DATA_PATH, EXPER_PATH
from utils.tools import dict_update
import cv2
import logging
from numpy.linalg import inv

from utils.utils import homography_scaling_torch as homography_scaling
from utils.utils import filter_points

from datasets.Coco import Coco



class Kitti_inh(Coco):

    ## copy the same thing from Coco
    default_config = {
        "labels": None,
        "cache_in_memory": False,
        "validation_size": 100,
        "truncate": None,
        "preprocessing": {"resize": [240, 320]},
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

    def __init__(
        self,
        export=False,
        transform=None,
        task="train",
        seed=0,
        sequence_length=1,
        **config,
    ):
        # Update config
        self.config = self.default_config
        self.config = dict_update(self.config, config)

        self.transforms = transform
        self.action = "train" if task == "train" else "val"

        # get files
        self.root = Path(self.config["root"])  # Path(KITTI_DATA_PATH)

        root_split_txt = self.config.get("root_split_txt", None)
        self.root_split_txt = Path(
            self.root if root_split_txt is None else root_split_txt
        )
        scene_list_path = (
            self.root_split_txt / "train.txt"
            if task == "train"
            else self.root_split_txt / "val.txt"
        )
        self.scenes = [
            # (label folder, raw image path)
            (Path(self.root / folder[:-1]), Path(self.root / folder[:-4] / 'image_02' / 'data') ) \
                for folder in open(scene_list_path)
        ]
        # self.scenes_imgs = [
        #     Path(self.root / folder[:-4] / 'image_02' / 'data') for folder in open(scene_list_path)
        # ]

        ## only for export??

        if self.config["labels"]:
            self.labels = True
            self.labels_path = Path(self.config["labels"], task)
            print("load labels from: ", self.config["labels"] + "/" + task)
        else:
            self.labels = False

        self.crawl_folders(sequence_length)

        # other variables
        self.init_var()

    def crawl_folders(self, sequence_length):
        sequence_set = []
        # demi_length = (sequence_length-1)//2
        demi_length = sequence_length - 1
        # shifts = list(range(-demi_length, demi_length + 1))
        # shifts.pop(demi_length)
        for (scene, scene_img_folder) in self.scenes:
            # intrinsics and imu_pose_matrixs are redundant for superpoint training
            intrinsics = np.eye(3)
            # imu_pose_matrixs = np.eye(4)
            # intrinsics = (
            #     np.genfromtxt(scene / "cam.txt").astype(np.float32).reshape((3, 3))
            # )
            # imu_pose_matrixs = (
            #     np.genfromtxt(scene / "imu_pose_matrixs.txt")
            #     .astype(np.float64)
            #     .reshape(-1, 4, 4)
            # )
            # imgs = sorted(scene.files('*.jpg'))

            ##### get images
            # print(f"scene_img_folder: {scene_img_folder}")
            image_paths = list(scene_img_folder.iterdir())
            names = [p.stem for p in image_paths]
            imgs = [str(p) for p in image_paths]
            # files = {"image_paths": image_paths, "names": names}
            #####


            # X_files = sorted(scene.glob('*.npy'))
            if len(imgs) < sequence_length:
                continue
            # for i in range(demi_length, len(imgs)-demi_length):
            for i in range(0, len(imgs) - demi_length):
                sample = None
                # sample = {'intrinsics': intrinsics, 'imu_pose_matrixs': [imu_pose_matrixs[i]], 'imgs': [imgs[i]], 'Xs': [load_as_array(X_files[i])], 'scene_name': scene.name, 'frame_ids': [i]}
                if self.labels:
                    p = Path(self.labels_path, scene.name, "{}.npz".format(names[i]))
                    # print(f"label Path: {p}")
                    if p.exists():
                        sample = {
                            "intrinsics": intrinsics,
                            # "imu_pose_matrixs": [imu_pose_matrixs[i]],
                            "image": [imgs[i]],
                            "scene_name": scene.name,
                            "frame_ids": [i],
                        }
                        sample.update({"name": [names[i]], "points": [str(p)]})
                else:
                    sample = {
                        "intrinsics": intrinsics,
                        # "imu_pose_matrixs": [imu_pose_matrixs[i]],
                        "imgs": [imgs[i]],
                        "scene_name": scene.name,
                        "frame_ids": [i],
                        "name": [names[i]],
                    }

                if sample is not None:
                    # for j in shifts:
                    for j in range(1, demi_length + 1):
                        sample["image"].append(imgs[i + j])
                        # sample["imu_pose_matrixs"].append(imu_pose_matrixs[i + j])
                        # sample['Xs'].append(load_as_array(X_files[i])) # [3, N]
                        sample["frame_ids"].append(i + j)
                    sequence_set.append(sample)
                # print(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set
        logging.info("Finished crawl_folders for KITTI.")

    def get_img_from_sample(self, sample):
        # imgs = [_preprocess(_read_image(img)[:, :, np.newaxis]) for img in sample['imgs']]
        # imgs = [_read_image(img) for img in sample['imgs']]
        imgs_path = sample["imgs"]
        # print(len(imgs_path))
        # print(str(imgs_path[0]))

        # assert len(imgs) == self.sequence_length
        return str(imgs_path[0])

    def get_from_sample(self, entry, sample):
        # print(f"sample: {sample}")
        return str(sample[entry][0])

    def format_sample(self, sample):
        sample_fix = {}
        if self.labels:
            entries = ["image", "points", "name"]
            # sample_fix['image'] = get_img_from_sample(sample)
            for entry in entries:
                sample_fix[entry] = self.get_from_sample(entry, sample)
        else:
            sample_fix["image"] = str(sample["imgs"][0])
            sample_fix["name"] = str(sample["scene_name"] + "/" + sample["name"][0])
            sample_fix["scene_name"] = str(sample["scene_name"])

        return sample_fix


if __name__ == "__main__":
    # main()
    pass
