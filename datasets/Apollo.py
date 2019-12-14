import numpy as np
# import tensorflow as tf
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

# from Kitti import Kitti
from datasets.Coco import Coco
import logging
from tqdm import tqdm
import glob


class Apollo(Coco):
    default_config = {
        "labels": None,
        "augmentation": {
            "photometric": {
                "enable": False,
                "primitives": "all",
                "params": {},
                "random_order": True,
            },
            "homographic": {"enable": False, "params": {}, "valid_border_margin": 0},
        },
        "homography_adaptation": {"enable": False},
    }

    def __init__(
        self,
        export=False,
        transform=None,
        task="train",
        seed=0,
        sequence_length=1,
        **config
    ):
        # Update config
        self.config = self.default_config
        self.config = dict_update(self.config, config)

        self.transforms = transform
        self.action = "train" if task == "train" else "val"

        # get files
        self.root =  Path(self.config['root'])
        self.task = task
        # get dataset split files
        root_split_txt = self.config.get('root_split_txt', None)
        self.root_split_txt = Path(self.root if root_split_txt is None else root_split_txt)
        # scene_list_path = (
        #     self.root_split_txt / "train.txt" if task == "train" else self.root_split_txt / "val.txt"
        # )
        self.scenes = [
            Path(self.root / "train")
        ]
        if self.config["labels"]:
            self.labels = True
            self.labels_path = Path(self.config["labels"], task)
            print("load labels from: ", self.config["labels"] + "/" + task)
        else:
            self.labels = False

        self.crawl_folders(sequence_length)

        # other variables
        self.init_var()

    @staticmethod
    def filter_list(list, select_word=''):
        return [l for l in list if select_word in l]

    def read_images_files_from_folder(self, drive_path, file='train.txt', cam_id=1):
        print(f"drive_path: {drive_path}")
        ## given that we have matched time stamps
        # arr = np.genfromtxt(f'{drive_path}/{file}',dtype='str') # [N, 1(path)]
        frame_list_path = f'{drive_path}/{file}'
        img_files = [
            # f"{frame[:-8]}/{frame[-7:-1]}" for frame in open(frame_list_path) # due to the output test file
            f"{str(drive_path)}/{frame[:-8]}/{frame[-7:-1]}" for frame in open(frame_list_path) # due to the output test file
        ]  # odo: Path(root/04_02) 0
        # img_files = np.char.add(str(drive_path)+'/image/', arr)
        # img_files = self.filter_list(img_files, select_word=f'Camera_{cam_id}')
        img_files = [Path(f+'.jpg') for f in img_files]
        img_files = sorted(img_files)
        print(f"{frame_list_path}, img: {img_files[0]}, len: {len(img_files)}")

        get_name = lambda frame: Path(f"{frame[:-8]}_{frame[-7:-1]}").stem
        img_names = [
            # f"{frame[:-8]}/{frame[-7:-1]}" for frame in open(frame_list_path) # due to the output test file
            f"{get_name(frame)}" for frame in open(frame_list_path) # due to the output test file
        ]

        ## no time stamps
        # img_dir = os.path.join(drive_path, "")
        # img_files = sorted(glob(img_dir + f"/{folder}/*.png"))
        # print(f"img_files: {img_files[0]}")
        return {'img_files': img_files, 'img_names': img_names}


    def crawl_folders(self, sequence_length):
        sequence_set = []
        # demi_length = (sequence_length-1)//2
        demi_length = sequence_length - 1  ### ??
        # shifts = list(range(-demi_length, demi_length + 1))
        # shifts.pop(demi_length)
        for scene in self.scenes:
            # intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            # imu_pose_matrixs = np.genfromtxt(scene/'imu_pose_matrixs.txt').astype(np.float64).reshape(-1, 4, 4)
            intrinsics = np.eye(3)
            imu_pose_matrixs = np.eye(4)

            ##### test
            # image_paths = list(scene.iterdir())
            # names = [p.stem for p in image_paths]
            # image_paths = [str(p) for p in image_paths]
            # files = {'image_paths': image_paths, 'names': names}
            #####
            print(f"scene: {scene}")
            # imgs = sorted(glob.glob(f"{scene}/rgb/*.png"))
            # self.read_images_files_from_folder()
            data = self.read_images_files_from_folder(
                # "", file=f"/{self.task}.txt", cam_id=1  
                self.root, file=f"/{self.task}.txt", cam_id=1
            )
            imgs = data['img_files']
            names = data['img_names']
            print(f"name: {names[0]}")
            ## make dir

            # names = [Path(p).stem for p in imgs]

            # X_files = sorted(scene.glob('*.npy'))
            if len(imgs) < sequence_length:
                continue
            # for i in range(demi_length, len(imgs)-demi_length):
            for i in tqdm(range(0, len(imgs) - demi_length)):
                sample = None
                # sample = {'intrinsics': intrinsics, 'imu_pose_matrixs': [imu_pose_matrixs[i]], 'imgs': [imgs[i]], 'Xs': [load_as_array(X_files[i])], 'scene_name': scene.name, 'frame_ids': [i]}
                if self.labels:
                    p = Path(
                        self.labels_path,
                        scene.name,
                        "{}.npz".format(names[i]),
                    )
                    if p.exists():
                        sample = {
                            "intrinsics": intrinsics,
                            "imu_pose_matrixs": [imu_pose_matrixs],
                            "image": [imgs[i]],
                            "scene_name": scene.name,
                            "frame_ids": [i],
                        }
                        sample.update({"name": [names[i]], "points": [str(p)]})
                else:
                    sample = {
                        "intrinsics": intrinsics,
                        "imu_pose_matrixs": [imu_pose_matrixs],
                        "imgs": [imgs[i]],
                        "scene_name": scene.name,
                        "frame_ids": [i],
                        "name": [names[i]],
                    }

                if sample is not None:
                    # for j in shifts:
                    for j in range(1, demi_length + 1):
                        sample["image"].append(imgs[i + j])
                        sample["imu_pose_matrixs"].append(imu_pose_matrixs[i + j])
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
        print(len(imgs_path))
        print(str(imgs_path[0]))

        # assert len(imgs) == self.sequence_length
        return str(imgs_path[0])

    def get_from_sample(self, entry, sample):
        return str(sample[entry][0])

    def format_sample(self, sample):
        sample_fix = {}
        if self.labels:
            entries = ["image", "points", "name"]
            # sample_fix['image'] = get_img_from_sample(sample)
            for entry in entries:
                sample_fix[entry] = self.get_from_sample(entry, sample)
        else:
            sample_fix['image'] = str(sample["imgs"][0])
            sample_fix['name'] = str(sample["scene_name"] + "/" + sample["name"][0])
            sample_fix['scene_name'] = str(sample["scene_name"])

        return sample_fix


if __name__ == "__main__":
    # main()
    pass
