"""
# deprecated synthetic data loader
"""

import torch.utils.data as data
import torch
import numpy as np
from imageio import imread
# from os import path as Path
import tensorflow as tf
from pathlib import Path
import tarfile
import os
import random
import logging
from .utils import pipeline
# from .utils.pipeline import parse_primitives
from utils.tools import dict_update

from datasets import synthetic_dataset
# from models.homographies import sample_homography

from tqdm import tqdm
import cv2
import shutil
from settings import DEBUG as debug
from settings import DATA_PATH
# DATA_PATH = '.'

def load_as_float(path):
    return imread(path).astype(np.float32)/255



class SyntheticDataset(data.Dataset):
    """
    """
    default_config = {
            'primitives': 'all',
            'truncate': {},
            'validation_size': -1,
            'test_size': -1,
            'on-the-fly': False,
            'cache_in_memory': False,
            'suffix': None,
            'add_augmentation_to_test_set': False,
            'num_parallel_calls': 10,
            'generation': {
                'split_sizes': {'training': 10000, 'validation': 200, 'test': 500},
                'image_size': [960, 1280],
                'random_seed': 0,
                'params': {
                    'generate_background': {
                        'min_kernel_size': 150, 'max_kernel_size': 500,
                        'min_rad_ratio': 0.02, 'max_rad_ratio': 0.031},
                    'draw_stripes': {'transform_params': (0.1, 0.1)},
                    'draw_multiple_polygons': {'kernel_boundaries': (50, 100)}
                },
            },
            'preprocessing': {
                'resize': [240, 320],
                'blur_size': 11,
            },
            'augmentation': {
                'photometric': {
                    'enable': False,
                    'primitives': 'all',
                    'params': {},
                    'random_order': True,
                },
                'homographic': {
                    'enable': False,
                    'params': {},
                    'valid_border_margin': 0,
                },
            }
    }

    debug = False

    if debug == True:
        drawing_primitives = [
            'draw_lines'
        ]
    else:
        drawing_primitives = [
                'draw_lines',
                'draw_polygon',
                'draw_multiple_polygons',
                'draw_ellipses',
                'draw_star',
                'draw_checkerboard',
                'draw_stripes',
                'draw_cube',
                'gaussian_noise'
        ]

    '''
    def dump_primitive_data(self, primitive, tar_path, config):
        pass
    '''
    def dump_primitive_data(self, primitive, tar_path, config):
        temp_dir = Path(os.environ['TMPDIR'], primitive)

        tf.logging.info('Generating tarfile for primitive {}.'.format(primitive))
        synthetic_dataset.set_random_state(np.random.RandomState(
            config['generation']['random_seed']))
        for split, size in self.config['generation']['split_sizes'].items():
            im_dir, pts_dir = [Path(temp_dir, i, split) for i in ['images', 'points']]
            im_dir.mkdir(parents=True, exist_ok=True)
            pts_dir.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(size), desc=split, leave=False):
                image = synthetic_dataset.generate_background(
                    config['generation']['image_size'],
                    **config['generation']['params']['generate_background'])
                points = np.array(getattr(synthetic_dataset, primitive)(
                    image, **config['generation']['params'].get(primitive, {})))
                points = np.flip(points, 1)  # reverse convention with opencv

                b = config['preprocessing']['blur_size']
                image = cv2.GaussianBlur(image, (b, b), 0)
                points = (points * np.array(config['preprocessing']['resize'], np.float)
                          / np.array(config['generation']['image_size'], np.float))
                image = cv2.resize(image, tuple(config['preprocessing']['resize'][::-1]),
                                   interpolation=cv2.INTER_LINEAR)

                cv2.imwrite(str(Path(im_dir, '{}.png'.format(i))), image)
                np.save(Path(pts_dir, '{}.npy'.format(i)), points)

        # Pack into a tar file
        tar = tarfile.open(tar_path, mode='w:gz')
        tar.add(temp_dir, arcname=primitive)
        tar.close()
        shutil.rmtree(temp_dir)
        tf.logging.info('Tarfile dumped to {}.'.format(tar_path))

    def parse_primitives(self, names, all_primitives):
        p = all_primitives if (names == 'all') \
            else (names if isinstance(names, list) else [names])
        assert set(p) <= set(all_primitives)
        return p

    def __init__(self, seed=None, train=True, sequence_length=3, transform=None, target_transform=None, getPts=False, warp_input=False, **config):
        from utils.homographies import sample_homography_np as sample_homography
        from utils.photometric import ImgAugTransform, customizedTransform
        from utils.utils import compute_valid_mask
        from utils.utils import inv_warp_image, warp_points

        torch.set_default_tensor_type(torch.FloatTensor)
        np.random.seed(seed)
        random.seed(seed)
        self.transform = transform
        self.sample_homography = sample_homography
        self.compute_valid_mask = compute_valid_mask
        self.inv_warp_image = inv_warp_image
        self.warp_points = warp_points
        self.ImgAugTransform = ImgAugTransform
        self.customizedTransform = customizedTransform

        self.action = 'training' if train == True else 'validation'
        self.warp_input = warp_input
        # Update config
        self.config = self.default_config
        self.config = dict_update(self.config, config)
        self.cell_size = 8
        self.getPts = getPts

        # Parse drawing primitives
        primitives = self.parse_primitives(config['primitives'], self.drawing_primitives)

        basepath = Path(
            DATA_PATH, 'synthetic_shapes' +
                       ('_{}'.format(config['suffix']) if config['suffix'] is not None else ''))
        basepath.mkdir(parents=True, exist_ok=True)

        splits = {s: {'images': [], 'points': []}
                  for s in [self.action] }
        for primitive in primitives:
            tar_path = Path(basepath, '{}.tar.gz'.format(primitive))
            if not tar_path.exists():
                self.dump_primitive_data(primitive, tar_path, self.config)

            # Untar locally
            logging.info('Extracting archive for primitive {}.'.format(primitive))
            # tar = tarfile.open(tar_path)
            # temp_dir = Path(os.environ['TMPDIR'])
            temp_dir = Path(os.environ['TMPDIR'])
            # tar.extractall(path=temp_dir)
            # tar.close()

            # Gather filenames in all splits, optionally truncate
            truncate = config['truncate'].get(primitive, 1)
            path = Path(temp_dir, primitive)
            for s in splits:
                e = [str(p) for p in Path(path, 'images', s).iterdir()]
                f = [p.replace('images', 'points') for p in e]
                f = [p.replace('.png', '.npy') for p in f]
                splits[s]['images'].extend(e[:int(truncate * len(e))])
                splits[s]['points'].extend(f[:int(truncate * len(f))])

        # Shuffle
        for s in splits:
            perm = np.random.RandomState(0).permutation(len(splits[s]['images']))
            for obj in ['images', 'points']:
                splits[s][obj] = np.array(splits[s][obj])[perm].tolist()

        self.crawl_folders(splits)

    def crawl_folders(self, splits):
        sequence_set = []
        for (img, pnts) in zip(splits[self.action]['images'], splits[self.action]['points']):
            sample = {'image': img, 'points': pnts}
            sequence_set.append(sample)
        self.samples = sequence_set
        #####

    def __getitem__(self, index):
        """
        :param index:
        :return:
            labels_2D: tensor(1, H, W)
            image: tensor(1, H, W)
        """
        def checkSat(img, name=''):
            if img.max() > 1:
                print(name, img.max())
            elif img.min() < 0:
                print(name, img.min())
        #########


        sample = self.samples[index]
        img = load_as_float(sample['image'])
        H, W = img.shape[0], img.shape[1]
        pnts = np.load(sample['points'])  # (y, x)
        # print('pnts: ', pnts[:5])
        pnts = torch.tensor(pnts).long()
        labels = torch.zeros(H,W)
        labels[pnts[:,0], pnts[:,1]] = 1


#         checkSat(img, 'load: ')

        # assert Hc == round(Hc) and Wc == round(Wc), "Input image size not fit in the block size"
        if self.config['augmentation']['photometric']['enable'] == True:
            augmentation = self.ImgAugTransform(**self.config['augmentation'])
            img = img[:,:,np.newaxis]
            img = augmentation(img)
            cusAug = self.customizedTransform()
            img = cusAug(img, **self.config['augmentation'])
            img = img.squeeze()
#             checkSat(img, 'photometric: ')

            #####
        if self.config['augmentation']['homographic']['enable'] == False:
            img = img[:,:,np.newaxis]
#             labels = labels[np.newaxis,:,:]
#             labels =  torch.from_numpy(labels)
            labels = labels.view(-1,H,W)
            if self.transform is not None:
                img = self.transform(img)
            sample = {'image': img, 'labels_2D': labels}
            valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=torch.eye(3))
            sample.update({'valid_mask': valid_mask})

#             checkSat(img, 'homographic: ')

        else:
            # img_warp = img
            from utils.utils import homography_scaling_torch as homography_scaling
            from utils.utils import filter_points
            from numpy.linalg import inv
            homography = self.sample_homography(np.array([2, 2]), shift=-1,
                                                **self.config['augmentation']['homographic']['params'])

            ##### use inverse from the sample homography
            homography = inv(homography)
            ######

            homography = torch.tensor(homography).to(torch.float32)
            # inv_homography = inv(homography)
            # inv_homography = torch.tensor(inv_homography).to(torch.float32)
            inv_homography = homography.inverse()
            img = torch.from_numpy(img)
            warped_img = self.inv_warp_image(img.squeeze(), inv_homography, mode='bilinear')
            warped_img = warped_img.squeeze().numpy()
            warped_img = warped_img[:,:,np.newaxis]

            # labels = torch.from_numpy(labels)
            # warped_labels = self.inv_warp_image(labels.squeeze(), inv_homography, mode='nearest').unsqueeze(0)
            warped_pnts = self.warp_points(torch.stack((pnts[:, 1], pnts[:, 0]), dim=1), homography_scaling(homography, H, W))
            warped_pnts = filter_points(warped_pnts, torch.tensor([W, H])).round().long()

            warped_labels = torch.zeros(H, W)
            warped_labels[warped_pnts[:, 1], warped_pnts[:, 0]] = 1
            warped_labels = warped_labels.view(-1, H, W)

            if self.transform is not None:
                warped_img = self.transform(warped_img)
            sample = {'image': warped_img, 'labels_2D': warped_labels}

            valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=inv_homography,
                            erosion_radius=self.config['augmentation']['homographic']['valid_border_margin'])  # can set to other value
            sample.update({'valid_mask': valid_mask})

        if self.getPts:
            sample.update({'pts': pnts})

        ######


        return sample

    def __len__(self):
        return len(self.samples)
