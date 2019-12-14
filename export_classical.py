"""export classical feature extractor (not tested)
"""

import argparse
import time
import csv
import yaml
import os
import logging
from pathlib import Path
import torch

import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter

from utils.utils import tensor2array, save_checkpoint, load_checkpoint, save_path_formatter
from settings import EXPER_PATH
from utils.loader import dataLoader, modelLoader, pretrainedLoader
from utils.utils import getWriterPath

# from utils.logging import *


def export_descriptor(config, output_dir, args):
    '''
    1) input 2 images, output keypoints and correspondence

    :param config:
    :param output_dir:
    :param args:
    :return:
    '''
    # config
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info('train on device: %s', device)
    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    writer = SummaryWriter(getWriterPath(task=args.command, date=True))

    ## save data
    from pathlib import Path
    # save_path = save_path_formatter(config, output_dir)
    save_path = Path(output_dir)
    save_output = save_path
    save_output = save_output / 'predictions'
    save_path = save_path / 'checkpoints'
    logging.info('=> will save everything to {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_output, exist_ok=True)

    # data loading
    from utils.loader import dataLoader_test as dataLoader
    data = dataLoader(config, dataset='hpatches')
    test_set, test_loader = data['test_set'], data['test_loader']

    from utils.print_tool import datasize
    datasize(test_loader, config, tag='test')

    from imageio import imread
    def load_as_float(path):
        return imread(path).astype(np.float32) / 255

    def squeezeToNumpy(tensor_arr):
        return tensor_arr.detach().cpu().numpy().squeeze()

    outputMatches = True
    count = 0
    max_length = 5
    method = config['model']['method']
    # tracker = PointTracker(max_length, nn_thresh=fe.nn_thresh)

    # for sample in tqdm(enumerate(test_loader)):
    for i, sample in tqdm(enumerate(test_loader)):

        img_0, img_1 = sample['image'], sample['warped_image']

        # first image, no matches
        img = img_0.numpy().squeeze()*255
        # H, W = img.shape[1], img.shape[2]
        # img = img.view(1,1,H,W)

        ##### add opencv functions here #####
        def classicalDetectors(image, method='sift'):
            round_method = False
            if round_method == True:
                from models.classical_detectors_descriptors import classical_detector_descriptor # with quantization
                points, desc = classical_detector_descriptor(image, **{'method': method})
                y, x = np.where(points)
                # pnts = np.stack((y, x), axis=1)
                pnts = np.stack((x, y), axis=1) # should be (x, y)
                ## collect descriptros
                desc = desc[y, x, :]
            else:
                # sift with subpixel accuracy
                from models.classical_detectors_descriptors import SIFT_det as classical_detector_descriptor
                pnts, desc = classical_detector_descriptor(image, image)

            print("desc shape: ", desc.shape)
            return pnts, desc

        pts, desc = classicalDetectors(img, method=method)
        print("total points: ", pts.shape)
        '''
        pts: list [numpy (N, 2)]
        desc: list [numpy (N, 128)]
        '''
        # save keypoints
        pred = {'image': squeezeToNumpy(img_0)}
        pred.update({'prob': pts,
                     'desc': desc})

        # second image, output matches
        img = img_1.numpy().squeeze()*255
        pred.update({'warped_image': squeezeToNumpy(img_1)})
        pts, desc = classicalDetectors(img, method=method)

        # if outputMatches == True:
        #     tracker.update(pts, desc)
        # pred.update({'matches': matches.transpose()})

        print("total points: ", pts.shape)
        pred.update({'warped_prob': pts,
                     'warped_desc': desc,
                     'homography': squeezeToNumpy(sample['homography'])
                     })
        # clean last descriptor
        '''
        pred:
            'image': np(320,240)
            'prob' (keypoints): np (N1, 2)
            'desc': np (N2, 256)
            'warped_image': np(320,240)
            'warped_prob' (keypoints): np (N2, 2)
            'warped_desc': np (N2, 256)
            'homography': np (3,3)

        '''

        # save data
        from pathlib import Path
        filename = str(count)
        path = Path(save_output, '{}.npz'.format(filename))
        np.savez_compressed(path, **pred)
        count += 1
    print("output pairs: ", count)
    save_file = save_output / "export.txt"
    with open(save_file, "a") as myfile:
        myfile.write("output pairs: " + str(count) + '\n')
    pass



if __name__ == '__main__':
    # global var
    torch.set_default_tensor_type(torch.FloatTensor)
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    # add parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # export command
    p_train = subparsers.add_parser('export_descriptor')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    # p_train.add_argument('exper', type=str)
    p_train.add_argument('--correspondence', action='store_true')
    p_train.add_argument('--eval', action='store_true')
    p_train.add_argument('--debug', action='store_true', default=False,
                         help='turn on debuging mode')
    p_train.set_defaults(func=export_descriptor)

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    os.makedirs(output_dir, exist_ok=True)

    # with capture_outputs(os.path.join(output_dir, 'log')):
    logging.info('Running command {}'.format(args.command.upper()))
    args.func(config, output_dir, args)

    # global variables

    # main()
