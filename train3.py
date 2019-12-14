import argparse
import time
import csv
import yaml
import os
import logging
from pathlib import Path

import numpy as np
import torch
import cv2
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm

from tensorboardX import SummaryWriter

from utils.utils import tensor2array, save_checkpoint, load_checkpoint, save_path_formatter
from utils.utils import getWriterPath
from settings import EXPER_PATH


## loaders: data, model, pretrained model
from utils.loader import dataLoader, modelLoader, pretrainedLoader
from models.model_wrap import SuperPointFrontend_torch, PointTracker




class SPInferLoader(object):
    def __init__(self, config, output_dir, args, cuda_num=0):
        torch.set_default_tensor_type(torch.FloatTensor)
        task = config['data']['name']
        self.device = torch.device("cuda:%d"%cuda_num if torch.cuda.is_available() else "cpu")

        # model loading
        model = config['model']['name']
        net = modelLoader(model=model)
        net.to(self.device)
        path = config['pretrained']
        print('==> Loading pre-trained network.')
        # This class runs the SuperPoint network and processes its outputs.
        nms_dist = config['model']['nms'] 
        conf_thresh = config['model']['detection_threshold']
        nn_thresh = config['model']['nn_thresh'] 
        print('nms_dist: ', nms_dist)
        print('conf_thresh: ', conf_thresh)
        print('nn_thresh: ', nn_thresh)

        from utils.print_tool import print_config
        print_config(config['model'])

        self.subpixel = config['model']['subpixel']

        self.fe = SuperPointFrontend_torch(config=config,
                                weights_path=path,
                                nms_dist=nms_dist,
                                conf_thresh=conf_thresh,
                                nn_thresh=nn_thresh,
                                cuda=False,
                                device=self.device
                                )
        print('==> Successfully loaded pre-trained network.')
        print(path)

        outputMatches = True
        count = 0
        max_length = 5
        sparse_desc_loss = True

        self.tracker = PointTracker(max_length, nn_thresh=self.fe.nn_thresh)

    def img_array_to_input(self, input_image):
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        input_image = input_image.astype('float32') / 255.0
        image = input_image[np.newaxis, np.newaxis, :, :]
        return image

    def process_output(self, outputs):
        # pts, pts_desc, dense_desc, heatmap = outputs
        return {'pts': outputs[0], 'pts_desc': outputs[1],
                'dense_desc': outputs[2], 'heatmap': outputs[3]}


    def run_two_imgs(self, img, img_warp):
        # get the inputs
        batch_size, H, W = img.shape[0], img.shape[1], img.shape[2]

        # first image, no matches
        pred = {}
        def run_one_img(img, name=''):
            img = torch.from_numpy(img).to(self.device)
            outputs = self.fe.run(img)
            outputs = self.process_output(outputs)
            """
            pts: list of np [np[3,N]]
            pts_desc: list of np [np[256,N]]
            """
            pts, pts_desc, dense_desc, heatmap = \
                outputs['pts'], outputs['pts_desc'], outputs['dense_desc'], outputs['heatmap']
            # if self.subpixel:
            #     pts = self.fe.soft_argmax_points()
            print("pts: ", pts[0][:,:5])
            '''
            img shape: tensor (batch_size, 1, H, W) 
            '''
            # save keypoints
            pred.update({name + 'image': img,
                        name + 'prob': pts,
                        name + 'desc': pts_desc,
                        name + 'heatmap': heatmap
                        })
        run_one_img(img, name='')
        run_one_img(img_warp, name='warped_')

        # second image, output matches
        # img = torch.from_numpy(img_warp).to(self.device)
        # warped_pts, warped_pts_desc, warped_dense_desc, warped_heatmap = self.fe.run(img)
        # pred.update({'warped_image': img_warp})
        # pred.update({'warped_prob': warped_pts,
        #              'warped_desc': warped_pts_desc,
        #              'warped_heatmap': warped_heatmap
        #              })

        return pred

    def get_matches(self, pred):
        def getMatches(tracker, pts, desc, warped_pts, warped_desc):
            tracker.update(pts, desc)
            tracker.update(warped_pts, warped_desc)
            matches = tracker.get_matches()
            print("SP matches: ", matches.shape[1])
            # clean last descriptor
            tracker.clear_desc()
            '''
            matches:
                np (n, 4)
            '''
            return matches.transpose()

        matches_batch = [getMatches(self.tracker, pred['prob'][i], pred['desc'][i], pred['warped_prob'][i], pred['warped_desc'][i])
            for i in range(len(pred['prob']))]
        return matches_batch


class SPInferLoader_heatmap(SPInferLoader):
    def __init__(self, config, output_dir, args, cuda_num=0, device='cpu'):
        # model loading
        from utils.loader import get_module
        Val_model_heatmap = get_module('', config['front_end_model'])
        val_agent = Val_model_heatmap(config['model'], device=device)
        val_agent.loadModel()
        print('==> Successfully loaded pre-trained network.')
        self.fe = val_agent
        self.device = device
        self.tracker = PointTracker(max_length=3, nn_thresh=self.fe.nn_thresh)
        pass

    def process_output(self, outputs):
        heatmap_batch = outputs
        val_agent = self.fe
        pts = val_agent.heatmap_to_pts()
        pts_subpixel = val_agent.soft_argmax_points(pts)
        desc_sparse = val_agent.desc_to_sparseDesc()
        return {'pts': pts_subpixel, 'pts_desc': desc_sparse,
                'dense_desc': {}, 'heatmap': heatmap_batch}
        pass

if __name__ == '__main__':
    # global var
    torch.set_default_tensor_type(torch.FloatTensor)
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    # add parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Training command
    p_train = subparsers.add_parser('train_base')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.add_argument('--eval', action='store_true')
    p_train.add_argument('--debug', action='store_true', default=False,
                         help='turn on debuging mode')
    p_train.set_defaults(func=train_base)

    # Validation command
    p_train = subparsers.add_parser('val_base')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.add_argument('--eval', action='store_true')
    p_train.add_argument('--debug', action='store_true', default=False,
                         help='turn on debuging mode')
    p_train.set_defaults(func=val_base)

    # Training command
    p_train = subparsers.add_parser('train_joint')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.add_argument('--eval', action='store_true')
    p_train.add_argument('--debug', action='store_true', default=False,
                         help='turn on debuging mode')
    p_train.set_defaults(func=train_joint)

    # Training command
    p_train = subparsers.add_parser('train_joint_dsac')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.add_argument('--eval', action='store_true')
    p_train.add_argument('--debug', action='store_true', default=False,
                         help='turn on debuging mode')
    p_train.set_defaults(func=train_joint_dsac)

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    # EXPER_PATH from settings.py
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    os.makedirs(output_dir, exist_ok=True)

    # with capture_outputs(os.path.join(output_dir, 'log')):
    logging.info('Running command {}'.format(args.command.upper()))
    args.func(config, output_dir, args)

    # global variables

    # main()

