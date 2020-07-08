"""script for subpixel experiment (not tested)
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
from utils.loader import dataLoader, modelLoader, pretrainedLoader
import logging

from utils.tools import dict_update

from utils.utils import labels2Dto3D, flattenDetection, labels2Dto3D_flattened

from utils.utils import pltImshow, saveImg
from utils.utils import precisionRecall_torch
from utils.utils import save_checkpoint

from pathlib import Path
from Train_model_frontend import Train_model_frontend

class Train_model_subpixel(Train_model_frontend):

    default_config = {
        'train_iter': 170000,
        'save_interval': 2000,
        'tensorboard_interval': 200,
        'model': {
            'subpixel': {
                'enable': False
            }
        }
    }
    def __init__(self, config, save_path=Path('.'), device='cpu', verbose=False):
        print("using: Train_model_subpixel")
        self.config = self.default_config
        self.config = dict_update(self.config, config)
        self.device=device
        self.save_path = save_path
        self.cell_size = 8
        self.max_iter = config['train_iter']
        self._train = True
        self._eval = True

        pass
    def print(self):
        print("hello")

    def loadModel(self):
        ###### check!
        model = self.config['model']['name']
        params = self.config['model']['params']
        print("model: ", model)
        net = modelLoader(model=model, **params).to(self.device)
        # net.init_
        logging.info('=> setting adam solver')
        # import torch.optim as optim
        # optimizer = optim.Adam(net.parameters(), lr=self.config['model']['learning_rate'], 
            # betas=(0.9, 0.999))
        optimizer = self.adamOptim(net, lr=self.config['model']['learning_rate'])

        n_iter = 0
        ## load pretrained
        if self.config['retrain'] == True:
            logging.info("New model")
            pass
        else:
            path = self.config['pretrained']
            mode = '' if path[:-3] == '.pth' else 'full'
            logging.info('load pretrained model from: %s', path)
            net, optimizer, n_iter = pretrainedLoader(net, optimizer, n_iter, path, mode=mode, full_path=True)
            logging.info('successfully load pretrained model from: %s', path)

        def setIter(n_iter):
            if self.config['reset_iter']:
                logging.info("reset iterations to 0")
                n_iter = 0
            return n_iter


        self.net = net
        self.optimizer = optimizer
        self.n_iter = setIter(n_iter)
        pass

    def train_val_sample(self, sample, n_iter=0, train=False):
        task = 'train' if train else 'val'
        tb_interval = self.config['tensorboard_interval']

        losses, tb_imgs, tb_hist = {}, {}, {}
        ## get the inputs
        # logging.info('get input img and label')
        img, labels_2D, mask_2D = sample['image'], sample['labels_2D'], sample['valid_mask']
        # img, labels = img.to(self.device), labels_2D.to(self.device)
        labels_res = sample['labels_res']

        # variables
        batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
        self.batch_size = batch_size
        # print("batch_size: ", batch_size)
        Hc = H // self.cell_size
        Wc = W // self.cell_size


        # zero the parameter gradients
        self.optimizer.zero_grad()

        # extract patches
        # extract the patches from labels 
        label_idx = labels_2D[...].nonzero()
        from utils.losses import extract_patches
        patch_size = self.config['model']['params']['patch_size']
        patches = extract_patches(label_idx.to(self.device), img.to(self.device), 
            patch_size=patch_size) # tensor [N, patch_size, patch_size]
        # patches = extract_patches(label_idx.to(device), labels_2D.to(device), patch_size=15) # tensor [N, patch_size, patch_size]
        # print("patches: ", patches.shape)

        patch_channels = self.config['model']['params'].get('subpixel_channel', 1)
        if patch_channels == 2:
            patch_heat = extract_patches(label_idx.to(self.device), img.to(self.device), 
                patch_size=patch_size) # tensor [N, patch_size, patch_size]

        def label_to_points(labels_res, points):
            labels_res = labels_res.transpose(1,2).transpose(2,3).unsqueeze(1)
            points_res = labels_res[points[:,0],points[:,1],points[:,2],points[:,3],:]  # tensor [N, 2]
            return points_res
            
        points_res = label_to_points(labels_res, label_idx)

        num_patches_max = 500
        # feed into the network
        pred_res = self.net(patches[:num_patches_max, ...].to(self.device)) # tensor [1, N, 2]



        # loss function
        def get_loss(points_res, pred_res):
            loss = (points_res - pred_res)
            loss = torch.norm(loss, p=2, dim=-1).mean()
            return loss

        loss = get_loss(points_res[:num_patches_max,...].to(self.device), 
                pred_res)
        self.loss = loss


        losses.update({'loss': loss})
        tb_hist.update({'points_res_0': points_res[:,0]})
        tb_hist.update({'points_res_1': points_res[:,1]})
        tb_hist.update({'pred_res_0': pred_res[:,0]})
        tb_hist.update({'pred_res_1': pred_res[:,1]})
        tb_imgs.update({'patches': patches[:,...].unsqueeze(1)})
        tb_imgs.update({'img': img})
        # forward + backward + optimize
        # if train:
        #     print("img: ", img.shape)
        #     outs, outs_warp = self.net(img.to(self.device)), self.net(img_warp.to(self.device), subpixel=self.subpixel)
        #     semi, coarse_desc = outs[0], outs[1]
        #     semi_warp, coarse_desc_warp = outs_warp[0], outs_warp[1]
        # else:
        #     with torch.no_grad():
        #         outs, outs_warp = self.net(img.to(self.device)), self.net(img_warp.to(self.device), subpixel=self.subpixel)
        #         semi, coarse_desc = outs[0], outs[1]
        #         semi_warp, coarse_desc_warp = outs_warp[0], outs_warp[1]
        #         pass

        # descriptor loss


        losses.update({'loss': loss})
        # print("losses: ", losses)

        if train:
            loss.backward()
            self.optimizer.step()

        self.tb_scalar_dict(losses, task)
        if n_iter % tb_interval == 0 or task == 'val':
            logging.info("current iteration: %d, tensorboard_interval: %d", n_iter, tb_interval)
            self.tb_images_dict(task, tb_imgs, max_img=5)
            self.tb_hist_dict(task, tb_hist)

        return loss.item()

    def tb_images_dict(self, task, tb_imgs, max_img=5):
        for element in list(tb_imgs):
            for idx in range(tb_imgs[element].shape[0]):
                if idx >= max_img: break
                self.writer.add_image(task + '-' + element + '/%d'%idx, 
                    tb_imgs[element][idx,...], self.n_iter)

    def tb_hist_dict(self, task, tb_dict):
        for element in list(tb_dict):
            self.writer.add_histogram(task + '-' + element, 
              tb_dict[element], self.n_iter)  
        pass




if __name__ == '__main__':
    filename = 'configs/magicpoint_shapes_subpix.yaml'
    import yaml
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_default_tensor_type(torch.FloatTensor)
    with open(filename, 'r') as f:
        config = yaml.load(f)

    from utils.loader import dataLoader as dataLoader
    # data = dataLoader(config, dataset='hpatches')
    task = config['data']['dataset']

    data = dataLoader(config, dataset=task, warp_input=True)
    # test_set, test_loader = data['test_set'], data['test_loader']
    train_loader, val_loader = data['train_loader'], data['val_loader']

    train_agent = Train_model_subpixel(config, device=device)
    train_agent.print()
    # writer from tensorboard
    from tensorboardX import SummaryWriter
    writer = SummaryWriter()
    train_agent.writer = writer

    # feed the data into the agent
    train_agent.train_loader = train_loader
    train_agent.val_loader = val_loader

    train_agent.loadModel()
    train_agent.dataParallel()

    try:
        # train function takes care of training and evaluation
        train_agent.train()
    except KeyboardInterrupt:
        print ("press ctrl + c, save model!")
        train_agent.saveModel()
        pass

    # try:
    #     # train function takes care of training and evaluation
    #     train_agent.train()
    # except KeyboardInterrupt:
    #     print ("press ctrl + c, save model!")
    #     train_agent.saveModel()
    #     pass
