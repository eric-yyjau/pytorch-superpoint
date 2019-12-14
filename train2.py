import argparse
import time
import csv
import yaml
import os
import logging
from pathlib import Path

import numpy as np
import torch
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
from utils.utils import precisionRecall_torch
from utils.utils import img_overlap, thd_img, toNumpy

import time
import scipy




##########

def train_base(config, output_dir, args):
    assert 'train_iter' in config

    # config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('train on device: %s', device)
    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    writer = SummaryWriter(getWriterPath(task=args.command, exper_name=args.exper_name, date=True))
    task = config['data']['dataset']
    retrain = config['retrain']


    ## save data
    # save_path = save_path_formatter(config, output_dir)
    save_path = Path(output_dir)
    save_path = save_path / 'checkpoints'
    logging.info('=> will save everything to {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)
    max_iter = config['train_iter']

    # data loading
    data = dataLoader(config, dataset=task, warp_input=True)
    train_loader, val_loader = data['train_loader'], data['val_loader']
    logging.info('== train split size %d in %d batches, val split size %d in %d batches'%\
        (len(train_loader)*config['model']['batch_size'], len(train_loader), len(val_loader)*config['model']['batch_size'], len(val_loader)))

    # model loading
#     net = modelLoader(model='SuperPointNet').cuda()
    # net = modelLoader(model='SuperPointNet')
    net = modelLoader(model='SuperPointNet_gauss2')
    # net.init_

    logging.info('=> setting adam solver')
    import torch.optim as optim
    optimizer = optim.Adam(net.parameters(), lr=config['model']['learning_rate'])
    n_iter = 0
    n_iter_val = 0
    epoch = 0

    ## load pretrained
    if retrain == True:
        logging.info("New model")
        # net.init_weights()
    else:
        try:
            path = config['pretrained']
            mode = '' if path[-3:] == 'pth' else 'full'
            logging.info('path: %s', path)
            net, optimizer, n_iter = pretrainedLoader(net, optimizer, n_iter, path, mode=mode, full_path=True)
            logging.info('successful load pretrained model from: %s', path)
            save_file = save_path / "training.txt"
            with open(save_file, "a") as myfile:
                myfile.write("model: " + path + "\n")
        except Exception:
            logging.info('no pretrained model')
            ## load checkpoint
            try:
                net, optimizer, n_iter = pretrainedLoader(net, optimizer, n_iter, save_path, mode='full')
                logging.info("load checkpoint. n_iter: %d", n_iter)
            except Exception:
                logging.info("no existed checkpoint")
                pass

    print("=== Let's use", torch.cuda.device_count(), "GPUs!")
    net = net.to(device)
    net = nn.DataParallel(net)
    ##### check: create 2nd optimizer #####
    optimizer = optim.Adam(net.parameters(), lr=config['model']['learning_rate'])

    if config['reset_iter']:
        logging.info("reset iterations to 0")
        n_iter = 0

    # train model
    # try:
    from utils.utils import labels2Dto3D, labels2Dto3D_flattened, flattenDetection, box_nms
    from utils.utils import saveLoss
    from utils.utils import pltImshow, saveImg
    def train_base_model(net, train_loader, val_loader, n_iter=0, n_iter_val=0, train=False, warped_pair=False):
        print("train show interval: ", config['train_show_interval'])
        save_interval = config['train_show_interval']
        print("train save interval: ", save_interval)


        from utils.utils import getPtsFromHeatmap
        cell_size = 8
        running_losses = []
        running_mAP = 0.0
        running_mAR = 0.0
        if train:
            task = 'train'
        else:
            task = 'val'

        # forward + backward + optimize
        loss_func = nn.CrossEntropyLoss(reduce=False).cuda()
        loss_func_BCE = nn.BCELoss(reduce=False).cuda()

        def get_loss(semi, labels3D_in_loss, mask_3D_flattened, config):
            if config['data']['gaussian_label']['enable']:
                loss = loss_func_BCE(nn.functional.softmax(semi, dim=1), labels3D_in_loss)
                loss = (loss.sum(dim=1) * mask_3D_flattened).sum()
            else:
                loss = loss_func(semi, labels3D_in_loss)
                loss = (loss * mask_3D_flattened).sum()
            loss = loss / (mask_3D_flattened.sum() + 1e-10)
            return loss

        def train_val_sample(net, sample, n_iter, train):
            task = 'training' if train else 'validating'
            ## get the inputs
            # logging.info('get input img and label')
            img, labels_2D, mask_2D = sample['image'], sample['labels_2D'], sample['valid_mask']
            if config['data']['gaussian_label']['enable']:
                labels_2D_gaussian = sample['labels_2D_gaussian']
            batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
            Hc, Wc = H//cell_size, W//cell_size
            if config['data']['gaussian_label']['enable']:
                labels3D_gaussian = labels2Dto3D(labels_2D_gaussian, cell_size=cell_size).float().cuda()
                # labels3D_flattened_gaussian = labels2Dto3D_flattened(labels_2D.cuda(), cell_size=cell_size)
            labels3D = labels2Dto3D(labels_2D.cuda(), cell_size=cell_size)
            labels3D_flattened = labels2Dto3D_flattened(labels_2D.cuda(), cell_size=cell_size)
            mask_3D = labels2Dto3D(mask_2D.cuda(), cell_size=cell_size, add_dustbin=False).float()
            mask_3D_flattened = torch.prod(mask_3D, 1) # [B, Hc, Wc]

            if config['data']['gaussian_label']['enable']:
                labels3D_in_loss = labels3D_gaussian
            else:
                labels3D_in_loss = labels3D_flattened

            if train:
                outs = net(img.cuda())
                # output = {'semi': semi, 'desc': desc}
                semi, coarse_desc = outs['semi'], outs['desc']
                loss = get_loss(semi, labels3D_in_loss, mask_3D_flattened, config)
                # zero the parameter gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    outs = net(img.cuda())
                    # semi, coarse_desc = outs[0], outs[1]
                    semi, coarse_desc = outs['semi'], outs['desc']
                    loss = get_loss(semi, labels3D_in_loss, mask_3D_flattened, config)

            writer.add_scalar(task + '-loss', loss, n_iter)

            if (train and n_iter % config['train_show_interval'] == 0) or (n_iter % 20 == 0 and not train):
                for idx in range(5):
                    # data save to tensorboard
                    writer.add_image(task+'-input'+'/%d'%idx, img[idx, :, :, :], n_iter)
                    writer.add_image(task+'-labels'+'/%d'%idx, labels_2D[idx,:,:,:], n_iter)
                    if config['data']['gaussian_label']['enable']:
                        writer.add_image(task+'-labels_gaussian'+'/%d'%idx, labels_2D_gaussian[idx,:,:,:], n_iter)
                    dust_resize = 0.3*scipy.misc.imresize(labels3D[idx:idx+1, -1, :, :].squeeze().cpu().float().numpy(), config['data']['preprocessing']['resize'], 'nearest')/255. + 0.3*img[idx, :, :, :].squeeze().cpu().float().numpy() + \
                            + scipy.misc.imresize(labels_2D[idx,:,:,:].squeeze().cpu().float().numpy(), config['data']['preprocessing']['resize'], 'nearest')/255
                    dust_resize_copy = dust_resize.copy()
                    dust_resize_copy[dust_resize_copy>1.] = 1.
                    writer.add_image(task+'-labels_dust_overlay'+'/%d'%idx, dust_resize_copy[np.newaxis, :, :], n_iter)
                    writer.add_image(task+'-mask_2D'+'/%d'%idx, mask_2D[idx:idx+1, :, :, :].squeeze(1), n_iter)
                    # writer.add_image(task+'-mask', mask[0:1, :, :, :].squeeze(1), n_iter)
                    writer.add_image(task+'-mask_3D_flattened'+'/%d'%idx, mask_3D_flattened[idx:idx+1, :, :].cpu().numpy(), n_iter) # should be the same as mask
                    mask_resize = scipy.misc.imresize(mask_3D_flattened[idx:idx+1, :, :].squeeze().cpu().float().numpy(), config['data']['preprocessing']['resize'], 'nearest')/255.*0.5 + img[idx, :, :, :].squeeze().cpu().float().numpy()*0.5
                    mask_resize_copy = mask_resize.copy()
                    mask_resize_copy[mask_resize_copy>1.] = 1.
                    writer.add_image(task+'-mask_3D_flattened_overlay'+'/%d'%idx, mask_resize_copy[np.newaxis, :, :], n_iter) # should be the same as mask

                    nms_dist = config['model']['nms'] # How to determine this?
                    conf_thresh = config['model']['detection_threshold']

                    # visualization
                    semi_flat = flattenDetection(semi[idx, :, :, :]) # [1, H, W]
                    writer.add_scalar(task+'-output_max', semi_flat.max(), n_iter)
                    semi_thd = thd_img(semi_flat, conf_thresh)
                    result_overlap = img_overlap(toNumpy(labels_2D[idx, :, :, :]), toNumpy(semi_thd.detach()), toNumpy(img[idx, :, :, :]))
                    writer.add_image(task+'-detector_output_thd_overlay'+'/%d'%idx, result_overlap, n_iter)

                precision_recall_list = []
                precision_recall_boxnms_list = []
                boxnms = False
                for idx in range(batch_size):
                    semi_flat_tensor = flattenDetection(semi[idx, :, :, :]).detach()
                    semi_flat = toNumpy(semi_flat_tensor)
                    semi_thd = np.squeeze(semi_flat, 0)
                    pts_nms = getPtsFromHeatmap(semi_thd, conf_thresh, nms_dist)
                    semi_thd_nms_sample = np.zeros_like(semi_thd)
                    semi_thd_nms_sample[pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)] = 1

                    label_sample = torch.squeeze(labels_2D[idx, :, :, :])
                    # pts_nms = getPtsFromHeatmap(label_sample.numpy(), conf_thresh, nms_dist)
                    # label_sample_rms_sample = np.zeros_like(label_sample.numpy())
                    # label_sample_rms_sample[pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)] = 1
                    label_sample_nms_sample = label_sample

                    if idx < 5:
                        result_overlap = img_overlap(np.expand_dims(label_sample_nms_sample, 0), np.expand_dims(semi_thd_nms_sample, 0), toNumpy(img[idx, :, :, :]))
                        writer.add_image(task+'-detector_output_thd_overlay-NMS'+'/%d'%idx, result_overlap, n_iter)
                    assert semi_thd_nms_sample.shape == label_sample_nms_sample.size()
                    precision_recall = precisionRecall_torch(torch.from_numpy(semi_thd_nms_sample), label_sample_nms_sample)
                    precision_recall_list.append(precision_recall)

                    if boxnms:
                        semi_flat_tensor_nms = box_nms(semi_flat_tensor.squeeze(), nms_dist, min_prob=conf_thresh).cpu()
                        semi_flat_tensor_nms = (semi_flat_tensor_nms>=conf_thresh).float()

                        if idx < 5:
                            result_overlap = img_overlap(np.expand_dims(label_sample_nms_sample, 0),
                                                         semi_flat_tensor_nms.numpy()[np.newaxis, :, :], toNumpy(img[idx, :, :, :]))
                            writer.add_image(task+'-detector_output_thd_overlay-boxNMS'+'/%d'%idx, result_overlap, n_iter)
                        precision_recall_boxnms = precisionRecall_torch(semi_flat_tensor_nms, label_sample_nms_sample)
                        precision_recall_boxnms_list.append(precision_recall_boxnms)

                precision = np.mean([precision_recall['precision'] for precision_recall in precision_recall_list])
                recall = np.mean([precision_recall['recall'] for precision_recall in precision_recall_list])
                writer.add_scalar(task+'-precision_nms', precision, n_iter)
                writer.add_scalar(task+'-recall_nms', recall, n_iter)
                print('-- [%s-%d-fast NMS] precision: %.4f, recall: %.4f'%(task, n_iter, precision, recall))
                if boxnms:
                    precision = np.mean([precision_recall['precision'] for precision_recall in precision_recall_boxnms_list])
                    recall = np.mean([precision_recall['recall'] for precision_recall in precision_recall_boxnms_list])
                    writer.add_scalar(task+'-precision_boxnms', precision, n_iter)
                    writer.add_scalar(task+'-recall_boxnms', recall, n_iter)
                    print('-- [%s-%d-boxNMS] precision: %.4f, recall: %.4f'%(task, n_iter, precision, recall))

            # save model
            if n_iter % save_interval == 0 and train:
                model_state_dict = net.module.state_dict()
                save_checkpoint(
                    save_path, {
                        'n_iter': n_iter + 1,
                        'model_state_dict': model_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    },
                    n_iter)
                logging.info("save model: %d", n_iter)
            return loss.item()


        # Train one epoch
        for i, sample_train in tqdm(enumerate(train_loader)):
            if train:
                # logging.info('====== Training...')
                net.train()
                loss_out = train_val_sample(net, sample_train, n_iter, True)
                n_iter += 1
                running_losses.append(loss_out)
            if args.eval and n_iter%config['validation_interval']==0:
                logging.info('====== Validating %s at train step %d'%(args.exper_name, n_iter))
                net.eval()
                for j, sample_val in enumerate(val_loader):
                    train_val_sample(net, sample_val, n_iter_val, False)  ##### check: in order to align val and training
                    n_iter_val += 1
                    if j > config['data']['validation_size']:  ##### check: how to limit the validation
                        break

            # interval = 200
            # if n_iter % interval == interval - 1:  # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.5f' %
            #           (n_iter + 1, i + 1, running_loss / interval))
            #     running_loss = 0.0

        # epoch_map = running_mAP/ (i+1)
        epoch_map = -1 # not computing the mAP for now; skip it by returning -1 @Rui
        epoch_loss = np.mean(np.asarray(running_losses))
        return epoch_loss, epoch_map, n_iter, n_iter_val

    loss = 0
    while True:
        epoch += 1
        train_loss, train_map, n_iter, n_iter_val = train_base_model(net, train_loader=train_loader, val_loader=val_loader,
                n_iter=n_iter, n_iter_val=n_iter_val, train=True, warped_pair=config['data']['warped_pair']['enable'])
        logging.info('n_iter: %d, ave training loss: %.5f, mAP: %.5f', n_iter, train_loss, train_map)
        save_file = save_path / "training.txt"
        saveLoss(save_file, n_iter, train_loss, mAP=train_map)
        loss = train_loss
        # if args.eval:
        #     val_loss, val_map, _ = train_base_model(net, data_loader=val_loader,
        #         n_iter=n_iter, train=False, warped_pair=config['data']['warped_pair']['enable'])
        #     logging.info('n_iter: %d, ave validation loss: %.5f, mAP: %.5f', n_iter, val_loss, val_map)
        #     saveLoss(save_file, n_iter, val_loss, task='val', mAP=val_map)

        if epoch % 10 == 0:
            model_state_dict = net.module.state_dict()
            save_checkpoint(
                save_path, {
                    'n_iter': n_iter + 1,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                 },
                n_iter)
            logging.info("save model: %d", n_iter)
        if n_iter > max_iter:
            break
    print('Finished Training')
    # pass
    # except KeyboardInterrupt:
    #     logging.info("ctrl + c is pressed. save model")
    #     # save checkpoint
    #     model_state_dict = net.module.state_dict()
    #     save_checkpoint(
    #         save_path, {
    #             'n_iter': n_iter + 1,
    #             'model_state_dict': model_state_dict,
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': loss,
    #         },
    #         n_iter)
    #     save_checkpoint(
    #         save_path, {
    #             'n_iter': n_iter + 1,
    #             'model_state_dict': model_state_dict,
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': loss,
    #         },
    #         "")
    #     pass
    # pass

def train_joint(config, output_dir, args):
    assert 'train_iter' in config

    # config
    from utils.utils import pltImshow
    from utils.utils import saveImg
    torch.set_default_tensor_type(torch.FloatTensor)
    task = config['data']['name']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('train on device: %s', device)
    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
#     writer = SummaryWriter(getWriterPath(task=args.command, date=True))
    writer = SummaryWriter(getWriterPath(task=args.command, exper_name=args.exper_name, date=True))
    retrain = config['retrain']
    # scale = config['warp_scale']

    ## save data
    save_path = Path(output_dir)
    save_output = save_path
    save_path = save_path / 'checkpoints'
    logging.info('=> will save everything to {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)
    max_iter = config['train_iter']

    # data loading
    data = dataLoader(config, dataset=task, warp_input=True)
    train_loader, val_loader = data['train_loader'], data['val_loader']

    # model loading
    net = modelLoader(model='SuperPointNet').cuda()
    # net.init_
    logging.info('=> setting adam solver')
    import torch.optim as optim
    optimizer = optim.Adam(net.parameters(), lr=config['model']['learning_rate'], betas=(0.9, 0.999))
    n_iter = 0

    ## load pretrained
    if retrain == True:
        logging.info("New model")
        pass
    else:
        try:
            path = config['pretrained']
            mode = '' if path[:-3] == '.pth' else 'full'
            logging.info('load pretrained model from: %s', path)
            net, optimizer, n_iter = pretrainedLoader(net, optimizer, n_iter, path, mode=mode, full_path=True)
            logging.info('successfully load pretrained model from: %s', path)
            save_file = save_path / "training.txt"
            with open(save_file, "a") as myfile:
                myfile.write("path: " + path + "\n")
        except Exception:
            logging.info('no pretrained model')
            ## load checkpoint
            try:
                net, optimizer, n_iter = pretrainedLoader(net, optimizer, n_iter, save_path, mode='full')
                logging.info("load checkpoint. n_iter: %d", n_iter)
            except Exception:
                logging.info("no existed checkpoint")
                pass

    print("=== Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)
#     optimizer = optim.Adam(net.parameters(), lr=config['model']['learning_rate'], betas=(0.9, 0.999))

    if config['reset_iter']:
        logging.info("reset iterations to 0")
        n_iter = 0

    # train model
    try:
        from utils.utils import warp_points
        from utils.utils import inv_warp_image_batch
        # from utils.utils import sample_homography
        from utils.utils import labels2Dto3D, flattenDetection, labels2Dto3D_flattened
        from utils.utils import descriptor_loss, saveLoss
        from utils.utils import sumto2D
        from utils.homographies import sample_homography_np as sample_homography

        descriptor_dist = config['model']['descriptor_dist']
        lambda_d = config['model']['lambda_d']
        def train_joint_model(net, data_loader, n_iter=0, train=False):
            running_loss_all = 0.0
            running_loss = 0.0
            train_map = 0

            def train_joint_sample(net, sample, n_iter=0, train=False):
                if train:
                    task = 'train'
                else:
                    task = 'val'
#                 print("add task: ", task)

                # import functions


                if verbose_time: print("load sample: ", time.time() - seconds)
                seconds = time.time()
                # config
                lamda = config['model']['lambda_loss']

                # get the inputs
                img, labels_2D, mask_2D = sample['image'], sample['labels_2D'], sample['valid_mask']
                batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
                cell_size = 8
                Hc = H // cell_size
                Wc = W // cell_size
                img, labels = img.to(device), labels_2D.to(device)
                labels = labels2Dto3D(labels, cell_size=cell_size)

                # get warped images
                img_warp = sample['warped_img']
                labels_warp_2D = sample['warped_labels']
                mask_warp = sample['warped_valid_mask']
                mask_warp_2D = mask_warp
                # mask
                mask_warp = labels2Dto3D(mask_warp, cell_size=cell_size)
                mask_warp = torch.prod(mask_warp[:, :cell_size * cell_size, :, :], dim=1)
                mask_warp = mask_warp.view(-1, 1, Hc, Wc)
                labels_warp = labels2Dto3D(labels_warp_2D, cell_size=cell_size)

                img_warp, labels_warp, mask_warp = \
                    img_warp.to(device), labels_warp.to(device), mask_warp.to(device)

                mat_H, mat_H_inv = sample['homographies'], sample['inv_homographies']
                mat_H, mat_H_inv = mat_H.to(device), mat_H_inv.to(device)



                if verbose_time: print("label 2D to 3D: ", time.time() - seconds)

                #                 writer.add_image('img', img[0, :, :, :], n_iter)
                #                 writer.add_image('img_warp', img_warp[0, :, :, :], n_iter)

                seconds = time.time()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                if train:
                    outs, outs_warp = net(img), net(img_warp)
                    semi, coarse_desc = outs[0], outs[1]
                    semi_warp, coarse_desc_warp = outs_warp[0], outs_warp[1]
                else:
                    with torch.no_grad():
                        outs, outs_warp = net(img), net(img_warp)
                        semi, coarse_desc = outs[0], outs[1]
                        semi_warp, coarse_desc_warp = outs_warp[0], outs_warp[1]

                if verbose_time: print("forward path: ", time.time() - seconds)
                seconds = time.time()

                # detector, flatten, visualization
                semi_flat = flattenDetection(semi[0, :, :, :])
                semi_warp_flat = flattenDetection(semi_warp[0, :, :, :])

                writer.add_scalar(task + 'output_max', semi_flat.max(), n_iter)

                thd = 0.015
                semi_thd = thd_img(semi_flat, thd=thd)
                semi_warp_thd = thd_img(semi_warp_flat, thd=thd)

#                 print("semi_thd shape", semi_thd.shape)

                # result_overlap = img_overlap(toNumpy(labels_warp_2D[1, :, :, :]), semi_warp_thd, toNumpy(img_warp[1, :, :, :]))
                # writer.add_image(task + '_warp_detector_output_thd_overlay_last', result_overlap, n_iter)

                if verbose_time: print("writer: ", time.time() - seconds)
                seconds = time.time()

                # loss
                mask_desc = mask_warp


                loss_desc, mask, positive_dist, negative_dist = \
                            descriptor_loss(coarse_desc, coarse_desc_warp, mat_H, mask_valid=mask_desc, device=device,
                                descriptor_dist=descriptor_dist, lambda_d=lambda_d)
#                                     descriptor_loss(coarse_desc, coarse_desc_warp, mat_H, device=device)

                ## detector loss



                ### new loss: check!!! #####
                loss_func = nn.CrossEntropyLoss(reduce=False).cuda()
                def get_loss(semi, labels3D_in_loss, mask_3D_flattened):
                    loss = loss_func(semi, labels3D_in_loss)
                    loss = (loss * mask_3D_flattened).sum()
                    loss = loss / (mask_3D_flattened.sum() + 1e-10)
                    return loss

                def getLabels(labels_2D, cell_size):
                    labels3D_flattened = labels2Dto3D_flattened(labels_2D.cuda(), cell_size=cell_size)
                    labels3D_in_loss = labels3D_flattened
                    return labels3D_in_loss
                def getMasks(mask_2D, cell_size):
                    mask_3D = labels2Dto3D(mask_2D.cuda(), cell_size=cell_size, add_dustbin=False).float()
                    mask_3D_flattened = torch.prod(mask_3D, 1)
                    return mask_3D_flattened
                labels3D_in_loss = getLabels(labels_2D, cell_size)
                mask_3D_flattened = getMasks(mask_2D, cell_size)

                loss_det = get_loss(semi, labels3D_in_loss, mask_3D_flattened)

                # warping
                labels3D_in_loss = getLabels(labels_warp_2D, cell_size)
                mask_3D_flattened = getMasks(mask_warp_2D, cell_size)
                loss_det_warp = get_loss(semi_warp, labels3D_in_loss, mask_3D_flattened)
#                 loss_det_warp = get_loss(semi, labels3D_in_loss, mask_3D_flattened)
#                 loss_det = nn.functional.binary_cross_entropy_with_logits(semi, labels)
#                 loss_det_warp = nn.functional.binary_cross_entropy_with_logits(semi_warp,
#                                                                                labels_warp, weight=mask_warp)
                # labels_warp)


                if verbose_time: print("loss, and descriptor loss: ", time.time() - seconds)
                seconds = time.time()

                ## detector + lamda * loss_desc
                # loss = loss_det + loss_det_warp + lamda * loss_desc
                if no_desc:
                    loss = loss_det + loss_det_warp
                else:
                    loss = loss_det + loss_det_warp + lamda * loss_desc

                #                 print(loss_det.item(), ", ", loss_det_warp.item(), ", ",loss_desc.item())
                if train:
                    loss.backward()
                    optimizer.step()

                if verbose_time: print("backpropagation: ", time.time() - seconds)
                seconds = time.time()


                def add2tensorboard():
                    #                 writer.add_image(task + 'desc_mask', sumto2D(mask)[:1, :, :], n_iter)
                # writer.add_image('desc_positive', sumto2D(positive_dist)[:1,:,:], n_iter)
                # writer.add_image('desc_negitive', sumto2D(negative_dist)[:1,:,:], n_iter)
                    result_overlap = img_overlap(toNumpy(labels_2D[0, :, :, :]), toNumpy(semi_thd), toNumpy(img[0, :, :, :]))

                    writer.add_image(task + '_detector_output_thd_overlay', result_overlap, n_iter)
                    saveImg(result_overlap.transpose([1, 2, 0])[..., [2, 1, 0]] * 255, 'test_0.png')  # rgb to bgr * 255

                    result_overlap = img_overlap(toNumpy(labels_warp_2D[0, :, :, :]),
                                                 toNumpy(semi_warp_thd), toNumpy(img_warp[0, :, :, :]))
                    writer.add_image(task + '_warp_detector_output_thd_overlay', result_overlap, n_iter)
                    saveImg(result_overlap.transpose([1, 2, 0])[..., [2, 1, 0]] * 255, 'test_1.png')  # rgb to bgr * 255

                    mask_overlap = img_overlap(toNumpy(1-mask_warp_2D[0,:,:,:])/2,
                                                np.zeros_like(toNumpy(img_warp[0, :, :, :])),
                                                toNumpy(img_warp[0, :, :, :]))

                    # writer.add_image(task + '_mask_valid_first_layer', mask_warp[0, :, :, :], n_iter)
                    # writer.add_image(task + '_mask_valid_last_layer', mask_warp[-1, :, :, :], n_iter)
                    ##### print to check
                    print("mask_2D shape: ", mask_warp_2D.shape)
                    print("mask_3D_flattened shape: ", mask_3D_flattened.shape)
                    writer.add_image(task + '_mask_warp_origin', mask_warp_2D[0, :, :, :], n_iter)
                    writer.add_image(task + '_mask_warp_3D_flattened', mask_3D_flattened[0, :, :], n_iter)
                    writer.add_image(task + '_mask_warp_origin-1', mask_warp_2D[1, :, :, :], n_iter)
                    writer.add_image(task + '_mask_warp_3D_flattened-1', mask_3D_flattened[1, :, :], n_iter)
                    writer.add_image(task + '_mask_warp_overlay', mask_overlap, n_iter)

                    writer.add_scalar(task + 'loss_det', loss_det, n_iter)
                    writer.add_scalar(task + 'loss_det_warp', loss_det_warp, n_iter)
                    writer.add_scalar(task + 'loss_desc', loss_desc, n_iter)
                    writer.add_scalar(task + 'loss_desc_pos', positive_dist, n_iter)
                    writer.add_scalar(task + 'loss_desc_neg', negative_dist, n_iter)
                    writer.add_scalar(task + 'loss', loss, n_iter)

                def add2tensorboard_nms():
                    from utils.utils import getPtsFromHeatmap
                    from utils.utils import box_nms
                    boxNms = False



                    nms_dist = config['model']['nms'] # How to determine this?
                    conf_thresh = config['model']['detection_threshold']
                    precision_recall_list = []
                    precision_recall_boxnms_list = []
                    for idx in range(batch_size):
                        semi_flat_tensor = flattenDetection(semi[idx, :, :, :]).detach()
                        semi_flat = toNumpy(semi_flat_tensor)
                        semi_thd = np.squeeze(semi_flat, 0)
                        pts_nms = getPtsFromHeatmap(semi_thd, conf_thresh, nms_dist)
                        semi_thd_nms_sample = np.zeros_like(semi_thd)
                        semi_thd_nms_sample[pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)] = 1

                        label_sample = torch.squeeze(labels_2D[idx, :, :, :])
                        # pts_nms = getPtsFromHeatmap(label_sample.numpy(), conf_thresh, nms_dist)
                        # label_sample_rms_sample = np.zeros_like(label_sample.numpy())
                        # label_sample_rms_sample[pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)] = 1
                        label_sample_nms_sample = label_sample

                        if idx < 5:
                            result_overlap = img_overlap(np.expand_dims(label_sample_nms_sample, 0),
                                                         np.expand_dims(semi_thd_nms_sample, 0), toNumpy(img[idx, :, :, :]))
                            writer.add_image(task+'-detector_output_thd_overlay-NMS'+'/%d'%idx, result_overlap, n_iter)
                        assert semi_thd_nms_sample.shape == label_sample_nms_sample.size()
                        precision_recall = precisionRecall_torch(torch.from_numpy(semi_thd_nms_sample), label_sample_nms_sample)
                        precision_recall_list.append(precision_recall)

                        if boxNms:
                            semi_flat_tensor_nms = box_nms(semi_flat_tensor.squeeze(), nms_dist, min_prob=conf_thresh).cpu()
                            semi_flat_tensor_nms = (semi_flat_tensor_nms>=conf_thresh).float()

                            if idx < 5:
                                result_overlap = img_overlap(np.expand_dims(label_sample_nms_sample, 0),
                                                             semi_flat_tensor_nms.numpy()[np.newaxis, :, :], toNumpy(img[idx, :, :, :]))
                                writer.add_image(task+'-detector_output_thd_overlay-boxNMS'+'/%d'%idx, result_overlap, n_iter)
                            precision_recall_boxnms = precisionRecall_torch(semi_flat_tensor_nms, label_sample_nms_sample)
                            precision_recall_boxnms_list.append(precision_recall_boxnms)

                    precision = np.mean([precision_recall['precision'] for precision_recall in precision_recall_list])
                    recall = np.mean([precision_recall['recall'] for precision_recall in precision_recall_list])
                    writer.add_scalar(task+'-precision_nms', precision, n_iter)
                    writer.add_scalar(task+'-recall_nms', recall, n_iter)
                    print('-- [%s-%d-fast NMS] precision: %.4f, recall: %.4f'%(task, n_iter, precision, recall))
                    if boxNms:
                        precision = np.mean([precision_recall['precision'] for precision_recall
                                             in precision_recall_boxnms_list])
                        recall = np.mean([precision_recall['recall'] for precision_recall in precision_recall_boxnms_list])
                        writer.add_scalar(task+'-precision_boxnms', precision, n_iter)
                        writer.add_scalar(task+'-recall_boxnms', recall, n_iter)
                        print('-- [%s-%d-boxNMS] precision: %.4f, recall: %.4f'%(task, n_iter, precision, recall))

                save_file = save_path / "training.txt"
                saveLoss(save_file, n_iter, loss.item(), 'train', loss_det=loss_det.item(),
                         loss_det_warp=loss_det_warp.item(), loss_desc=loss_desc.item())

                # print statistics
#                 running_loss += loss.item()
#                 running_loss_all += loss.item()
                interval = 100
                if n_iter % interval == 0 or task == 'val':
                    add2tensorboard()
                    if n_iter % interval*3 == 0:
                        add2tensorboard_nms()
                return loss




            for i, sample in tqdm(enumerate(data_loader)):
                net.train()
                loss = train_joint_sample(net, sample, n_iter, True)
                logging_interval = 100
                if n_iter % logging_interval == logging_interval-1:  # print every 100 mini-batches
                    print('[%d, %5d] loss: %.5f' %
                          (n_iter + 1, i + 1, loss))



                if args.eval and n_iter%config['validation_interval']==0:
                    logging.info('====== Validating... ' + args.exper_name)
                    net.eval()
                    for j, sample_val in enumerate(val_loader):
                        train_joint_sample(net, sample_val, n_iter, False)  ##### check: in order to align val and training
                        if j > config['data']['validation_size']:  ##### check: how to limit the validation
                            break
                n_iter += 1
                # save models
                save_interval = config['validation_interval']
                if n_iter % save_interval == save_interval - 1 and train:
                    model_state_dict = net.module.state_dict()
                    save_checkpoint(
                        save_path, {
                            'n_iter': n_iter + 1,
                            'model_state_dict': model_state_dict,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                        },
                        n_iter + 1)

                if n_iter > max_iter: break
#             train_loss = running_loss_all / len(data_loader)

            train_loss = -1,
            train_map = -1
            return loss, train_map, n_iter

        verbose_time = False
#         print("scale: ", scale)
        no_desc = False
        if no_desc: print("note!! No training descriptor!!")
        if config['reset_iter']:
            logging.info("reset iterations to 0")
            n_iter = 0



        epoch = 0
        loss = -1
        print("use batch size: ", config['model']['batch_size'])
        logging.info('== train split size %d in %d batches, val split size %d in %d batches'%\
            (len(train_loader)*config['model']['batch_size'], len(train_loader),
             len(val_loader)*config['model']['batch_size'], len(val_loader)))
        while n_iter < max_iter:
            epoch += 1
            print("epoch: ", epoch)
            running_loss = 0.0
            seconds = time.time()
            train_loss, _, n_iter = train_joint_model(net, data_loader=train_loader, n_iter=n_iter, train=True)
#             logging.info('n_iter: %d, ave training loss: %.5f', n_iter, train_loss)
            save_file = save_path / "training.txt"
#             saveLoss(save_file, n_iter, train_loss, task='train')
            loss = train_loss
#             if args.eval:
#                 val_loss, _map, _ = train_joint_model(net, data_loader=val_loader, n_iter=n_iter, train=False)
#                 logging.info('n_iter: %d, ave validation loss: %.5f', n_iter, val_loss)
#                 saveLoss(save_file, n_iter, val_loss, task='val')

            # epoch_loss = loss
            # logging.info('n_iter: %d, ave loss: %.5f', n_iter, epoch_loss)
            if epoch % 1 == 0:
                model_state_dict = net.module.state_dict()
                save_checkpoint(
                    save_path, {
                        'n_iter': n_iter + 1,
                        'model_state_dict': model_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    },
                    n_iter)

        print('Finished Training')
        pass
    except KeyboardInterrupt:
        logging.info("ctrl + c is pressed. save model")
        # is_best = True
        # save checkpoint
        model_state_dict = net.module.state_dict()
        save_checkpoint(
            save_path, {
                'n_iter': n_iter + 1,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            },
            n_iter)
        save_checkpoint(
            save_path, {
                'n_iter': n_iter + 1,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            },
            "")
        pass
        pass

    pass

def train_joint_dsac(config, output_dir, args):
    assert 'train_iter' in config

    # config
    from utils.utils import pltImshow
    from utils.utils import saveImg
    torch.set_default_tensor_type(torch.FloatTensor)
    task = config['data']['name']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('train on device: %s', device)
    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    # writer = SummaryWriter('runs/'+args.command)
    # writer = SummaryWriter(getWriterPath(task=args.command, date=True), comment='pretrained_superpoint')
    writer = SummaryWriter(getWriterPath(task=args.command, date=True))

    ## save data
    # save_path = save_path_formatter(config, output_dir)
    save_path = Path(output_dir)
    save_path = save_path / 'checkpoints'
    logging.info('=> will save everything to {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)
    n_iter = config['train_iter']

    # data loading

    data = dataLoader(config, dataset=task, warp_input=True)
    train_loader, val_loader = data['train_loader'], data['val_loader']


    # model loading
    net = modelLoader(model='SuperPointNet')
    net.to(device)
    # net.init_
    logging.info('=> setting adam solver')
    import torch.optim as optim
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

    epoch = 0
    ## load pretrained
    retrain = False
    if retrain == True:
        logging.info("New model")
        pass
    else:
        ## load pretrained
        try:
            path = config['pretrained']
            print('==> Loading pre-trained network.')
            # This class runs the SuperPoint network and processes its outputs.
            nms_dist = 4
            conf_thresh = 0.015
            nn_thresh = 0.7

            fe = SuperPointFrontend_torch(weights_path=path,
                                    nms_dist=nms_dist,
                                    conf_thresh=conf_thresh,
                                    nn_thresh=nn_thresh,
                                    cuda=False,
                                    device=device
                                    )
            print('==> Successfully loaded pre-trained network.')
            print(path)
        except Exception:
            pass

    outputMatches = True
    count = 0
    max_length = 5
    sparse_desc_loss = True

    tracker = PointTracker(max_length, nn_thresh=fe.nn_thresh)
    from utils.utils import pltImshow
    # train model
    try:
        for epoch in tqdm(range(epoch, n_iter)):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, sample in tqdm(enumerate(train_loader)):
                # import functions
                from utils.utils import warp_points
                from utils.utils import inv_warp_image_batch
                from utils.utils import sample_homography
                from utils.utils import labels2Dto3D, flattenDetection
                from utils.utils import descriptor_loss
                from utils.utils import sumto2D

                # config
                lamda = 0.0001

                # get the inputs
                img, labels = sample['image'], sample['labels_2D']
                batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
                img, labels = img.to(device), labels.to(device)

                ##############################

                ## sample homography matrix
                mat_H = [sample_homography() for i in range(batch_size)]
                # mat_H = [np.identity(3) for i in range(batch_size)]
                mat_H = np.stack(mat_H, axis=0)
                mat_H = torch.tensor(mat_H, dtype=torch.float32)
                mat_H = mat_H.to(device)

                mat_H_inv = torch.stack([torch.inverse(mat_H[i,:,:]) for i in range(batch_size)])
                mat_H_inv = torch.tensor(mat_H_inv, dtype=torch.float32)
                mat_H_inv = mat_H_inv.to(device)
                ## warp images
                img_warp = inv_warp_image_batch(img, mat_H_inv, device=device)
                labels_warp = inv_warp_image_batch(labels, mat_H_inv, device=device)

                writer.add_image('labels', labels[0,:,:,:], epoch)
                writer.add_image('labels_warp', labels_warp[0,:,:,:], epoch)

                # labels = labels2Dto3D(labels, cell_size=8)
                # labels_warp = labels2Dto3D(labels_warp, cell_size=8)
                img_warp, labels_warp = img_warp.to(device), labels_warp.to(device)

                writer.add_image('img', img[0, :, :, :], epoch)
                writer.add_image('img_warp', img_warp[0, :, :, :], epoch)

                sample['homography'] = mat_H
                ##############################

                # zero the parameter gradients
                optimizer.zero_grad()

                # first image, no matches
                img = img
                pts, pts_desc, dense_desc, heatmap = fe.run(img)
                '''
                img shape: tensor (batch_size, 1, H, W)
                pts:
                    [batches x tensor(3, N1)]
                desc:
                    [batches x tensor(256, N1)]

                '''
                # save keypoints
                pred = {'image': img}
                pred.update({'prob': pts,
                             'desc': pts_desc,
                             'heatmap': heatmap
                               })

                # second image, output matches
                img = img_warp
                warped_pts, warped_pts_desc, warped_dense_desc, warped_heatmap = fe.run(img)
                pred.update({'warped_image': img_warp})
                pred.update({'warped_prob': warped_pts,
                             'warped_desc': warped_pts_desc,
                             'homography': sample['homography'],
                             'warped_heatmap': warped_heatmap
                             })


                # forward + backward + optimize
                loss_det = nn.functional.binary_cross_entropy_with_logits(heatmap, labels)
                loss_det_warp = nn.functional.binary_cross_entropy_with_logits(warped_heatmap, labels_warp)
                if sparse_desc_loss:
                    from utils.utils import descriptor_sparse_loss
                # loss_desc  = descriptor_sparse_loss(dense_desc, warped_dense_desc, mat_H, pts, warped_pts)
                # loss = loss_det + loss_det_warp + lamda * loss_desc
                loss = loss_det + loss_det_warp
                loss.backward()
                optimizer.step()

                def getMatches(tracker, pts, desc, warped_pts, warped_desc):
                    tracker.update(pts, desc)
                    tracker.update(warped_pts, warped_desc)
                    matches = tracker.get_matches()
                    print("matches: ", matches.shape)
                    # clean last descriptor
                    tracker.clear_desc()
                    '''
                    matches:
                        np (n, 4)
                    '''
                    return matches.transpose()

                if outputMatches == True:
                    matches_batch = [getMatches(tracker, pts[i], pts_desc[i], warped_pts[i], warped_pts_desc[i])
                                     for i in range(batch_size)]

                    # pred.update({'matches': matches.transpose()})
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
                # testing
                def testing():
                    from utils.utils import pltImshow
                    h = heatmap.cpu().detach().numpy()
                    pltImshow(h[0,0,:,:])
                    img = pred['image']
                    img = img.cpu().detach().numpy()
                    pltImshow(img[0,0,:,:])

                print("loss: ", loss)
                # print statistics
                # running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            epoch_loss = running_loss / len(train_loader)
            logging.info('epoch: %d, ave loss: %.5f', epoch, epoch_loss)
            if epoch % 1000 == 0:
                save_checkpoint(
                    save_path, {
                        'epoch': epoch + 1,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    },
                    epoch)

        print('Finished Training')
        pass
    except KeyboardInterrupt:
        logging.info("ctrl + c is pressed. save model")
        # is_best = True
        # save checkpoint
        save_checkpoint(
            save_path, {
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            },
            epoch)
        save_checkpoint(
            save_path, {
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            },
            "")
        pass

    pass

def validate_with_gt():
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
    print("check config: ", config)
    # EXPER_PATH from settings.py
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    os.makedirs(output_dir, exist_ok=True)

    # with capture_outputs(os.path.join(output_dir, 'log')):
    logging.info('Running command {}'.format(args.command.upper()))
    args.func(config, output_dir, args)

    # global variables

    # main()

