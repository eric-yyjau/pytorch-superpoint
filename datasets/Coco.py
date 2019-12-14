import numpy as np
# import tensorflow as tf
import torch
from pathlib import Path
import torch.utils.data as data

# from .base_dataset import BaseDataset
from settings import DATA_PATH, EXPER_PATH
from utils.tools import dict_update
import cv2
from utils.utils import homography_scaling_torch as homography_scaling
from utils.utils import filter_points

class Coco(data.Dataset):
    default_config = {
        'labels': None,
        'cache_in_memory': False,
        'validation_size': 100,
        'truncate': None,
        'preprocessing': {
            'resize': [240, 320]
        },
        'num_parallel_calls': 10,
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
        },
        'warped_pair': {
            'enable': False,
            'params': {},
            'valid_border_margin': 0,
        },
        'homography_adaptation': {
            'enable': False
        }
    }

    def __init__(self, export=False, transform=None, task='train', **config):

        # Update config
        self.config = self.default_config
        self.config = dict_update(self.config, config)

        self.transforms = transform
        self.action = 'train' if task == 'train' else 'val'

        # get files
        base_path = Path(DATA_PATH, 'COCO/' + task + '2014/')
        # base_path = Path(DATA_PATH, 'COCO_small/' + task + '2014/')
        image_paths = list(base_path.iterdir())
        # if config['truncate']:
        #     image_paths = image_paths[:config['truncate']]
        names = [p.stem for p in image_paths]
        image_paths = [str(p) for p in image_paths]
        files = {'image_paths': image_paths, 'names': names}


        sequence_set = []
        # labels
        self.labels = False
        if self.config['labels']:
            self.labels = True
            # from models.model_wrap import labels2Dto3D
            # self.labels2Dto3D = labels2Dto3D
            print("load labels from: ", self.config['labels']+'/'+task)
            count = 0
            for (img, name) in zip(files['image_paths'], files['names']):
                p = Path(self.config['labels'], task, '{}.npz'.format(name))
                if p.exists():
                    sample = {'image': img, 'name': name, 'points': str(p)}
                    sequence_set.append(sample)
                    count += 1
                # if count > 100:
                #     print ("only load %d image!!!", count)
                #     print ("only load one image!!!")
                #     print ("only load one image!!!")
                #     break
            pass
        else:
            for (img, name) in zip(files['image_paths'], files['names']):
                sample = {'image': img, 'name': name}
                sequence_set.append(sample)
        self.samples = sequence_set

        self.init_var()

        pass

    def init_var(self):
        torch.set_default_tensor_type(torch.FloatTensor)
        from utils.homographies import sample_homography_np as sample_homography
        from utils.utils import inv_warp_image
        from utils.utils import compute_valid_mask
        from utils.photometric import ImgAugTransform, customizedTransform
        from utils.utils import inv_warp_image, inv_warp_image_batch, warp_points
        
        self.sample_homography = sample_homography
        self.inv_warp_image = inv_warp_image
        self.inv_warp_image_batch = inv_warp_image_batch
        self.compute_valid_mask = compute_valid_mask
        self.ImgAugTransform = ImgAugTransform
        self.customizedTransform = customizedTransform
        self.warp_points = warp_points

        self.enable_photo_train = self.config['augmentation']['photometric']['enable']
        self.enable_homo_train = self.config['augmentation']['homographic']['enable']

        self.enable_homo_val = False
        self.enable_photo_val = False

        self.cell_size = 8
        if self.config['preprocessing']['resize']:
            self.sizer = self.config['preprocessing']['resize']

        self.gaussian_label = False
        if self.config['gaussian_label']['enable']:
            self.gaussian_label = True
            y, x = self.sizer
            # self.params_transform = {'crop_size_y': y, 'crop_size_x': x, 'stride': 1, 'sigma': self.config['gaussian_label']['sigma']}
        pass

    def putGaussianMaps(self, center, accumulate_confid_map):
        crop_size_y = self.params_transform['crop_size_y']
        crop_size_x = self.params_transform['crop_size_x']
        stride = self.params_transform['stride']
        sigma = self.params_transform['sigma']

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

    def get_img_from_sample(self, sample):
        return sample['image'] # path

    def format_sample(self, sample):
        return sample

    def __getitem__(self, index):
        '''

        :param index:
        :return:
            image: tensor (H, W, channel=1)
        '''
        def _read_image(path):
            cell = 8
            input_image = cv2.imread(path)
            # print(f"path: {path}, image: {image}")
            # print(f"path: {path}, image: {input_image.shape}")
            input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
                                     interpolation=cv2.INTER_AREA)
            H, W = input_image.shape[0], input_image.shape[1]
            # H = H//cell*cell
            # W = W//cell*cell
            # input_image = input_image[:H,:W,:]
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

            input_image = input_image.astype('float32') / 255.0
            return input_image

        def _preprocess(image):
            if self.transforms is not None:
                image = self.transforms(image)
            return image

        def get_labels_gaussian(pnts, subpixel=False):
            heatmaps = np.zeros((H, W))
            if subpixel:
                print("pnt: ", pnts.shape)
                for center in pnts:
                    heatmaps = self.putGaussianMaps(center, heatmaps)
            else:
                aug_par = {'photometric': {}}
                aug_par['photometric']['enable'] = True
                aug_par['photometric']['params'] = self.config['gaussian_label']['params']
                augmentation = self.ImgAugTransform(**aug_par)
                # get label_2D
                labels = points_to_2D(pnts, H, W)
                labels = labels[:,:,np.newaxis]
                heatmaps = augmentation(labels)

            # warped_labels_gaussian = torch.tensor(heatmaps).float().view(-1, H, W)
            warped_labels_gaussian = torch.tensor(heatmaps).type(torch.FloatTensor).view(-1, H, W)
            warped_labels_gaussian[warped_labels_gaussian>1.] = 1.
            return warped_labels_gaussian


        from datasets.data_tools import np_to_tensor

        # def np_to_tensor(img, H, W):
        #     img = torch.tensor(img).type(torch.FloatTensor).view(-1, H, W)
        #     return img


        from datasets.data_tools import warpLabels

        def imgPhotometric(img):
            """

            :param img:
                numpy (H, W)
            :return:
            """
            augmentation = self.ImgAugTransform(**self.config['augmentation'])
            img = img[:,:,np.newaxis]
            img = augmentation(img)
            cusAug = self.customizedTransform()
            img = cusAug(img, **self.config['augmentation'])
            return img

        def points_to_2D(pnts, H, W):
            labels = np.zeros((H, W))
            pnts = pnts.astype(int)
            labels[pnts[:, 1], pnts[:, 0]] = 1
            return labels
            

        to_floatTensor = lambda x: torch.tensor(x).type(torch.FloatTensor)

        from numpy.linalg import inv
        sample = self.samples[index]
        sample = self.format_sample(sample)
        input  = {}
        input.update(sample)
        # image
        # img_o = _read_image(self.get_img_from_sample(sample))
        img_o = _read_image(sample['image'])
        H, W = img_o.shape[0], img_o.shape[1]
        # print(f"image: {image.shape}")
        img_aug = img_o.copy()
        if (self.enable_photo_train == True and self.action == 'train') or (self.enable_photo_val and self.action == 'val'):
            img_aug = imgPhotometric(img_o) # numpy array (H, W, 1)


        # img_aug = _preprocess(img_aug[:,:,np.newaxis])
        img_aug = torch.tensor(img_aug, dtype=torch.float32).view(-1, H, W)

        valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=torch.eye(3))
        input.update({'image': img_aug})
        input.update({'valid_mask': valid_mask})

        if self.config['homography_adaptation']['enable']:
            # img_aug = torch.tensor(img_aug)
            homoAdapt_iter = self.config['homography_adaptation']['num']
            homographies = np.stack([self.sample_homography(np.array([2, 2]), shift=-1,
                           **self.config['homography_adaptation']['homographies']['params'])
                           for i in range(homoAdapt_iter)])
            ##### use inverse from the sample homography
            homographies = np.stack([inv(homography) for homography in homographies])
            homographies[0,:,:] = np.identity(3)
            # homographies_id = np.stack([homographies_id, homographies])[:-1,...]

            ######

            homographies = torch.tensor(homographies, dtype=torch.float32)
            inv_homographies = torch.stack([torch.inverse(homographies[i, :, :]) for i in range(homoAdapt_iter)])

            # images
            warped_img = self.inv_warp_image_batch(img_aug.squeeze().repeat(homoAdapt_iter,1,1,1), inv_homographies, mode='bilinear').unsqueeze(0)
            warped_img = warped_img.squeeze()
            # masks
            valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=inv_homographies,
                                                 erosion_radius=self.config['augmentation']['homographic'][
                                                     'valid_border_margin'])
            input.update({'image': warped_img, 'valid_mask': valid_mask, 'image_2D':img_aug})
            input.update({'homographies': homographies, 'inv_homographies': inv_homographies})

        # laebls
        if self.labels:
            pnts = np.load(sample['points'])['pts']
            # pnts = pnts.astype(int)
            # labels = np.zeros_like(img_o)
            # labels[pnts[:, 1], pnts[:, 0]] = 1
            labels = points_to_2D(pnts, H, W)
            labels_2D = to_floatTensor(labels[np.newaxis,:,:])
            input.update({'labels_2D': labels_2D})

            ## residual
            labels_res = torch.zeros((2, H, W)).type(torch.FloatTensor)
            input.update({'labels_res': labels_res})

            if (self.enable_homo_train == True and self.action == 'train') or (self.enable_homo_val and self.action == 'val'):
                homography = self.sample_homography(np.array([2, 2]), shift=-1,
                                                    **self.config['augmentation']['homographic']['params'])

                ##### use inverse from the sample homography
                homography = inv(homography)
                ######

                inv_homography = inv(homography)
                inv_homography = torch.tensor(inv_homography).to(torch.float32)
                homography = torch.tensor(homography).to(torch.float32)
                #                 img = torch.from_numpy(img)
                warped_img = self.inv_warp_image(img_aug.squeeze(), inv_homography, mode='bilinear').unsqueeze(0)
                # warped_img = warped_img.squeeze().numpy()
                # warped_img = warped_img[:,:,np.newaxis]

                ##### check: add photometric #####

                # labels = torch.from_numpy(labels)
                # warped_labels = self.inv_warp_image(labels.squeeze(), inv_homography, mode='nearest').unsqueeze(0)
                ##### check #####
                warped_set = warpLabels(pnts, H, W, homography)
                warped_labels = warped_set['labels']
                # if self.transform is not None:
                    # warped_img = self.transform(warped_img)
                valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=inv_homography,
                            erosion_radius=self.config['augmentation']['homographic']['valid_border_margin'])

                input.update({'image': warped_img, 'labels_2D': warped_labels, 'valid_mask': valid_mask})


            if self.config['warped_pair']['enable']:
                homography = self.sample_homography(np.array([2, 2]), shift=-1,
                                           **self.config['warped_pair']['params'])

                ##### use inverse from the sample homography
                homography = np.linalg.inv(homography)
                #####
                inv_homography = np.linalg.inv(homography)

                homography = torch.tensor(homography).type(torch.FloatTensor)
                inv_homography = torch.tensor(inv_homography).type(torch.FloatTensor)

                # photometric augmentation from original image

                # warp original image
                warped_img = torch.tensor(img_o, dtype=torch.float32)
                warped_img = self.inv_warp_image(warped_img.squeeze(), inv_homography, mode='bilinear').unsqueeze(0) 
                if (self.enable_photo_train == True and self.action == 'train') or (self.enable_photo_val and self.action == 'val'):
                    warped_img = imgPhotometric(warped_img.numpy().squeeze()) # numpy array (H, W, 1)
                    warped_img = torch.tensor(warped_img, dtype=torch.float32)
                    pass
                warped_img = warped_img.view(-1, H, W)

                # warped_labels = warpLabels(pnts, H, W, homography)
                warped_set = warpLabels(pnts, H, W, homography, bilinear=True)
                warped_labels = warped_set['labels']
                warped_res = warped_set['res']
                warped_res = warped_res.transpose(1,2).transpose(0,1)
                # print("warped_res: ", warped_res.shape)
                if self.gaussian_label:
                    # print("do gaussian labels!")
                    # warped_labels_gaussian = get_labels_gaussian(warped_set['warped_pnts'].numpy())
                    from utils.var_dim import squeezeToNumpy
                    # warped_labels_bi = self.inv_warp_image(labels_2D.squeeze(), inv_homography, mode='nearest').unsqueeze(0) # bilinear, nearest
                    warped_labels_bi = warped_set['labels_bi']
                    warped_labels_gaussian = self.gaussian_blur(squeezeToNumpy(warped_labels_bi))
                    warped_labels_gaussian = np_to_tensor(warped_labels_gaussian, H, W)
                    input['warped_labels_gaussian'] = warped_labels_gaussian
                    input.update({'warped_labels_bi': warped_labels_bi})

                input.update({'warped_img': warped_img, 'warped_labels': warped_labels, 'warped_res': warped_res})

                # print('erosion_radius', self.config['warped_pair']['valid_border_margin'])
                valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=inv_homography,
                            erosion_radius=self.config['warped_pair']['valid_border_margin'])  # can set to other value
                input.update({'warped_valid_mask': valid_mask})
                input.update({'homographies': homography, 'inv_homographies': inv_homography})

            # labels = self.labels2Dto3D(self.cell_size, labels)
            # labels = torch.from_numpy(labels[np.newaxis,:,:])
            # input.update({'labels': labels})

            if self.gaussian_label:
                # warped_labels_gaussian = get_labels_gaussian(pnts)
                labels_gaussian = self.gaussian_blur(squeezeToNumpy(labels_2D))
                labels_gaussian = np_to_tensor(labels_gaussian, H, W)
                input['labels_2D_gaussian'] = labels_gaussian

        name = sample['name']
        to_numpy = False
        if to_numpy:
            image = np.array(img)

        input.update({'name': name, 'scene_name': "./"}) # dummy scene name
        return input

    def __len__(self):
        return len(self.samples)

    ## util functions
    def gaussian_blur(self, image):
        """
        image: np [H, W]
        return:
            blurred_image: np [H, W]
        """
        aug_par = {'photometric': {}}
        aug_par['photometric']['enable'] = True
        aug_par['photometric']['params'] = self.config['gaussian_label']['params']
        augmentation = self.ImgAugTransform(**aug_par)
        # get label_2D
        # labels = points_to_2D(pnts, H, W)
        image = image[:,:,np.newaxis]
        heatmaps = augmentation(image)
        return heatmaps.squeeze()

