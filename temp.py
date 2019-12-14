## magic point evaluation
def _metrics(self, outputs, inputs, **config):
    pred = outputs['pred']
    labels = inputs['keypoint_map']
    
    precision = tf.reduce_sum(pred * labels) / tf.reduce_sum(pred)
    recall = tf.reduce_sum(pred * labels) / tf.reduce_sum(labels)
    
    return {'precision': precision, 'recall': recall}




## load kitti
if 'path' in dict.keys():
    path_datasets = config['data']['path']
from datasets.sequence_folders import SequenceFolder
import custom_transforms

normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
train_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
train_set = SequenceFolder(
    path_datasets,
    transform=train_transform,
    train=True,
    seed=1,
)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=config['model']['batch_size'], shuffle=True,
    pin_memory=True)
# val_set =

# testing
'''
import matplotlib.pyplot as plt
sample = train_set[0]
plt.imshow(sample[0])
'''


def saveImg(img, filename):
    import cv2
    cv2.imwrite(filename, img)

def pltImshow(img):
    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.show()


sample = train_set[0]
print("Original image")
# print(sample[0])
img = sample[0].numpy()
img = np.transpose(img, (1, 2, 0))
filename = "test.png"
saveImg(img, filename)
# print("")
# cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# cv2.waitKey(0)


for epoch in range(2):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(train_loader):
        # get the inputs
        # inputs = torch.from_numpy(tgt_img)
        scale = 8
        img = tgt_img[:tgt_img.shape[0] // scale, :tgt_img.shape[1] // scale]
        print(img.shape)
        # np.repeat(tgt_img,
        
        # labels =
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = npet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            
            
#########
semi = semi.data.cpu().numpy().squeeze()
    # --- Process points.
    dense = np.exp(semi) # Softmax.
    dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
    # Remove dustbin.
    nodust = dense[:-1, :, :]
    # Reshape to get full resolution heatmap.
    Hc = int(H / self.cell)
    Wc = int(W / self.cell)
    nodust = nodust.transpose(1, 2, 0)
    heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
    heatmap = np.transpose(heatmap, [0, 2, 1, 3])
    heatmap = np.reshape(heatmap, [Hc*self.cell, Wc*self.cell])
    xs, ys = np.where(heatmap >= self.conf_thresh) # Confidence threshold.
    
    
#########
Hc, Wc = semi.shape[2], semi.shape[3]
cell_size = 8

semi.exponential_()
d = semi.sum(dim=1)+0.00001
d = d.view(d.shape[0], 1, d.shape[1], d.shape[2])
semi = semi/d  # how to /(64,15,20)

semi = semi[:,:-1,:,:]
semi.transpose_(1,2)
semi.transpose_(2,3)
semi = semi.view(-1,Hc,Wc,cell_size, cell_size)
semi.transpose_(2,3)
semi = semi.contiguous()
semi = semi.view(-1,1,Hc*cell_size, Wc*cell_size)

#############
from utils.utils import saveImg
img = data['image']
labels = np.zeros((img.shape[0],img.shape[1],3))
pnts = true_warped_keypoints.astype(int)
labels[pnts[:, 1], pnts[:, 0],0] = 1
pnts = warped_keypoints.astype(int)
labels[pnts[:, 1], pnts[:, 0],1] = 1
labels[:,:,2] = img
saveImg(labels*255, 'test.png')

#############
from utils.utils import saveImg
img = data['image']
labels = np.zeros((img.shape[0],img.shape[1],3))
pnts = keypoints.astype(int)
labels[pnts[:, 1], pnts[:, 0],0] = 1
pnts = warped_keypoints.astype(int)
labels[pnts[:, 1], pnts[:, 0],1] = 1
labels[:,:,2] = img
saveImg(labels*255, 'test.png')

############
# def inv_warp_image_batch(img, mat_homo_inv, device='cpu'):
from utils.utils import inv_warp_image_batch
from numpy.linalg import inv
H = data['homography']
inv_warp_image_batch(torch.tensor(img), torch.tensor(inv(H)))

###########
img1 = img.cpu().numpy()
img1 = img1[0,0,:,:].squeeze()
pltImshow(img1)

img2 = img_warp.cpu().numpy()
img2 = img2[0,0,:,:]
pltImshow(img2)

a = sumto2D(mask).cpu().numpy()
pltImshow(a[0,:,:])

#############

h = torch.tensor(np.identity(3)[np.newaxis,:,:]).type(torch.float32).to(device)
coor_cells = torch.stack(torch.meshgrid(torch.arange(Hc), torch.arange(Wc)), dim=2)
coor_cells = coor_cells.type(torch.FloatTensor).to(device)
coor_cells = coor_cells * cell_size + cell_size // 2
## coord_cells is now a grid containing the coordinates of the Hc x Wc
## center pixels of the 8x8 cells of the image

coor_cells = coor_cells.view([-1, Hc, Wc, 1, 1, 2])
# warped_coor_cells = warp_points(coor_cells.view([-1, 2]), homographies, device)
coor_cells = normPts(coor_cells.view([-1, 2]), shape)
coor_cells = torch.stack((coor_cells[:, 1], coor_cells[:, 0]), dim=1)  # (y, x) to (x, y)
warped_coor_cells = warp_points(coor_cells, h[np.newaxis, :,:], device)
warped_coor_cells = torch.stack((warped_coor_cells[:, :, 1], warped_coor_cells[:, :, 0]), dim=2)  # (batch, x, y) to (batch, y, x)
warped_coor_cells = denormPts(warped_coor_cells, shape)
warped_coor_cells = warped_coor_cells.view([-1, 1, 1, Hc, Wc, 2])

##############

coor = coor_cells.cpu().numpy().reshape([-1, 2]).astype(int)
img = np.zeros((240, 320))
img[coor[:,0], coor[:,1]] = 1
from utils.utils import pltImshow
pltImshow(img)

warped_coor = warped_coor_cells.cpu().numpy().reshape([-1, 2]).astype(int)
img_warp = np.zeros((240, 320))
img_warp[warped_coor[:,0], warped_coor[:,1]] = 1
from utils.utils import pltImshow
pltImshow(img_warp)


#########
n = negative_dist
n_2 = n.view(-1,30,40)
n_1 = n.view(-1,1200)
n_1 = n_1.sum(dim=1)
idx = n_1.max(0)[1]
a = n_1[idx]


mask_o = mask.sum(dim=1).sum(dim=1)
img = (mask_o).cpu().numpy()[0,:,:][:,:,np.newaxis]
img = img/img.max()*255
saveImg(img, 'mask_1.png')

mask_o = mask.sum(dim=3).sum(dim=3)
img = (mask_o).cpu().numpy()[0,:,:][:,:,np.newaxis]
img = img/img.max()*255
saveImg(img, 'mask_2.png')

mask_v =  mask_valid[0,:,:][:,:,np.newaxis]
saveImg(mask_v*255, 'mask_valid.png')

mask_v =  mask_valid[0,:,:].cpu().numpy()[:,:,np.newaxis]
saveImg(mask_v*255, 'mask_valid.png')

#########
from utils.utils import pltImshow
pltImshow(mask_warp[0,0,:,:].cpu().numpy())