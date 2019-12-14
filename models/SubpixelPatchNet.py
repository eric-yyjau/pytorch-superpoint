import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from models.unet_parts import *

# from models.SubpixelNet import SubpixelNet
class SubpixelPatchNet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self, subpixel_channel=1, patch_size=32):
    super(SubpixelPatchNet, self).__init__()
    self.relu = torch.nn.ReLU(inplace=True)
    print("patch size: ", patch_size)
    
    c1, c2, c3= 64, 128, 256
    f1 = 512
    out = 2
    # self.inc = inconv(1, c1)
    # self.down1 = down(subpixel_channel, c1)
    # self.down2 = down(c1, c2)
    # self.down3 = down(c2, c3)

    self.conv1a = torch.nn.Conv2d(subpixel_channel, c1, kernel_size=3, stride=2, padding=1)
    self.bn1a =  nn.BatchNorm2d(c1)
    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
    self.bn2a =  nn.BatchNorm2d(c2)
    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
    self.bn3a =  nn.BatchNorm2d(c3)

    flatten = int(patch_size/8)**2
    self.fc1   = torch.nn.Linear(c3*flatten, f1)
    self.bn1   = nn.BatchNorm1d(f1)
    self.fc2   = torch.nn.Linear(f1, out)


    # self.up1 = up(c4+c3, c2)
    # self.up2 = up(c2+c2, c1)
    # self.up3 = up(c1+c1, c1)
    # self.outc = outconv(c1, subpixel_channel)


  @staticmethod
  def soft_argmax_2d(patches):
    """
    params:
        patches: (B, N, H, W)
    return:
        coor: (B, N, 2)  (x, y)

    """
    import torchgeometry as tgm
    m = tgm.contrib.SpatialSoftArgmax2d()
    coords = m(patches)  # 1x4x2
    return coords

  def forward(self, x, subpixel=False):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
    # Let's stick to this version: first BN, then relu
    # x = self.down1(x)
    # x = self.down2(x)
    # x = self.down3(x)

    x = self.bn1a(self.relu(self.conv1a(x)))
    x = self.bn2a(self.relu(self.conv2a(x)))
    x = self.bn3a(self.relu(self.conv3a(x)))

    x = x.view(x.size(0), -1)
    x = self.relu(self.bn1(self.fc1(x)))
    # x = self.relu(self.fc1(x))
    x = self.fc2(x)

    return x



if __name__ == '__main__':

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = SubpixelPatchNet()
  model = model.to(device)


  # check keras-like model summary using torchsummary
  from torchsummary import summary
  summary(model, input_size=(1, 32, 32))
