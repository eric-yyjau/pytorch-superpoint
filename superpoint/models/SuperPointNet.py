import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_

# class SuperPointNet(torch.nn.Module):
#   """ Pytorch definition of SuperPoint Network. """

#   def vgg_block(self, filters_in, filters_out, kernel_size, padding, name, batch_normalization=True, relu=True, kernel_reg=0., **params):
#     modules = [nn.Conv2d(filters_in, filters_out, kernel_size=kernel_size, stride=1, padding=padding)]
#     if batch_normalization:
#         modules.append(nn.BatchNorm2d(filters_out))
#     if relu:
#         modules.append(self.relu)
#     else:
#         print('NO RELU!!!!!!!!!!!!!!!')
#     return nn.Sequential(*modules)

#     # x2 = conv_layer(inputs) ## HOW TO IMPLEMENT KERNEL REG?
#     # if relu:
#     #     x2 = self.relu(x2)
#     # else:
#     #     print('NO RELU!!!!!!!!!!!!!!!')
#     # if batch_normalization:
#     #     x2 = nn.BatchNorm2d(filters_out)(x2)
#     # return x2

#   def __init__(self):
#     super(SuperPointNet, self).__init__()
#     self.relu = nn.ReLU(inplace=True)
#     self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#     params_conv = {'batch_normalization': False, 'relu': True, 'kernel_reg': 0.}

#     self.conv1_1 = self.vgg_block(1, 64, 3, 1, 'conv1_1', **params_conv)
#     self.conv1_2 = self.vgg_block(64, 64, 3, 1, 'conv1_2', **params_conv)

#     self.conv2_1 = self.vgg_block(64, 64, 3, 1, 'conv2_1', **params_conv)
#     self.conv2_2 = self.vgg_block(64, 64, 3, 1, 'conv2_2', **params_conv)

#     self.conv3_1 = self.vgg_block(64, 128, 3, 1, 'conv3_1', **params_conv)
#     self.conv3_2 = self.vgg_block(128, 128, 3, 1, 'conv3_2', **params_conv)

#     self.conv4_1 = self.vgg_block(128, 128, 3, 1, 'conv4_1', **params_conv)
#     self.conv4_2 = self.vgg_block(128, 128, 3, 1, 'conv4_2', **params_conv)

#     # Detector Head.
#     self.det_conv1_1 = self.vgg_block(128, 256, 3, 1, 'det_conv1_1', **params_conv)
#     self.det_conv1_2 = self.vgg_block(256, 65, 1, 0, 'det_conv1_2', batch_normalization=False, relu=False)

#     # Descriptor Head.
#     self.des_conv1_1 = self.vgg_block(128, 256, 3, 1, 'des_conv1_1', **params_conv)
#     self.des_conv1_2 = self.vgg_block(256, 256, 1, 0, 'des_conv1_2', batch_normalization=False, relu=False)
#     # desc = nn.functional.normalize(desc, p=2, dim=1)

#   def forward(self, x):
#     """ Forward pass that jointly computes unprocessed point and descriptor
#     tensors.
#     Input
#       x: Image pytorch tensor shaped N x 1 x H x W.
#     Output
#       semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
#       desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
#     """
#     # Shared Encoder.
#     # print(x.size(), '!!!!!!!!!!!')
#     x = self.pool(self.conv1_2(self.conv1_1(x)))
#     # print(x.size(), self.conv1_1(x).size(), '!!!!!!!!!!!')
#     x = self.pool(self.conv2_2(self.conv2_1(x)))
#     x = self.pool(self.conv3_2(self.conv3_1(x)))
#     x = self.conv4_2(self.conv4_1(x))
#     # Detector Head.
#     semi = self.det_conv1_2(self.det_conv1_1(x))
#     # print(semi.size(), '!!!!!!!!!!!')
#     # Descriptor Head.
#     desc = self.des_conv1_2(self.des_conv1_1(x))
#     desc = nn.functional.normalize(desc, p=2, dim=1)

#     return semi, desc

# ###############################



class SuperPointNet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self):
    super(SuperPointNet, self).__init__()
    def predict_flow(in_planes):
        return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

    def convrelu(in_channels, out_channels, kernel, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.ReLU(inplace=True),
        )

    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    self.unpool = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    det_h = 65
    gn = 64
    useGn = False
    self.reBn = True
    if self.reBn:
        print ("model structure: relu - bn - conv")
    else:
        print ("model structure: bn - relu - conv")


    if useGn:
        print ("apply group norm!")
    else:
        print ("apply batch norm!")


    # Shared Encoder.
    self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
    self.bn1a = nn.GroupNorm(gn, c1) if useGn else nn.BatchNorm2d(c1)
    self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    self.bn1b = nn.GroupNorm(gn, c1) if useGn else nn.BatchNorm2d(c1)
    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.bn2a = nn.GroupNorm(gn, c2) if useGn else nn.BatchNorm2d(c2)
    self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    self.bn2b = nn.GroupNorm(gn, c2) if useGn else nn.BatchNorm2d(c2)
    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    self.bn3a = nn.GroupNorm(gn, c3) if useGn else nn.BatchNorm2d(c3)
    self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    self.bn3b = nn.GroupNorm(gn, c3) if useGn else nn.BatchNorm2d(c3)
    self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.bn4a = nn.GroupNorm(gn, c4) if useGn else nn.BatchNorm2d(c4)
    self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    self.bn4b = nn.GroupNorm(gn, c4) if useGn else nn.BatchNorm2d(c4)
    # Detector Head.
    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.bnPa = nn.GroupNorm(gn, c5) if useGn else nn.BatchNorm2d(c5)
    self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
    self.bnPb = nn.GroupNorm(det_h, det_h) if useGn else nn.BatchNorm2d(65)
    # Descriptor Head.
    self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.bnDa = nn.GroupNorm(gn, c5) if useGn else nn.BatchNorm2d(c5)
    self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
    self.bnDb = nn.GroupNorm(gn, d1) if useGn else nn.BatchNorm2d(d1)
    # subpixel head
    # self.predict_flow4 = predict_flow(c4)
    # self.predict_flow3 = predict_flow(c3 + 2)
    # self.predict_flow2 = predict_flow(c2 + 2)
    # self.predict_flow1 = predict_flow(c1 + 2)
    # self.conv_up3 = convrelu(512 + 1024, 512, 3, 1)
    # self.conv_up2 = convrelu(512 + 512, 512, 3, 1)
    # self.conv_up1 = convrelu(256 + 512, 256, 3, 1)
    # self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

  def forward(self, x, subpixel=False):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """

    # Let's stick to this version: first BN, then relu
    if self.reBn:
        # Shared Encoder.
        x = self.relu(self.bn1a(self.conv1a(x)))
        conv1 = self.relu(self.bn1b(self.conv1b(x)))
        x, ind1 = self.pool(conv1)
        x = self.relu(self.bn2a(self.conv2a(x)))
        conv2 = self.relu(self.bn2b(self.conv2b(x)))
        x, ind2 = self.pool(conv2)
        x = self.relu(self.bn3a(self.conv3a(x)))
        conv3 = self.relu(self.bn3b(self.conv3b(x)))
        x, ind3 = self.pool(conv3)
        x = self.relu(self.bn4a(self.conv4a(x)))
        x = self.relu(self.bn4b(self.conv4b(x)))
        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x)))
        semi = self.bnPb(self.convPb(cPa))
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x)))
        desc = self.bnDb(self.convDb(cDa))
    else:
        # Shared Encoder.
        x = self.bn1a(self.relu(self.conv1a(x)))
        x = self.bn1b(self.relu(self.conv1b(x)))
        x = self.pool(x)
        x = self.bn2a(self.relu(self.conv2a(x)))
        x = self.bn2b(self.relu(self.conv2b(x)))
        x = self.pool(x)
        x = self.bn3a(self.relu(self.conv3a(x)))
        x = self.bn3b(self.relu(self.conv3b(x)))
        x = self.pool(x)
        x = self.bn4a(self.relu(self.conv4a(x)))
        x = self.bn4b(self.relu(self.conv4b(x)))
        # Detector Head.
        cPa = self.bnPa(self.relu(self.convPa(x)))
        semi = self.bnPb(self.convPb(cPa))
        # Descriptor Head.
        cDa = self.bnDa(self.relu(self.convDa(x)))
        desc = self.bnDb(self.convDb(cDa))

    dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    output = {'semi': semi, 'desc': desc}

    if subpixel:
        pass
        # subPred = self.predict_flow4(x)
        # subPred = self.unpool(subPred, ind3)
        # concat3 = torch.cat((subPred,conv3),1) 
        # subPred = self.predict_flow3(concat3)
        # subPred = self.unpool(subPred, ind2)
        # concat2 = torch.cat((subPred,conv2),1) 
        # subPred = self.predict_flow2(concat2)
        # subPred = self.unpool(subPred, ind1)
        # concat1 = torch.cat((subPred,conv1),1) 
        # subPred = self.predict_flow1(concat1)
        # print("subPred: ", subPred.shape)
        # return semi, desc, subPred


    return output

def forward_original(self, x):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
    # Shared Encoder.
    x = self.relu(self.conv1a(x))
    x = self.relu(self.conv1b(x))
    x = self.pool(x)
    x = self.relu(self.conv2a(x))
    x = self.relu(self.conv2b(x))
    x = self.pool(x)
    x = self.relu(self.conv3a(x))
    x = self.relu(self.conv3b(x))
    x = self.pool(x)
    x = self.relu(self.conv4a(x))
    x = self.relu(self.conv4b(x))
    # Detector Head.
    cPa = self.relu(self.convPa(x))
    semi = self.convPb(cPa)

    # Descriptor Head.
    cDa = self.relu(self.convDa(x))
    desc = self.convDb(cDa)
    dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    return semi, desc





###############################

def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )




if __name__ == '__main__':

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = SuperPointNet()
  model = model.to(device)

  # check keras-like model summary using torchsummary
  from torchsummary import summary
  summary(model, input_size=(1, 224, 224))
