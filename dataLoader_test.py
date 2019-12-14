#!/usr/bin/env python
# coding: utf-8

# # test data loader

# In[1]:


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)



def loadConfig(filename):
    import yaml
    with open(filename, 'r') as f:
        config = yaml.load(f)
    return config

if __name__ == '__main__':
    # load config
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


    filename = 'configs/magicpoint_kitti_train.yaml'
    config = loadConfig(filename)


    # In[2]:


    from utils.loader import dataLoader
    task = 'kitti'
    data = dataLoader(config, dataset=task, warp_input=True)


    # In[ ]:


    # data
    train_loader, val_loader = data['train_loader'], data['val_loader']

    logging.info('== train split size %d in %d batches, val split size %d in %d batches'%\
            (len(train_loader)*config['model']['batch_size'], len(train_loader),
             len(val_loader)*config['model']['batch_size'], len(val_loader)))

    for i, sample in enumerate(train_loader):
        if i% 100 == 0:
            print(list(sample))
            break


# In[ ]:


# list(val_loader)


# In[ ]:



sys.executable


# In[ ]:





# In[ ]:


import mmdet

