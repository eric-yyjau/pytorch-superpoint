"""tools to print object shape or type

"""


# from utils.print_tool import print_config
def print_config(config, file=None):
    print('='*10, ' important config: ', '='*10, file=file)
    for item in list(config):
        print(item, ": ", config[item], file=file)
    
    print('='*32)

# from utils.print_tool import print_statistics
def print_statistics(arr, name='array'):
    print(f"{name}: lenght={len(arr)}, mean={arr.mean()}, max={arr.max()}, min={arr.min()}")


# from utils.print_tool import print_dict_attr
def print_dict_attr(dictionary, attr=None, file=None):
    for item in list(dictionary):
        d = dictionary[item]
        if attr == None:
            print(item, ": ", d, file=file)
        else:
            if hasattr(d, attr):
                print(item, ": ", getattr(d, attr), file=file)
            else:
                print(item, ": ", len(d), file=file)

import logging
# from utils.print_tool import datasize
def datasize(train_loader, config, tag='train'):
    logging.info('== %s split size %d in %d batches'%\
    (tag, len(train_loader)*config['model']['batch_size'], len(train_loader)))
    pass

# from utils.print_tool import plot_grad_flow
def plot_grad_flow(named_parameters):
    import matplotlib.pyplot as plt
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")