# Training tutorial
- This tutorial can walk through how to load pretrained model and run the training script.

## Required package
- roi_pool: 
	- https://github.com/open-mmlab/mmdetection/tree/master/mmdet/ops/roi_pool
	- put 'roi_pool_cuda.cpython-36m-x86_64-linux-gnu.so' in 'utils/roi_pool/'

## load model
- module test:
```
ipython
run models/SuperPointNet_gauss2.py
```

- refer to 'deepSfm/Train_model_heatmap.py':loadModel(self):
	- https://github.com/eric-yyjau/deepSfm/blob/c74ec7bdd3c191a80ecbcc0279f9283f7ba934cb/Train_model_heatmap.py#L124