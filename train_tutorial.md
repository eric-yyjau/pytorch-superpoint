# Training tutorial
- This tutorial can walk through how to load pretrained model and run the training script.

## environment
```
conda create --name py36-torch python=3.6
pip install -r requirements.txt
```

## Required package
- roi_pool: 
	- install the package: https://github.com/open-mmlab/mmdetection/blob/master/INSTALL.md
	- https://github.com/open-mmlab/mmdetection/tree/master/mmdet/ops/roi_pool
	- put 'roi_pool_cuda.cpython-36m-x86_64-linux-gnu.so' in 'utils/roi_pool/'

## Required settings
- check the config file
	- check the model path
		- set in 'pretrained'
		- set 'retrain' to false
		- (set 'reset_iter' to false)
	- check the data path: (can use hyperlink: ln -s)
		- put data in the path 'dataset'
            - kitti: 'datasets/kitti_wVal'(default in setting.py)
		- the folder name should match the one listed on ['data']['dataset']
		- put the files in 'datasets/kitti_split' to 'datasets/kitti_wVal'
		```
		cp datasets/kitti_split/train.txt datasets/kitti_wVal/
		cp datasets/kitti_split/val.txt datasets/kitti_wVal/
		```
	- check the labels path
		- check the path: ['data']['labels']
            - kitti: logs/magicpoint_synth20_homoAdapt100_kitti_h384/predictions (default)
		- the path uses base path EXPER_PATH (listed in settings.py)

## Run the code
```
python train4.py <train task> <config file> <export folder>
python train4.py train_joint --debug configs/superpoint_kitti_train_heatmap.yaml superpoint_kitti --eval

```

## Related files
- train4.py: training script (load 'train_model_frontend')
- train_model_frontend.py: class for training
- configs/superpoint_coco_train.yaml: path and parameter settings

## Code logic

## testing log
- 2019/7/11
    - test python train4.py train_joint --debug configs/superpoint_kitti_train_heatmap.yaml superpoint_kitti --eval
    - environment: python: 3.6, pytorch: 1.1, cuda:10
