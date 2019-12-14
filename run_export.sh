## quick script to run export and evaluation

export_folder='superpoint_coco_heat2_0_170k_nms4_det0.015'
# export_folder='superpoint_kitti_heat2_0'
echo $export_folder
# python3 export.py export_descriptor configs/magicpoint_repeatability_heatmap.yaml $export_folder
python3 evaluation.py /home/yyjau/Documents/deepSfm_test/logs/$export_folder/predictions --repeatibility --homography --outputImg --plotMatching

