
export_folder='superpoint_coco_heat2_0_170k_nms4_det0.015'
# export_folder='superpoint_kitti_heat2_0'
echo $export_folder
# python3 export2.py export_descriptor configs/hpatches_rep/magicpoint_repeatability_heatmap.yaml $export_folder
# python3 export.py export_descriptor configs/magicpoint_repeatability_heatmap.yaml $export_folder
# python3 export.py export_descriptor_coco configs/magicpoint_repeatability.yaml $export_folder
# python3 evaluation.py /home/yyjau/Documents/deepSfm/logs/$export_folder/predictions --repeatibility --outputImg --homography --plotMatching
python3 evaluation.py /home/yyjau/Documents/deepSfm_test/logs/$export_folder/predictions --repeatibility --homography --outputImg --plotMatching



# export_folder='superpoint_kitti40_2_90k_hpatches_nms4_det0.001'
# echo $export_folder
# python3 export.py export_descriptor configs/magicpoint_repeatability_1.yaml $export_folder
# #python3 export.py export_descriptor_coco configs/magicpoint_repeatability.yaml $export_folder
# python3 evaluation.py /home/yoyee/Documents/deepSfm/logs/$export_folder/predictions --repeatibility --outputImg --homography --plotMatching

## export2.py
# export_folder='superpoint_coco_heat1_2_120k_hpatches_sub'
# echo $export_folder
# python3 export2.py export_descriptor configs/magicpoint_repeatability_heatmap.yaml $export_folder
# #python3 export.py export_descriptor_coco configs/magicpoint_repeatability.yaml $export_folder
# python3 evaluation.py /home/yoyee/Documents/deepSfm/logs/$export_folder/predictions --repeatibility --outputImg --homography --plotMatching

# export_folder='superpoint_spollo_v0'
# echo $export_folder
# python3 export2.py export_descriptor configs/hpatches_rep/magicpoint_repeatability_heatmap.yaml $export_folder
# python3 export2.py export_descriptor configs/magicpoint_repeatability_heatmap.yaml $export_folder
#python3 export.py export_descriptor_coco configs/magicpoint_repeatability.yaml $export_folder
# python3 evaluation.py /home/yyjau/Documents/deepSfm/logs/$export_folder/predictions --repeatibility --outputImg --homography --plotMatching

# export_folder='superpoint_coco_heat2_1_50k_hpatches_sub'
# echo $export_folder
# # python3 export2.py export_descriptor configs/hpatches_rep/magicpoint_repeatability_heatmap.yaml $export_folder
# python3 export2.py export_descriptor configs/hpatches_rep/magicpoint_repeatability_heatmap_2.yaml $export_folder
# #python3 export.py export_descriptor_coco configs/magicpoint_repeatability.yaml $export_folder
# python3 evaluation.py /home/yoyee/Documents/deepSfm/logs/$export_folder/predictions --repeatibility --outputImg --homography --plotMatching

# export_folder='superpoint_kitti_heat2_0_50k_hpatches_sub'
# echo $export_folder
# # python3 export2.py export_descriptor configs/hpatches_rep/magicpoint_repeatability_heatmap.yaml $export_folder
# python3 export2.py export_descriptor configs/hpatches_rep/magicpoint_repeatability_heatmap_3.yaml $export_folder
# #python3 export.py export_descriptor_coco configs/magicpoint_repeatability.yaml $export_folder
# python3 evaluation.py /home/yoyee/Documents/deepSfm/logs/$export_folder/predictions --repeatibility --outputImg --homography --plotMatching

# sift
# export_folder='sift_hpatches_3_subpixel'
# echo $export_folder
# python3 export_classical.py export_descriptor configs/classical_descriptors.yaml $export_folder
# python3 evaluation.py /home/yoyee/Documents/deepSfm/logs/$export_folder/predictions --repeatibility --outputImg --homography

