"""copy labels out of images (step 2)
"""

import subprocess
from glob import glob
import os

source_folder = 'magicpoint_synth20_homoAdapt100_kitti_h384'
target_folder = f"{source_folder}_labels"
base_path = '/data/kitti'
middle_path = 'predictions/'
final_folder = 'train'
folders = glob(f'{base_path}/{source_folder}/{middle_path}/{final_folder}/*') 

# print(f"folders: {folders}")
for f in folders:
    if os.path.isdir(f) == False:
        continue
    f_target = str(f).replace(source_folder, target_folder)
    command = f'rsync -rh {f}/*.npz {f_target}'
    print(f"command: {command}")
    subprocess.run(f"{command}", shell=True, check=True)

print(f"total folders: {len(folders)}")
