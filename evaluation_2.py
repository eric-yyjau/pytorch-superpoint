"""evaluation class (in progress)
please refer to evaluation.py
"""

import numpy as np
from evaluations.descriptor_evaluation import compute_homography
from evaluations.detector_evaluation import compute_repeatability
import cv2
import matplotlib.pyplot as plt
import logging
import os
from tqdm import tqdm
from utils.draw import plot_imgs

# from utils.draw import draw_matches_cv
# from utils.utils import find_files_with_ext
# from utils.var_dim import to3dim


def evaluate(args, **options):
    # path = '/home/yoyee/Documents/SuperPoint/superpoint/logs/outputs/superpoint_coco/'


    # for i in range(2):
    #     f = files[i]

    from numpy.linalg import norm
    from utils.draw import draw_keypoints
    from utils.utils import saveImg

    from evaluate_frontend import evaluate_frontend
    eva_frontend = evaluate_frontend(args)

    eva_frontend.run()

    eva_frontend.print_results()

    eva_frontend.save_results()



if __name__ == '__main__':
    import argparse


    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-o', '--outputImg', action='store_true')
    parser.add_argument('-r', '--repeatibility', action='store_true')
    parser.add_argument('-homo', '--homography', action='store_true')
    parser.add_argument('-plm', '--plotMatching', action='store_true')
    args = parser.parse_args()
    evaluate(args)
