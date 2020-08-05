"""Script for evaluation
This is the evaluation script for image denoising project.

Author: You-Yi Jau, Yiqian Wang
Date: 2020/03/30
"""

import matplotlib
matplotlib.use('Agg') # solve error of tk

import numpy as np
from evaluations.descriptor_evaluation import compute_homography
from evaluations.detector_evaluation import compute_repeatability
import cv2
import matplotlib.pyplot as plt

import logging
import os
from tqdm import tqdm
from utils.draw import plot_imgs
from utils.logging import *

def draw_matches_cv(data, matches, plot_points=True):
    if plot_points:
        keypoints1 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints1']]
        keypoints2 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints2']]
    else:
        matches_pts = data['matches']
        keypoints1 = [cv2.KeyPoint(p[0], p[1], 1) for p in matches_pts]
        keypoints2 = [cv2.KeyPoint(p[2], p[3], 1) for p in matches_pts]
        print(f"matches_pts: {matches_pts}")
        # keypoints1, keypoints2 = [], []

    inliers = data['inliers'].astype(bool)
    # matches = np.array(data['matches'])[inliers].tolist()
    # matches = matches[inliers].tolist()
    def to3dim(img):
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        return img
    img1 = to3dim(data['image1'])
    img2 = to3dim(data['image2'])
    img1 = np.concatenate([img1, img1, img1], axis=2)
    img2 = np.concatenate([img2, img2, img2], axis=2)
    return cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches,
                           None, matchColor=(0,255,0), singlePointColor=(0, 0, 255))

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def find_files_with_ext(directory, extension='.npz', if_int=True):
    # print(os.listdir(directory))
    list_of_files = []
    import os
    if extension == ".npz":
        for l in os.listdir(directory):
            if l.endswith(extension):
                list_of_files.append(l)
                # print(l)
    if if_int:
        list_of_files = [e for e in list_of_files if isfloat(e[:-4])]
    return list_of_files


def to3dim(img):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    return img

def evaluate(args, **options):
    # path = '/home/yoyee/Documents/SuperPoint/superpoint/logs/outputs/superpoint_coco/'
    path = args.path
    files = find_files_with_ext(path)
    correctness = []
    est_H_mean_dist = []
    repeatability = []
    mscore = []
    mAP = []
    localization_err = []
    rep_thd = 3
    save_file = path + "/result.txt"
    inliers_method = 'cv'
    compute_map = True
    verbose = True
    top_K = 1000
    print("top_K: ", top_K)

    reproduce = True
    if reproduce:
        logging.info("reproduce = True")
        np.random.seed(0)
        print(f"test random # : np({np.random.rand(1)})")


    # create output dir
    if args.outputImg:
        path_warp = path+'/warping'
        os.makedirs(path_warp, exist_ok=True)
        path_match = path + '/matching'
        os.makedirs(path_match, exist_ok=True)
        path_rep = path + '/repeatibility' + str(rep_thd)
        os.makedirs(path_rep, exist_ok=True)

    # for i in range(2):
    #     f = files[i]
    print(f"file: {files[0]}")
    files.sort(key=lambda x: int(x[:-4]))
    from numpy.linalg import norm
    from utils.draw import draw_keypoints
    from utils.utils import saveImg

    for f in tqdm(files):
        f_num = f[:-4]
        data = np.load(path + '/' + f)
        print("load successfully. ", f)

        # unwarp
        # prob = data['prob']
        # warped_prob = data['prob']
        # desc = data['desc']
        # warped_desc = data['warped_desc']
        # homography = data['homography']
        real_H = data['homography']
        image = data['image']
        warped_image = data['warped_image']
        keypoints = data['prob'][:, [1, 0]]
        print("keypoints: ", keypoints[:3,:])
        warped_keypoints = data['warped_prob'][:, [1, 0]]
        print("warped_keypoints: ", warped_keypoints[:3,:])
        # print("Unwrap successfully.")

        if args.repeatibility:
            rep, local_err = compute_repeatability(data, keep_k_points=top_K, distance_thresh=rep_thd, verbose=False)
            repeatability.append(rep)
            print("repeatability: %.2f"%(rep))
            if local_err > 0:
                localization_err.append(local_err)
                print('local_err: ', local_err)
            if args.outputImg:
                # img = to3dim(image)
                img = image
                pts = data['prob']
                img1 = draw_keypoints(img*255, pts.transpose())

                # img = to3dim(warped_image)
                img = warped_image
                pts = data['warped_prob']
                img2 = draw_keypoints(img*255, pts.transpose())

                plot_imgs([img1.astype(np.uint8), img2.astype(np.uint8)], titles=['img1', 'img2'], dpi=200)
                plt.title("rep: " + str(repeatability[-1]))
                plt.tight_layout()
                
                plt.savefig(path_rep + '/' + f_num + '.png', dpi=300, bbox_inches='tight')
                pass


        if args.homography:
            # estimate result
            ##### check
            homography_thresh = [1,3,5,10,20,50]
            #####
            result = compute_homography(data, correctness_thresh=homography_thresh)
            correctness.append(result['correctness'])
            # est_H_mean_dist.append(result['mean_dist'])
            # compute matching score
            def warpLabels(pnts, homography, H, W):
                import torch
                """
                input:
                    pnts: numpy
                    homography: numpy
                output:
                    warped_pnts: numpy
                """
                from utils.utils import warp_points
                from utils.utils import filter_points
                pnts = torch.tensor(pnts).long()
                homography = torch.tensor(homography, dtype=torch.float32)
                warped_pnts = warp_points(torch.stack((pnts[:, 0], pnts[:, 1]), dim=1),
                                          homography)  # check the (x, y)
                warped_pnts = filter_points(warped_pnts, torch.tensor([W, H])).round().long()
                return warped_pnts.numpy()

            from numpy.linalg import inv
            H, W = image.shape
            unwarped_pnts = warpLabels(warped_keypoints, inv(real_H), H, W)
            score = (result['inliers'].sum() * 2) / (keypoints.shape[0] + unwarped_pnts.shape[0])
            print("m. score: ", score)
            mscore.append(score)
            # compute map
            if compute_map:
                def getMatches(data):
                    from models.model_wrap import PointTracker

                    desc = data['desc']
                    warped_desc = data['warped_desc']

                    nn_thresh = 1.2
                    print("nn threshold: ", nn_thresh)
                    tracker = PointTracker(max_length=2, nn_thresh=nn_thresh)
                    # matches = tracker.nn_match_two_way(desc, warped_desc, nn_)
                    tracker.update(keypoints.T, desc.T)
                    tracker.update(warped_keypoints.T, warped_desc.T)
                    matches = tracker.get_matches().T
                    mscores = tracker.get_mscores().T

                    # mAP
                    # matches = data['matches']
                    print("matches: ", matches.shape)
                    print("mscores: ", mscores.shape)
                    print("mscore max: ", mscores.max(axis=0))
                    print("mscore min: ", mscores.min(axis=0))

                    return matches, mscores

                def getInliers(matches, H, epi=3, verbose=False):
                    """
                    input:
                        matches: numpy (n, 4(x1, y1, x2, y2))
                        H (ground truth homography): numpy (3, 3)
                    """
                    from evaluations.detector_evaluation import warp_keypoints
                    # warp points 
                    warped_points = warp_keypoints(matches[:, :2], H) # make sure the input fits the (x,y)

                    # compute point distance
                    norm = np.linalg.norm(warped_points - matches[:, 2:4],
                                            ord=None, axis=1)
                    inliers = norm < epi
                    if verbose:
                        print("Total matches: ", inliers.shape[0], ", inliers: ", inliers.sum(),
                                          ", percentage: ", inliers.sum() / inliers.shape[0])

                    return inliers

                def getInliers_cv(matches, H=None, epi=3, verbose=False):
                    import cv2
                    # count inliers: use opencv homography estimation
                    # Estimate the homography between the matches using RANSAC
                    H, inliers = cv2.findHomography(matches[:, [0, 1]],
                                                    matches[:, [2, 3]],
                                                    cv2.RANSAC)
                    inliers = inliers.flatten()
                    print("Total matches: ", inliers.shape[0], 
                          ", inliers: ", inliers.sum(),
                          ", percentage: ", inliers.sum() / inliers.shape[0])
                    return inliers
            
            
                def computeAP(m_test, m_score):
                    from sklearn.metrics import average_precision_score

                    average_precision = average_precision_score(m_test, m_score)
                    print('Average precision-recall score: {0:0.2f}'.format(
                        average_precision))
                    return average_precision

                def flipArr(arr):
                    return arr.max() - arr
                
                if args.sift:
                    assert result is not None
                    matches, mscores = result['matches'], result['mscores']
                else:
                    matches, mscores = getMatches(data)
                
                real_H = data['homography']
                if inliers_method == 'gt':
                    # use ground truth homography
                    print("use ground truth homography for inliers")
                    inliers = getInliers(matches, real_H, epi=3, verbose=verbose)
                else:
                    # use opencv estimation as inliers
                    print("use opencv estimation for inliers")
                    inliers = getInliers_cv(matches, real_H, epi=3, verbose=verbose)
                    
                ## distance to confidence
                if args.sift:
                    m_flip = flipArr(mscores[:])  # for sift
                else:
                    m_flip = flipArr(mscores[:,2])
        
                if inliers.shape[0] > 0 and inliers.sum()>0:
#                     m_flip = flipArr(m_flip)
                    # compute ap
                    ap = computeAP(inliers, m_flip)
                else:
                    ap = 0
                
                mAP.append(ap)


            if args.outputImg:
                # draw warping
                output = result
                # img1 = image/255
                # img2 = warped_image/255
                img1 = image
                img2 = warped_image

                img1 = to3dim(img1)
                img2 = to3dim(img2)
                H = output['homography']
                warped_img1 = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
                # from numpy.linalg import inv
                # warped_img1 = cv2.warpPerspective(img1, inv(H), (img2.shape[1], img2.shape[0]))
                img1 = np.concatenate([img1, img1, img1], axis=2)
                warped_img1 = np.stack([warped_img1, warped_img1, warped_img1], axis=2)
                img2 = np.concatenate([img2, img2, img2], axis=2)
                plot_imgs([img1, img2, warped_img1], titles=['img1', 'img2', 'warped_img1'], dpi=200)
                plt.tight_layout()
                plt.savefig(path_warp + '/' + f_num + '.png')

                ## plot filtered image
                img1, img2 = data['image'], data['warped_image']
                warped_img1 = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
                plot_imgs([img1, img2, warped_img1], titles=['img1', 'img2', 'warped_img1'], dpi=200)
                plt.tight_layout()
                # plt.savefig(path_warp + '/' + f_num + '_fil.png')
                plt.savefig(path_warp + '/' + f_num + '.png')

                # plt.show()

                # draw matches
                result['image1'] = image
                result['image2'] = warped_image
                matches = np.array(result['cv2_matches'])
                ratio = 0.2
                ran_idx = np.random.choice(matches.shape[0], int(matches.shape[0]*ratio))

                img = draw_matches_cv(result, matches[ran_idx], plot_points=True)
                # filename = "correspondence_visualization"
                plot_imgs([img], titles=["Two images feature correspondences"], dpi=200)
                plt.tight_layout()
                plt.savefig(path_match + '/' + f_num + 'cv.png', bbox_inches='tight')
                plt.close('all')
                # pltImshow(img)

        if args.plotMatching:
            matches = result['matches'] # np [N x 4]
            if matches.shape[0] > 0:
                from utils.draw import draw_matches
                filename = path_match + '/' + f_num + 'm.png'
                ratio = 0.1
                inliers = result['inliers']

                matches_in = matches[inliers == True]
                matches_out = matches[inliers == False]

                def get_random_m(matches, ratio):
                    ran_idx = np.random.choice(matches.shape[0], int(matches.shape[0]*ratio))               
                    return matches[ran_idx], ran_idx
                image = data['image']
                warped_image = data['warped_image']
                ## outliers
                matches_temp, _ = get_random_m(matches_out, ratio)
                # print(f"matches_in: {matches_in.shape}, matches_temp: {matches_temp.shape}")
                draw_matches(image, warped_image, matches_temp, lw=0.5, color='r',
                            filename=None, show=False, if_fig=True)
                ## inliers
                matches_temp, _ = get_random_m(matches_in, ratio)
                draw_matches(image, warped_image, matches_temp, lw=1.0, 
                        filename=filename, show=False, if_fig=False)






    if args.repeatibility:
        repeatability_ave = np.array(repeatability).mean()
        localization_err_m = np.array(localization_err).mean()
        print("repeatability: ", repeatability_ave)
        print("localization error over ", len(localization_err), " images : ", localization_err_m)
    if args.homography:
        correctness_ave = np.array(correctness).mean(axis=0)
        # est_H_mean_dist = np.array(est_H_mean_dist)
        print("homography estimation threshold", homography_thresh)
        print("correctness_ave", correctness_ave)
        # print(f"mean est H dist: {est_H_mean_dist.mean()}")
        mscore_m = np.array(mscore).mean(axis=0)
        print("matching score", mscore_m)
        if compute_map:
            mAP_m = np.array(mAP).mean()
            print("mean AP", mAP_m)

        print("end")



    # save to files
    with open(save_file, "a") as myfile:
        myfile.write("path: " + path + '\n')
        myfile.write("output Images: " + str(args.outputImg) + '\n')
        if args.repeatibility:
            myfile.write("repeatability threshold: " + str(rep_thd) + '\n')
            myfile.write("repeatability: " + str(repeatability_ave) + '\n')
            myfile.write("localization error: " + str(localization_err_m) + '\n')
        if args.homography:
            myfile.write("Homography estimation: " + '\n')
            myfile.write("Homography threshold: " + str(homography_thresh) + '\n')
            myfile.write("Average correctness: " + str(correctness_ave) + '\n')

            # myfile.write("mean est H dist: " + str(est_H_mean_dist.mean()) + '\n')

            if compute_map:
                myfile.write("nn mean AP: " + str(mAP_m) + '\n')
            myfile.write("matching score: " + str(mscore_m) + '\n')



        if verbose:
            myfile.write("====== details =====" + '\n')
            for i in range(len(files)):

                myfile.write("file: " + files[i])
                if args.repeatibility:
                    myfile.write("; rep: " + str(repeatability[i]))
                if args.homography:
                    myfile.write("; correct: " + str(correctness[i]))
                    # matching
                    myfile.write("; mscore: " + str(mscore[i]))
                    if compute_map:
                        myfile.write(":, mean AP: " + str(mAP[i]))
                myfile.write('\n')
            myfile.write("======== end ========" + '\n')

    dict_of_lists = {
        'repeatability': repeatability,
        'localization_err': localization_err,
        'correctness': np.array(correctness),
        'homography_thresh': homography_thresh,
        'mscore': mscore,
        'mAP': np.array(mAP),
        # 'est_H_mean_dist': est_H_mean_dist
    }

    filename = f'{save_file[:-4]}.npz'
    logging.info(f"save file: {filename}")
    np.savez(
        filename,
        **dict_of_lists,
    )


if __name__ == '__main__':
    import argparse


    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--sift', action='store_true', help='use sift matches')
    parser.add_argument('-o', '--outputImg', action='store_true')
    parser.add_argument('-r', '--repeatibility', action='store_true')
    parser.add_argument('-homo', '--homography', action='store_true')
    parser.add_argument('-plm', '--plotMatching', action='store_true')
    args = parser.parse_args()
    evaluate(args)
