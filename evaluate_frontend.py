
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
from utils.loader import dataLoader, modelLoader, pretrainedLoader
import logging

from utils.tools import dict_update

from utils.utils import labels2Dto3D, flattenDetection, labels2Dto3D_flattened

from utils.utils import pltImshow, saveImg
from utils.utils import precisionRecall_torch
from utils.utils import save_checkpoint

from pathlib import Path



class evaluate_frontend(object):
    """ Wrapper around pytorch net to help with pre and post image processing. """
    '''
    * SuperPointFrontend_torch:
    ** note: the input, output is different from that of SuperPointFrontend
    heatmap: torch (batch_size, H, W, 1)
    dense_desc: torch (batch_size, H, W, 256)
    pts: [batch_size, np (N, 3)]
    desc: [batch_size, np(256, N)]
    '''
    default_config = {
        'train_iter': 170000,
        'save_interval': 2000,
        'tensorboard_interval': 200
    }

    def __init__(self, config, save_path=Path('.'), device='cpu', verbose=False, rep_thd=3):
        
        # create lists
        error_names = ['correctness', 'repeatability', 'mscore', 'mAP', 'localization_err']
        errors = {error_name:[] for error_name in error_names}
        print(errors)

        # correctness = []
        # repeatability = []
        # mscore = []
        # mAP = []
        # localization_err = []
        self.rep_thd = rep_thd
        save_file = path + "/result.txt"
        inliers_method = 'cv'
        compute_map = True
        verbose = True

        self.files = read_files(args.path)
        self.create_folders()


    @staticmethod
    def read_files(path):
        files = find_files_with_ext(path)
        files.sort(key=lambda x: int(x[:-4]))
        return files

    def create_folders(self):
       # create output dir
       if args.outputImg:
           path_warp = path+'/warping'
           os.makedirs(path_warp, exist_ok=True)
           path_match = path + '/matching'
           os.makedirs(path_match, exist_ok=True)
           path_rep = path + '/repeatibility' + str(rep_thd)
           os.makedirs(path_rep, exist_ok=True)
       pass 

    def run():
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
            warped_keypoints = data['warped_prob'][:, [1, 0]]
            if args.repeatibility:
                self.eva_repeatibility()
                if args.outputImg:
                    output_img()
            if args.homography:
                self.eva_homography_estimation()
                if args.outputImg:
                    output_img()
            if args.plotMatching:

    def save_results():
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
            pass

    def print_results():
        if args.repeatibility:
            repeatability_ave = np.array(repeatability).mean()
            localization_err_m = np.array(localization_err).mean()
            print("repeatability: ", repeatability_ave)
            print("localization error over ", len(localization_err), " images : ", localization_err_m)
        if args.homography:
            correctness_ave = np.array(correctness).mean(axis=0)
            print("homography estimation threshold", homography_thresh)
            print("correctness_ave", correctness_ave)
            mscore_m = np.array(mscore).mean(axis=0)
            print("matching score", mscore_m)
            if compute_map:
                mAP_m = np.array(mAP).mean()
                print("mean AP", mAP_m)

            print("end")

    def eva_homography_estimation():
        homography_thresh = [1,3,5]
        #####
        result = compute_homography(data, correctness_thresh=homography_thresh)
        correctness.append(result['correctness'])


    def eva_repeatibility():
        rep, local_err = compute_repeatability(data, keep_k_points=300, distance_thresh=rep_thd, verbose=False)
        repeatability.append(rep)
        print("repeatability: %.2f"%(rep))
        if local_err > 0:
            localization_err.append(local_err)
            print('local_err: ', local_err)

    def eva_mscore():
        from numpy.linalg import inv
        from utils.utils import warpLabels
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
                
            m_flip = flipArr(mscores[:,2])
        
            if inliers.shape[0] > 0 and inliers.sum()>0:
                # m_flip = flipArr(m_flip)
                # compute ap
                ap = computeAP(inliers, m_flip)
            else:
                ap = 0
            
            mAP.append(ap)


    def output_img():
        img = image
        pts = data['prob']
        img1 = draw_keypoints(img*255, pts.transpose())

        # img = to3dim(warped_image)
        img = warped_image
        pts = data['warped_prob']
        img2 = draw_keypoints(img*255, pts.transpose())

        plot_imgs([img1.astype(int), img2.astype(int)], titles=['img1', 'img2'], dpi=200)
        plt.title("rep: " + str(repeatability[-1]))
        plt.tight_layout()

        plt.savefig(path_rep + '/' + f_num + '.png', dpi=300, bbox_inches='tight')
        pass

    def plot_matching():
        matches = data['matches']
        if matches.shape[0] > 0:
            from utils.draw import draw_matches
            filename = path_match + '/' + f_num + 'm.png'
            draw_matches(image, warped_image, matches, filename=filename)






