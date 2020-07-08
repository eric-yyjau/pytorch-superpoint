import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import ConnectionPatch


def toHomogeneous(coords):
    tmp = np.expand_dims(np.ones(len(coords)), axis=0)
    return np.concatenate((coords, tmp.T), axis=1)

def fromHomogeneous(coords):
    return np.array([np.array([p[0]/p[2], p[1]/p[2]]) for p in coords])

def get_inverse(mat):
    R = mat[0:3, 0:3].T
    t = mat[0:3, 3]
    t = -R @ t
    z = np.array([[0, 0, 0, 1]])
    tmp = np.concatenate((R, t.reshape(-1, 1)), axis=1)
    return np.concatenate((tmp, z), axis=0)

def plot_corr(img1, img2, img1_pts, img2_pts, vsplit=False, figsize=(20, 15)):
    plot1 = 211
    plot2 = 212
    if vsplit:
        plot1 = 121
        plot2 = 122

    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(plot1)
    ax1.imshow(img1, cmap='gray', aspect = "auto")
    ax1.scatter(img1_pts[:, 0], img1_pts[:, 1], marker='+')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    ax2 = plt.subplot(plot2)
    ax2.imshow(img2, cmap='gray', aspect = "auto")
    ax2.scatter(img2_pts[:, 0], img2_pts[:, 1], marker='+')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    for i in range(len(img1_pts)):
        xy1 = (img1_pts[i,0],img1_pts[i,1])
        xy2 = (img2_pts[i,0],img2_pts[i,1])
        con = ConnectionPatch(xyA=xy2, xyB=xy1, coordsA="data", coordsB="data",
                          axesA=ax2, axesB=ax1, color='#53F242')
        ax2.add_artist(con)

    plt.subplots_adjust(wspace=0, hspace=0)
#     plt.tight_layout()
    plt.show()


def plot_img(img, pts, pts2=None, figsize=(20, 15)):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')
    plt.scatter(pts[:, 0], pts[:, 1], c='c', marker='+')
    if pts2 is not None:
        plt.scatter(pts2[:, 0], pts2[:, 1], c='red', marker='x')
        
    plt.tight_layout()
    plt.show()


# CONVENTION FOR QUATERNION: [qx, qy, qz, qw]
_EPS = np.finfo(float).eps * 4.0
def from_quaternion(quaternion):
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3]],
        [    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3]],
        [    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1]]])

def to_quaternion(matrix, isprecise=False):
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[0, 1, 2, 3], np.argmax(w)]
    if q[3] < 0.0:
        np.negative(q, q)
    return q

def quaternion_multiply(quaternion1, quaternion0):
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array([x1*w0 + y1*z0 - z1*y0 + w1*x0,
                        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                         x1*y0 - y1*x0 + z1*w0 + w1*z0,
                        -x1*x0 - y1*y0 - z1*z0 + w1*w0], dtype=np.float64)

def quaternion_inverse(quaternion):
    q = np.array(quaternion, dtype=np.float64, copy=True)
    np.negative(q[:3], q[:3])
    return q / np.dot(q, q)

def dump_pose_TUM(out_file, poses, times):
    # First frame as the origin
    first_pose = np.linalg.inv(poses[0])
    with open(out_file, 'w') as f:
        for p in range(len(times)):
            this_pose = poses[p]
            this_pose = first_pose @ this_pose
            tx = this_pose[0, 3]
            ty = this_pose[1, 3]
            tz = this_pose[2, 3]
            rot = this_pose[:3, :3]
            qx, qy, qz, qw = to_quaternion(rot)
    #         print('%f %f %f %f %f %f %f %f\n' % (times[p], tx, ty, tz, qx, qy, qz, qw))
            f.write('%s %f %f %f %f %f %f %f\n' % (times[p], tx, ty, tz, qx, qy, qz, qw))

def build_pose(R, t):
    z = np.array([[0, 0, 0, 1]])
    if len(t.shape) == 1:
        assert t.shape[0] == 3
        tmp = np.concatenate((R, np.expand_dims(t, 1)), axis=1)
        return np.concatenate((tmp, z), axis=0)

    if len(t.shape) == 2:
        l = []
        for i in range(t.shape[0]):
            tmp = np.concatenate((R[i], np.expand_dims(t[i], 1)), axis=1)
            l += [np.concatenate((tmp, z), axis=0)]

        return np.array(l)
    else:
        raise ValueError("translation vector has a shape that is more than 2 dimensions.")
        
# def eval(gt_dir, pred_dir):
#     pred_files = glob.glob(pred_dir + '/*.txt')
#     ate_all = []
#     for i in range(len(pred_files)):
#         gtruth_file = os.path.join(gt_dir, os.path.basename(pred_files[i]))

#         if not os.path.exists(gtruth_file):
#             print("Ground truth file not found!")
#             print('\t> ground truth file: ' + gtruth_file)
#             print('\t> pred file: ' + pred_files[i])
#             continue

#         ate = compute_ate(gtruth_file, pred_files[i])
#         if ate == False:
#             continue
#         ate_all.append(ate)

#     ate_all = np.array(ate_all)
# #     print("Predictions dir: %s" % pred_dir)
#     print("ATE mean: %.4f, std: %.4f" % (np.mean(ate_all), np.std(ate_all)))