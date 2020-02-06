import numpy as np
import argparse
import cv2

min_depth = 0
max_depth = 0
num_samples = 200

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351

# depth
def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def load_gt_disp_kitti(gt_path):
    gt_disparities = []
    for i in range(200):
        # TODO
        disp = cv2.imread(gt_path + "/training/disp_noc_0/" + str(i).zfill(6) + "_10.png", -1)
        disp = disp.astype(np.float32) / 256
        gt_disparities.append(disp)

    return gt_disparities

def disps_to_depths(gt_disps, pred_disps):
    gt_depths = []
    pred_depths = []
    pred_disps_resized = []

    for i in range(len(gt_disps)):
        gt_disp = gt_disps[i]

        height, width = gt_disp.shape

        pred_disp = pred_disps[i]
        pred_disp = width * cv2.resize(pred_disp, (width, height), interpolation=cv2.INTER_LINEAR)
        pred_disps_resized.append(pred_disp)

        mask = gt_disp > 0
        gt_depth = width_to_focal[width] * 0.54 / (gt_disp + (1.0 - mask))
        pred_depth = width_to_focal[width] * 0.54 / pred_disp

        gt_depths.append(gt_depth)
        pred_depths.append(pred_depth)

    return gt_depths, pred_depths, pred_disps_resized

def main():
    pred_disparities = np.load(predicted_disp_path)

    gt_disparities = load_gt_disp_kitti(gt_path)
    gt_depths, pred_depths, pred_disparities_resized = disps_to_depths(gt_disparities, pred_disparities)

    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1_all = np.zeros(num_samples, np.float32)
    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)

    for i in range(num_samples):

        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        gt_disp = gt_disparities[i]
        pred_disp = pred_disparities_resized[i]

        # clip
        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth

        disp_mask = gt_disp > 0
        disp_diff = np.abs(gt_disp[disp_mask] - pred_disp[disp_mask])
        bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[disp_mask]) >= 0.05)
        d1_all[ii] = 100 * bad_pixels.sum() / disp_mask.sum()

        abs_rel[i], sq_rel[i], rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[disp_mask], pred_depth[disp_mask])

