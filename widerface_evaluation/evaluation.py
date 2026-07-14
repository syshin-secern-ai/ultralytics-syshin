from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy import ndarray
from scipy.io import loadmat
from tqdm import tqdm

PredictionDict = dict[str, dict[str, np.ndarray]]


def get_gt_boxes(gt_dir: Path) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    gt_mat = loadmat(gt_dir / "wider_face_val.mat")
    hard_mat = loadmat(gt_dir / "wider_hard_val.mat")
    medium_mat = loadmat(gt_dir / "wider_medium_val.mat")
    easy_mat = loadmat(gt_dir / "wider_easy_val.mat")
    return (
        gt_mat["face_bbx_list"],
        gt_mat["event_list"],
        gt_mat["file_list"],
        hard_mat["gt_list"],
        medium_mat["gt_list"],
        easy_mat["gt_list"],
    )


def norm_score(predictions: PredictionDict) -> PredictionDict:
    min_score = np.inf
    max_score = -np.inf
    for event_predictions in predictions.values():
        for boxes in event_predictions.values():
            if boxes.size == 0:
                continue
            min_score = min(min_score, float(np.min(boxes[:, -1])))
            max_score = max(max_score, float(np.max(boxes[:, -1])))

    if not np.isfinite(min_score) or not np.isfinite(max_score):
        return predictions
    diff = max_score - min_score
    for event_predictions in predictions.values():
        for boxes in event_predictions.values():
            if boxes.size == 0:
                continue
            if diff <= np.finfo(np.float32).eps:
                boxes[:, -1] = 1.0
            else:
                boxes[:, -1] = (boxes[:, -1] - min_score) / diff
    return predictions


def image_eval(pred: ndarray, gt: ndarray, ignore: ndarray, iou_thresh: float) -> tuple[ndarray, ndarray]:
    pred = pred.copy()
    gt = gt.copy()
    pred_recall = np.zeros(pred.shape[0])
    recall_list = np.zeros(gt.shape[0])
    proposal_list = np.ones(pred.shape[0])

    pred[:, 2] += pred[:, 0]
    pred[:, 3] += pred[:, 1]
    gt[:, 2] += gt[:, 0]
    gt[:, 3] += gt[:, 1]

    for pred_idx in range(pred.shape[0]):
        gt_overlap = bbox_overlap(gt, pred[pred_idx])
        max_overlap = gt_overlap.max()
        max_idx = gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[pred_idx] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1
        pred_recall[pred_idx] = np.count_nonzero(recall_list == 1)
    return pred_recall, proposal_list


def bbox_overlap(boxes: ndarray, query_box: ndarray) -> ndarray:
    x1 = np.maximum(boxes[:, 0], query_box[0])
    y1 = np.maximum(boxes[:, 1], query_box[1])
    x2 = np.minimum(boxes[:, 2], query_box[2])
    y2 = np.minimum(boxes[:, 3], query_box[3])
    widths = x2 - x1 + 1
    heights = y2 - y1 + 1
    inter = widths * heights
    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    query_area = (query_box[2] - query_box[0] + 1) * (query_box[3] - query_box[1] + 1)
    union = box_areas + query_area - inter
    overlaps = np.divide(inter, union, out=np.zeros_like(inter), where=union != 0)
    overlaps[widths <= 0] = 0
    overlaps[heights <= 0] = 0
    return overlaps


def img_pr_info(
    thresh_num: int,
    pred_info: ndarray,
    proposal_list: ndarray,
    pred_recall: ndarray,
) -> ndarray:
    pr_info = np.zeros((thresh_num, 2))
    for threshold_idx in range(thresh_num):
        thresh = 1 - (threshold_idx + 1) / thresh_num
        pred_indices = np.where(pred_info[:, 4] >= thresh)[0]
        if len(pred_indices) == 0:
            continue
        pred_idx = pred_indices[-1]
        valid_pred_indices = np.where(proposal_list[: pred_idx + 1] == 1)[0]
        pr_info[threshold_idx, 0] = len(valid_pred_indices)
        pr_info[threshold_idx, 1] = pred_recall[pred_idx]
    return pr_info


def dataset_pr_info(thresh_num: int, pr_curve: ndarray, count_face: int) -> ndarray:
    output = np.zeros((thresh_num, 2))
    output[:, 0] = np.divide(
        pr_curve[:, 1],
        pr_curve[:, 0],
        out=np.zeros(thresh_num),
        where=pr_curve[:, 0] > 0,
    )
    if count_face > 0:
        output[:, 1] = pr_curve[:, 1] / count_face
    return output


def voc_ap(recalls: ndarray, precisions: ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = np.maximum(mpre[idx - 1], mpre[idx])
    change_indices = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change_indices + 1] - mrec[change_indices]) * mpre[change_indices + 1]))


def evaluation(predictions: PredictionDict, gt_dir: Path, iou_thresh: float = 0.5) -> list[float]:
    pred = norm_score(predictions)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_dir)
    event_num = len(event_list)
    thresh_num = 1000
    settings = ["easy", "medium", "hard"]
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    aps = []
    for setting_id in range(3):
        # different setting
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2))
        # [hard, medium, easy]
        for i in tqdm(range(event_num), f"Processing {settings[setting_id]}", dynamic_ncols=True):
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            pred_list = pred.get(event_name, {})
            sub_gt_list = gt_list[i][0]
            gt_bbx_list = facebox_list[i][0]

            for j in range(len(img_list)):
                pred_info = pred_list.get(str(img_list[j][0][0]), np.zeros((0, 5)))

                gt_boxes = gt_bbx_list[j][0].astype(np.float64)
                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index - 1] = 1
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)

                _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

                pr_curve += _img_pr_info
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = voc_ap(recall, propose)
        aps.append(ap)
    return aps
