import cv2
import numpy as np
# Reading Image
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, recall_score, precision_score


def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (512, 512))
    ori_x = x
    x = x / 255.0
    x = x.astype(np.float32)
    return ori_x, x


# Reading Groundtruth
def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (512, 512))
    ori_x = x
    x = x / 255.0
    x = x.astype(np.int32)
    return ori_x, x


# Saving Image
def save_results(y_pred, save_image_path, mode="single", ori_y=None):
    if mode == "single":
        y_pred = np.expand_dims(y_pred, axis=-1)
        y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) * 255
        cv2.imwrite(save_image_path, y_pred)

    elif mode == "combined":
        pred_image = np.zeros((y_pred.shape[0], y_pred.shape[1], 3))
        # _y_pred = y_pred[:, :, channel]
        # _ori_y = ori_y[:, :, channel]
        pred_image[:, :, 0] = ((y_pred > 0.5) & (ori_y <= 128)) * 255
        pred_image[:, :, 1] = ((y_pred > 0.5) & (ori_y > 128)) * 255
        pred_image[:, :, 2] = ((ori_y > 128) & (y_pred <= 0.5)) * 255

        print(" saving result", save_image_path)
        cv2.imwrite(save_image_path, pred_image)


# Saving Image
def save_results_od_bv(y_pred, save_image_path, mode="single", ori_y=None, channel=None):
    if mode == "single":
        y_pred = y_pred[:, :, channel]
        y_pred = np.expand_dims(y_pred, axis=-1)
        y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) * 255
        cv2.imwrite(save_image_path, y_pred)

    elif mode == "combined":
        pred_image = np.zeros((y_pred.shape[0], y_pred.shape[1], 3))
        _y_pred = y_pred[:, :, channel]
        _ori_y = ori_y
        pred_image[:, :, 0] = ((_y_pred > 0.5) & (_ori_y <= 128)) * 255
        pred_image[:, :, 1] = ((_y_pred > 0.5) & (_ori_y > 128)) * 255
        pred_image[:, :, 2] = ((_ori_y > 128) & (_y_pred <= 0.5)) * 255
        print("saving result", save_image_path)
        cv2.imwrite(save_image_path, pred_image)

def confusion_matrix (y_pred,ori_y,threshold):
    results = []
    FP = np.sum(((y_pred > threshold) & (ori_y <= threshold)))
    TP = np.sum(((y_pred > threshold) & (ori_y > threshold)))
    FN = np.sum(((y_pred <= threshold) & (ori_y > threshold)))
    TN = np.sum(((y_pred <= threshold) & (ori_y <= threshold)))
    results.append([TP,FP,FN,TN])
    return results

def save_results_zip(ori_x, ori_y, y_pred, save_image_path):
    line = np.ones((512, 10, 3)) * 255

    ori_y = np.expand_dims(ori_y, axis=-1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) * 255

    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)


def save_results_zip_no_gt(ori_x, y_pred, save_image_path):
    line = np.ones((512, 10, 3)) * 255

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) * 255

    cat_images = np.concatenate([ori_x, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)


# Calculate Metrics
def calculate_metrics(y, y_pred, count):
    """ Calculate the metrics """
    score = []
    acc_value = accuracy_score(y, y_pred)
    f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
    jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
    recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary")
    precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
    image_name = f"image_{count}"
    score.append([image_name, acc_value, f1_value, jac_value, recall_value, precision_value])
    return score


# Calculate Metrics
def calculate_metrics_od_bv(y, y_pred, threshold = 0.5, count = 0, channel = 1):
    """ Calculate the metrics """
    score = []
    bv_pred = y_pred[:, :, channel].flatten()
    # bv_pred_prob = y_pred_prob[:, :, bv_channel].flatten()
    bv_gt = y.flatten()
    acc_value = accuracy_score(bv_gt > threshold, bv_pred > threshold)
    f1_value = f1_score(bv_gt > threshold, bv_pred > threshold, labels=[0, 1], average="binary")
    jac_value = jaccard_score(bv_gt > threshold, bv_pred > threshold, labels=[0, 1], average="binary")
    recall_value = recall_score(bv_gt > threshold, bv_pred > threshold, labels=[0, 1], average="binary")
    # recall_computed = np.sum((bv_gt > threshold) & (bv_pred > threshold)) / np.sum(bv_gt > threshold)
    precision_value = precision_score(bv_gt > threshold, bv_pred > threshold, labels=[0, 1], average="binary")
    # auc_score = auc(bv_gt > threshold, bv_pred_prob)
    # SCORE_BV.append([name, acc_value, f1_value, jac_value, recall_value, precision_value, recall_computed, auc_score])
    image_name = f"image_{count}"
    score.append([image_name, acc_value, f1_value, jac_value, recall_value, precision_value])
    return score

# New
def save_results_zip_od_bv(ori_x, ori_y, y_pred, save_image_path, count, channel):
    line = np.ones((512, 10, 3)) * 255

    y_pred = y_pred[:, :, channel]

    ori_y = np.expand_dims(ori_y, axis=-1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) * 255

    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred], axis=1)
    # image_name = f"image_{count}.png"

    # final_save_path = f"{save_image_path}_{image_name}"
    cv2.imwrite(save_image_path, cat_images)

def save_results_zip_no_gt_od_bv(ori_x, y_pred, save_image_path, channel=1):
        line = np.ones((512, 10, 3)) * 255

        y_pred = y_pred[:, :, channel]

        y_pred = np.expand_dims(y_pred, axis=-1)
        y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) * 255

        cat_images = np.concatenate([ori_x, line, y_pred], axis=1)
        cv2.imwrite(save_image_path, cat_images)