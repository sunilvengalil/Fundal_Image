import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou
from utils import get_filenames_sorted

H = 512
W = 768

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    # x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    return ori_x, x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (512, 512)
    # x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = x[:,:,0:2] # Discard the last channel 
    return ori_x, x

def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*.jpg")))
    y = sorted(glob(os.path.join(path, "mask", "*.jpg")))
    return x, y

def save_results(ori_x, ori_y, y_pred, save_image_path, channel):
    line = np.ones((y_pred.shape[0], 10, 3)) * 255

    pred_image = np.zeros((y_pred.shape[0], y_pred.shape[1], 3))
    _y_pred = y_pred[:, :, channel]
    _ori_y = ori_y[:, :, channel]
    pred_image[:, :, 0] = ((_y_pred > 0.5) & (_ori_y <= 128)) * 255
    pred_image[:, :, 1] = ((_y_pred > 0.5) & (_ori_y  > 128)) * 255
    pred_image[:, :, 2] = ((_ori_y  > 128) & (_y_pred <= 0.5 )) * 255

    print(" saving result", save_image_path)
    cv2.imwrite(save_image_path, pred_image)

if __name__ == "__main__":

    data_dir = "new_data"
    od_channel, bv_channel = 0, 1

    od_result_dir = "files_demo_od"
    bv_result_dir = "files_demo_bv"

    create_dir(od_result_dir)
    create_dir(bv_result_dir)

    model_dir = "files"

    """ Load the model """
    model_file_name = f"{model_dir}/model.h5"
    print(model_file_name)
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model(model_file_name)

    """ Load the dataset """
    # dataset_path = os.path.join(data_dir, "test")
    test_x, test_y = get_filenames_sorted(data_dir + "/test/image/", data_dir + "/test/mask/" )

    """ Make the prediction and calculate the metrics values """
    SCORE_BV, SCORE_OD = [], []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extracting name """
        name = x.rsplit("/", 1)[1].rsplit(".", 1)[0]
        print(name)

        """ Read the image and mask """
        ori_x, x = read_image(x)
        ori_y, y = read_mask(y)

        """ Prediction """
        y_pred = model.predict(np.expand_dims(x, axis=0))[0]
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.float32)

        print(np.max(ori_y), np.max(y))

        """ Saving the images """
        save_image_path_od = f"{od_result_dir}/{name}.png"
        save_results(ori_x, ori_y, y_pred, save_image_path_od, od_channel)

        save_image_path_bv = f"{bv_result_dir}/{name}.png"
        save_results(ori_x, ori_y, y_pred, save_image_path_bv, bv_channel)


    #     """ Calculate the bv metrics """
        bv_pred = y_pred[:, :, bv_channel].flatten()
        bv_gt = y[:, :, bv_channel].flatten()
        acc_value = accuracy_score(bv_gt > 0.5, bv_pred>0.5)
        f1_value = f1_score(bv_gt > 0.5, bv_pred>0.5, labels=[0, 1], average="binary")
        jac_value = jaccard_score(bv_gt > 0.5, bv_pred>0.5, labels=[0, 1], average="binary")
        recall_value = recall_score(bv_gt > 0.5, bv_pred>0.5, labels=[0, 1], average="binary")
        recall_computed = np.sum((bv_gt > 0.5) & (bv_pred > 0.5)) / np.sum(bv_gt > 0.5)
        precision_value = precision_score(bv_gt > 0.5, bv_pred>0.5, labels=[0, 1], average="binary")
        SCORE_BV.append([name, acc_value, f1_value, jac_value, recall_value, precision_value, recall_computed])

    #     """ Calculate the od metrics """
        bv_pred = y_pred[:, :, od_channel].flatten()
        bv_gt = y[:, :, od_channel].flatten()
        acc_value = accuracy_score(bv_gt > 0.5, bv_pred>0.5)
        f1_value = f1_score(bv_gt > 0.5, bv_pred>0.5, labels=[0, 1], average="binary")
        jac_value = jaccard_score(bv_gt > 0.5, bv_pred>0.5, labels=[0, 1], average="binary")
        recall_value = recall_score(bv_gt > 0.5, bv_pred>0.5, labels=[0, 1], average="binary")
        recall_computed = np.sum((bv_gt > 0.5) & (bv_pred > 0.5)) / np.sum(bv_gt > 0.5)
        precision_value = precision_score(bv_gt > 0.5, bv_pred>0.5, labels=[0, 1], average="binary")
        SCORE_OD.append([name, acc_value, f1_value, jac_value, recall_value, precision_value, recall_computed])
        
    print("\n")
    for SCORE in [SCORE_OD, SCORE_BV]:
        if SCORE == SCORE_OD:
            print("****** OD Metrics")
        else:
            print("****** BV Metrics")
        score = [s[1:] for s in SCORE]
        score = np.mean(score, axis=0)
        print(f"Accuracy: {score[0]:0.5f}")
        print(f"F1: {score[1]:0.5f} (dice score)")
        print(f"AUC: {score[1]:0.5f} (Auc score)")
        print(f"Jaccard: {score[2]:0.5f}")
        print(f"Recall: {score[3]:0.5f}")
        print(f"Precision: {score[4]:0.5f}")

        # """ Saving """
        if SCORE == SCORE_OD:
            df = pd.DataFrame(SCORE, columns=["Image", "Acc", "F1", "Jaccard", "Recall", "Precision", "Recall Computed"])
            df.to_csv(f"{od_result_dir}/score.csv")
        else:
            df = pd.DataFrame(SCORE, columns=["Image", "Acc", "F1", "Jaccard", "Recall", "Precision", "Recall Computed"])
            df.to_csv(f"{bv_result_dir}/score.csv")
        print("\n")