### To Run The Code, use the following command - streamlit run fia_server.py in command prompt
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from utils import read_image, read_mask, save_results, calculate_metrics, save_results_zip, save_results_zip_no_gt
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import streamlit as st
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import zipfile
import shutil
from glob import glob
from tqdm import tqdm

from quality_check.quality_check import get_model_name, loss_fn2, loss_fn1, mcc, get_image_quality
from tensorflow.keras.models import load_model

np.random.seed(42)
save_dir = 'models/'

# Title
st.title("Segmentation of Fundal Images")

# Create dataset checkboxes
_radio = st.sidebar.radio("",
                          ("Home", "Run The Project", "Enter Your Feedback"))

if _radio == "Home":
    # Display project Description
    st.image("images/Fundal_Image.png")
    with open("intro.md") as intro_file:
        st.markdown(intro_file.read(), unsafe_allow_html=True)
elif _radio == "Run The Project":
    model_dir = "models"
    bv_model_path = f"{model_dir}/bv_model.h5"
    bv_model = None
    od_model = None
    # Dataset upload
    st.subheader("Choose the dataset: ")

    # Create dataset checkboxes
    # drive_checkbox = st.sidebar.checkbox("DRIVE")
    # hrf_checkbox = st.sidebar.checkbox("HRF")
    # custom_checkbox = st.sidebar.checkbox("Upload your own data")
    radio_upload = st.radio("",
                            ("Upload Image For Prediction",
                             "Upload multiple files as zip",
                             "Predictions on Drive Dataset",
                             "Predictions on HRF Dataset"))

    # quality assessment code which takes the path of one image file as input
    fold_var = 0
    model = load_model(save_dir + get_model_name(fold_var),
                       custom_objects={'loss_fn1': loss_fn1, 'loss_fn2': loss_fn2, 'mcc': mcc})

    if radio_upload == "Upload Image For Prediction":
        # Create the drag-drop uploader
        st.header("Load the Fundal Image")
        img_data = st.file_uploader(label="Image", accept_multiple_files=True)

        # Predicting and Displaying Image
        if img_data is not None and len(img_data) > 0:
            count = 0
            final_score_bv = []
            final_score_od = []

            #            for images in img_data:
            with st.form(key="quality_check"):
                # st.write("Performing Image Quality Analysis on uploaded Image")

                with st.spinner('Performing Image Quality Analysis on uploaded Image...'):
                    img = Image.open(img_data[count])  # read in the image

                    # Display image
                    with st.expander("View uploaded image", expanded=True):
                        st.image(img)

                    image_quality, quality_score = get_image_quality(img_data, img, model)

                    qcheck_submitted = st.form_submit_button("Rerun quality check")

            # if image_quality[0:4] == "good":
            with st.form(key="bv_segmentation"):
                
                st.code(f"Qualtiy Score: {quality_score:0.2f} [Score 0-1 - O is poor and 1 is best]"
                        , language="python")
                save_uploaded_image = Image.open(img_data[count])
                # st.image(save_uploaded_image)
                img_array = np.array(save_uploaded_image)
                img_shape = img_array.shape
                # print(img_shape)
                text_line = f"Image size: {img_shape}."

                st.write(text_line)

                # st.write("Your image will be rescaled to (512, 512, 3) for further computation.")

                st.header("Select the Analysis to perform: ")

                # Create the checkboxes
                bv_checkbox = st.checkbox("Blood Vessel Segmentation")
                od_checkbox = st.checkbox("Optic Disc Segmentation")

                st.header("Upload Ground truth images (Optional)")
                # Select the ground truths to test
                # bv_checkbox_gt = st.checkbox("Blood Vessel Ground truth")
                # od_checkbox_gt = st.checkbox("Optic Disc Ground truth")

                # if bv_checkbox_gt:
                bv_data = st.file_uploader(label="BV_GT", accept_multiple_files=True)

                # if od_checkbox_gt:
                od_data = st.file_uploader(label="OD_GT", accept_multiple_files=True)

                print(bv_data)

                if len(bv_data) > 0 or len(od_data) > 0:
                    radio_type = st.radio("",
                                          ("Single", "Combined"))

            

                with st.spinner('Computing the Prediction Results...'):

                    st.title("Predictions")  # Line 402
                    save_uploaded_image = Image.open(img_data[count])
                    # st.image(save_uploaded_image)
                    img_array = np.array(save_uploaded_image)
                    img_shape = img_array.shape
                    # print(img_shape)
                    if not os.path.isdir("history"):
                        os.mkdir("history")
                    image_name = f'history/test_{count}.jpg'
                    cv2.imwrite(image_name, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

                    # Load Image For Prediction
                    # img_path = "01_h_0.png"
                    ori_x, x = read_image(image_name)

                    if len(bv_data) > 0:
                        # Display BV Mask
                        try:
                            save_uploaded_image = Image.open(bv_data[count])
                            print(bv_data[count])
                            # st.image(save_uploaded_image)
                            img_array = np.array(save_uploaded_image)
                            if not os.path.isdir("history"):
                                os.mkdir("history")
                            bv_name = f'history/bv_gt_{count}.jpg'
                            cv2.imwrite(bv_name, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                            ori_y_bv, y_bv = read_mask(bv_name)
                        except:
                            st.write("Waiting for upload of groundtruth...")

                    if len(od_data) > 0:
                        # Display OD Mask
                        try:
                            save_uploaded_image = Image.open(od_data[count])
                            # st.image(save_uploaded_image)
                            img_array = np.array(save_uploaded_image)
                            if not os.path.isdir("history"):
                                os.mkdir("history")
                            od_name = f'history/od_gt_{count}.jpg'
                            cv2.imwrite(od_name, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                            ori_y_od, y_od = read_mask(od_name)
                        except:
                            st.write("Waiting for upload of ground truth...")

                    if bv_checkbox:
                        if bv_model is None:
                            print(f"Loading Blood Vessel Segmentation model from {bv_model_path}")
                            bv_model = load_model(bv_model_path, compile=False)
                        y_pred = bv_model.predict(np.expand_dims(x, axis=0))[0]
                        y_pred = y_pred > 0.5
                        y_pred = y_pred.astype(np.int32)
                        y_pred = np.squeeze(y_pred, axis=-1)
                        if not os.path.isdir("history"):
                            os.mkdir("history")
                        # pred_name_bv = f"history/bv_prediction_{count}.jpg"

                        # if not bv_checkbox_gt or not od_checkbox_gt:
                        if len(bv_data) == 0:
                            pred_name_bv = f"history/bv_prediction_{count}.jpg"
                            save_results(y_pred, pred_name_bv)

                            final_image = Image.open(pred_name_bv)
                            final_image_array = np.asarray(final_image)
                            reshaped_array = cv2.resize(final_image_array, (img_shape[1], img_shape[0]))
                            # print(reshaped_array.shape)

                        elif radio_type == "Single":
                            try:
                                pred_name_bv = f"history/bv_prediction_{count}.jpg"
                                save_results(y_pred, pred_name_bv)

                                final_image = Image.open(pred_name_bv)
                                final_image_array = np.asarray(final_image)
                                reshaped_array = cv2.resize(final_image_array, (img_shape[1], img_shape[0]))
                                # print(reshaped_array.shape)
                            except:
                                st.write("Waiting for upload of ground truth...")

                        elif radio_type == "Combined":
                            try:
                                pred_name_bv_combined = f"history/bv_prediction_combined_{count}.jpg"
                                save_results(y_pred, pred_name_bv_combined, mode="combined", ori_y=ori_y_bv)

                                final_image = Image.open(pred_name_bv_combined)
                                final_image_array = np.asarray(final_image)
                                reshaped_array = cv2.resize(final_image_array, (img_shape[1], img_shape[0]))
                                # print(reshaped_array.shape)

                            except:
                                st.write("Waiting for upload of ground truth...")

                        if len(bv_data) > 0:
                            try:
                                col1, col2, col3 = st.columns(3)

                                original = Image.open(img_data[count])
                                col1.header("Image")
                                col1.image(original, use_column_width=True)

                                gnd = Image.open(bv_data[count])
                                col2.header("Ground truth")
                                col2.image(gnd, use_column_width=True)

                                col3.header("BV Prediction")
                                col3.image(reshaped_array, use_column_width=True)

                                y_bv = y_bv.flatten()
                                y_pred = y_pred.flatten()

                                SCORE = calculate_metrics(y_bv, y_pred, count)
                                final_score_bv.extend(SCORE)

                                score = [s[1:] for s in SCORE]
                                score = np.mean(score, axis=0)
                                st.write("The Calculated BV Metrics is as follows: ")

                                df = pd.DataFrame(SCORE,
                                                  columns=["Image", "Acc", "F1", "Jaccard", "Recall", "Precision"])
                                st.table(df)

                            except:
                                pass

                        else:
                            col1, col2 = st.columns(2)

                            original = Image.open(img_data[count])
                            col1.header("Image")
                            col1.image(original, use_column_width=True)

                            col2.header("BV Prediction")
                            col2.image(reshaped_array, use_column_width=True)

                    if od_checkbox:
                        od_model = load_model("models/od_model.h5", compile=False)
                        y_pred = od_model.predict(np.expand_dims(x, axis=0))[0]
                        y_pred = y_pred > 0.5
                        y_pred = y_pred.astype(np.int32)
                        y_pred = np.squeeze(y_pred, axis=-1)
                        if not os.path.isdir("history"):
                            os.mkdir("history")
                        # pred_name_od = f"history/od_prediction_{count}.jpg"

                        # if not bv_checkbox_gt or not od_checkbox_gt:
                        if len(od_data) == 0:
                            pred_name_od = f"history/od_prediction_{count}.jpg"
                            save_results(y_pred, pred_name_od)

                            final_image = Image.open(pred_name_od)
                            final_image_array = np.asarray(final_image)
                            reshaped_array = cv2.resize(final_image_array, (img_shape[1], img_shape[0]))
                            # print(reshaped_array.shape)

                        elif radio_type == "Single":
                            try:
                                pred_name_od = f"history/od_prediction_{count}.jpg"
                                save_results(y_pred, pred_name_od)

                                final_image = Image.open(pred_name_od)
                                final_image_array = np.asarray(final_image)
                                reshaped_array = cv2.resize(final_image_array, (img_shape[1], img_shape[0]))
                            except:
                                st.write("Waiting for upload of ground truth...")

                        elif radio_type == "Combined":
                            try:
                                pred_name_od_combined = f"history/od_prediction_combined_{count}.jpg"
                                save_results(y_pred, pred_name_od_combined, mode="combined", ori_y=ori_y_od)

                                final_image = Image.open(pred_name_od_combined)
                                final_image_array = np.asarray(final_image)
                                reshaped_array = cv2.resize(final_image_array, (img_shape[1], img_shape[0]))
                            except:
                                st.write("Waiting for upload of ground truth...")

                        if len(od_data) > 0:
                            try:

                                col1, col2, col3 = st.columns(3)

                                original = Image.open(img_data[count])
                                col1.header("Image")
                                col1.image(original, use_column_width=True)

                                gnd = Image.open(od_data[count])
                                col2.header("Groundtruth")
                                col2.image(gnd, use_column_width=True)

                                col3.header("OD Prediction")
                                col3.image(reshaped_array, use_column_width=True)

                                y_od = y_od.flatten()
                                y_pred = y_pred.flatten()

                                SCORE = calculate_metrics(y_od, y_pred, count)
                                final_score_od.extend(SCORE)

                                score = [s[1:] for s in SCORE]
                                score = np.mean(score, axis=0)

                                st.write("The Calculated OD Metrics is as follows: ")
                                df = pd.DataFrame(SCORE,
                                                  columns=["Image", "Acc", "F1", "Jaccard", "Recall", "Precision"])

                                st.table(df)

                            except:
                                st.write("Waiting for upload of ground truth...")

                        else:
                            col1, col2 = st.columns(2)

                            original = Image.open(img_data[count])
                            col1.header("Image")
                            col1.image(original, use_column_width=True)

                            col2.header("OD Prediction")
                            col2.image(reshaped_array, use_column_width=True)

                    count += 1

                    if len(bv_data) > 0:

                        # """ Saving """

                        # with open(pred_name_bv, "rb") as file:
                        #     btn = st.download_button(
                        #             label="Download BV Prediction",
                        #             data=file,
                        #             file_name=pred_name_bv,
                        #             mime="image/png"
                        #         )

                        df_bv = pd.DataFrame(final_score_bv,
                                             columns=["Image", "Acc", "F1", "Jaccard", "Recall", "Precision"])
                        if not os.path.isdir("history"):
                            os.mkdir("history")
                        # TODO add few more fields to this DF
                        # Date, Imagepath, od gt path, bv gt path,
                        df_bv.to_csv(f"history/bv_score.csv", index=False)

                    if len(od_data) > 0:

                        # """ Saving """

                        # with open(pred_name_od, "rb") as file:
                        #     btn = st.download_button(
                        #             label="Download OD Prediction",
                        #             data=file,
                        #             file_name=pred_name_od,
                        #             mime="image/png"
                        #         )

                        df_bv = pd.DataFrame(final_score_od,
                                             columns=["Image", "Acc", "F1", "Jaccard", "Recall", "Precision"])
                        if not os.path.isdir("history"):
                            os.mkdir("history")
                        df_bv.to_csv(f"history/od_score.csv", index=False)

                    submitted = st.form_submit_button("Run Prediction")

                # # Code Snippet For Download Button Line 429 - 490

                # if bv_checkbox:

                #     """ Saving """
                #     if not bv_checkbox_gt or not od_checkbox_gt:
                #        with open(pred_name_bv, "rb") as file:
                #             btn = st.download_button(
                #                     label="Download BV Prediction",
                #                     data=file,
                #                     file_name=pred_name_bv,
                #                     mime="image/png"
                #                 )

                #     elif radio_type == "Single":
                #         with open(pred_name_bv, "rb") as file:
                #             btn = st.download_button(
                #                     label="Download BV Prediction",
                #                     data=file,
                #                     file_name=pred_name_bv,
                #                     mime="image/png"
                #                 )

                #     elif radio_type == "Combined":
                #         with open(pred_name_bv_combined, "rb") as file:
                #             btn = st.download_button(
                #                     label="Download BV Prediction",
                #                     data=file,
                #                     file_name=pred_name_bv_combined,
                #                     mime="image/png"
                #                 )

                # if od_checkbox:

                #     """ Saving """
                #     if not bv_checkbox_gt or not od_checkbox_gt:
                #         with open(pred_name_od, "rb") as file:
                #             btn = st.download_button(
                #                     label="Download OD Prediction",
                #                     data=file,
                #                     file_name=pred_name_od,
                #                     mime="image/png"
                #                 )

                #     elif radio_type == "Single":
                #         with open(pred_name_od, "rb") as file:
                #             btn = st.download_button(
                #                     label="Download OD Prediction",
                #                     data=file,
                #                     file_name=pred_name_od,
                #                     mime="image/png"
                #                 )

                #     elif radio_type == "Combined":
                #         with open(pred_name_od_combined, "rb") as file:
                #             btn = st.download_button(
                #                     label="Download OD Prediction",
                #                     data=file,
                #                     file_name=pred_name_od_combined,
                #                     mime="image/png"
                #                 )

            # else:
            #     st.write("Image Quality Is Too Poor To Continue")

    elif radio_upload == "Upload multiple files as zip":
        model_dir = "models"
        bv_model_path = f"{model_dir}/bv_model.h5"
        bv_model = load_model(bv_model_path, compile=False)
        od_model = load_model("models/od_model.h5", compile=False)

        if os.path.exists("data.zip"):
            os.remove("data.zip")

        if os.path.exists("bv_data.zip"):
            os.remove("bv_data.zip")

        if os.path.exists("od_data.zip"):
            os.remove("od_data.zip")

        if os.path.exists("data/"):
            shutil.rmtree("data")

        if os.path.exists("results/"):
            shutil.rmtree("results")

        if os.path.exists("results_od/"):
            shutil.rmtree("results_od")

        if os.path.exists("files/"):
            shutil.rmtree("files")


        def create_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)


        create_dir("data")
        create_dir("results")
        create_dir("files")
        create_dir("results_od")

        # user_consent_2 = st.checkbox("Click on the checkbox to allow us to save information for further advancements")

        st.header("Select the option to perform the respective segmentation")
        with open("upload_files_discripion.md") as desc_file:
            st.markdown(desc_file.read(), unsafe_allow_html=True)

        zip_data = st.file_uploader(label="Upload The Zip File", type="zip")

        if zip_data is not None:
            # with zipfile.ZipFile("data.zip", 'r') as zip_ref:
            #     zip_ref.extractall("data")
            with open(zip_data.name, 'wb') as f:
                # print(f)
                f.write(zip_data.getbuffer())

            with zipfile.ZipFile('data.zip', 'r') as zip_ref:
                zip_ref.extractall('')

        bv_zip_checkbox = st.checkbox("Blood Vessels")
        od_zip_checkbox = st.checkbox("Optic Disc")

        if bv_zip_checkbox:

            if os.path.exists("data/bv_gt"):
                bv_value = True

            else:
                bv_value = False


            def load_data(path, default=False):
                x = sorted(glob(os.path.join(path, "image", "*.png")))
                if default == True:
                    y = sorted(glob(os.path.join(path, "bv_gt", "*.png")))
                else:
                    y = None

                return x, y


            dataset_path = "data"
            test_x, test_y = load_data(dataset_path, default=bv_value)

            with st.spinner('Computing the Prediction Results...'):
                SCORE = []

                if bv_value == True:
                    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
                        name = x.split("/")[-1].split(".")[0]
                        # print(name)

                        ori_x, x = read_image(x)
                        ori_y, y = read_mask(y)

                        y_pred = bv_model.predict(np.expand_dims(x, axis=0))[0]
                        y_pred = y_pred > 0.5
                        y_pred = y_pred.astype(np.int32)
                        y_pred = np.squeeze(y_pred, axis=-1)

                        save_image_path = f"results/{name}.png"

                        save_results_zip(ori_x, ori_y, y_pred, save_image_path)

                        y = y.flatten()
                        y_pred = y_pred.flatten()
                        acc_value = accuracy_score(y, y_pred)
                        f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
                        jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
                        recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary")
                        precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
                        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

                    score = [s[1:] for s in SCORE]
                    score = np.mean(score, axis=0)
                    print(f"Accuracy: {score[0]:0.5f}")
                    print(f"F1: {score[1]:0.5f}")
                    print(f"Jaccard: {score[2]:0.5f}")
                    print(f"Recall: {score[3]:0.5f}")
                    print(f"Precision: {score[4]:0.5f}")

                    df = pd.DataFrame(SCORE, columns=["Image", "Acc", "F1", "Jaccard", "Recall", "Precision"])
                    df.to_csv("files/score.csv")

                    st.subheader("Blood Vessel Segmentation Results")
                    directory = os.listdir("results")

                    # with st.expander("View Images"):
                    #     st.write("-----------------Image--------------------------------Groundtruth-------------------------Prediction------------------")

                    with open("table.md") as table_file:
                        st.markdown(table_file.read(), unsafe_allow_html=True)

                        for files in directory:
                            new_image = Image.open(f"results/{files}")
                            st.write(files[0:7])
                            st.image(new_image)

                    with st.expander("View Scores"):
                        df_drive = pd.read_csv("files/score.csv")
                        st.table(df_drive)

                else:
                    for x in tqdm(test_x, total=len(test_x)):
                        name = x.split("/")[-1].split(".")[0]
                        # print(name)

                        ori_x, x = read_image(x)
                        # ori_y, y = read_mask(y)

                        y_pred = bv_model.predict(np.expand_dims(x, axis=0))[0]
                        y_pred = y_pred > 0.5
                        y_pred = y_pred.astype(np.int32)
                        y_pred = np.squeeze(y_pred, axis=-1)

                        save_image_path = f"results/{name}.png"

                        save_results_zip_no_gt(ori_x, y_pred, save_image_path)

                    st.subheader("Blood Vessel Segmentation Results")
                    directory = os.listdir("results")

                    # with st.expander("View Images"):
                    #     st.write("-----------------Image--------------------------------Groundtruth-------------------------Prediction------------------")

                    with open("table_2.md") as table_file:
                        st.markdown(table_file.read(), unsafe_allow_html=True)

                        for files in directory:
                            new_image = Image.open(f"results/{files}")
                            st.write(files[0:7])
                            st.image(new_image)

        if od_zip_checkbox:

            if os.path.exists("data/od_gt"):
                od_gt_selected = True

            else:
                od_gt_selected = False


            def load_data(path, default=False):
                x = sorted(glob(os.path.join(path, "image", "*.png")))
                if default == True:
                    y = sorted(glob(os.path.join(path, "od_gt", "*.png")))

                else:
                    y = None

                return x, y


            dataset_path = "data"
            test_x, test_y = load_data(dataset_path, default=od_gt_selected)

            with st.spinner('Computing the Prediction Results...'):
                SCORE = []
                if od_gt_selected == True:
                    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
                        name = x.split("/")[-1].split(".")[0]
                        # print(name)

                        ori_x, x = read_image(x)
                        ori_y, y = read_mask(y)

                        y_pred = od_model.predict(np.expand_dims(x, axis=0))[0]
                        y_pred = y_pred > 0.5
                        y_pred = y_pred.astype(np.int32)
                        y_pred = np.squeeze(y_pred, axis=-1)

                        save_image_path = f"results_od/{name}.png"
                        save_results_zip(ori_x, ori_y, y_pred, save_image_path)

                        y = y.flatten()
                        y_pred = y_pred.flatten()
                        acc_value = accuracy_score(y, y_pred)
                        f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
                        jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
                        recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary")
                        precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
                        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

                    # Compute scores
                    score = [s[1:] for s in SCORE]
                    score = np.mean(score, axis=0)
                    print(f"Accuracy: {score[0]:0.5f}")
                    print(f"F1: {score[1]:0.5f}")
                    print(f"Jaccard: {score[2]:0.5f}")
                    print(f"Recall: {score[3]:0.5f}")
                    print(f"Precision: {score[4]:0.5f}")

                    df = pd.DataFrame(SCORE, columns=["Image", "Acc", "F1", "Jaccard", "Recall", "Precision"])
                    df.to_csv("files/score_od.csv")

                    st.subheader("Optic Disc Segmentation Results")
                    directory = os.listdir("results_od")

                    # with st.expander("View Images"):
                    #     st.write("-----------------Image--------------------------------Groundtruth-------------------------Prediction------------------")

                    with open("table.md") as table_file:
                        st.markdown(table_file.read(), unsafe_allow_html=True)

                        for files in directory:
                            new_image = Image.open(f"results_od/{files}")
                            st.write(files[0:7])
                            st.image(new_image)

                    with st.expander("View Scores"):
                        df_drive = pd.read_csv("files/score_od.csv")
                        st.table(df_drive)
                else:
                    # OD GT is not selected
                    for x in tqdm(test_x, total=len(test_x)):
                        name = x.split("/")[-1].split(".")[0]
                        # print(name)

                        ori_x, x = read_image(x)
                        # ori_y, y = read_mask(y)

                        y_pred = od_model.predict(np.expand_dims(x, axis=0))[0]
                        y_pred = y_pred > 0.5
                        y_pred = y_pred.astype(np.int32)
                        y_pred = np.squeeze(y_pred, axis=-1)

                        save_image_path = f"results_od/{name}.png"
                        save_results_zip_no_gt(ori_x, y_pred, save_image_path)

                    st.subheader("Optic Disc Segmentation")
                    directory = os.listdir("results_od")

                    # with st.expander("View Images"):
                    #     st.write("-----------------Image--------------------------------Groundtruth-------------------------Prediction------------------")

                    # Display the table and results
                    with open("table_2.md") as table_file:
                        st.markdown(table_file.read(), unsafe_allow_html=True)

                        for files in directory:
                            new_image = Image.open(f"results_od/{files}")
                            st.write(files[0:7])
                            st.image(new_image)

        # if not user_consent_2:
        #     if os.path.exists("data.zip"):
        #         os.remove("data.zip")

        #     if os.path.exists("bv_data.zip"):
        #         os.remove("bv_data.zip")

        #     if os.path.exists("od_data.zip"):
        #         os.remove("od_data.zip")

        #     if os.path.exists("data/"):
        #         shutil.rmtree("data")

        #     if os.path.exists("results/"):
        #         shutil.rmtree("results")

        #     if os.path.exists("results_od/"):
        #         shutil.rmtree("results_od")

        #     if os.path.exists("files/"):
        #         shutil.rmtree("files")

    elif radio_upload == "Predictions on Drive Dataset":
        st.subheader("Blood Vessel Segmentation Results on DRIVE dataset")
        directory = os.listdir("results_75")
        with st.expander("View Images"):

            # st.write("-----------------Image--------------------------------Groundtruth-------------------------Prediction------------------")
            # drive_table1 = pd.read_csv("results_75_csv/display_drive.csv")
            # st.table(drive_table1)

            with open("table.md") as table_file:
                st.markdown(table_file.read(), unsafe_allow_html=True)

            for files in directory:
                new_image = Image.open(f"results_75/{files}")
                st.write(files[0:7])
                st.image(new_image)

        with st.expander("View Scores"):
            df_drive = pd.read_csv("results_75_csv/score_75.csv")
            st.table(df_drive)

        st.subheader("Optic Disc Segmentation Results On DRIVE dataset")
        directory = os.listdir("results_75_od")

        with st.expander("View Images"):
            # st.write("-----------------Image--------------------------------Groundtruth-------------------------Prediction------------------")

            with open("table.md") as table_file:
                st.markdown(table_file.read(), unsafe_allow_html=True)

            for files in directory:
                new_image = Image.open(f"results_75_od/{files}")
                st.write(files[0:7])
                st.image(new_image)

        with st.expander("View Scores"):
            df_drive = pd.read_csv("results_75_csv/score_75_od.csv")
            st.table(df_drive)

    elif radio_upload == "Predictions on HRF Dataset":
        st.subheader("Blood Vessel Segmentation Results on HRF dataset")
        directory = os.listdir("results_hrf")

        with st.expander("View Images"):
            # st.write("-----------------Image--------------------------------Groundtruth-------------------------Prediction------------------")

            with open("table.md") as table_file:
                st.markdown(table_file.read(), unsafe_allow_html=True)

            for files in directory:
                new_image = Image.open(f"results_hrf/{files}")
                st.write(files[0:5])
                st.image(new_image)

        with st.expander("View Scores"):
            df_drive = pd.read_csv("results_hrf_csv/final_score_5_aspect.csv")
            st.table(df_drive)

        st.subheader("Optic Disc Segmentation Results On HRF dataset: ")
        directory = os.listdir("results_hrf_od")
        # print(directory)

        with st.expander("View Images"):
            # st.write("-----------------Image--------------------------------Groundtruth-------------------------Prediction------------------")

            with open("table.md") as table_file:
                st.markdown(table_file.read(), unsafe_allow_html=True)

            for files in directory:
                new_image = Image.open(f"results_hrf_od/{files}")
                st.write(files[0:5])
                st.image(new_image)

        with st.expander("View Scores"):
            df_drive = pd.read_csv("results_hrf_csv/final_score_5_od.csv")
            st.table(df_drive)


elif _radio == "Enter Your Feedback":

    if not os.path.isdir("feedback"):
        os.mkdir("feedback")

    # Run this first
    # file_count = open('feedback/count_track.txt', 'w')
    # file_count.write("0")
    # file_count.close()

    email = st.text_area("Enter Your Email ID (Optional)", height=10)
    message = st.text_area("Enter Your Feedback", height=100)

    if st.button('Submit Feedback'):
        st.write("Thank You For submitting the feedback! Your feedback has been registered sucessfully!")

        # Calculating the total count
        count_no = open('feedback/count_track.txt', 'r')
        final_count = count_no.readline()
        final_count = int(final_count) + 1
        count_no.close()

        # Run this next
        file_count = open('feedback/count_track.txt', 'w')
        file_count.write(str(final_count))
        file_count.close()

        # Calculating the total count
        count_no = open('feedback/count_track.txt', 'r')
        total_count = count_no.readline()
        count_no.close()

        # Final Feedback
        file_feedback = open(f'feedback/feedback_{total_count}.txt', 'w')
        file_feedback.write(message)
        file_feedback.close()

        # Calculating the total count for email
        email_count_no = open('feedback/email_count_track.txt', 'r')
        email_final_count = count_no.readline()
        email_final_count = int(final_count) + 1
        email_count_no.close()

        # Run this next
        email_file_count = open('feedback/email_count_track.txt', 'w')
        email_file_count.write(str(final_count))
        email_file_count.close()

        # Calculating the total count
        email_count_no = open('feedback/email_count_track.txt', 'r')
        email_total_count = count_no.readline()
        email_count_no.close()

        # Final Feedback
        email_file_feedback = open(f'feedback/email_{total_count}.txt', 'w')
        email_file_feedback.write(message)
        email_file_feedback.close()