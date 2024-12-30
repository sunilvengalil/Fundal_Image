# Common utility functions for processing fundal images
from typing import List
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import glob

import cv2
from albumentations import HorizontalFlip, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion

W, H = 512, 512

supported_extensions = ["png", "jpg", "tif", "jpeg", "gif"]

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def image_files(path, file_prefix=None):
    all_image_files = []
    if file_prefix is None:
        file_names = list(glob.glob(path + "/*.*" ))
    else:
        prefix___ = path + "/" + file_prefix + ".*"
        print(prefix___)
        file_names = list(glob.glob(prefix___))


    for filename in file_names:
        ext = filename.rsplit(".", 1)[1]
        if ext in supported_extensions:
            all_image_files.append(filename)
    return sorted(all_image_files)


def get_filenames_sorted(image_path, gt_path=None):
    image_file_names = image_files(image_path)
    gt_file_names = None
    if gt_path is not None:
        # Get the test files also
        gt_file_names = sorted(image_files(gt_path))
    return image_file_names, gt_file_names


def get_filenames_sorted_train_test(path):
    """
    Use this api for getting the image and ground truth filenames
    :param path: Base path where drive images are stored
    :return: List of filenames in sorted order
    """
    print(path)
    print(os.path.join(path, "training", "images", "*.tif"))
    train_x = sorted(glob.glob(os.path.join(path, "training", "images", "*.tif")))
    train_y = sorted(glob.glob(os.path.join(path, "training", "od_and_bv_gt_combined", "*.png")))

    test_x = sorted(glob.glob(os.path.join(path, "test", "images", "*.tif")))
    test_y = sorted(glob.glob(os.path.join(path, "test", "od_and_bv_gt_combined", "*.png")))

    return (train_x, train_y), (test_x, test_y)


def create_data(images_file_names, gt_filenames, save_path, augment=False):
    for idx, (x, y) in tqdm(enumerate(zip(images_file_names, gt_filenames)), total=len(images_file_names)):
        print(x,y)
        """ Extracting names """
        name = x.rsplit("/", 1)[1].rsplit(".", 1)[0]
        print("name", name)

        # """ Reading image and mask """
        # if rsplit(".", 1)[1]  == "gif":
        #     X = Image.open(x)

        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = GridDistortion(p=1)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            X = [x, x1, x2, x3, x4, x5]
            Y = [y, y1, y2, y3, y4, y5]
        else:
            X = [x]
            Y = [y]
        index = 0

        # Resizing and saving the images and gt as png
        for i, m in zip(X, Y):
            print(i.shape, m.shape)
            i = cv2.resize(i, (W, H))
            m = cv2.resize(m, (W, H))

            if len(X) == 1:
                tmp_image_name = f"{name}.png"
                tmp_mask_name = f"{name}.png"
            else:
                tmp_image_name = f"{name}_{index}.png"
                tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)
            print("Saving to ", image_path)
            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)
            index += 1


def generate_od_and_bv_gt(od_gt_path, bv_gt_path, output_path, mask_path, mask_ext="gif"):
    """
    Reads the manually annotated images and generates a binary mask images
    :param od_gt_path: Path to location where OD ground truth images are saved
    :param bv_gt_path: Path to location where BV ground truth images are saved
    :param output_path: Path to save the combined images
    :param mask_path: Path where retinal mask images are saved
    :return: None
    """
    gt_images, files = _generate_od_gt(input_path=od_gt_path, output_path=None, mask_path=mask_path, ext=mask_ext)
    out_files = [file.rsplit("/", 1)[1] for file in files]
    images_border_removed = remove_mask_border(input_path=None, output_path=None, images=gt_images, out_file_names=None)
    combine_binary_images(images_border_removed, bv_gt_path, output_path, od_files=out_files)


def generate_od_gt(input_path, output_path, mask_path):
    """
    Reads the manually annotated images and generates a binary mask images
    :param input_path:
    :param output_path:
    :return: None
    """
    gt_images, files = _generate_od_gt(input_path=input_path, output_path=None, mask_path=mask_path)
    out_file_names = [file.rsplit("/", 1)[1] for file in files]
    remove_mask_border(input_path=None, output_path=output_path, images=gt_images, out_file_names=out_file_names)


def _generate_od_gt(input_path, output_path, mask_path, ext="gif") -> (List[np.ndarray], List[str]):
    """
    Reads the manually annotated images and generates a binary mask images
    :param input_path: 
    :param output_path: 
    :return: gt_images, files
    """
    files = image_files(input_path)
    if len(files) == 0:
        raise Exception(f"No image files present in directory {input_path}. Supported image files are {supported_extensions}")

    if output_path is not None and not os.path.isdir(output_path):
        os.mkdir(output_path)

    gt_images = []
    for file in files:
        print("Reading from file", file)
        img = cv2.imread(file)
        print(img.shape)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh, img_bw = cv2.threshold(img_gray, maxval=255, thresh=10, type=cv2.THRESH_BINARY)

        out_filename = file.rsplit("/", 1)[1]
        out_filename_without_ext = out_filename.rsplit(".", 1)[0]
        mask_file_name = out_filename_without_ext + f"_mask.{ext}"

        mask_file_name_with_path = mask_path + mask_file_name
        print(f"reading mask {mask_file_name_with_path}")
        img_mask = np.asarray(Image.open(mask_file_name_with_path))
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
        print("Mask shape", img_mask.shape, np.min(img_mask), np.max(img_mask))

        # print(f"Mask Image shape: {img_mask.shape}" + " Minimum:" + str(np.min(img_mask)) +" Maximum:"+ str(np.max(img_mask)))

        indices_of_0 = img_bw < 128
        back_ground_indices = img_mask < 128
        # print(img_mask.shape)
        inverted_image = np.zeros(img_bw.shape, dtype=np.float32)
        inverted_image[indices_of_0] = 255.0
        inverted_image[back_ground_indices] = 0.0
        inverted_image = cv2.cvtColor(inverted_image, cv2.COLOR_GRAY2BGR)
        gt_images.append(inverted_image)

        if output_path is not None and os.path.isdir(output_path):
            output_file_name_with_full_path = output_path + out_filename_without_ext + ".png"
            cv2.imwrite(output_file_name_with_full_path, inverted_image)
    return gt_images, files


def load_images(path, file_prefix=None):
    files = image_files(path, file_prefix)
    if len(files) == 0:
        raise Exception(f"No images present in path {path}. Supported image formats are {supported_extensions}")
    images = []
    for file in files:
        image = np.asarray(Image.open(file))
        images.append(image)
    file_names = [file.rsplit("/", 1)[1] for file in files]
    return images, file_names


def validate_image_path_or_images(od, od_files):
    if od is None:
        raise Exception("Parameter od should not be None")
    if isinstance(od, str):
        od_images, od_files = load_images(od)
    elif isinstance(od, List) and len(od) > 0:
        if od_files is None or len(od_files) == 0:
            raise Exception("Filenames should be should be passed in prameter od_files")
        od_images = od
    else:
        raise Exception("Parameter od ")

    return od_images, od_files


def combine_binary_images(od, bv_path, out_path, od_files=None,):
    od_images, od_files = validate_image_path_or_images(od, od_files)
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    for image_no, image_od in enumerate(od_images):
        bv_file_name = od_files[image_no].rsplit(".", 1)[0].replace("test", "manual1")
        print(od_files[image_no], bv_file_name)
        images_bv, bv_file_names = load_images(bv_path, bv_file_name)
        if len(images_bv) > 1:
            raise Exception("Multiple bv ground truth file present with same name and different format. Files found bv_file_names")
        image_bv = images_bv[0]

        # Create a combined image for ground truth with two output channels
        combined_image = np.zeros( (image_od.shape[0], image_od.shape[1], 3), dtype=np.float32 )
        if len(image_od.shape) == 3:
            combined_image[:, :, 0] = image_od[:,:,0]
        else:
            combined_image[:, :, 0] = image_od

        if len(image_bv.shape) == 3:
            combined_image[:, :, 1] = image_bv[:,:,0]
        else:
            combined_image[:, :, 1] = image_bv

        out_file_name = out_path + "/" + od_files[image_no].rsplit(".", 1)[0] + ".png"
        print(f"OD Shape:{image_od.shape} BV Shape:{image_bv.shape}  Combined shape:{combined_image.shape}")
        print(f"Writing output to {out_file_name}")
        cv2.imwrite(out_file_name, combined_image)


def remove_mask_border(input_path,
                       output_path=None,
                       images=None,
                       out_file_names=None,
                       max_width=400,
                       max_height=400,
                       min_width=20,
                       min_height=20
                       ):
    if output_path is not None and not os.path.isdir(output_path):
        os.mkdir(output_path)

    input_image_filenames = []
    if (images is None or len(images) == 0) and input_path is not None and os.path.isdir(input_path):
        images, input_image_filenames = load_images(input_path)

    # print("Processing "+ str(len(images)) + "images")
    result_images = []
    for image_no, image in enumerate(images):
        # print(image.shape, np.min(image), np.max(image))
        column_wise_sum = np.sum(image, axis=(1, 2))
        row_with_max_white_pixels = np.argmax(column_wise_sum)
        row_wise_sum = np.sum(image, axis=(0,2))
        column_with_max_white_pixels = np.argmax(row_wise_sum)
        width = min_width
        height = min_height

        # cv2.rectangle(image,
        #               (column_with_max_white_pixels - width, row_with_max_white_pixels - height),
        #               (column_with_max_white_pixels + width, row_with_max_white_pixels + height), (0, 0, 255), 2)


        # cv2.rectangle(image, (10, 10), (image.shape[1], row_with_max_white_pixels- width), (255, 0, 0), 2 )
        # print("image shape",image.shape, height, width, row_with_max_white_pixels + height)
        # print(row_with_max_white_pixels, row_with_max_white_pixels + height)
        # print(column_with_max_white_pixels, row_with_max_white_pixels + width)

        num_white_pixels_below = np.sum(image[row_with_max_white_pixels + height,
                           column_with_max_white_pixels - width:column_with_max_white_pixels + width]
                                  )
        # print(file, num_white_pixels_below)

        while num_white_pixels_below > 0 and height < max_height:
            height += 1
            num_white_pixels_below = np.sum(image[row_with_max_white_pixels + height,
                                            column_with_max_white_pixels - width:column_with_max_white_pixels + width] > 128
                                            )
            # print("height", height)

        num_white_pixels_left = np.sum(image[row_with_max_white_pixels - height : row_with_max_white_pixels + height,
                                        column_with_max_white_pixels - width] > 128
                                        )
        # print("num_white_pixels_left",num_white_pixels_left)
        while num_white_pixels_left > 0 and width < max_width:
            width += 1
            num_white_pixels_left = np.sum(image[row_with_max_white_pixels - height : row_with_max_white_pixels + height,
                                            column_with_max_white_pixels - width] > 128
                                            )
            # print("width", width, num_white_pixels_left)

        num_white_pixels_right = np.sum(image[row_with_max_white_pixels - height : row_with_max_white_pixels + height,
                                        column_with_max_white_pixels + width] > 128
                                        )
        # print("num_white_pixels_right",num_white_pixels_right)
        while num_white_pixels_right > 0 and width < max_width:
            width += 1
            num_white_pixels_right = np.sum(image[row_with_max_white_pixels - height : row_with_max_white_pixels + height,
                                            column_with_max_white_pixels + width] > 128
                                            )
            # print("width", width, num_white_pixels_right)

        # cv2.rectangle(image,
        #               (column_with_max_white_pixels - width, row_with_max_white_pixels - height),
        #               (column_with_max_white_pixels + width, row_with_max_white_pixels + height), (0, 255, 0), 2)
        # cv2.rectangle(image, (10, 10), (image.shape[1], row_with_max_white_pixels- height), (255, 0, 0), 2 )
        # print(file, num_white_pixels_below)

        image[0: row_with_max_white_pixels - height, :] = 0
        image[row_with_max_white_pixels + height:, :] = 0
        image[:, 0:column_with_max_white_pixels - width:] = 0
        image[:, column_with_max_white_pixels + width:] = 0

        # print(row_with_max_white_pixels, column_with_max_white_pixels)
        result_images.append(image)
        if output_path is not None and os.path.isdir(output_path):
            out_file_name = None
            if input_image_filenames is not None and len(input_image_filenames) > 0:
                out_file_name = input_image_filenames[image_no].rsplit(".", 1)
            elif out_file_names is not None and len(out_file_names) > 0:
                out_file_name, ext = out_file_names[image_no].rsplit(".", 1)

            if out_file_name is not None:
                out_file_png = out_file_name + ".png"
                print(f"Saving image to {out_file_png}")
                cv2.imwrite(output_path + "/" + out_file_png, image)
    return result_images


if __name__ == "__main__":
    root_path = "/Users/sunilv/gitprojects/medical_image_processing/fundal_image/"
    drive_test_images = image_files(f"{root_path}datasets/drive/test/images/")
    print(drive_test_images)