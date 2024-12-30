# Common utility functions for processing fundal images
from typing import Tuple
import os
import numpy as np
from PIL import Image

import glob
import cv2

def generate_patches(images:np.ndarray,
                     patch_size:Tuple
                     ):
    pass


def generate_od_gt(input_path, output_path, mask_path):
    """
    Reads the manually annotated images and generates a binary mask images
    :param input_path: 
    :param output_path: 
    :return: None
    """
    input_path += "*"
    for file in glob.glob(input_path):
        print("Reading from file", file)
        img = cv2.imread(file)
        print(img.shape)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # (thresh, img_bw) = cv2.threshold(img_gray, maxval=255, thresh=128, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh, img_bw = cv2.threshold(img_gray, maxval=255, thresh=10, type=cv2.THRESH_BINARY)

        out_filename = file.rsplit("\\", 1)[1]
        print(out_filename)
        out_filename_without_ext = out_filename.rsplit(".", 1)[0]
        mask_file_name = out_filename_without_ext + "_mask.gif"
        # mask_file_name = out_filename
        output_file_name_with_full_path = output_path + out_filename_without_ext + ".png"
        mask_file_name_with_path = mask_path + mask_file_name
        print(f"reading mask {mask_file_name_with_path}")
        # img_mask = cv2.imread(mask_file_name_with_path)
        img_mask = np.asarray(Image.open(mask_file_name_with_path))
        # img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)


        print(img_mask.shape)
        print(np.min(img_mask), np.max(img_mask))

        indices_of_0 = img_bw < 128
        back_ground_indices = img_mask < 128
        print(img_mask.shape)
        inverted_image = np.zeros(img_bw.shape)
        inverted_image[indices_of_0] = 255
        inverted_image[back_ground_indices] = 0

        cv2.imwrite(output_file_name_with_full_path, inverted_image)


def combine_od_and_bv_gt(od_path, bv_path, mask_path, out_path):
    od_files = sorted(os.listdir(od_path))
    bv_files = sorted(os.listdir(bv_path))
    mask_files = sorted(os.listdir(mask_path))
    count = 0
    for i, j, k in zip(od_files, bv_files, mask_files):
        print(od_files, bv_files)
        image_od = Image.open(od_path + "/" + i)
        image_bv = Image.open(bv_path + "/" + j)
        image_mask = Image.open(mask_path + "/" + k)

        od_np = np.asarray(image_od)
        bv_np = np.asarray(image_bv)
        mask_np = np.asarray(image_mask)

        print(od_np.shape)
        print(bv_np.shape)
        count += 1

        # Create a combined image for ground truth with two output channels
        color_image_shape = list(od_np.shape)
        combined_image = np.zeros( (od_np.shape[0], od_np.shape[1], 3) )
        # set all retina pixels to white
        # combined_image[retina_indices] = 255
        if len(od_np.shape) == 3:
            combined_image[:, :, 0] = od_np[:,:,0]
        else:
            combined_image[:, :, 0] = od_np

        if len(bv_np.shape) == 3:
            combined_image[:, :, 1] = bv_np[:,:,0]
        else:
            combined_image[:, :, 1] = bv_np


        out_filename = out_path + "/" + i
        print(f"Writing output to {out_filename}")
        #set Blood
        cv2.imwrite(out_path + "/" + i,  combined_image)

        retina_indices = mask_np > 128
        non_retina_indices = mask_np <= 128

        blood_vessel_indices = bv_np > 128
        non_blood_vessel_indices = bv_np <= 128

        od_indices = bv_np > 128
        non_od_indices = bv_np <= 128

        # GT for 4 channels

        # np3 = numpy_data1 + numpy_data2
        #
        # data = Image.fromarray(np3)
        #
        # data.save(dir1 + str(count) + '.png')


def remove_mask_border(in_path, out_path, max_width, max_height, min_width, min_height):
    #max_height = 70
    image_files = sorted(os.listdir(in_path))
    for file in image_files:
        image = cv2.imread(in_path + "/" + file)
        print(image.shape)
        column_wise_sum = np.sum(image, axis=(1,2))
        row_with_max_white_pixels = np.argmax(column_wise_sum)
        row_wise_sum = np.sum(image, axis=(0,2))
        column_with_max_white_pixels = np.argmax(row_wise_sum)
        width = min_width
        height = min_height


        # print(row_with_max_white_pixels)
        # print(column_with_max_white_pixels)

        cv2.rectangle(image,
                      (column_with_max_white_pixels - width, row_with_max_white_pixels - height),
                      (column_with_max_white_pixels + width, row_with_max_white_pixels + height), (0, 0, 255), 2)

        cv2.rectangle(image, (10, 10), (image.shape[1], row_with_max_white_pixels- width), (255, 0, 0), 2 )
        print("image shape",image.shape, height, width, row_with_max_white_pixels + height)
        print(row_with_max_white_pixels, row_with_max_white_pixels + height)
        print(column_with_max_white_pixels, row_with_max_white_pixels + width)

        num_white_pixels_below = np.sum(image[row_with_max_white_pixels + height,
                           column_with_max_white_pixels - width:column_with_max_white_pixels + width]
                                  )
        print(file, num_white_pixels_below)

        while num_white_pixels_below > 0 and height < max_height:
            height += 1
            num_white_pixels_below = np.sum(image[row_with_max_white_pixels + height,
                                            column_with_max_white_pixels - width:column_with_max_white_pixels + width] > 128
                                            )
            #print("height", height)

        num_white_pixels_left = np.sum(image[row_with_max_white_pixels - height : row_with_max_white_pixels + height,
                                        column_with_max_white_pixels - width] > 128
                                        )
        print("num_white_pixels_left",num_white_pixels_left)
        while num_white_pixels_left > 0 and width < max_width:
            width += 1
            num_white_pixels_left = np.sum(image[row_with_max_white_pixels - height : row_with_max_white_pixels + height,
                                            column_with_max_white_pixels - width] > 128
                                            )
            print("width", width, num_white_pixels_left)

        print(row_with_max_white_pixels)
        print(column_with_max_white_pixels)

        num_white_pixels_right = np.sum(image[row_with_max_white_pixels - height : row_with_max_white_pixels + height,
                                        column_with_max_white_pixels + width] > 128
                                        )
        print("num_white_pixels_right",num_white_pixels_right)
        while num_white_pixels_right > 0 and width < max_width:
            width += 1
            num_white_pixels_right = np.sum(image[row_with_max_white_pixels - height : row_with_max_white_pixels + height,
                                            column_with_max_white_pixels + width] > 128
                                            )
            print("width", width, num_white_pixels_right)
        


        # cv2.rectangle(image,
        #               (column_with_max_white_pixels - width, row_with_max_white_pixels - height),
        #               (column_with_max_white_pixels + width, row_with_max_white_pixels + height), (0, 255, 0), 2)
        # cv2.rectangle(image, (10, 10), (image.shape[1], row_with_max_white_pixels- height), (255, 0, 0), 2 )


        print(file, num_white_pixels_below)

        image[0: row_with_max_white_pixels - height,:] = 0
        image[row_with_max_white_pixels + height:,:] = 0

        image[:, 0:column_with_max_white_pixels - width:] = 0
        image[:, column_with_max_white_pixels + width:] = 0

        print(row_with_max_white_pixels, column_with_max_white_pixels)
        cv2.imwrite(out_path + "/" + file, image)