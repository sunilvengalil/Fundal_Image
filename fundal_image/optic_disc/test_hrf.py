from fundal_image.utils import _generate_od_gt, combine_binary_images, remove_mask_border, generate_od_gt, generate_od_and_bv_gt
#root_path = "D:/IIIT-B/Pycharm/fundal_image/datasets/HRF/"
root_path = "/Users/sunilv/gitprojects/medical_image_processing/fundal_image/datasets/HRF/"
datasets = ["training", "test"]


"""
HRF
"""

for dataset in datasets:
    bv_directory = root_path + f"{dataset}/BV_Masks/"
    od_directory = root_path + f"{dataset}/groundtruth/"
    out_directory = root_path + f"{dataset}/od_and_bv_gt_combined/"

    generate_od_and_bv_gt(od_gt_path = root_path + f"{dataset}/od_annotations/",
                          bv_gt_path=bv_directory,
                          output_path = out_directory,
                          mask_path = root_path  + f"{dataset}/mask/",
                          mask_ext="tif"
                          )