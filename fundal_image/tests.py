from fundal_image.utils import _generate_od_gt, combine_binary_images, remove_mask_border, generate_od_gt, generate_od_and_bv_gt
root_path = "/Users/sunilv/gitprojects/medical_image_processing/"
split_name = ["training", "test"]
dataset = "idrid"

# """
# IDRID Dataset
# https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
# """
#
# split_name = "test"
# od_directory = root_path + f"fundal_image/datasets/{dataset}/{split_name}/od_gt_final/"
# exudates_directory = root_path + f"fundal_image/datasets/{dataset}/{split_name}/exudate_gt/"
# out_directory = root_path + f"fundal_image/datasets/{dataset}/{split_name}/od_and_exudate_gt_combined/"
#
# combine_binary_images(od_directory, exudates_directory, out_directory, suffix_1="OD", suffx_2="EX")


# """
# DRIVE
# """
# datasets = ["test"]
#
# for dataset in datasets:
#     bv_directory = root_path + f"fundal_image/datasets/drive/{dataset}/groundtruth1/"
#     od_directory = root_path + f"fundal_image/datasets/drive/{dataset}/od_gt_final1/"
#     out_directory = root_path + f"fundal_image/datasets/drive/{dataset}/od_and_bv_gt_combined1/"
#
#     generate_od_and_bv_gt(od_gt_path = root_path + f"fundal_image/datasets/drive/{dataset}/od_annotations/",
#                           bv_gt_path=bv_directory,
#                           output_path = out_directory,
#                           mask_path = root_path  + f"fundal_image/datasets/drive/{dataset}/mask/",
#                           )


# """
# CHASE_DB
# """
datasets = ["chase_db"]
split_names =["split_1/training"]

for dataset in datasets:
    for split_name in split_names:
        bv_directory = root_path + f"fundal_image/datasets/{dataset}/{split_name}/groundtruth1/"
        od_directory = root_path + f"fundal_image/datasets/{dataset}/{split_name}/od_gt_final1/"
        out_directory = root_path + f"fundal_image/datasets/{dataset}/{split_name}/od_and_bv_gt_combined1/"
        od_gt_path_op = root_path +  f"fundal_image/datasets/{dataset}/{split_name}/od_and_mask/"
        od_gt_path_with_border_op = root_path +  f"fundal_image/datasets/{dataset}/{split_name}/od_and_mask_with_border/"

        generate_od_and_bv_gt(od_gt_path = root_path + f"fundal_image/datasets/{dataset}/{split_name}/od_annotations/",
                              bv_gt_path=bv_directory,
                              output_path = out_directory,
                              mask_path = root_path  + f"fundal_image/datasets/{dataset}/{split_name}/mask/",
                              od_gt_path_op=od_gt_path_op,
                              od_gt_path_with_border_op=od_gt_path_with_border_op
                              )


# for dataset in datasets:
#     generate_od_gt(input_path = root_path + f"fundal_image/datasets/drive/{dataset}/od_annotations/",
#                    output_path = root_path + f"fundal_image/datasets/drive/{dataset}/od_gt_final1/",
#                    mask_path = root_path  + f"fundal_image/datasets/drive/{dataset}/mask/"
#                    )
#
# for dataset in datasets:
#
#     combine_binary_images(od_directory,
#                           bv_directory,
#                           out_directory
#                          )

# for dataset in datasets:
#     _generate_od_gt(input_path = root_path + f"fundal_image/datasets/drive/{dataset}/od_annotations/",
#                    output_path = root_path + f"fundal_image/datasets/drive/{dataset}/od_and_mask/",
#                    mask_path = root_path  + f"fundal_image/datasets/drive/{dataset}/mask/"
#                    )


# for dataset in datasets:
#     remove_mask_border(in_path=root_path + f"fundal_image/datasets/drive/{dataset}/od_and_mask/",
#                        out_path=root_path + f"fundal_image/datasets/drive/{dataset}/od_gt_final/"
#                        )


"""
diabetic retinopathy
"""

# generate_od_gt(input_path = root_path + f"fundal_image/datasets/diabetic_retinopathy_test/od_annotations/",
#                output_path = root_path + f"fundal_image/datasets/diabetic_retinopathy_test/od_and_mask/",
#                mask_path = root_path  + f"fundal_image/datasets/diabetic_retinopathy_test/mask/"
#                )


# remove_mask_border(root_path + f"fundal_image/datasets/diabetic_retinopathy_test/od_and_mask",
#                    root_path + f"fundal_image/datasets/diabetic_retinopathy_test/od_gt_final",
#                    400,
#                    400,
#                    20,
#                    20
#                    )
#
#
# bv_directory = root_path + f"fundal_image/datasets/diabetic_retinopathy_test//groundtruth1"
# od_directory = root_path + f"fundal_image/datasets/diabetic_retinopathy_test/od_gt_final"
# mask_directory = root_path + f"fundal_image/datasets/diabetic_retinopathy_test/mask"
# out_directory = root_path + f"fundal_image/datasets/diabetic_retinopathy_test/od_and_bv_gt_combined"

# combine_binary_images(od_directory,
#                      bv_directory,
#                      mask_directory,
#                      out_directory
#                      )
