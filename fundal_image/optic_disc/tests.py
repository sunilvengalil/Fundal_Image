from utils import generate_od_gt, combine_od_and_bv_gt, remove_mask_border
root_path = ""
datasets = ["training", "test"]


"""
DRIVE
"""
# for dataset in datasets:
#     generate_od_gt(input_path = root_path + f"fundal_image/datasets/drive/{dataset}/od_annotations/",
#                    output_path = root_path + f"fundal_image/datasets/drive/{dataset}/od_and_mask/",
#                    mask_path = root_path  + f"fundal_image/datasets/drive/{dataset}/mask/"
#                    )
#
# for dataset in datasets:
#     remove_mask_border(root_path + f"fundal_image/datasets/drive/{dataset}/od_and_mask",
#                        root_path + f"fundal_image/datasets/drive/{dataset}/od_gt_final"
#                        )

# for dataset in datasets:
#     generate_od_gt(input_path = root_path + f"{dataset}",
#                    output_path = root_path + f"{dataset}",
#                    mask_path = root_path  + f"{dataset}"
#                    )

# for dataset in datasets:
#     remove_mask_border(root_path + f"{dataset}",
#                        root_path + f"{dataset}",
#                        300, 
#                        300,
#                        15,
#                        15
#                        )

# for dataset in datasets:
od_ground = root_path + f"ACTUAL_OD_GROUND_CORRECT"
od_preds = root_path + f"TEST_Predictions_DRIVE"
mask_directory = root_path + f"mask"
out_directory = root_path + f"Combined_Analysis"

combine_od_and_bv_gt(od_ground,
                    od_preds,
                    mask_directory,
                    out_directory
                    )

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


# bv_directory = root_path + f"fundal_image/datasets/diabetic_retinopathy_test//groundtruth1"
# od_directory = root_path + f"fundal_image/datasets/diabetic_retinopathy_test/od_gt_final"
# mask_directory = root_path + f"fundal_image/datasets/diabetic_retinopathy_test/mask"
# out_directory = root_path + f"fundal_image/datasets/diabetic_retinopathy_test/od_and_bv_gt_combined"

# combine_od_and_bv_gt(od_directory,
#                      bv_directory,
#                      mask_directory,
#                      out_directory
#                      )
