"""
This Python File contains Functions to use YOLACT-Model from xXAI-botXx

Make sure to run this file in yolact folder or one folder on top

Features: Training, Inference, Evaluation, Visualization, Installation-Checking

Recommended Environment:
Python = 3.12.3
PyTorch = 2.3.0
"""



###############
### Imports ### 
###############
import sys
sys.path.append("./")
sys.path.append("./comfortable-yolact")

# Common Utils
import os
import json
import argparse

# Image Utils
import numpy as np
import cv2
import matplotlib.pyplot as plt

# PyTorch & Torch-Utils
import torch

from torch.utils.data import DataLoader

# Model imports
from yolact import Yolact
from data import Config
from backbone import ResNetBackbone
from layers.output_utils import postprocess



#############
### Utils ###
#############
# def clear_output():
#     """
#     Clears the print output
#     """
#     os.system('cls')



def get_device():
    """
    Returns a torch device, gpu if available else cpu
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device



def get_configuration(name="inference",
                      max_size=550,
                      decay=5e-4,
                      gamma=0.1,
                      lr_steps=(280000, 600000, 700000, 750000),
                      lr_warmup_init=1e-4,
                      lr_warmup_until=500,
                      freeze_bn=True,
                      conf_alpha=1,
                      bbox_alpha=1.5,
                      mask_alpha=0.4 / 256 * 140 * 140, 
                      use_semantic_segmentation_loss=True,
                      semantic_segmentation_alpha=1,
                      use_mask_scoring=False,
                      mask_scoring_alpha=1,
                      use_focal_loss=False,
                      focal_loss_alpha=0.25,
                      focal_loss_gamma=2,
                      focal_loss_init_pi=0.01,
                      max_num_detections=100,
                      eval_mask_branch=True,
                      nms_top_k=200,
                      nms_conf_thresh=0.005,
                      nms_thresh=0.5,
                      mask_type=1,
                      mask_size=6.125,
                      masks_to_train=100,
                      mask_proto_src=0,
                      mask_proto_net=[(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})],
                      mask_proto_bias=False,
                      mask_proto_prototype_activation="relu",
                      mask_proto_mask_activation="sigmoid",
                      mask_proto_coeff_activation="tanh",
                      mask_proto_crop=True,
                      mask_proto_crop_expand=0,
                      mask_proto_loss=None,
                      mask_proto_binarize_downsampled_gt=True,
                      mask_proto_normalize_mask_loss_by_sqrt_area=False,
                      mask_proto_reweight_mask_loss=False,
                      mask_proto_grid_file='data/grid.npy',
                      mask_proto_use_grid= False,
                      mask_proto_coeff_gate=False,
                      mask_proto_prototypes_as_features=False,
                      mask_proto_prototypes_as_features_no_grad=False,
                      mask_proto_remove_empty_masks=False,
                      mask_proto_reweight_coeff=1,
                      mask_proto_coeff_diversity_loss=False,
                      mask_proto_coeff_diversity_alpha=1,
                      mask_proto_normalize_emulate_roi_pooling=True,
                      mask_proto_double_loss=False,
                      mask_proto_double_loss_alpha=1,
                      mask_proto_split_prototypes_by_head=False,
                      mask_proto_crop_with_pred_box=False,
                      mask_proto_debug=False,
                      discard_box_width=4 / 550,
                      discard_box_height=4 / 550,
                      share_prediction_module=True,
                      ohem_use_most_confident=False,
                      use_class_balanced_conf=False,
                      use_sigmoid_focal_loss=False,
                      use_objectness_score=False,
                      use_class_existence_loss=False,
                      class_existence_alpha=1,
                      use_change_matching=False,
                      extra_head_net=[(256, 3, {'padding': 1})],
                      head_layer_params={'kernel_size': 3, 'padding': 1},
                      extra_layers=(0, 0, 0),
                      positive_iou_threshold=0.5,
                      negative_iou_threshold=0.4,
                      ohem_negpos_ratio=3,
                      crowd_iou_threshold=0.7,
                      force_cpu_nms=True,
                      use_coeff_nms=False,
                      use_instance_coeff=False,
                      num_instance_coeffs=64,
                      train_masks=True,
                      train_boxes=True,
                      use_gt_bboxes=False,
                      preserve_aspect_ratio=False,
                      use_prediction_module=False,
                      use_yolo_regressors=False,
                      use_prediction_matching=False,
                      delayed_settings=[],
                      no_jit=False,
                      mask_dim=None,
                      use_maskiou=True,
                      maskiou_net=[(8, 3, {'stride': 2}), 
                                   (16, 3, {'stride': 2}), 
                                   (32, 3, {'stride': 2}), 
                                   (64, 3, {'stride': 2}), 
                                   (128, 3, {'stride': 2})],
                      discard_mask_area=5*5, # -1,
                      maskiou_alpha=25, # 6.125,
                      rescore_mask=True,
                      rescore_bbox=False,
                      maskious_to_train=-1,
                      augment_photometric_distort=True,
                      augment_expand=True,
                      augment_random_sample_crop=True,
                      augment_random_mirror=True,
                      augment_random_flip=False,
                      augment_random_rot90=False,
                      data_name="inference data",
                      data_has_gt=False,
                      data_class_names=["object"]*80,
                      backbone_name="ResNet101",
                      backbone_weight_path="resnet101_reducedfc.pth",
                      backbone_type=ResNetBackbone,
                      backbone_args=([3, 4, 23, 3],),
                      backbone_channel_order='RGB',
                      backbone_normalize=True,
                      backbone_substract_means=False,
                      backbone_to_float=False,
                      backbone_selected_layers=list(range(1, 4)),
                      backbone_pred_scales=[[24], [48], [96], [192], [384]],
                      backbone_pred_aspect_ratios=[ [[1, 1/2, 2]] ]*5,
                      backbone_use_pixel_scales=True,
                      backbone_preapply_sqrt=False,
                      backbone_use_square_anchors=True,
                      fpn_num_features=256,
                      fpn_interpolation_mode='bilinear',
                      fpn_num_downsamples=2,
                      fpn_use_conv_downsample=True,
                      fpn_pad=True,
                      fpn_relu_downsample_layers=False,
                      fpn_relu_pred_layers=True):
    """
    Creates a configuration for the YOLACT model.

    See yolact/data/config.py for more details.
    Or here: https://github.com/xXAI-botXx/yolact/blob/master/data/config.py

    Parameters:
    name (str): The name of the configuration.
    max_size (int): Maximum size of the input images.
    decay (float): Weight decay parameter.
    gamma (float): Learning rate decay factor.
    lr_steps (tuple): Steps at which the learning rate is decayed.
    lr_warmup_init (float): Initial learning rate for warmup.
    lr_warmup_until (int): Number of iterations for learning rate warmup.
    freeze_bn (bool): Whether to freeze batch normalization layers.
    conf_alpha (float): Scaling factor for confidence loss.
    bbox_alpha (float): Scaling factor for bounding box loss.
    mask_alpha (float): Scaling factor for mask loss.
    use_semantic_segmentation_loss (bool): Whether to use semantic segmentation loss.
    semantic_segmentation_alpha (float): Scaling factor for semantic segmentation loss.
    use_mask_scoring (bool): Whether to use mask scoring.
    mask_scoring_alpha (float): Scaling factor for mask scoring loss.
    use_focal_loss (bool): Whether to use focal loss.
    focal_loss_alpha (float): Alpha parameter for focal loss.
    focal_loss_gamma (float): Gamma parameter for focal loss.
    focal_loss_init_pi (float): Initial pi parameter for focal loss.
    max_num_detections (int): Maximum number of detections per image.
    eval_mask_branch (bool): Whether to evaluate the mask branch.
    nms_top_k (int): Number of top scoring boxes to keep before NMS.
    nms_conf_thresh (float): Confidence threshold for NMS.
    nms_thresh (float): IoU threshold for NMS.
    mask_type (int): Type of mask to use.
    mask_size (float): Size of the mask.
    masks_to_train (int): Number of masks to train.
    mask_proto_src (int): Source layer for mask prototype.
    mask_proto_net (list): Network configuration for mask prototype.
    mask_proto_bias (bool): Whether to use bias in mask prototype network.
    mask_proto_prototype_activation (str): Activation function for mask prototype.
    mask_proto_mask_activation (str): Activation function for mask.
    mask_proto_coeff_activation (str): Activation function for mask coefficients.
    mask_proto_crop (bool): Whether to crop the mask prototypes.
    mask_proto_crop_expand (int): Amount to expand the crop.
    mask_proto_loss (None): Custom loss for mask prototypes.
    mask_proto_binarize_downsampled_gt (bool): Whether to binarize downsampled ground truth masks.
    mask_proto_normalize_mask_loss_by_sqrt_area (bool): Whether to normalize mask loss by the square root of the area.
    mask_proto_reweight_mask_loss (bool): Whether to reweight mask loss.
    mask_proto_grid_file (str): File path for mask prototype grid.
    mask_proto_use_grid (bool): Whether to use grid for mask prototype.
    mask_proto_coeff_gate (bool): Whether to use coefficient gate for mask prototype.
    mask_proto_prototypes_as_features (bool): Whether to use prototypes as features.
    mask_proto_prototypes_as_features_no_grad (bool): Whether to use prototypes as features without gradient.
    mask_proto_remove_empty_masks (bool): Whether to remove empty masks.
    mask_proto_reweight_coeff (float): Coefficient for reweighting mask loss.
    mask_proto_coeff_diversity_loss (bool): Whether to use coefficient diversity loss.
    mask_proto_coeff_diversity_alpha (float): Alpha parameter for coefficient diversity loss.
    mask_proto_normalize_emulate_roi_pooling (bool): Whether to normalize to emulate ROI pooling.
    mask_proto_double_loss (bool): Whether to use double loss for mask prototypes.
    mask_proto_double_loss_alpha (float): Alpha parameter for double loss.
    mask_proto_split_prototypes_by_head (bool): Whether to split prototypes by head.
    mask_proto_crop_with_pred_box (bool): Whether to crop with predicted box.
    mask_proto_debug (bool): Whether to enable debug mode for mask prototypes.
    discard_box_width (float): Minimum width to discard a box.
    discard_box_height (float): Minimum height to discard a box.
    share_prediction_module (bool): Whether to share prediction module.
    ohem_use_most_confident (bool): Whether to use most confident samples for OHEM.
    use_class_balanced_conf (bool): Whether to use class-balanced confidence.
    use_sigmoid_focal_loss (bool): Whether to use sigmoid focal loss.
    use_objectness_score (bool): Whether to use objectness score.
    use_class_existence_loss (bool): Whether to use class existence loss.
    class_existence_alpha (float): Alpha parameter for class existence loss.
    use_change_matching (bool): Whether to use change matching.
    extra_head_net (list): Additional layers for the head network.
    head_layer_params (dict): Parameters for head layer.
    extra_layers (tuple): Additional layers configuration.
    positive_iou_threshold (float): IoU threshold for positive samples.
    negative_iou_threshold (float): IoU threshold for negative samples.
    ohem_negpos_ratio (int): Ratio of negative to positive samples for OHEM.
    crowd_iou_threshold (float): IoU threshold for crowd samples.
    force_cpu_nms (bool): Whether to force CPU for NMS.
    use_coeff_nms (bool): Whether to use coefficient NMS.
    use_instance_coeff (bool): Whether to use instance coefficients.
    num_instance_coeffs (int): Number of instance coefficients.
    train_masks (bool): Whether to train masks.
    train_boxes (bool): Whether to train bounding boxes.
    use_gt_bboxes (bool): Whether to use ground truth bounding boxes.
    preserve_aspect_ratio (bool): Whether to preserve aspect ratio.
    use_prediction_module (bool): Whether to use prediction module.
    use_yolo_regressors (bool): Whether to use YOLO regressors.
    use_prediction_matching (bool): Whether to use prediction matching.
    delayed_settings (list): List of delayed settings.
    no_jit (bool): Whether to disable JIT.
    mask_dim (None): Dimension of the mask.
    use_maskiou (bool): Whether to use MaskIoU.
    maskiou_net (list): Network configuration for MaskIoU.
    discard_mask_area (int): Minimum area to discard a mask.
    maskiou_alpha (float): Alpha parameter for MaskIoU.
    rescore_mask (bool): Whether to rescore mask.
    rescore_bbox (bool): Whether to rescore bounding box.
    maskious_to_train (int): Number of MaskIoUs to train.
    augment_photometric_distort (bool): Whether to apply photometric distortions during augmentation.
    augment_expand (bool): Whether to apply expansion during augmentation.
    augment_random_sample_crop (bool): Whether to apply random sample cropping during augmentation.
    augment_random_mirror (bool): Whether to apply random mirroring during augmentation.
    augment_random_flip (bool): Whether to apply random flipping during augmentation.
    augment_random_rot90 (bool): Whether to apply random 90 degree rotations during augmentation.
    data_name (str): Name of the dataset.
    data_has_gt (bool): Whether the dataset has ground truth annotations.
    data_class_names (list): List of class names in the dataset.
    backbone_name (str): Name of the backbone network.
    backbone_weight_path (str): Path to the backbone weights file.
    backbone_type (type): Type of the backbone network.
    backbone_args (tuple): Arguments for the backbone network.
    backbone_channel_order (str): Channel order for the backbone network.
    backbone_normalize (bool): Whether to normalize input for the backbone network.
    backbone_substract_means (bool): Whether to subtract means for the backbone network.
    backbone_to_float (bool): Whether to convert input to float for the backbone network.
    backbone_selected_layers (list): List of selected layers for the backbone network.
    backbone_pred_scales (list): List of scales for predictions from the backbone network.
    backbone_pred_aspect_ratios (list): List of aspect ratios for predictions from the backbone network.
    backbone_use_pixel_scales (bool): Whether to use pixel scales for predictions from the backbone network.
    backbone_preapply_sqrt (bool): Whether to preapply square root for predictions from the backbone network.
    backbone_use_square_anchors (bool): Whether to use square anchors for predictions from the backbone network.
    fpn_num_features (int): Number of features in the FPN.
    fpn_interpolation_mode (str): Interpolation mode for the FPN.
    fpn_num_downsamples (int): Number of downsample operations in the FPN.
    fpn_use_conv_downsample (bool): Whether to use convolution for downsampling in the FPN.
    fpn_pad (bool): Whether to apply padding in the FPN.
    fpn_relu_downsample_layers (bool): Whether to apply ReLU to downsample layers in the FPN.
    fpn_relu_pred_layers (bool): Whether to apply ReLU to prediction layers in the FPN.

    Returns:
    dict: A dictionary containing the configuration for the YOLACT model.
    """
    activation_func = {
        'tanh':    torch.tanh,
        'sigmoid': torch.sigmoid,
        'softmax': lambda x: torch.nn.functional.softmax(x, dim=-1),
        'relu':    lambda x: torch.nn.functional.relu(x, inplace=True),
        'none':    lambda x: x,
    }

    data_config = Config({
        'name': data_name,

        'has_gt': data_has_gt,

        'class_names': data_class_names,    # ["BG"]
    })

    resnet_transform = Config({
        'channel_order': backbone_channel_order,
        'normalize': backbone_normalize,
        'subtract_means': backbone_substract_means,
        'to_float': backbone_to_float,
    })

    backbone_config = Config({
        'name': backbone_name,
        'path': backbone_weight_path,
        'type': backbone_type,
        'args': backbone_args,
        'transform': resnet_transform,

        'selected_layers': backbone_selected_layers,
        'pred_scales': backbone_pred_scales,
        'pred_aspect_ratios': backbone_pred_aspect_ratios,

        'use_pixel_scales': backbone_use_pixel_scales,
        'preapply_sqrt': backbone_preapply_sqrt,
        'use_square_anchors': backbone_use_square_anchors,
    })

    fpn_config = Config({
        'num_features': fpn_num_features,
        'interpolation_mode': fpn_interpolation_mode,
        'num_downsample': fpn_num_downsamples,
        'use_conv_downsample': fpn_use_conv_downsample,
        'pad': fpn_pad,
        'relu_downsample_layers': fpn_relu_downsample_layers,
        'relu_pred_layers': fpn_relu_pred_layers,
    })

    your_custom_yolact_config = Config({

        'name': name,

        'dataset': data_config,
        'num_classes': len(data_config.class_names) + 1, # This should include the background class
        'max_size': max_size,

        # Randomize hue, vibrance, etc.
        'augment_photometric_distort': augment_photometric_distort,
        # Have a chance to scale down the image and pad (to emulate smaller detections)
        'augment_expand': augment_expand,
        # Potentialy sample a random crop from the image and put it in a random place
        'augment_random_sample_crop': augment_random_sample_crop,
        # Mirror the image with a probability of 1/2
        'augment_random_mirror': augment_random_mirror,
        # Flip the image vertically with a probability of 1/2
        'augment_random_flip': augment_random_flip,
        # With uniform probability, rotate the image [0,90,180,270] degrees
        'augment_random_rot90': augment_random_rot90,
        'freeze_bn': freeze_bn,


        'fpn': fpn_config,

        'decay': decay,
        'gamma': gamma,
        'lr_steps': lr_steps,
        'lr_warmup_init': lr_warmup_init,
        'lr_warmup_until': lr_warmup_until,

        #    backbone
        'backbone': backbone_config,

        #     scale loss
        'conf_alpha': conf_alpha,
        'bbox_alpha': bbox_alpha,
        'mask_alpha': mask_alpha, 

        'use_semantic_segmentation_loss': use_semantic_segmentation_loss,
        'semantic_segmentation_alpha': semantic_segmentation_alpha,

        'use_mask_scoring': use_mask_scoring,
        'mask_scoring_alpha': mask_scoring_alpha,

        'use_focal_loss': use_focal_loss,
        'focal_loss_alpha': focal_loss_alpha,
        'focal_loss_gamma': focal_loss_gamma,
        'focal_loss_init_pi': focal_loss_init_pi,


        'max_num_detections': max_num_detections,
        'eval_mask_branch': True,
        
        'nms_top_k': nms_top_k,
        'nms_conf_thresh': nms_conf_thresh,
        'nms_thresh': nms_thresh,

        'mask_type': mask_type,
        'mask_size': mask_size,
        'masks_to_train': masks_to_train,
        'mask_proto_src': mask_proto_src,
        'mask_proto_net': mask_proto_net,
        'mask_proto_bias': mask_proto_bias,
        'mask_proto_prototype_activation': activation_func[mask_proto_prototype_activation],
        'mask_proto_mask_activation': activation_func[mask_proto_mask_activation],
        'mask_proto_coeff_activation': activation_func[mask_proto_coeff_activation],
        'mask_proto_crop': mask_proto_crop,
        'mask_proto_crop_expand': mask_proto_crop_expand,
        'mask_proto_loss': mask_proto_loss,
        'mask_proto_binarize_downsampled_gt': mask_proto_binarize_downsampled_gt,
        'mask_proto_normalize_mask_loss_by_sqrt_area': mask_proto_normalize_mask_loss_by_sqrt_area,
        'mask_proto_reweight_mask_loss': mask_proto_reweight_mask_loss,
        'mask_proto_grid_file': mask_proto_grid_file,
        'mask_proto_use_grid':  mask_proto_use_grid,
        'mask_proto_coeff_gate': mask_proto_coeff_gate,
        'mask_proto_prototypes_as_features': mask_proto_prototypes_as_features,
        'mask_proto_prototypes_as_features_no_grad': mask_proto_prototypes_as_features_no_grad,
        'mask_proto_remove_empty_masks': mask_proto_remove_empty_masks,
        'mask_proto_reweight_coeff': mask_proto_reweight_coeff,
        'mask_proto_coeff_diversity_loss': mask_proto_coeff_diversity_loss,
        'mask_proto_coeff_diversity_alpha': mask_proto_coeff_diversity_alpha,
        'mask_proto_normalize_emulate_roi_pooling': mask_proto_normalize_emulate_roi_pooling,
        'mask_proto_double_loss': mask_proto_double_loss,
        'mask_proto_double_loss_alpha': mask_proto_double_loss_alpha,
        'mask_proto_split_prototypes_by_head': mask_proto_split_prototypes_by_head,
        'mask_proto_crop_with_pred_box': mask_proto_crop_with_pred_box,
        'mask_proto_debug': mask_proto_debug,

        'discard_box_width': discard_box_width,
        'discard_box_height': discard_box_height,

        'share_prediction_module': share_prediction_module,
        'ohem_use_most_confident': ohem_use_most_confident,

        'use_class_balanced_conf': use_class_balanced_conf,

        'use_sigmoid_focal_loss': use_sigmoid_focal_loss,

        'use_objectness_score': use_objectness_score,

        'use_class_existence_loss': use_class_existence_loss,
        'class_existence_alpha': class_existence_alpha,

        'use_change_matching': use_change_matching,

        'extra_head_net': extra_head_net,

        'head_layer_params': head_layer_params,

        'extra_layers': extra_layers,

        'positive_iou_threshold': positive_iou_threshold,
        'negative_iou_threshold': negative_iou_threshold,

        'ohem_negpos_ratio': ohem_negpos_ratio,

        'crowd_iou_threshold': crowd_iou_threshold,
        
        'force_cpu_nms': force_cpu_nms,

        'use_coeff_nms': use_coeff_nms,

        'use_instance_coeff': use_instance_coeff,
        'num_instance_coeffs': num_instance_coeffs,

        'train_masks': train_masks,
        'train_boxes': train_boxes,
        'use_gt_bboxes': use_gt_bboxes,

        'preserve_aspect_ratio': preserve_aspect_ratio,

        'use_prediction_module': use_prediction_module,

        'use_yolo_regressors': use_yolo_regressors,
        
        'use_prediction_matching': use_prediction_matching,

        'delayed_settings': delayed_settings,

        'no_jit': no_jit,

        'mask_dim': mask_dim,

        'use_maskiou': use_maskiou,  
        
        'maskiou_net': maskiou_net,

        'discard_mask_area': discard_mask_area, 

        'maskiou_alpha': maskiou_alpha, 
        'rescore_mask': rescore_mask,
        'rescore_bbox': rescore_bbox,
        'maskious_to_train': maskious_to_train
    })

    return your_custom_yolact_config



def get_configuration_from_json(json_file):
    """
    Extracts the configurations from a json file and puts them into a configuration

    Duplicate names will lead to only one change
    """
    config = get_configuration()

    with open(json_file, "r") as f:
        data = json.load(f)

        for key, value in data.items():
            if key in config.keys():
                config[key] = value
            elif key in config.dataset.keys():
                config.dataset[key] = value
            elif key in config.backbone.keys():
                config.backbone[key] = value
            elif key in config.fpn.keys():
                config.fpn[key] = value

    return config



def load_datanames(path_to_images,
                    amount,     # for random mode
                    start_idx,  # for range mode
                    end_idx,    # for range mode
                    image_name, # for single mode
                    data_mode="all",
                    should_print=True):
    """
    Loads file paths from a specified directory based on the given mode.

    Parameters:
    path_to_images (str): The path to the directory containing images.
    amount (int): Number of random images to select (used in 'random' mode).
    start_idx (int): The starting index of the range of images to select (used in 'range' mode).
    end_idx (int): The ending index of the range of images to select (used in 'range' mode).
    image_name (str): The name of a single image to select (used in 'single' mode).
    data_mode (str, optional): The mode for selecting images. It can be one of the following:
        - 'all': Selects all images.
        - 'random': Selects a random set of images up to the specified amount.
        - 'range': Selects a range of images from start_idx to end_idx.
        - 'single': Selects a single image specified by image_name.
        Default is 'all'.
    should_print (bool, optional): Whether to print information about selected images. Default is True.

    Returns:
    list: A list of file-names of the selected images.

    Raises:
    ValueError: If an invalid data_mode is provided.

    Example:
    >>> load_data_paths('/path/to/images', amount=10, start_idx=0, end_idx=10, image_name='image1.jpg', data_mode='random')
    ['image2.jpg', 'image5.jpg', 'image8.jpg', ...]

    Notice: Detects all forms of files and directories and doesn't filter on them.
    """
    all_images = os.listdir(path_to_images)

    images = []

    # Validation / Test

    if data_mode == "all":
        data_indices = np.arange(0, len(all_images))
    elif data_mode == "random":
        data_indices = np.random.randint(0, len(all_images), amount)
    elif data_mode == "range":
        end_idx = min(len(all_images)-1, end_idx)
        data_indices = np.arange(start_idx, end_idx+1)
    elif data_mode == "single":
        data_indices = []
        images += [image_name]
    else:
        raise ValueError(f"INFERENCE_DATA_MODE has a illegal value: {data_mode}")

    for cur_idx in data_indices:
        images += [all_images[cur_idx]]

    if should_print:
        print(f"Inference Image Indices:\n{data_indices}")
        print(f"Inference Data Amount: {len(images)}")

    # data_amount = len(images)
    return images



def get_bounding_boxes(mask):
    """
    Generate percentage bounding boxes from a mask image.

    This function takes a mask image where each unique pixel value corresponds to a different class. 
    It computes the bounding boxes for each class, represented as percentages of the image dimensions. 
    The output is a numpy array containing the bounding box coordinates and the class ID for each box.

    Parameters:
    mask (numpy.ndarray): A 2D numpy array where each unique value represents a different class. 
                          The mask should have integer values, with 0 typically representing the background.

    Returns:
    numpy.ndarray: A 2D numpy array of shape (N, 5), where N is the number of bounding boxes. 
                   Each row corresponds to a bounding box in the format [xmin, ymin, xmax, ymax, class_id], 
                   where the coordinates are normalized to the range [0, 1].

    Example:
    --------
    >>> mask = np.array([[0, 0, 1, 1],
                         [0, 2, 2, 1],
                         [3, 3, 3, 3]])
    >>> get_bounding_boxes(mask)
    array([[1.0       , 0.        , 1.        , 0.5       , 1.        ],
           [0.25      , 0.25      , 0.75      , 0.5       , 1.        ],
           [0.        , 0.66666667, 1.        , 1.        , 1.        ]])
    """
    height, width = mask.shape
    unique_classes = np.unique(mask)
    
    # Initialize an empty array with the shape (0, 5) to store the boxes and classes
    boxes_and_classes = np.empty((0, 5))

    for class_id in unique_classes:
        # Skip background
        if class_id == 0:
            continue  

        pos = np.where(mask == class_id)
        xmin = np.min(pos[1]) / width
        xmax = np.max(pos[1]) / width
        ymin = np.min(pos[0]) / height
        ymax = np.max(pos[0]) / height
        
        # Create a numpy array for the current bounding box and class
        cur_box_and_class = np.array([[xmin, ymin, xmax, ymax, 1]])

        # Stack the new array onto the existing one
        boxes_and_classes = np.vstack((boxes_and_classes, cur_box_and_class))

    return boxes_and_classes



def transform_mask(mask, one_dimensional=False, input_color_map=None):
    """
    Transform a mask into a visible mask with color coding.

    This function converts a segmentation mask into a color image for visualization.
    Each unique value in the mask is assigned a unique color. Optionally, a predefined 
    colormap can be provided. The output can be either a 1-dimensional or 3-dimensional 
    color image.

    Parameters:
    -----------
    mask : numpy.ndarray
        A 2D array representing the segmentation mask where each unique value corresponds 
        to a different object or class.
    one_dimensional : bool, optional
        If True, the output will be a 1-dimensional color image (grayscale). If False, the 
        output will be a 3-dimensional color image (RGB). Default is False.
    input_color_map : list of tuples or list of int, optional
        A list of predefined colors to be used for the objects. Each element should be a 
        tuple of 3 integers for RGB mode or a single integer for grayscale mode. The number 
        of colors in the colormap must be at least equal to the number of unique values in 
        the mask (excluding the background). Default is None.

    Returns:
    --------
    tuple:
        - numpy.ndarray: The color image representing the mask.
        - list: The colormap used for the transformation.

    Raises:
    -------
    ValueError:
        If the dimensions of the input colormap do not match the specified dimensionality 
        (1D or 3D).
    """
    if one_dimensional:
        dimensions = 1
    else:
        dimensions = 3

    if input_color_map and len(input_color_map[0]) != dimensions:
        raise ValueError(f"Dimension of Input Colormap {len(input_color_map[0])} doesn't fit to used dimensions: {dimensions}")

    color_map = []

    # convert mask map to color image
    color_image = np.zeros((mask.shape[0], mask.shape[1], dimensions), dtype=np.uint8)

    # assign a random color to each object and skip the background
    unique_values = np.unique(mask)
    idx = 0
    for value in unique_values:
        if value != 0:
            if input_color_map:
                color = input_color_map[idx]
            else:
                if dimensions == 3:
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                else:
                    color = (np.random.randint(20, 255))
            color_image[mask == value] = color
            color_map += [color]
            idx += 1
    return color_image, color_map



def extract_and_visualize_mask(masks, image=None, ax=None, visualize=True, color_map=None, soft_join=False):
    """
    Extracts masks from a 3D mask array and optionally visualizes them.

    This function takes a 3D array of masks and combines them into a single 2D mask image.
    Optionally, it can visualize the result by overlaying the mask on an input image and 
    displaying it using matplotlib. It returns the extracted 2D mask, and optionally the 
    colorized mask and the colormap.

    Parameters:
    -----------
    masks : numpy.ndarray
        A 3D array of shape (width, height, num_masks) where each mask corresponds to a 
        different object or class.
    image : numpy.ndarray, optional
        An optional 3D array representing the image on which the mask will be overlaid for 
        visualization. It should be of shape (width, height, 3). Default is None.
    ax : matplotlib.axes.Axes, optional
        A matplotlib Axes object to plot the visualization. If None, a new figure and axes 
        will be created. Default is None.
    visualize : bool, optional
        If True, the function will visualize the mask overlay on the image. Default is True.
    color_map : list of tuples, optional
        A list of predefined colors to be used for the objects in the mask. Each element 
        should be a tuple of 3 integers representing an RGB color. Default is None.
    soft_join : bool, optional
        If True, the mask will be softly blended with the input image. If False, the mask 
        will be directly overlaid on the image. Default is False.

    Returns:
    --------
    numpy.ndarray
        The 2D mask image of shape (width, height) where each unique value corresponds to 
        a different object or class.
    numpy.ndarray, optional
        The colorized mask image of shape (width, height, 3) if `visualize` is True.
    list, optional
        The colormap used for the transformation if `visualize` is True.

    Raises:
    -------
    IndexError
        If there is an error in accessing the mask slices due to incorrect shape or size.
    """
    shape = (masks.shape[0], masks.shape[1], 1)

    # calc mask
    if masks.size == 0:
        result_mask = np.full(shape, 0, np.uint8)
    else:
        try:
            result_mask = np.where(masks[:, :, 0], 1, 0)
        except IndexError as e:
            raise IndexError(f"error in 'result_mask = np.where(masks[:, :, 0], 1, 0)'.\nShape of mask: {masks.shape}\nSize of mask: {masks.size}")
        for idx in range(1, masks.shape[2]):
            cur_mask = np.where(masks[:, :, idx], idx+1, 0)
            result_mask = np.where(cur_mask == 0, result_mask, cur_mask)
        result_mask = result_mask.astype("int")

    # visualize
    if visualize:
        color_image, color_map = transform_mask(result_mask, one_dimensional=False, input_color_map=color_map)

        if image is not None:
            color_image = color_image.astype(int) 

            w, h, c = color_image.shape

            if soft_join == False:
                for cur_row_idx in range(w):
                    for cur_col_idx in range(h):
                        if color_image[cur_row_idx, cur_col_idx].sum() != 0:
                            image[cur_row_idx, cur_col_idx] = color_image[cur_row_idx, cur_col_idx]
                color_image = image
            else:
                # remove black baclground
                for cur_row_idx in range(w):
                    for cur_col_idx in range(h):
                        if color_image[cur_row_idx, cur_col_idx].sum() == 0:
                            color_image[cur_row_idx, cur_col_idx] = image[cur_row_idx, cur_col_idx]

                # Set the transparency levels (alpha and beta)
                alpha = 0.5  # transparency of 1. image
                beta = 1 - alpha  # transparency of 2. image

                # Blend the images
                color_image = cv2.addWeighted(image, alpha, color_image, beta, 0)
                # color_image = cv2.add(color_image, image)

        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 15), sharey=True)
            fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=None)
            
        ax.imshow(color_image, vmin=0, vmax=255)
        ax.set_title("Instance Segmentation Mask")
        ax.axis("off")

        return result_mask, color_image, color_map

    return result_mask



def calc_pixel_accuracy(mask_1, mask_2):
    """
    Calculate the pixel accuracy between two masks.

    Args:
        mask_1 (np.ndarray): The first mask.
        mask_2 (np.ndarray): The second mask.

    Returns:
        float: The pixel accuracy between the two masks.

    Raises:
        ValueError: If the shapes of the masks are different.
    """
    if mask_1.shape != mask_2.shape:
        raise ValueError(f"Can't calculate the pixel accuracy between the 2 masks because of different shapes: {mask_1.shape} and {mask_2.shape}")
    
    matching_pixels = np.sum(mask_1 == mask_2)
    all_pixels = np.prod(mask_1.shape)
    return matching_pixels / all_pixels



def calc_intersection_over_union(mask_1, mask_2):
    """
    Calculate the Intersection over Union (IoU) between two masks.

    Args:
        mask_1 (np.ndarray): The first mask.
        mask_2 (np.ndarray): The second mask.

    Returns:
        float: The IoU between the two masks.

    Raises:
        ValueError: If the shapes of the masks are different.
    """
    if mask_1.shape != mask_2.shape:
        raise ValueError(f"Can't calculate the IoU between the 2 masks because of different shapes: {mask_1.shape} and {mask_2.shape}")
    
    intersection = np.logical_and(mask_1, mask_2)
    union = np.logical_or(mask_1, mask_2)
    
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)
    
    return intersection_area / union_area



def calc_precision_and_recall(mask_1, mask_2, only_bg_and_fg=False, aggregation="mean"):
    """
    Calculate the precision and recall between two masks.

    Args:
        mask_1 (np.ndarray): The first mask.
        mask_2 (np.ndarray): The second mask.
        only_bg_and_fg (bool): Whether to calculate only for background and foreground. Defaults to False.
        aggregation (str): Method to aggregate precision and recall values. Options are "sum", "mean", "median", "std", "var". Defaults to "mean".

    Returns:
        tuple: Precision and recall values.

    Raises:
        ValueError: If the shapes of the masks are different.
    """
    if mask_1.shape != mask_2.shape:
        raise ValueError(f"Can't calculate precision and recall between the 2 masks because of different shapes: {mask_1.shape} and {mask_2.shape}")
    
    if only_bg_and_fg:
        TP = np.sum((mask_1 > 0) & (mask_2 > 0))
        FP = np.sum((mask_1 > 0) & (mask_2 == 0))
        FN = np.sum((mask_1 == 0) & (mask_2 > 0))

        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
    else:
        precision = []
        recall = []
        unique_labels = np.unique(np.concatenate((mask_1.flatten(), mask_2.flatten())))
        for class_label in unique_labels:
            if class_label != 0:
                TP = np.sum((mask_1 == class_label) & (mask_2 == class_label))
                FP = np.sum((mask_1 == class_label) & (mask_2 != class_label))
                FN = np.sum((mask_1 != class_label) & (mask_2 == class_label))

                precision.append(TP / (TP + FP) if TP + FP != 0 else 0)
                recall.append(TP / (TP + FN) if TP + FN != 0 else 0)

        if aggregation.lower() == "sum":
            precision = np.sum(precision)
            recall = np.sum(recall)
        elif aggregation.lower() in ["avg", "mean"]:
            precision = np.mean(precision)
            recall = np.mean(recall)
        elif aggregation.lower() == "median":
            precision = np.median(precision)
            recall = np.median(recall)
        elif aggregation.lower() == "std":
            precision = np.std(precision)
            recall = np.std(recall)
        elif aggregation.lower() == "var":
            precision = np.var(precision)
            recall = np.var(recall)
    
    return precision, recall



def eval_pred(pred, ground_truth, should_print=True):
    """
    Evaluate prediction against ground truth by calculating pixel accuracy, IoU, precision, and recall.

    Args:
        pred (np.ndarray): The predicted mask.
        ground_truth (np.ndarray): The ground truth mask.
        should_print (bool): Whether to print the evaluation results. Defaults to True.

    Returns:
        tuple: Pixel accuracy, IoU, precision, and recall.
    """
    pixel_acc = calc_pixel_accuracy(pred, ground_truth)
    iou = calc_intersection_over_union(pred, ground_truth)
    precision, recall = calc_precision_and_recall(pred, ground_truth, only_bg_and_fg=True)

    if should_print:
        print("\nEvaluation:")
        print(f"    - Pixel Accuracy = {round(pixel_acc * 100, 2)}%")
        print(f"    - IoU = {iou}")
        print(f"    - Precision = {round(precision * 100, 2)}%\n        -> How many positive predicted are really positive\n        -> Only BG/FG")
        print(f"    - Recall = {round(recall * 100, 2)}%\n        -> How many positive were found\n        -> Only BG/FG")

    return pixel_acc, iou, precision, recall



class Custom_YOLACT_inference_Dataset(torch.utils.data.Dataset):
    """
    A custom dataset class for YOLACT inference, extending PyTorch's Dataset class.

    This dataset class loads images and their corresponding masks (if available) for inference. It performs
    necessary preprocessing, such as resizing and color conversion, and verifies the existence
    of the specified images and masks.

    Attributes:
    -----------
    images : list
        A list of image filenames.
    img_folder : str
        The path to the folder containing the images.
    mask_folder : str
        The path to the folder containing the masks.
    data_type : str
        The file extension of the images (default is ".png").
    size : int
        The size to which the images and masks are resized (default is 550).
    should_print : bool
        A flag indicating whether to print status messages (default is True).

    Methods:
    --------
    __len__():
        Returns the number of images in the dataset.
    __getitem__(idx):
        Returns the preprocessed image, its path, and optionally its bounding boxes and mask.
    verify_data():
        Verifies the existence of the specified images and masks, and updates the image list
        to include only valid entries.
    """
    def __init__(self, images, img_folder_path, mask_folder_path,
                    data_type=".png", size=550, should_print=True):
        self.images = images
        self.img_folder = img_folder_path
        self.mask_folder = mask_folder_path
        self.data_type = data_type
        self.size = size
        self.should_print = should_print
        if len(self.images) == 0:
            raise ValueError("There are no images to train!")
        self.verify_data()
        if len(self.images) == 0:
            raise ValueError("There are no images to train!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load image
        img_path = os.path.join(self.img_folder, self.images[idx])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File '{img_path}' not found!")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, [self.size, self.size])
        # prepared_image = FastBaseTransform()(image.unsqueeze(0))
        prepared_image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).float()     # .unsqueeze(0)

        # load mask
        if self.mask_folder:
            mask_path = os.path.join(self.mask_folder,  self.images[idx])
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"File '{mask_path}' not found!")
            masks = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            boxes_and_classes = get_bounding_boxes(masks)

            return prepared_image, img_path, boxes_and_classes, torch.from_numpy(masks)
        
        return prepared_image, img_path

    def verify_data(self):
        """
        Verifies the existence of the specified images and masks, and updates the image list.

        This method checks each image and mask file for existence and updates the list of images
        to include only those with valid corresponding files. It also prints verification results
        if the should_print attribute is True.
        """
        updated_images = []
        if self.should_print:
            print(f"\n{'-'*32}\nVerifying Data...")

        images_found = 0
        images_not_found = []

        masks_found = 0
        masks_not_found = []

        for cur_image in self.images:
            image_path = os.path.join(self.img_folder, cur_image)
            if self.mask_folder:
                mask_path = os.path.join(self.mask_folder, cur_image)

            image_exists = os.path.exists(image_path) and os.path.isfile(image_path) and any([image_path.endswith(ending) for ending in [".png", ".jpg", ".jpeg"]])
            if image_exists:
                images_found += 1
            else:
                images_not_found += [image_path]

            if self.mask_folder:
                mask_exists = os.path.exists(mask_path) and os.path.isfile(mask_path) and any([mask_path.endswith(ending) for ending in [".png", ".jpg", ".jpeg"]])
                if mask_exists:
                    masks_found += 1
                else:
                    masks_not_found += [mask_path]
            else:
                mask_exists = True

            if image_exists and mask_exists:
                updated_images += [cur_image]

        if self.should_print:
            print(f"\n> > > Images < < <\nFound: {round((images_found/len(self.images))*100, 2)}% ({images_found}/{len(self.images)})")
            if len(images_not_found) > 0:
                print("\n Not Found:")
            for not_found in images_not_found:
                print(f"    -> {not_found}")

            if self.mask_folder:
                print(f"\n> > > Masks < < <\nFound: {round((masks_found/len(self.images))*100, 2)}% ({masks_found}/{len(self.images)})")
                if len(masks_not_found) > 0:
                    print("\n Not Found:")
                for not_found in masks_not_found:
                    print(f"    -> {not_found}")

            print(f"\nUpdating Images...")
            print(f"From {len(self.images)} to {len(updated_images)} Images\n    -> Image amount reduced by {round(( 1-(len(updated_images)/len(self.images)) )*100, 2)}%")
        
        self.images = updated_images

        if self.should_print:
            print(f"{'-'*32}\n")



################
### Training ###
################
# FIXME



#################
### Inference ###
#################
def single_inference(model, image, configuration):
    """
    Makes one single inference
    """
    preds = model(image)

    if len(image.shape) == 4:
        image = image.squeeze(0)
    
    color_channels, w, h = image.shape
    classes, scores, boxes, masks = postprocess(preds, w, h, batch_idx=0, interpolation_mode='bilinear',
                                        visualize_lincomb=False, crop_masks=True, score_threshold=0,
                                        configuration=configuration)
    
    
    # change to widht, height, masks
    masks = masks.permute(1, 2, 0).numpy().astype(int)

    return classes, scores, boxes, masks

def inference(MODEL_SAVE_PATH, 
              WEIGHTS_NAME,
              PATH_TO_INFERENCE_IMAGES,
              PATH_TO_INFERENCE_MASKS,
              INFERENCE_DATA_MODE,
              INFERENCE_DATA_AMOUNT,
              INFERENCE_START_IDX,
              INFERENCE_END_IDX,
              INFERENCE_IMAGE_NAME,
              IMG_MAX_SIZE,
              OUTPUT_DIR,
              OUTPUT_TYPE="png",
              INTERACTIVE=False,
              SHOULD_SAVE=True,
              SHOULD_VISUALIZE=False,
              SAVE_VISUALIZATION=True,
              SHOULD_PRINT=True):
    """
    Makes an inference from a model.
    """
    # get current device
    device = get_device()

    # load image names
    inference_images = load_datanames(path_to_images=PATH_TO_INFERENCE_IMAGES,
                                        amount=INFERENCE_DATA_AMOUNT,     # for random mode
                                        start_idx=INFERENCE_START_IDX,  # for range mode
                                        end_idx=INFERENCE_END_IDX,    # for range mode
                                        image_name=INFERENCE_IMAGE_NAME, # for single mode
                                        data_mode=INFERENCE_DATA_MODE,
                                        should_print=SHOULD_PRINT)
    
    # load data
    inference_dataset = Custom_YOLACT_inference_Dataset(
        images=inference_images, 
        img_folder_path=PATH_TO_INFERENCE_IMAGES, 
        mask_folder_path=PATH_TO_INFERENCE_MASKS, 
        data_type=".png",
        size=IMG_MAX_SIZE,
        should_print=SHOULD_PRINT
    )

    inference_loader = DataLoader(inference_dataset, 
                                    batch_size=1,
                                    pin_memory=True,
                                    generator=torch.Generator(device=device))
    
    # create configuration file
    configuration = get_configuration(name="inference",
                                        max_size=IMG_MAX_SIZE,
                                        decay=5e-4,
                                        gamma=0.1,
                                        lr_steps=(280000, 600000, 700000, 750000),
                                        lr_warmup_init=1e-4,
                                        lr_warmup_until=500,
                                        freeze_bn=True,
                                        conf_alpha=1,
                                        bbox_alpha=1.5,
                                        mask_alpha=0.4 / 256 * 140 * 140, 
                                        use_semantic_segmentation_loss=True,
                                        semantic_segmentation_alpha=1,
                                        use_mask_scoring=False,
                                        mask_scoring_alpha=1,
                                        use_focal_loss=False,
                                        focal_loss_alpha=0.25,
                                        focal_loss_gamma=2,
                                        focal_loss_init_pi=0.01,
                                        max_num_detections=100,
                                        eval_mask_branch=True,
                                        nms_top_k=200,
                                        nms_conf_thresh=0.005,
                                        nms_thresh=0.5,
                                        mask_type=1,
                                        mask_size=6.125,
                                        masks_to_train=100,
                                        mask_proto_src=0,
                                        mask_proto_net=[(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})],
                                        mask_proto_bias=False,
                                        mask_proto_prototype_activation="relu",
                                        mask_proto_mask_activation="sigmoid",
                                        mask_proto_coeff_activation="tanh",
                                        mask_proto_crop=True,
                                        mask_proto_crop_expand=0,
                                        mask_proto_loss=None,
                                        mask_proto_binarize_downsampled_gt=True,
                                        mask_proto_normalize_mask_loss_by_sqrt_area=False,
                                        mask_proto_reweight_mask_loss=False,
                                        mask_proto_grid_file='data/grid.npy',
                                        mask_proto_use_grid= False,
                                        mask_proto_coeff_gate=False,
                                        mask_proto_prototypes_as_features=False,
                                        mask_proto_prototypes_as_features_no_grad=False,
                                        mask_proto_remove_empty_masks=False,
                                        mask_proto_reweight_coeff=1,
                                        mask_proto_coeff_diversity_loss=False,
                                        mask_proto_coeff_diversity_alpha=1,
                                        mask_proto_normalize_emulate_roi_pooling=True,
                                        mask_proto_double_loss=False,
                                        mask_proto_double_loss_alpha=1,
                                        mask_proto_split_prototypes_by_head=False,
                                        mask_proto_crop_with_pred_box=False,
                                        mask_proto_debug=False,
                                        discard_box_width=4 / 550,
                                        discard_box_height=4 / 550,
                                        share_prediction_module=True,
                                        ohem_use_most_confident=False,
                                        use_class_balanced_conf=False,
                                        use_sigmoid_focal_loss=False,
                                        use_objectness_score=False,
                                        use_class_existence_loss=False,
                                        class_existence_alpha=1,
                                        use_change_matching=False,
                                        extra_head_net=[(256, 3, {'padding': 1})],
                                        head_layer_params={'kernel_size': 3, 'padding': 1},
                                        extra_layers=(0, 0, 0),
                                        positive_iou_threshold=0.5,
                                        negative_iou_threshold=0.4,
                                        ohem_negpos_ratio=3,
                                        crowd_iou_threshold=0.7,
                                        force_cpu_nms=True,
                                        use_coeff_nms=False,
                                        use_instance_coeff=False,
                                        num_instance_coeffs=64,
                                        train_masks=True,
                                        train_boxes=True,
                                        use_gt_bboxes=False,
                                        preserve_aspect_ratio=False,
                                        use_prediction_module=False,
                                        use_yolo_regressors=False,
                                        use_prediction_matching=False,
                                        delayed_settings=[],
                                        no_jit=False,
                                        mask_dim=None,
                                        use_maskiou=True,
                                        maskiou_net=[(8, 3, {'stride': 2}), 
                                                    (16, 3, {'stride': 2}), 
                                                    (32, 3, {'stride': 2}), 
                                                    (64, 3, {'stride': 2}), 
                                                    (128, 3, {'stride': 2})],
                                        discard_mask_area=5*5, # -1,
                                        maskiou_alpha=25, # 6.125,
                                        rescore_mask=True,
                                        rescore_bbox=False,
                                        maskious_to_train=-1,
                                        augment_photometric_distort=True,
                                        augment_expand=True,
                                        augment_random_sample_crop=True,
                                        augment_random_mirror=True,
                                        augment_random_flip=False,
                                        augment_random_rot90=False,
                                        data_name="inference data",
                                        data_has_gt=False,
                                        data_class_names=["object"]*80,
                                        backbone_name="ResNet101",
                                        backbone_weight_path="resnet101_reducedfc.pth",
                                        backbone_type=ResNetBackbone,
                                        backbone_args=([3, 4, 23, 3],),
                                        backbone_channel_order='RGB',
                                        backbone_normalize=True,
                                        backbone_substract_means=False,
                                        backbone_to_float=False,
                                        backbone_selected_layers=list(range(1, 4)),
                                        backbone_pred_scales=[[24], [48], [96], [192], [384]],
                                        backbone_pred_aspect_ratios=[ [[1, 1/2, 2]] ]*5,
                                        backbone_use_pixel_scales=True,
                                        backbone_preapply_sqrt=False,
                                        backbone_use_square_anchors=True,
                                        fpn_num_features=256,
                                        fpn_interpolation_mode='bilinear',
                                        fpn_num_downsamples=2,
                                        fpn_use_conv_downsample=True,
                                        fpn_pad=True,
                                        fpn_relu_downsample_layers=False,
                                        fpn_relu_pred_layers=True)

    if SHOULD_SAVE:
        # make sure outputdir exists and reset it
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        else:
            for cur_file in os.listdir(OUTPUT_DIR):
                os.remove(os.path.join(OUTPUT_DIR, cur_file))

    if SHOULD_PRINT:
        print(f"Instance Segmentation with YOLACT (inference)\n")

    with torch.no_grad():
        # create and init model
        if SHOULD_PRINT:
            print("Loading Neuronal Network...")
        model = Yolact(configuration=configuration)
        model.eval()

        if SHOULD_PRINT:
            print("Loading Weights...")

        if MODEL_SAVE_PATH is not None and WEIGHTS_NAME is not None:
            weight_path = os.path.join(MODEL_SAVE_PATH, WEIGHTS_NAME)
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"Can't find Weights: {weight_path}")
        
            if device.type == "cpu":
                model.load_weights(weight_path, device=device)
            else:
                model.load_weights(weight_path)
        
        model.to(device)

        # do inference
        for data in inference_loader:
            # clear_output()

            if INTERACTIVE and SHOULD_PRINT:
                print(f"\n{'-'*16}\n>>> Look for the interactive questioning!\n{'-'*16}\n\n")

            if PATH_TO_INFERENCE_MASKS:
                image, name, bboxes_and_class, mask = data
            else:
                image = data[0]
                name = data[1]
                mask = None

            name = name[0]
            cleaned_name = ".".join(name.split(".")[:-1]).split("/")[-1]

            if SHOULD_PRINT:
                print("Inference Network...")
            classes, scores, boxes, masks = single_inference(model=model, image=image, configuration=configuration)

            if len(image.shape) == 4:
                image = image.squeeze(0)

            image = image.permute(1, 2, 0).numpy().astype(int) 

            extracted_mask = extract_and_visualize_mask(masks, image=None, ax=None, visualize=False)
            
            if SHOULD_SAVE:
                if OUTPUT_TYPE in ["numpy", "npy"]:
                    np.save(os.path.join(OUTPUT_DIR, f'{cleaned_name}.npy'), extracted_mask)
                else:
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f'{cleaned_name}.png'), extracted_mask)

            if SHOULD_VISUALIZE:
                ncols = 3

                fig, ax = plt.subplots(ncols=ncols, nrows=1, figsize=(20, 15), sharey=True)
                fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.02, hspace=None)
                
                # plot original image
                ax[0].imshow(image)
                ax[0].set_title("Original")
                ax[0].axis("off")

                # plot mask alone
                _, color_image, color_map = extract_and_visualize_mask(masks, image=None, ax=ax[1], visualize=True)
                ax[1].set_title("Prediction Mask")
                ax[1].axis("off")

                # plot result
                _, _, _ = extract_and_visualize_mask(masks, image=image, ax=ax[2], visualize=True, color_map=color_map)
                ax[2].set_title("Result")
                ax[2].axis("off")

                if SAVE_VISUALIZATION:
                    plt.savefig(os.path.join(OUTPUT_DIR, f'visualization_{cleaned_name}.jpg'), dpi=fig.dpi)


                if should_print:
                    print("\nShowing Visualization*")

                plt.show()



            # eval and plot ground truth comparisson
            if mask is not None:
                mask = mask.squeeze(0).numpy()
                mask = cv2.resize(mask, [IMG_MAX_SIZE, IMG_MAX_SIZE])
                eval_pred(extracted_mask, mask)

                if SHOULD_VISUALIZE:
                    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20, 15), sharey=True)
                    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)
                    
                    # plot ground_truth
                    mask, _ = transform_mask(mask, one_dimensional=False)
                    ax[0].imshow(mask)
                    ax[0].set_title("Ground Truth Mask")
                    ax[0].axis("off")

                    # plot prediction mask
                    _, color_image, color_map = extract_and_visualize_mask(masks, image=None, ax=ax[1], visualize=True, color_map=color_map)
                    ax[1].set_title("Predicted Mask")
                    ax[1].axis("off")

                    if SAVE_VISUALIZATION:
                        plt.savefig(os.path.join(OUTPUT_DIR, f'visualization_{cleaned_name}_gt.jpg'), dpi=fig.dpi)

                    if should_print:
                        print("\nShowing Ground Truth Visualization*")

                    plt.show()

            if INTERACTIVE:
                user_input = input("\nUser Input: next (enter) | exit (x) ->")
                if user_input in ["exit", "quit", "q", "x"]:
                    if should_print:
                        print("\nSee you later! I hope you enjoyed YOLACT!\n")
                    break
                else:
                    pass

                plt.clf()


##################
### Validation ###
##################
# FIXME



####################
### Running Code ###
####################

if __name__ == "__main__":
    # process arguments
    parser = argparse.ArgumentParser(description="Comfortable Yolact Runner")

    parser.add_argument("--mode", "-m", help="What should be runned? inference or train", default="inference", required=True, type=str)
    parser.add_argument("--model_save_path", "-msp", help="Path to the folder where the weights/model is", default=None, required=False, type=str)
    parser.add_argument("--weights_name", "-wn", help="Weights file name", default=None, required=False, type=str)
    parser.add_argument("--data_path", "-d", help="Path to where the data is", default="~/data/", required=False, type=str)
    parser.add_argument("--image_path", "-ip", help="image-folder name", default="depth-ims", required=False, type=str)
    parser.add_argument("--mask_path", "-map", help="masks-folder name", default=None, required=False, type=str)
    parser.add_argument("--data_mode", "-dm", help="Mode of getting the data (all, single, range, random)", default=None, required=False, type=str)
    parser.add_argument("--data_amount", "-da", help="Amount of data, when using random data mode", default=None, required=False, type=int)
    parser.add_argument("--inference_start_idx", "-isi", help="Start index for inference data", default=0, required=False, type=int)
    parser.add_argument("--inference_end_idx", "-iei", help="End index for inference data", default=None, required=False, type=int)
    parser.add_argument("--inference_image_name", "-iin", help="Name of the inference image", default=None, required=False, type=str)
    parser.add_argument("--img_max_size", "-ims", help="Maximum image size", default=1024, required=False, type=int)
    parser.add_argument("--output_dir", "-od", help="Directory to save output", default="./output", required=False, type=str)
    parser.add_argument("--output_type", "-ot", help="Type of output files", default="png", required=False, type=str)
    parser.add_argument("--interactive", "-int", help="Run in interactive mode", default=False, type=bool) #action='store_true')
    parser.add_argument("--should_save", "-ss", help="Should save output", default=True, type=bool) #action='store_true')
    parser.add_argument("--should_visualize", "-sv", help="Should visualize output", default=False, type=bool) #action='store_true')
    parser.add_argument("--save_visualization", "-svs", help="Save visualized output", default=True, type=bool) #action='store_true')
    parser.add_argument("--should_print", "-sp", help="Should print output", default=True, type=bool) #action='store_true')

    args = parser.parse_args()

    # pull arguments (or defaults)
    mode = args.mode
    model_save_path = args.model_save_path
    weights_name = args.weights_name
    data_path = args.data_path
    image_path = args.image_path
    mask_path = args.mask_path
    inference_data_mode = args.data_mode
    data_amount = args.data_amount
    inference_start_idx = args.inference_start_idx
    inference_end_idx = args.inference_end_idx
    inference_image_name = args.inference_image_name
    img_max_size = args.img_max_size
    output_dir = args.output_dir
    output_type = args.output_type
    interactive = args.interactive
    should_save = args.should_save
    should_visualize = args.should_visualize
    save_visualization = args.save_visualization
    should_print = args.should_print

    path_to_images = os.path.join(data_path, image_path)
    if mask_path is not None:
        path_to_masks = os.path.join(data_path, mask_path)
    else:
        path_to_masks = None

    if mode.lower() == "inference":
        inference(MODEL_SAVE_PATH=model_save_path, 
                    WEIGHTS_NAME=weights_name,
                    PATH_TO_INFERENCE_IMAGES=path_to_images,
                    PATH_TO_INFERENCE_MASKS=path_to_masks,
                    INFERENCE_DATA_MODE=inference_data_mode,
                    INFERENCE_DATA_AMOUNT=data_amount,
                    INFERENCE_START_IDX=inference_start_idx,
                    INFERENCE_END_IDX=inference_end_idx,
                    INFERENCE_IMAGE_NAME=inference_image_name,
                    IMG_MAX_SIZE=img_max_size,
                    OUTPUT_DIR=output_dir,
                    OUTPUT_TYPE=output_type,
                    INTERACTIVE=interactive,
                    SHOULD_SAVE=should_save,
                    SHOULD_VISUALIZE=should_visualize,
                    SAVE_VISUALIZATION=save_visualization,
                    SHOULD_PRINT=should_print)
    elif mode.lower() == "train":
        pass
    else:
        print(f"There is no mode '{mode}'.")



# python yolact_toolkit.py \
#     --mode inference \
#     --model_save_path ../weights \
#     --weights_name 2024-06-05_15-39_YOLACT_test_training_with_WISDOM-Sim_399_50000.pth \
#     --data_path /home/tobia/data/wisdom/wisdom-sim/ \
#     --image_path depth_ims \
#     --mask_path modal_segmasks \
#     --data_mode all \
#     --data_amount 5 \
#     --inference_start_idx 0 \
#     --inference_end_idx 50 \
#     --inference_image_name image_00011811.png \
#     --img_max_size 550 \
#     --output_dir ../output \
#     --output_type png \
#     --interactive True \
#     --should_save True \
#     --should_visualize True \
#     --save_visualization True \
#     --should_print True


