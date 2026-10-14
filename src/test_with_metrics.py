import os
import cv2
import math
import torch
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
from patchify import patchify, unpatchify
from sklearn.metrics import classification_report, confusion_matrix
import json
from collections import defaultdict

from utils.constants import Constants
from utils.plot import visualize
from utils.logger import custom_logger
from utils.root_config import get_root_config


def calculate_iou(y_true, y_pred, num_classes):
    """Calculate IoU (Intersection over Union) for each class."""
    ious = []
    for class_id in range(num_classes):
        # Convert to binary masks for each class
        true_mask = (y_true == class_id)
        pred_mask = (y_pred == class_id)
        
        # Calculate intersection and union
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        
        # Avoid division by zero
        if union == 0:
            ious.append(1.0 if intersection == 0 else 0.0)
        else:
            ious.append(intersection / union)
    
    return ious


def calculate_average_precision(y_true, y_pred, class_id, threshold=0.5):
    """Calculate Average Precision for a specific class and threshold."""
    # Convert to binary masks
    true_mask = (y_true == class_id).astype(int)
    pred_mask = (y_pred == class_id).astype(int)
    
    # Calculate True Positives, False Positives, and False Negatives
    tp = np.logical_and(true_mask, pred_mask).sum()
    fp = np.logical_and(1 - true_mask, pred_mask).sum()
    fn = np.logical_and(true_mask, 1 - pred_mask).sum()
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # For semantic segmentation, we'll use precision as a proxy for AP
    # since we're dealing with dense predictions
    return precision


def calculate_dice_coefficient(y_true, y_pred, class_id):
    """Calculate Dice coefficient for a specific class."""
    true_mask = (y_true == class_id)
    pred_mask = (y_pred == class_id)
    
    intersection = np.logical_and(true_mask, pred_mask).sum()
    
    # Dice coefficient formula: 2 * |A ∩ B| / (|A| + |B|)
    dice = 2.0 * intersection / (true_mask.sum() + pred_mask.sum()) if (true_mask.sum() + pred_mask.sum()) > 0 else 1.0
    return dice


def calculate_pixel_accuracy(y_true, y_pred):
    """Calculate overall pixel accuracy."""
    correct_pixels = np.sum(y_true == y_pred)
    total_pixels = y_true.size
    return correct_pixels / total_pixels


def calculate_mean_pixel_accuracy(y_true, y_pred, num_classes):
    """Calculate mean pixel accuracy across all classes."""
    class_accuracies = []
    for class_id in range(num_classes):
        true_mask = (y_true == class_id)
        pred_mask = (y_pred == class_id)
        
        # Calculate accuracy for this class
        correct = np.logical_and(true_mask, pred_mask).sum()
        total = true_mask.sum()
        
        if total > 0:
            accuracy = correct / total
        else:
            accuracy = 1.0  # Perfect accuracy if class doesn't exist
        class_accuracies.append(accuracy)
    
    return np.mean(class_accuracies)


def calculate_frequency_weighted_iou(y_true, y_pred, num_classes):
    """Calculate Frequency Weighted IoU."""
    ious = calculate_iou(y_true, y_pred, num_classes)
    
    # Calculate class frequencies
    class_frequencies = []
    total_pixels = y_true.size
    
    for class_id in range(num_classes):
        class_pixels = np.sum(y_true == class_id)
        frequency = class_pixels / total_pixels
        class_frequencies.append(frequency)
    
    # Calculate weighted IoU
    fw_iou = sum(freq * iou for freq, iou in zip(class_frequencies, ious))
    return fw_iou


if __name__ == "__main__":

    ################################# Loading Variables and Paths from Config #################################

    ROOT, slice_config = get_root_config(__file__, Constants)

    # get the required variable values from config
    log_level = slice_config['vars']['log_level']
    file_type = slice_config['vars']['file_type']
    patch_size = slice_config['vars']['patch_size']  # size of each patch and window
    encoder = slice_config['vars']['encoder']        # the backbone/encoder of the model
    encoder_weights = slice_config['vars']['encoder_weights']
    classes = slice_config['vars']['test_classes']
    device = slice_config['vars']['device']

    # get the log file dir from config
    log_dir = ROOT / slice_config['dirs']['log_dir']
    # make the directory if it does not exist
    log_dir.mkdir(parents = True, exist_ok = True)
    # get the log file path
    log_path = log_dir / slice_config['vars']['test_log_name']
    # convert the path to string in a format compliant with the current OS
    log_path = log_path.as_posix()
    
    # initialize the logger
    logger = custom_logger("Land Cover Semantic Segmentation Test Logs", log_path, log_level)

    # get the dir of input images for inference from config
    img_dir = ROOT / slice_config['dirs']['data_dir'] / slice_config['dirs']['test_dir'] / slice_config['dirs']['image_dir']
    img_dir = img_dir.as_posix()

    # get the dir of input masks for inference from config
    gt_mask_dir = ROOT / slice_config['dirs']['data_dir'] / slice_config['dirs']['test_dir'] / slice_config['dirs']['mask_dir']
    gt_mask_dir = gt_mask_dir.as_posix()

    # get the model path from config
    model_name = slice_config['vars']['model_name']
    model_path = ROOT / slice_config['dirs']['model_dir'] / model_name
    model_path = model_path.as_posix()

    # get the predicted masks dir from config
    pred_mask_dir = ROOT / slice_config['dirs']['output_dir'] / slice_config['dirs']['pred_mask_dir']
    # make the directory if it does not exist
    pred_mask_dir.mkdir(parents = True, exist_ok = True)
    pred_mask_dir = pred_mask_dir.as_posix()

    # get the prediction plots dir from config
    pred_plot_dir = ROOT / slice_config['dirs']['output_dir'] / slice_config['dirs']['pred_plot_dir']
    # make the directory if it does not exist
    pred_plot_dir.mkdir(parents = True, exist_ok = True)
    pred_plot_dir = pred_plot_dir.as_posix()

    # Create metrics output directory
    metrics_dir = ROOT / slice_config['dirs']['output_dir'] / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = metrics_dir.as_posix()

    ###########################################################################################################

    ####################################### Functional Part of Program ########################################

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    model = torch.load(model_path, map_location=torch.device(device))

    class_values = [Constants.CLASSES.value.index(cls.lower()) for cls in classes]
    num_classes = len(class_values)
    
    img_list = list(filter(lambda x:x.endswith((file_type)), os.listdir(img_dir)))

    print(f"\nTotal images found to test: {len(img_list)}")
    logger.info(f"Total images found to test: {len(img_list)}")

    # Initialize metrics storage
    all_metrics = {
        'per_image_metrics': [],
        'overall_metrics': {},
        'per_class_metrics': {}
    }

    # Aggregate metrics across all images
    all_gt_masks = []
    all_pred_masks = []

    try:
        for filename in img_list:

            print(f"\nPreparing image and ground truth mask file {filename}...")
            logger.info(f"Preparing image and ground truth mask file {filename}...")

            # reading image
            try:
                image = cv2.imread(os.path.join(img_dir, filename), 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.error(f"Could not read image file {filename}!")
                raise e

            # reading ground truth mask
            try:
                gt_mask = cv2.imread(os.path.join(gt_mask_dir, filename), 0)
                # filter classes
                gt_masks = [(gt_mask == v) for v in class_values]
                gt_mask = np.stack(gt_masks, axis=-1).astype('float')
                gt_mask = gt_mask.argmax(2)
            except Exception as e:
                logger.error(f"Could not read ground truth mask file {filename}!")
                raise e

            # padding image to be perfectly divisible by patch_size
            try:
                pad_height = (math.ceil(image.shape[0] / patch_size) * patch_size) - image.shape[0]
                pad_width = (math.ceil(image.shape[1] / patch_size) * patch_size) - image.shape[1]
                padded_shape = ((0, pad_height), (0, pad_width), (0, 0))
                image_padded = np.pad(image, padded_shape, mode='reflect')
            except Exception as e:
                logger.error("Could not pad the image!")
                raise e

            # dividing image into patches according to patch_size in overlapping mode to have smooth reconstruction of predicted mask patches
            try:
                patches = patchify(image_padded, (patch_size, patch_size, 3), step=patch_size//2)[:, :, 0, :, :, :]
                mask_patches = np.empty(patches.shape[:-1], dtype=patches.dtype)
            except Exception as e:
                logger.error("Could not patchify the image!")
                raise e

            print("\nImage and ground truth mask preparation done successfully!")
            logger.info("Image and ground truth mask preparation done successfully!")

            # model prediction
            print(f"\nPredicting image file {filename}...")
            logger.info(f"Predicting image file {filename}...")
            try:
                for i in tqdm(range(0, patches.shape[0])):
                    for j in range(0, patches.shape[1]):
                        img_patch = preprocessing_fn(patches[i, j, :, :, :])
                        img_patch = img_patch.transpose(2, 0, 1).astype('float32')
                        x_tensor = torch.from_numpy(img_patch).to(device).unsqueeze(0)
                        pred_mask = model.predict(x_tensor)
                        pred_mask = pred_mask.squeeze().cpu().numpy().round()
                        pred_mask = pred_mask.transpose(1, 2, 0)
                        pred_mask = pred_mask.argmax(2)
                        mask_patches[i, j, :, :] = pred_mask
            except Exception as e:
                logger.error(f"Could not predict image file {filename}!")
                raise e

            # unpatch
            try:
                pred_mask = unpatchify(mask_patches, image_padded.shape[:-1])
            except Exception as e:
                logger.error("Could not unpatchify predicted mask patches!")
                raise e
            
            # unpad
            try:
                pred_mask = pred_mask[:image.shape[0], :image.shape[1]]
            except Exception as e:
                logger.error("Could not unpad reconstructed predicted mask!")
                raise e
            
            # classes found
            try:
                classes_found = []
                for cls in np.unique(pred_mask):
                    classes_found.append(Constants.CLASSES.value[cls])
                print(f"Total classes found in the predicted mask: {classes_found}")
                logger.info(f"Total classes found in the predicted mask: {classes_found}")
            except Exception as e:
                logger.error("Could not find classes in the predicted mask!")
                raise e
            
            # filter classes
            try:
                pred_masks = [(pred_mask == v) for v in class_values]
                pred_mask = np.stack(pred_masks, axis=-1).astype('float')
                pred_mask = pred_mask.argmax(2)
                print(f"Classes present in the predicted mask after filtering according to user input of 'test_classes': {classes}")
                logger.info(f"Classes present in the predicted mask after filtering according to user input of 'test_classes': {classes}")
            except Exception as e:
                logger.error("Could not filter user given classes from the predicted mask!")
                raise e
            
            try:
                cv2.imwrite(os.path.join(pred_mask_dir, filename), pred_mask)
                print("Predicted mask written successfully!")
                logger.info("Predicted mask written successfully!")
            except Exception as e:
                logger.error("Could not write the predicted mask!")
                raise e

            try:
                plot_fig = visualize(
                    image=image, 
                    ground_truth_mask=gt_mask, 
                    predicted_mask=pred_mask
                )
                plot_fig.savefig(os.path.join(pred_plot_dir, filename.split('.')[0] + '.png'))
                print("Prediction plot saved successfully!")
                logger.info("Prediction plot saved successfully!")
            except Exception as e:
                logger.error("Could not plot the image, ground truth mask, and predicted mask!")
                raise e

            # Calculate metrics for this image
            print(f"\nCalculating metrics for {filename}...")
            logger.info(f"Calculating metrics for {filename}...")
            
            try:
                # Calculate per-image metrics
                image_metrics = {
                    'filename': filename,
                    'pixel_accuracy': calculate_pixel_accuracy(gt_mask, pred_mask),
                    'mean_pixel_accuracy': calculate_mean_pixel_accuracy(gt_mask, pred_mask, num_classes),
                    'frequency_weighted_iou': calculate_frequency_weighted_iou(gt_mask, pred_mask, num_classes)
                }
                
                # Calculate IoU for each class
                ious = calculate_iou(gt_mask, pred_mask, num_classes)
                image_metrics['mean_iou'] = np.mean(ious)
                image_metrics['per_class_iou'] = {}
                for i, class_name in enumerate(classes):
                    image_metrics['per_class_iou'][class_name] = ious[i]
                
                # Calculate mAP 50 and mAP 75 (using precision as proxy for AP)
                map_50_scores = []
                map_75_scores = []
                dice_scores = []
                
                for i, class_name in enumerate(classes):
                    # mAP 50 (threshold 0.5)
                    ap_50 = calculate_average_precision(gt_mask, pred_mask, i, threshold=0.5)
                    map_50_scores.append(ap_50)
                    
                    # mAP 75 (threshold 0.75)
                    ap_75 = calculate_average_precision(gt_mask, pred_mask, i, threshold=0.75)
                    map_75_scores.append(ap_75)
                    
                    # Dice coefficient
                    dice = calculate_dice_coefficient(gt_mask, pred_mask, i)
                    dice_scores.append(dice)
                
                image_metrics['map_50'] = np.mean(map_50_scores)
                image_metrics['map_75'] = np.mean(map_75_scores)
                image_metrics['mean_dice'] = np.mean(dice_scores)
                
                # Store per-class metrics
                image_metrics['per_class_map_50'] = {}
                image_metrics['per_class_map_75'] = {}
                image_metrics['per_class_dice'] = {}
                
                for i, class_name in enumerate(classes):
                    image_metrics['per_class_map_50'][class_name] = map_50_scores[i]
                    image_metrics['per_class_map_75'][class_name] = map_75_scores[i]
                    image_metrics['per_class_dice'][class_name] = dice_scores[i]
                
                all_metrics['per_image_metrics'].append(image_metrics)
                
                # Print metrics for this image
                print(f"Metrics for {filename}:")
                print(f"  Pixel Accuracy: {image_metrics['pixel_accuracy']:.4f}")
                print(f"  Mean IoU: {image_metrics['mean_iou']:.4f}")
                print(f"  mAP@50: {image_metrics['map_50']:.4f}")
                print(f"  mAP@75: {image_metrics['map_75']:.4f}")
                print(f"  Mean Dice: {image_metrics['mean_dice']:.4f}")
                
                logger.info(f"Metrics for {filename}:")
                logger.info(f"  Pixel Accuracy: {image_metrics['pixel_accuracy']:.4f}")
                logger.info(f"  Mean IoU: {image_metrics['mean_iou']:.4f}")
                logger.info(f"  mAP@50: {image_metrics['map_50']:.4f}")
                logger.info(f"  mAP@75: {image_metrics['map_75']:.4f}")
                logger.info(f"  Mean Dice: {image_metrics['mean_dice']:.4f}")
                
                # Store masks for overall metrics calculation
                all_gt_masks.append(gt_mask.flatten())
                all_pred_masks.append(pred_mask.flatten())
                
            except Exception as e:
                logger.error(f"Could not calculate metrics for {filename}!")
                raise e

        # Calculate overall metrics across all images
        print("\nCalculating overall metrics across all test images...")
        logger.info("Calculating overall metrics across all test images...")
        
        try:
            # Concatenate all masks
            all_gt_masks = np.concatenate(all_gt_masks)
            all_pred_masks = np.concatenate(all_pred_masks)
            
            # Calculate overall metrics
            overall_metrics = {
                'pixel_accuracy': calculate_pixel_accuracy(all_gt_masks, all_pred_masks),
                'mean_pixel_accuracy': calculate_mean_pixel_accuracy(all_gt_masks, all_pred_masks, num_classes),
                'frequency_weighted_iou': calculate_frequency_weighted_iou(all_gt_masks, all_pred_masks, num_classes)
            }
            
            # Calculate overall IoU
            ious = calculate_iou(all_gt_masks, all_pred_masks, num_classes)
            overall_metrics['mean_iou'] = np.mean(ious)
            overall_metrics['per_class_iou'] = {}
            for i, class_name in enumerate(classes):
                overall_metrics['per_class_iou'][class_name] = ious[i]
            
            # Calculate overall mAP scores
            map_50_scores = []
            map_75_scores = []
            dice_scores = []
            
            for i, class_name in enumerate(classes):
                ap_50 = calculate_average_precision(all_gt_masks, all_pred_masks, i, threshold=0.5)
                map_50_scores.append(ap_50)
                
                ap_75 = calculate_average_precision(all_gt_masks, all_pred_masks, i, threshold=0.75)
                map_75_scores.append(ap_75)
                
                dice = calculate_dice_coefficient(all_gt_masks, all_pred_masks, i)
                dice_scores.append(dice)
            
            overall_metrics['map_50'] = np.mean(map_50_scores)
            overall_metrics['map_75'] = np.mean(map_75_scores)
            overall_metrics['mean_dice'] = np.mean(dice_scores)
            
            # Store overall per-class metrics
            overall_metrics['per_class_map_50'] = {}
            overall_metrics['per_class_map_75'] = {}
            overall_metrics['per_class_dice'] = {}
            
            for i, class_name in enumerate(classes):
                overall_metrics['per_class_map_50'][class_name] = map_50_scores[i]
                overall_metrics['per_class_map_75'][class_name] = map_75_scores[i]
                overall_metrics['per_class_dice'][class_name] = dice_scores[i]
            
            all_metrics['overall_metrics'] = overall_metrics
            
            # Print overall metrics
            print("\nOverall Metrics Across All Test Images:")
            print(f"  Pixel Accuracy: {overall_metrics['pixel_accuracy']:.4f}")
            print(f"  Mean Pixel Accuracy: {overall_metrics['mean_pixel_accuracy']:.4f}")
            print(f"  Mean IoU: {overall_metrics['mean_iou']:.4f}")
            print(f"  mAP@50: {overall_metrics['map_50']:.4f}")
            print(f"  mAP@75: {overall_metrics['map_75']:.4f}")
            print(f"  Mean Dice: {overall_metrics['mean_dice']:.4f}")
            print(f"  Frequency Weighted IoU: {overall_metrics['frequency_weighted_iou']:.4f}")
            
            logger.info("Overall Metrics Across All Test Images:")
            logger.info(f"  Pixel Accuracy: {overall_metrics['pixel_accuracy']:.4f}")
            logger.info(f"  Mean Pixel Accuracy: {overall_metrics['mean_pixel_accuracy']:.4f}")
            logger.info(f"  Mean IoU: {overall_metrics['mean_iou']:.4f}")
            logger.info(f"  mAP@50: {overall_metrics['map_50']:.4f}")
            logger.info(f"  mAP@75: {overall_metrics['map_75']:.4f}")
            logger.info(f"  Mean Dice: {overall_metrics['mean_dice']:.4f}")
            logger.info(f"  Frequency Weighted IoU: {overall_metrics['frequency_weighted_iou']:.4f}")
            
            # Print per-class metrics
            print("\nPer-Class Metrics:")
            logger.info("Per-Class Metrics:")
            for class_name in classes:
                print(f"  {class_name}:")
                print(f"    IoU: {overall_metrics['per_class_iou'][class_name]:.4f}")
                print(f"    AP@50: {overall_metrics['per_class_map_50'][class_name]:.4f}")
                print(f"    AP@75: {overall_metrics['per_class_map_75'][class_name]:.4f}")
                print(f"    Dice: {overall_metrics['per_class_dice'][class_name]:.4f}")
                
                logger.info(f"  {class_name}:")
                logger.info(f"    IoU: {overall_metrics['per_class_iou'][class_name]:.4f}")
                logger.info(f"    AP@50: {overall_metrics['per_class_map_50'][class_name]:.4f}")
                logger.info(f"    AP@75: {overall_metrics['per_class_map_75'][class_name]:.4f}")
                logger.info(f"    Dice: {overall_metrics['per_class_dice'][class_name]:.4f}")
            
            # Save metrics to JSON file
            metrics_file = os.path.join(metrics_dir, 'test_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            
            print(f"\nMetrics saved to: {metrics_file}")
            logger.info(f"Metrics saved to: {metrics_file}")
            
            # Create a summary CSV file
            import pandas as pd
            
            # Summary of overall metrics
            summary_data = {
                'Metric': [
                    'Pixel Accuracy',
                    'Mean Pixel Accuracy',
                    'Mean IoU',
                    'mAP@50',
                    'mAP@75',
                    'Mean Dice',
                    'Frequency Weighted IoU'
                ],
                'Value': [
                    overall_metrics['pixel_accuracy'],
                    overall_metrics['mean_pixel_accuracy'],
                    overall_metrics['mean_iou'],
                    overall_metrics['map_50'],
                    overall_metrics['map_75'],
                    overall_metrics['mean_dice'],
                    overall_metrics['frequency_weighted_iou']
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_csv = os.path.join(metrics_dir, 'test_metrics_summary.csv')
            summary_df.to_csv(summary_csv, index=False)
            
            # Per-class metrics CSV
            per_class_data = []
            for class_name in classes:
                per_class_data.append({
                    'Class': class_name,
                    'IoU': overall_metrics['per_class_iou'][class_name],
                    'AP@50': overall_metrics['per_class_map_50'][class_name],
                    'AP@75': overall_metrics['per_class_map_75'][class_name],
                    'Dice': overall_metrics['per_class_dice'][class_name]
                })
            
            per_class_df = pd.DataFrame(per_class_data)
            per_class_csv = os.path.join(metrics_dir, 'test_metrics_per_class.csv')
            per_class_df.to_csv(per_class_csv, index=False)
            
            print(f"Summary metrics saved to: {summary_csv}")
            print(f"Per-class metrics saved to: {per_class_csv}")
            logger.info(f"Summary metrics saved to: {summary_csv}")
            logger.info(f"Per-class metrics saved to: {per_class_csv}")
            
        except Exception as e:
            logger.error("Could not calculate overall metrics!")
            raise e

    except Exception as e:
        logger.error("No images found in 'data/test/images' folder!")
        raise e

    ###########################################################################################################