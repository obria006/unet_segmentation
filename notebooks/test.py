import cv2
import numpy as np
def overlay_labels(image, gt, pred, gt_color=(0, 255, 0), pred_color=(255, 0, 0), transparency=0.5):
    # Normalize the grayscale image to the range [0, 255]
    image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Create an RGB version of the grayscale image
    image_rgb = cv2.cvtColor(image_normalized, cv2.COLOR_GRAY2RGB)

    # Convert gt and pred to binary masks
    gt_mask = np.clip((gt > 0).astype(np.uint8) * 255,0,255)
    pred_mask = np.clip((pred > 0).astype(np.uint8) * 255,0,255)

    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2RGB)
    pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)

    print(np.where(gt_mask>0))


    # gt_mask[np.where(gt_mask>0)] = gt_color
    # pred_mask[np.where(pred_mask>0)] = pred_color

    # # Apply transparency to the masks
    # overlay_gt = cv2.addWeighted(image_rgb, 1 - transparency, cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2RGB), transparency, 0)
    # overlay_pred = cv2.addWeighted(image_rgb, 1 - transparency, cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB), transparency, 0)

    # # Set color channels for gt and pred overlays
    # overlay_gt[:, :, 0] = 0  # Set blue channel to 0 (remove blue)
    # overlay_gt[:, :, 2] = 0  # Set red channel to 0 (remove red)

    # overlay_pred[:, :, 0] = 0  # Set green and red channels to 0 (remove green and red)
    # overlay_pred[:, :, 1] = 0

    # print(overlay_gt.shape)
    # print(overlay_pred.shape)

    # # Apply the specified colors to gt and pred overlays
    # overlay_gt = cv2.addWeighted(overlay_gt, 1, np.array(gt_color, dtype=np.float64), 1 - transparency, 0)
    # overlay_pred = cv2.addWeighted(overlay_pred, 1, np.array(pred_color, dtype=np.float64), 1 - transparency, 0)

    # # Combine the overlays
    # result = cv2.addWeighted(overlay_gt, 1, overlay_pred, 1, 0)

    # return result