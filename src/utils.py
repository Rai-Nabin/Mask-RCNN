import random

import cv2 as cv
import numpy as np
import torch

from src.coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

# Create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))


def get_outputs(batched_image, model, threshold):
    # Forward pass of the image through model
    with torch.no_grad():
        # Outputs a list containing a dictionary
        model_outputs = model(batched_image)

    # Get all the scores and load it onto the CPU
    scores = list(model_outputs[0]['scores'].detach().cpu().numpy())

    # Indices of those scores which are above threshold
    thresholded_prediction_indices = [scores.index(
        score) for score in scores if score > threshold]
    thresholded_predictions_count = len(thresholded_prediction_indices)

    # Get the masks which are greater than 0.5
    masks = (model_outputs[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()

    # Discard masks for the objects which are below threshold
    masks = masks[:thresholded_predictions_count]

    # Get the bounding boxes in (x1, y1), (x2, y2) format
    bounding_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]
                      for i in model_outputs[0]['boxes'].detach().cpu()]

    # Discard bounding boxes below treshold value
    bounding_boxes = bounding_boxes[:thresholded_predictions_count]

    # Get the class labels
    output_labels = [coco_names[i] for i in model_outputs[0]['labels']]

    return masks, bounding_boxes, output_labels


def draw_segmentation_map(image, masks, bounding_boxes, output_labels):
    alpha = 1
    beta = 0.6      # Transparency for the segmentation map
    gamma = 0       # Scalar added to each sum

    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)

        # Apply a random color mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[masks[i] == 1], green_map[masks[i]
                                          == 1], blue_map[masks[i] == 1] = color

        # Combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)

        # Apply mask on the image
        cv.addWeighted(image, alpha, segmentation_map, beta, gamma, image)

        # Draw the bounding boxes around the objects
        cv.rectangle(image, bounding_boxes[i][0], bounding_boxes[i][1], color=color,
                     thickness=2)
        # Put the labels text above the objects
        cv.putText(image, output_labels[i], (bounding_boxes[i][0][0], bounding_boxes[i][0][1]-10),
                   cv.FONT_HERSHEY_SIMPLEX, 1, color,
                   thickness=2, lineType=cv.LINE_AA)
    return image
