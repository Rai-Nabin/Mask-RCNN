import argparse

import cv2 as cv
import torch
import torchvision
from torchvision.transforms import transforms as T

from src.utils import draw_segmentation_map, get_outputs


def show_image(image, window_name):
    cv.imshow(window_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main(image_path, threshold):

    # Constructs a Mask R-CNN model with a ResNet-50-FPN backbone.
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True, progress=True, num_classes=91)

    # Set the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model on to the computation device
    model.to(device)

    # Set model to evaluation mode
    model.eval()

    image = cv.imread(image_path, cv.COLOR_BGR2RGB)
    original_image = image.copy()

    # show_image(image, "Input Image")

    # Transform the image to tensor
    transform = T.Compose([T.ToTensor()])
    transformed_image = transform(image)

    # Add a batch dimesion
    batched_image = transformed_image.unsqueeze(0).to(device)

    masks, bounding_boxes, output_labels = get_outputs(
        batched_image, model, threshold)

    detected_image = draw_segmentation_map(
        original_image, masks, bounding_boxes, output_labels)

    # Visualize the image
    show_image(detected_image, "Output Image")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_image', required=True,
                        help='Path to the input image')
    parser.add_argument('-t', '--threshold', default=0.9, type=float,
                        help='Score threshold for discarding detection')

    args = vars(parser.parse_args())

    image_path = args['input_image']
    threshold = args['threshold']

    main(image_path, threshold)
