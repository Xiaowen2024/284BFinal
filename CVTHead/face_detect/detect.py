from typing import Tuple, Union
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import math
import mediapipe as mp
import numpy as np

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def scale(
    bbox,
    height,
    width,
    scaling_factor=0.5,
):
    new_start_point = np.clip(int(bbox.origin_x - bbox.width * scaling_factor), 0, width), \
        np.clip(int(bbox.origin_y - bbox.height * scaling_factor), 0, height)
    new_end_point = np.clip(int(bbox.origin_x + bbox.width * (1 + scaling_factor)), 0, width), \
        np.clip(int(bbox.origin_y + bbox.height * (1 + scaling_factor)), 0, height)
    return (new_start_point, new_end_point)


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape

  for i, detection in enumerate(detection_result.detections):
    # Draw bounding_box
    bbox = detection.bounding_box
    # start_point = bbox.origin_x, bbox.origin_y
    start_point, end_point = scale(bbox, height, width)
    print("Scaled start and end points:", start_point, end_point)
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    # Draw keypoints
    for keypoint in detection.keypoints:
      keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                     width, height)
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    
    # Extract individual face images
    cv2.imwrite(f"results/detection_{i}.jpg", cv2.cvtColor(image[start_point[1]:end_point[1], start_point[0]:end_point[0]], cv2.COLOR_BGR2RGB))

    # TODO: Is it worth implementing features to try to segment the face?TODO
  return annotated_image

def detect(image_file):
    image_file = './image.jpg'
    img = cv2.imread(image_file)


    # Create an FaceDetector object.
    base_options = python.BaseOptions(model_asset_path='face_detect/detector.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    # Load the input image.
    image = mp.Image.create_from_file(image_file)

    # Detect faces in the input image.
    detection_result = detector.detect(image)

    # Process the detection result. In this case, write it back to disk.
    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("results/rgb.jpg", rgb_annotated_image)

if __name__ == "__main__":
    detect('image.jpg')