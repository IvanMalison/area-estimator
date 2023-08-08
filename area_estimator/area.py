import cv2
import math
import logging
import numpy as np

from typing import Sequence

from . import labelbox


logger = logging.getLogger(__name__)


def calculate_area(mask, diagonal_fov_degrees, distance_to_subject):
    image_width_pixels = float(mask.shape[1])
    # Compute the real-world diagonal length of the scene
    diagonal_fov_radians = np.radians(diagonal_fov_degrees)
    scene_diagonal = 2.0 * float(distance_to_subject) * np.tan(diagonal_fov_radians / 2.0)

    # Compute the real-world width of the scene (assuming square image)
    scene_width = scene_diagonal / np.sqrt(2.0)

    # Compute the real-world distance represented by a pixel
    d = scene_width / image_width_pixels
    A_pixel = d * d

    # Compute the center of the mask
    center_y, center_x = float(mask.shape[0]) / 2.0, float(mask.shape[1]) / 2.0

    total_area = 0.0

    # Iterate over the mask
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != 0.0:  # If the pixel is part of the region of interest
                x = np.sqrt((float(i) - center_y) ** 2 + (float(j) - center_x) ** 2) * d
                R = np.sqrt(distance_to_subject ** 2 + x ** 2)
                A_x = A_pixel * (R / distance_to_subject) ** 2
                total_area += A_x

    return total_area


def get_aspect_ratio(image):
    # The aspect ratio of an image is the ratio of its width to its height
    height, width = image.shape[:2]
    return width / height


def calculate_fov(diagonal_fov_degrees, aspect_ratio):
    # Convert the diagonal FOV to radians
    diagonal_fov_radians = math.radians(diagonal_fov_degrees)

    # Calculate the horizontal and vertical FOV
    horizontal_fov_radians = 2 * math.atan(aspect_ratio / 2 * math.tan(diagonal_fov_radians / 2))
    vertical_fov_radians = 2 * math.atan(1 / 2 * math.tan(diagonal_fov_radians / 2))

    # Convert the horizontal and vertical FOV to degrees
    horizontal_fov_degrees = math.degrees(horizontal_fov_radians)
    vertical_fov_degrees = math.degrees(vertical_fov_radians)

    return horizontal_fov_degrees, vertical_fov_degrees


class AreaCalculator:

    def __init__(self, diagonal_fov_degrees: int, aspect_ratio: float = 4.0 / 3.0):
        self.diagonal_fov_degrees = diagonal_fov_degrees
        self.horizontal_fov, self.vertical_fov = calculate_fov(diagonal_fov_degrees, aspect_ratio)
        self.aspect_ratio = aspect_ratio

    def get_fovs(self, aspect_ratio):
        return calculate_fov(self.diagonal_fov_degrees, aspect_ratio)

    def calculate_area_for_mask(self, mask, height):
        image = self._get_image(mask)
        return calculate_area(image, self.diagonal_fov_degrees, height)

    def _get_image(self, image):
        if isinstance(image, np.ndarray):
            return image
        return cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    def calculate_area(self, masks, height):
        return sum(self.calculate_area_for_mask(mask, height) for mask in masks)


class ProjectAreaEstimator:

    height_feature_id = "clkww3kwq01ro07z33qxv0v74"
    default_height = 300

    def __init__(
            self, calculator: AreaCalculator,
            projects: Sequence[labelbox.CachedProject],
            extractor: labelbox.LabelExtractor,
    ):
        self.calculator = calculator
        self.projects = projects
        self.extractor = extractor

    def get_all(self):
        return {
            sample_id: self.calculate_area(project, sample_id)
            for project in self.projects
            for sample_id in project.get_available_ids()
        }

    def calculate_area(self, project, sample_id):
        objs = self.extractor.get_objects_by_type(
            project.get_all_project_annotations(sample_id)
        )
        images = [label.image for label in objs["Sargassum"]]

        try:
            height = self.get_height(project, sample_id)
        except Exception as e:
            logger.warn(
                f"Encountered exception {e} getting height for {sample_id}, "
                f"using default of {self.default_height}",
            )
            height = self.default_height
        else:
            if height is None:
                logger.warn(
                    f"No height found for {sample_id}, using default of {self.default_height}",
                )
                height = self.default_height

        return self.calculator.calculate_area(images, height)

    def get_height(self, project, sample_id):
        project_info = project.lookup_by_id(sample_id)['projects'][project._project_id]
        classifications = project_info['labels'][0]['annotations']['classifications']

        for c in classifications:
            if c["feature_schema_id"] == self.height_feature_id:
                return int(c['text_answer']['content'])
