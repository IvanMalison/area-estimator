import cv2
import math
import logging
import numpy as np

from typing import Sequence

from . import labelbox


logger = logging.getLogger(__name__)


def calculate_area(mask, h_fov, v_fov, height):
    # Count the number of white pixels in the mask
    mask_pixels = cv2.countNonZero(mask)

    # Image dimensions
    image_height, image_width = mask.shape

    # Convert the FOV from degrees to radians
    h_fov_rad = np.radians(h_fov)
    v_fov_rad = np.radians(v_fov)

    # Calculate the width and height of the area covered by the image
    area_width = 2 * height * np.tan(h_fov_rad / 2)
    area_height = 2 * height * np.tan(v_fov_rad / 2)

    # Calculate the area of one pixel
    pixel_area = (area_width * area_height) / (image_width * image_height)

    # Calculate the total area of the mask
    mask_area = mask_pixels * pixel_area

    return mask_area


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
        horizontal, vertical = self.get_fovs(get_aspect_ratio(image))
        return calculate_area(image, horizontal, vertical, height)

    def _get_image(self, image):
        if isinstance(image, np.ndarray):
            return image
        return cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    def calculate_area(self, masks, height):
        return sum(self.calculate_area_for_mask(mask, height) for mask in masks)


class ProjectAreaEstimator:

    height_feature_id = "clkww5y9g0002356lz6558syg"
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
            if c['feature_id'] == self.height_feature_id:
                return int(c['text_answer']['content'])
