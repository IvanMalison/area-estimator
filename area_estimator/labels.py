import abc
import cv2
import enum
import functools
import math
import numpy as np
import os
import shapely

from scipy.spatial import ConvexHull
from typing import Sequence, Optional, Union, Tuple, TypeVar, List

from . import util
from .line import Line


_auto = object()


class _AnnotationMeta(abc.ABCMeta):

    def __init__(self, name, bases, attrs):
        if self.name is not None:
            self.name = util.space_camel_case(self.__name__) if self.name is _auto else self.name
            self.name_to_class[self.name] = self


T = TypeVar('T', bound='Annotation')


class Annotation(abc.ABC, metaclass=_AnnotationMeta):

    name: Optional[Union[object, str]]
    name = None

    name_to_class = {}  # type: ignore

    @abc.abstractmethod
    def to_yolo_format(self, img_width, img_height) -> str:
        ...

    @abc.abstractmethod
    def reflect_horizontally(self: T, img_width: int) -> T:
        ...

    @abc.abstractmethod
    def rotate_90(self: T, img_shape: tuple, multiple: int) -> T:
        ...

    @property
    @abc.abstractmethod
    def max_x(self: T) -> int:
        ...

    @property
    @abc.abstractmethod
    def max_y(self: T) -> int:
        ...

    @property
    @abc.abstractmethod
    def min_x(self: T) -> int:
        ...

    @property
    @abc.abstractmethod
    def min_y(self: T) -> int:
        ...

    @property
    @abc.abstractmethod
    def polygon(self: T) -> List[Tuple[int, int]]:
        ...

    @functools.cached_property
    def shapely(self):
        return shapely.Polygon(self.polygon)


class BoundingBox(Annotation):

    @classmethod
    def from_center(cls, center_x, center_y, width, height):
        left = center_x - width / 2
        top = center_y - height / 2
        return cls(left, top, width, height)

    @classmethod
    def from_labelbox_data(cls, annotation):
        return cls(**annotation["bounding_box"])

    def __init__(self, left, top, width, height):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    @functools.cached_property
    def center(self):
        center_x = self.left + self.width / 2
        center_y = self.top + self.height / 2
        return center_x, center_y

    def distance_from_center(self, x, y):
        center_x, center_y = self.center
        return math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    def distance(self, x, y):
        dist_x = max(self.left - x, x - (self.left + self.width), 0)
        dist_y = max(self.top - y, y - (self.top + self.height), 0)

        return math.sqrt(dist_x ** 2 + dist_y ** 2)

    def __repr__(self):
        return f"<{type(self).__name__} {str(self.__dict__)}>"

    def to_yolo_format(self, img_width, img_height):
        xc, yc = self.center
        x_center = xc / img_width
        y_center = yc / img_height
        width_yolo = self.width / img_width
        height_yolo = self.height / img_height

        values = [x_center, y_center, width_yolo, height_yolo]
        if any(value > 1.0 or value < 0.0 for value in values):
            raise ValueError("Bounding box Out of bounds")
        return ' '.join(map(str, values))

    def reflect_horizontally(self, img_width):
        flipped_left = img_width - self.left - self.width
        return type(self)(flipped_left, self.top, self.width, self.height)

    def rotate_90(self, img_shape, multiple):
        if multiple % 4 == 1:
            return type(self)(
                self.top,
                img_shape[1] - self.left - self.width,
                self.height,
                self.width
            )
        elif multiple % 4 == 2:
            return type(self)(
                img_shape[1] - self.left - self.width,
                img_shape[0] - self.top - self.height,
                self.width,
                self.height)
        elif multiple % 4 == 3:
            return type(self)(
                img_shape[0] - self.top - self.height,
                self.left,
                self.height,
                self.width
            )
        else:
            return self

    @functools.cached_property
    def polygon(self):
        top_left = (self.left, self.top)
        top_right = (self.left + self.width, self.top)
        bottom_right = (self.left + self.width, self.top + self.height)
        bottom_left = (self.left, self.top + self.height)

        return [top_left, top_right, bottom_right, bottom_left]

    @functools.cached_property
    def shapely(self):
        return shapely.Polygon(self.polygon)

    @property
    def max_x(self):
        return self.left + self.width

    @property
    def min_x(self):
        return self.left

    @property
    def min_y(self):
        return self.top

    @property
    def max_y(self):
        return self.top + self.height

    @property
    def longest_side_length(self):
        return max(self.width, self.height)

    def crop(self, mins, maxes):
        min_x, min_y = mins
        max_x, max_y = maxes
        # Ensure the bounding box does not go outside the new limits
        left = max(self.left, min_x) - min_x
        top = max(self.top, min_y) - min_y

        right = min(self.left + self.width, max_x) - min_x
        bottom = min(self.top + self.height, max_y) - min_y

        # If the bounding box is completely outside the new limits, raise an error
        if right - left <= 0 or bottom - top <= 0:
            raise ValueError("The bounding box is outside the crop area.")

        # Return a new BoundingBox with the updated dimensions
        return type(self)(left, top, right - left, bottom - top)

    def distance_to_line(self, line):
        if isinstance(line, Line):
            line = line.shapely_ray
        return self.shapely.distance(line)

    def shapely_distance(self, other):
        if hasattr(other, "shapely"):
            other = other.shapely
        return self.shapely.distance(other)


S = TypeVar('S', bound='SegmentationAnnotation')


class SegmentationAnnotation(Annotation, abc.ABC):

    @property
    @abc.abstractmethod
    def image(self):
        ...

    @property
    @abc.abstractmethod
    def scale(self):
        ...

    @functools.cached_property
    def largest_contour(self):
        # Find contours
        contours, _ = cv2.findContours(
            self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Assuming the largest contour corresponds to the object of interest
        contour = max(contours, key=cv2.contourArea)

        return contour.reshape(-1, 2)

    @functools.cached_property
    def bounding_box(self):
        # Calculate minimum area bounding box
        rect = cv2.minAreaRect(self.largest_contour)
        box = cv2.boxPoints(rect)
        return box

    @functools.cached_property
    def center_line(self):
        # Calculate center line through longest dimension of bounding box
        rect = cv2.minAreaRect(self.largest_contour)
        center = np.array(rect[0])
        dimensions = np.array(rect[1])
        rotation = rect[2]

        # Calculate endpoints of the line segment
        if dimensions[0] < dimensions[1]:
            rotation = (rotation + 90) % 180
        rotation_rad = np.deg2rad(rotation)
        displacement = np.array([np.cos(rotation_rad), np.sin(rotation_rad)]) * dimensions.max() / 2

        point1 = tuple(map(int, center - displacement))
        point2 = tuple(map(int, center + displacement))

        return point1, point2

    @functools.cached_property
    def longest_line(self):
        contour_points = self.largest_contour
        p1, p2 = max(
            (
                (contour_points[i], contour_points[j])
                for i in range(len(contour_points))
                for j in range(i + 1, len(contour_points))
            ),
            key=lambda pair: np.linalg.norm(pair[0] - pair[1])
        )
        return tuple(p1), tuple(p2)

    def to_yolo_format(self, width: int, height: int):
        # normalize the polygon coordinates
        normalized_polygon = [
            [point[0] / width, point[1] / height] for point in self.polygon
        ]

        return ' '.join(
            [' '.join(map(str, point)) for point in normalized_polygon]
        )

    @functools.cached_property
    def max_x(self: T) -> int:
        return max(point[0] for point in self.polygon)

    @functools.cached_property
    def max_y(self: T) -> int:
        return max(point[1] for point in self.polygon)

    @functools.cached_property
    def min_x(self: T) -> int:
        return min(point[0] for point in self.polygon)

    @functools.cached_property
    def min_y(self: T) -> int:
        return min(point[1] for point in self.polygon)


class ImageSegmentationMask(SegmentationAnnotation):

    @functools.cached_property
    def polygon(self):
        contour = self.largest_contour
        epsilon = 0.02 * cv2.arcLength(contour, True)  # 2% of the arc length
        approx = cv2.approxPolyDP(contour, epsilon, True)

        while len(approx) < 4:
            epsilon *= 0.5
            approx = cv2.approxPolyDP(contour, epsilon, True)

        # Return the points of the approximated polygon
        return approx.squeeze().tolist()

    @property
    def scale(self):
        return tuple(self.image.shape)

    def reflect_horizontally(self):
        # Flip the image horizontally
        flipped_img = cv2.flip(self.image, 1)

        # Create a new instance with the flipped image
        return type(self)(flipped_img)

    def rotate_90(self, shape, multiple):
        # Rotate the image 90 degrees multiple times
        rotated_img = np.rot90(self.image, k=-multiple)

        # Create a new instance with the rotated image
        return type(self)(rotated_img)

    def crop(self, mins: Tuple[int, int], maxes: Tuple[int, int]):
        min_x, min_y = mins
        max_x, max_y = maxes
        cropped_mask = self.image[min_y:max_y, min_x:max_x]
        return NDArraySegmentationMask(cropped_mask)


class NDArraySegmentationMask(ImageSegmentationMask):

    def __init__(self, image: np.ndarray):
        self._image = image

    @property
    def image(self):
        return self._image


class FileSegmentationMask(ImageSegmentationMask):

    def __init__(self, filepath: os.PathLike):
        self._filepath = filepath

    @functools.cached_property
    def image(self):
        with open(self._filepath, 'rb') as f:
            img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)


class PolygonSegmentation(SegmentationAnnotation):

    def __init__(self, polygon: List[Tuple[int, int]], width: int, height: int):
        self._polygon = polygon
        self.width = width
        self.height = height

    @property
    def scale(self):
        return (self.height, self.width)

    @property
    def polygon(self):
        return self._polygon

    @functools.cached_property
    def image(self):
        # create an empty image of required size
        img = np.zeros((self.height, self.width), dtype=np.uint8)

        # convert polygon points to integers
        pts = np.array(self.polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))

        # fill polygon with white color
        cv2.fillPoly(img, [pts], (255))

        return img

    @functools.cached_property
    def largest_contour(self):
        # Convert polygon to contour
        contour = np.array(self.polygon, dtype=np.int32)

        return contour.reshape(-1, 2)

    def reflect_horizontally(self, img_width):
        # Create a new polygon with the x-coordinates flipped
        flipped_polygon = [(img_width - x, y) for x, y in self.polygon]

        # Create a new instance with the flipped polygon
        return type(self)(flipped_polygon, self.width, self.height)

    def rotate_90(self, img_shape, multiple, dir=-1):
        multiple *= dir
        rotated_polygon = None
        if multiple % 4 == 1:  # 90 degrees clockwise rotation
            new_height, new_width = self.width, self.height
            rotated_polygon = [(y, img_shape[1] - x) for x, y in self.polygon]
        elif multiple % 4 == 2:  # 180 degrees rotation
            new_height, new_width = self.height, self.width
            rotated_polygon = [
                (img_shape[1] - x, img_shape[0] - y) for x, y in self.polygon
            ]
        elif multiple % 4 == 3:  # 270 degrees clockwise rotation
            new_height, new_width = self.width, self.height
            rotated_polygon = [(img_shape[0] - y, x) for x, y in self.polygon]
        else:
            return self

        # Create a new instance with the rotated polygon
        return type(self)(rotated_polygon, new_width, new_height)

    def crop(self, mins: Tuple[int, int], maxes: Tuple[int, int]):
        min_x, min_y = mins
        max_x, max_y = maxes
        # Validate crop boundaries
        if (
                min_x < 0 or min_y < 0 or max_x > self.width or
                max_y > self.height or max_x <= min_x or max_y <= min_y
        ):
            raise ValueError("Invalid crop boundaries.")

        # Adjust coordinates of the vertices by subtracting the top-left corner
        # coordinates of the crop region
        cropped_points = [(x - min_x, y - min_y) for x, y in self.polygon]
        return type(self)(cropped_points, max_x - min_x, max_y - min_y)


class Sargassum(FileSegmentationMask):
    name = _auto


def get_bounding_polygon(labels: Sequence[Annotation]):
    points = []
    for label in labels:
        for point in label.polygon:
            points.append(point)

    hull = ConvexHull(points)

    return np.array([points[i] for i in hull.vertices])


def get_bounding_coordinates(labels: Sequence[Annotation]):
    max_x = max(label.max_x for label in labels)
    max_y = max(label.max_y for label in labels)

    min_x = min(label.min_x for label in labels)
    min_y = min(label.min_y for label in labels)

    return (round(min_x), round(min_y)), (round(max_x), round(max_y))
