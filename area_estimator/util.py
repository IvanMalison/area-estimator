import argparse
import cv2
import functools
import itertools
import logging
import numpy as np
import os
import re
import requests

from collections import defaultdict


logger = logging.getLogger(__name__)


def space_camel_case(s, words_to_lower=("to", "and", "of", "the", "a", "is")):
    # Use regex to add space before capital letters
    spaced = re.sub('([A-Z])', r' \1', s)

    # Split the string into words and change the case for specified words
    spaced_words = spaced.split()
    result = []
    for word in spaced_words:
        if word.lower() in words_to_lower:
            result.append(word.lower())
        else:
            result.append(word)

    # Join the words together and strip the leading space
    result_string = ' '.join(result).strip()

    # Capitalize the first letter of the result string
    result_string = result_string[0].upper() + result_string[1:]

    return result_string


def replace_subpath(full_path, target_segment, replacement_segment):
    path_parts = full_path.split(os.sep)
    try:
        idx = len(path_parts) - 1 - path_parts[::-1].index(target_segment)
        path_parts[idx] = replacement_segment
    except ValueError:
        raise ValueError(
            f"The target_segment '{target_segment}' "
            f"does not exist in the full_path '{full_path}'"
        )
    return os.sep.join(path_parts)


def download_with_headers(url, headers=None):
    logger.info(f"Downloading from url: {url}")
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    return response.content


def new_session_with_headers(headers):
    session = requests.Session()
    session.headers = headers
    return session


def split_list_by_proportion(input_list, proportion):
    if not 0 <= proportion <= 1:
        raise ValueError("Proportion must be between 0 and 1")

    split_index = round(len(input_list) * proportion)
    return input_list[:split_index], input_list[split_index:]


MIME_TYPE_TO_EXTENSION = {
    'image/png': 'png',
    'image/jpg': 'jpg',
    'image/jpeg': 'jpeg',
}


def mime_type_to_extension(mime_type):
    return MIME_TYPE_TO_EXTENSION[mime_type]


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def fade_to_black_around_polygon(img, vertices):
    # blank mask, ensure it's single channel
    polygon_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # fill the mask
    cv2.fillPoly(polygon_mask, [np.array(vertices, dtype=np.int32)], 255)

    around_polygon_mask = cv2.bitwise_not(polygon_mask)

    # Calculate the distance to the nearest zero for each pixel in the mask
    dist_mask = cv2.distanceTransform(
        around_polygon_mask, cv2.DIST_L2, 5
    )

    # Change the drop-off rate
    dist_mask = np.power(dist_mask, 0.25)  # You can adjust this value

    # Normalize the distance mask to the range 0-255
    cv2.normalize(dist_mask, dist_mask, 0, 255, cv2.NORM_MINMAX)

    # Invert the distance mask so the edges of the polygon are white
    dist_mask = 255 - dist_mask

    # Convert the distance mask to the same type as the input image
    dist_mask = dist_mask.astype(img.dtype)

    faded = cv2.multiply(img, dist_mask, scale=1 / 255)

    return blur_outside_poly(faded, vertices)


def blur_outside_poly(img, vertices):
    # Create an initial mask filled with zeros, same size as image
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Fill the polygon area in the mask with ones
    cv2.fillPoly(mask, [np.array(vertices, dtype=np.int32)], 1)

    # Blur the entire original image
    blurred_img = cv2.GaussianBlur(img, (51, 51), 0)

    # Blend the original image and the blurred image using the mask
    result = (mask * img) + ((1 - mask) * blurred_img)

    # Convert the result to integer type
    result = result.astype(np.uint8)

    return result


def apply_region_of_interest_mask(img, vertices):
    # blank mask
    mask = np.zeros_like(img)

    # fill the mask
    cv2.fillPoly(mask, [np.array(vertices, dtype=np.int32)], 255)

    # Erode the mask to create a smooth transition
    for i in range(50):  # Change this to control the size of the transition
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # Apply the mask to the original image
    result = cv2.bitwise_and(img, mask)

    return result


def detect_line_segments(image):
    lsd = cv2.createLineSegmentDetector(0)

    # Detect lines in the image
    return lsd.detect(image)[0]


def canny(img):
    return cv2.Canny(img, 10, 50, apertureSize=3)


def is_iterable(value):
    try:
        iter(value)
        return True
    except TypeError:
        return False


def _makelist(value):
    if not isinstance(value, list):
        if is_iterable(value):
            if isinstance(value, str):
                return [value]
            return list(value)
        else:
            return [value]
    return value


def makelist(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        return _makelist(function(*args, **kwargs))

    return wrapped


def circular_iter(values, n):
    length = len(values)
    for i in range(length):
        yield [values[(i + j) % length] for j in range(n)]


def flatten(list_of_lists):
    return [
        item for sublist in list_of_lists for item in sublist
    ]


def weighted_median(data, weights):
    # First, sort the data
    sorted_data, sorted_weights = zip(*sorted(zip(data, weights)))
    # Calculate the cumulative sum of the weights
    cum_weights = np.cumsum(sorted_weights)
    # Find the point where the cumulative sum is half
    cutoff_index = np.searchsorted(cum_weights, cum_weights[-1] / 2.)
    # If the cumulative sum is exactly half at the cutoff point,
    # then the weighted median is the average of the data at this index and the next
    # else it's the data at the cutoff index
    if cum_weights[-1] % 2 == 0:
        return (sorted_data[cutoff_index - 1] + sorted_data[cutoff_index]) / 2.
    else:
        return sorted_data[cutoff_index]


def weighted_average_hough_line(lines):
    pass


def hough_to_slope_intercept_form(alpha, rho):
    alpha = alpha + np.pi / 2
    # Avoid division by zero
    if np.sin(alpha) == 0:
        m = float('inf')
    else:
        m = -np.cos(alpha) / np.sin(alpha)

    if np.sin(alpha) != 0:
        b = rho / np.sin(alpha)
    else:
        b = rho

    return m, b


def hough_to_line_segment_within_rectangle(
    alpha, rho, min_x, min_y, max_x, max_y,
):
    m, b = hough_to_slope_intercept_form(alpha, rho)

    if m == float('inf'):  # Vertical line
        if min_x <= rho <= max_x:
            return ((rho, min_y), (rho, max_y))
    elif m == 0:  # Horizontal line
        if min_y <= b <= max_y:
            return ((min_x, b), (max_x, b))

    intersections = []

    # Intersect with start_y and end_y
    x_at_start_y = (min_y - b) / m if m != 0 else float('inf')
    x_at_end_y = (max_y - b) / m if m != 0 else float('inf')
    intersections.extend([
        (x, y) for x, y in [(x_at_start_y, min_y), (x_at_end_y, max_y)]
        if min_x <= x <= max_x
    ])

    # Intersect with start_x and end_x
    y_at_start_x = m * min_x + b
    y_at_end_x = m * max_x + b
    intersections.extend([
        (x, y) for x, y in [(min_x, y_at_start_x), (max_x, y_at_end_x)]
        if min_y <= y <= max_y
    ])

    if len(intersections) < 2:
        raise ValueError('Line does not intersect rectangle, or intersects at only one point')
    elif len(intersections) > 2:
        intersections.sort(
            key=lambda p: (p[0] - min_x) ** 2 + (p[1] - min_y) ** 2,
        )  # sort by distance to rectangle's start point
        intersections = [intersections[0], intersections[-1]]  # take first and last

    return (intersections[0], intersections[1])


def normalize_alpha_rho(alpha, rho):
    if rho < 0:
        rho = -rho
        alpha = alpha + np.pi
    alpha = alpha % (2 * np.pi)  # ensure alpha is in the range [0, 2pi)
    return alpha, rho


def YOLO(*args, **kwargs):
    from .hacks import ultralytics
    # Ultralytics is doing dumb stuff that makes their import times really
    # slow... so do it lazily
    return ultralytics().YOLO(*args, **kwargs)


def StreamProcessor(*args, **kwargs):
    from . import hacks
    return hacks.StreamProcessor(*args, **kwargs)


def group_by(iterable, key_func):
    groups = defaultdict(list)
    for item in iterable:
        key = key_func(item)
        groups[key].append(item)
    return groups


def find_disjoint(collections):
    # Generate all pairs of collections
    for collection1, collection2 in itertools.combinations(collections, 2):
        # Convert collections to sets for easy intersection checking
        set1 = set(collection1)
        set2 = set(collection2)
        # If intersection is empty, these collections are disjoint
        if len(set1.intersection(set2)) == 0:
            return (collection1, collection2)
    # If no disjoint pairs found, return None
    return None


def find_middle_point(object1, object2, object3, point_function=lambda x: x):
    # Convert objects to points using the provided function

    x1, y1 = point_function(object1)
    x2, y2 = point_function(object2)
    x3, y3 = point_function(object3)

    delta_x = max(x1, x2, x3) - min(x1, x2, x3)
    delta_y = max(y1, y2, y3) - min(y1, y2, y3)

    if delta_x > delta_y:  # Points are arranged more horizontally
        if (x1 <= x2 and x1 >= x3) or (x1 >= x2 and x1 <= x3):
            return object1
        elif (x2 <= x1 and x2 >= x3) or (x2 >= x1 and x2 <= x3):
            return object2
        else:
            return object3
    else:  # Points are arranged more vertically
        if (y1 <= y2 and y1 >= y3) or (y1 >= y2 and y1 <= y3):
            return object1
        elif (y2 <= y1 and y2 >= y3) or (y2 >= y1 and y2 <= y3):
            return object2
        else:
            return object3


class StoreBooleanAction(argparse.Action):

    false_strings = ('false', 'no', 'f', 'n', '0')
    true_strings = ('true', 'yes', 't', 'y', '1')

    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, True)
        elif values.lower() in self.false_strings:
            setattr(namespace, self.dest, False)
        elif values.lower() in self.true_strings:
            setattr(namespace, self.dest, True)
        else:
            parser.error(f'Unknown value {values} for {self.dest}')


def closest_pocket(x, y, pockets, max_distance=None):
    closest = min(pockets, key=lambda p: p.distance(x, y))
    if closest and max_distance is not None:
        if closest.distance(x, y) > max_distance:
            return None
    return closest
