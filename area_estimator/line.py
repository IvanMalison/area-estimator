import functools
import math
import numpy as np
import shapely

from typing import Tuple

from . import util


class Line:

    @classmethod
    def from_hough(cls, *args, **kwargs):
        return cls.from_tuples(
            *util.hough_to_line_segment_within_rectangle(*args, **kwargs),
        )

    @classmethod
    def from_tuples(cls, start, end):
        return cls(start[0], start[1], end[0], end[1])

    @classmethod
    def from_labelbox_data(cls, annotation):
        return cls(
            start_x=annotation["line"][0]["x"],
            start_y=annotation["line"][0]["y"],
            end_x=annotation["line"][1]["x"],
            end_y=annotation["line"][1]["y"],
        )

    def __init__(self, start_x, start_y, end_x, end_y):
        self.x1 = start_x
        self.y1 = start_y
        self.x2 = end_x
        self.y2 = end_y

    @functools.cached_property
    def length(self):
        return math.sqrt(
            (self.x1 - self.x2) ** 2 + (self.y1 - self.y2) ** 2
        )

    @functools.cached_property
    def max_x(self):
        return max(self.x1, self.x2)

    @functools.cached_property
    def max_y(self):
        return max(self.y1, self.y2)

    @functools.cached_property
    def max_scale(self):
        return max(self.max_x, self.max_y)

    @functools.cached_property
    def tuple(self):
        return self.x1, self.y1, self.x2, self.y2

    @functools.cached_property
    def start(self):
        return self.x1, self.y1

    @functools.cached_property
    def end(self):
        return self.x2, self.y2

    @functools.cached_property
    def tuples(self):
        return self.start, self.end

    @functools.cached_property
    def slope(self):
        # Check for vertical line
        if self.x1 == self.x2:
            return float('inf')

        # Calculate the slope (rise over run)
        return (self.y2 - self.y1) / (self.x2 - self.x1)

    @functools.cached_property
    def intercept(self):
        return self.y1 - self.slope * self.x1

    @property
    def slope_intercept(self):
        return self.slope, self.intercept

    @functools.cached_property
    def angle(self):
        # return math.atan(self.slope)
        dy = self.y2 - self.y1
        dx = self.x2 - self.x1
        return math.atan2(dy, dx)

    @property
    def alpha(self):
        return self.angle

    @functools.cached_property
    def signed_origin_distance(self):
        return self.signed_distance(0.0, 0.0)

    @property
    def rho(self):
        return self.signed_origin_distance

    @property
    def hough(self):
        return self.angle, self.signed_origin_distance

    @functools.cached_property
    def shapely(self):
        return shapely.LineString(self.tuples)

    @functools.cached_property
    def shapely_ray(self):
        first_point = np.array(self.shapely.coords[0])
        last_point = np.array(self.shapely.coords[-1])

        direction_vector = last_point - first_point
        direction_vector /= np.linalg.norm(direction_vector)

        very_large_value = 1e6
        first_point_extended = first_point - very_large_value * direction_vector
        last_point_extended = last_point + very_large_value * direction_vector

        return shapely.LineString([first_point_extended, last_point_extended])

    def signed_distance(self, x, y):
        A = self.y2 - self.y1
        B = self.x1 - self.x2
        C = self.x2 * self.y1 - self.x1 * self.y2
        return (A * x + B * y + C) / self.length

    def distance(self, x, y):
        return abs(self.signed_distance(x, y))

    def angle_difference(self, other):
        diff = abs(self.angle - other.angle)
        return min(diff, abs(np.pi - diff))

    def nearly_extends(self, other, angle_tolerance=0.01, dist_tolerance=1.0):
        # Calculate the difference in angles
        angle_diff = self.angle_difference(other)

        # Calculate the distances from the endpoints of 'other' to 'self'
        dist_start = self.distance(other.start_x, other.start_y)
        dist_end = self.distance(other.end_x, other.end_y)

        # Check if the lines are nearly parallel and the distances are within the tolerance
        return angle_diff <= angle_tolerance and max(dist_start, dist_end) <= dist_tolerance

    def combine(self, other):
        # Check if the lines nearly extend each other
        if not self.nearly_extends(other):
            raise ValueError("Lines do not nearly extend each other")

        # Calculate all possible combinations of start and end points
        combinations = [
            (self.x1, self.y1, other.end_x, other.end_y),
            (other.start_x, other.start_y, self.x2, self.y2),
            (self.x2, self.y2, other.start_x, other.start_y),
            (other.end_x, other.end_y, self.x1, self.y1),
        ]

        # Choose the longest line using max() with key parameter
        longest = max(
            combinations,
            key=lambda coords: math.sqrt(
                (coords[0] - coords[2]) ** 2 + (coords[1] - coords[3]) ** 2
            )
        )

        # Return a new Line object that represents the combined line
        return Line(*longest)

    def increasing_x(self):
        if self.x1 > self.x2:
            return type(self)(self.x2, self.y2, self.x1, self.y1)
        return self

    def project_point(self, point: Tuple[float]):
        point_vector = np.array(point) - np.array([self.x1, self.y1])
        projected_length = np.dot(point_vector, self.line_unit_vector)
        return np.array([self.x1, self.y1]) + self.line_unit_vector * projected_length

    def scalar_projection(self, point):
        """Scalar projection of point onto line."""
        projected_point = self.project_point(point)
        distance = np.linalg.norm(np.array([self.x1, self.y1]) - np.array(projected_point))
        return distance / self.length

    @functools.cached_property
    def line_vector(self):
        return np.array(self.end) - np.array(self.start)

    @functools.cached_property
    def line_unit_vector(self):
        return self.line_vector / self.length

    def point_at_scalar(self, scalar: float):
        return self.start + self.line_vector * scalar

    def __repr__(self):
        return f"<{type(self).__name__} {str(self.__dict__)}>"

    def hough_intersection(self, other):
        if self.angle == other.angle:
            return None

        A = np.array([
            [np.cos(self.angle), np.sin(self.angle)],
            [np.cos(other.angle), np.sin(other.angle)]
        ])

        b = np.array([self.rho, other.rho])

        return np.linalg.solve(A, b)

    def intersection(self, other):
        if self.slope == other.slope:
            return None

        x = (other.intercept - self.intercept) / (self.slope - other.slope)

        y = self.slope * x + self.intercept

        return x, y


class HoughLine:

    def __init__(self, rho, theta):
        self.rho = rho
        self.theta = theta

    # Define a method to calculate a point a given distance away from the 'start' of the line
    def point_at_scalar(self, scalar):
        x = self.rho * np.cos(self.theta) + scalar * np.sin(self.theta)
        y = self.rho * np.sin(self.theta) - scalar * np.cos(self.theta)
        return (x, y)

    def scalar_projection(self, point):
        x, y = point
        # Project the point onto the line
        projection = (x * np.cos(self.theta) + y * np.sin(self.theta)) - self.rho
        return projection
