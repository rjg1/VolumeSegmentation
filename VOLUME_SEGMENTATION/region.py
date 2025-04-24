import numpy as np
from scipy.spatial import ConvexHull
import random
from shapely.geometry import Polygon, Point
from scipy.spatial import distance

# Seed for random operation random
RANDOM_SEED = 10
# Points to sample for radius
RADIUS_SAMPLE_POINTS = 20

def calculate_area_of_roi(boundary_points):
    """
    Calculate the area of a region using the boundary points.
    Assumes the boundary points form a closed polygon in 2D (x, y).
    """
    # Extract 2D points (x, y) from the boundary points
    polygon_points = [(x, y) for x, y, z in boundary_points]  # Use only the x and y coordinates
    
    if len(polygon_points) < 3:
        # print(f"Invalid ROI: Only {len(polygon_points)} points provided.")
        return 0.0  # Return 0 area for invalid polygons
    polygon = Polygon(polygon_points)  # Create a polygon from the points
    return polygon.area

def find_centroid_3d(points):
    """Calculate the centroid of a list of points (x, y, z)."""
    points = np.array(points)
    centroid = np.mean(points, axis=0)  # [mean_x, mean_y, mean_z]
    return centroid

def calculate_avg_radius(boundary_points, centroid, num_samples, random_generator):
    """
    Calculate an approximate average radius of an ROI by randomly sampling points
    from the boundary points. The radius is calculated based on the distances
    between the centroid and the sampled points.
    """
    num_samples = min(num_samples, len(boundary_points))

    # Randomly sample points from the boundary using the provided random generator
    sampled_points = random_generator.sample(boundary_points, num_samples)

    # Calculate distances from the centroid to each sampled point
    distances = [distance.euclidean(centroid[:2], (x, y)) for x, y, z in sampled_points]

    # Return the average distance (average radius)
    return sum(distances) / len(distances) if distances else 0

class Region:
    def __init__(self, points=None, original_index = None, precalc_area = False, precalc_centroid = False, precalc_radius = False, intensity = 0):
        # Initialise max bound points
        self.random_generator = random.Random(RANDOM_SEED)
        self.xmin = np.inf
        self.ymin = np.inf
        self.zmin = np.inf
        self.xmax = -np.inf
        self.ymax = -np.inf
        self.zmax = -np.inf
        self.area = None
        self.centroid = None
        self.avg_radius = None
        self.original_index = original_index
        self.intensity = intensity
        if points:
            self.points = points
            # Update max and min points
            for point in points:
                self.check_point_values(point)
        else:
            self.points = []

        # Pre-calc area if applicable
        if precalc_area:
            self.area = calculate_area_of_roi(self.points)
        # Pre-calc centroid if applicable
        if precalc_centroid:
            self.centroid = find_centroid_3d(self.points)
        # Pre-calc roi radius if applicable
        if precalc_radius:
            self.avg_radius = calculate_avg_radius(self.points, self.get_centroid(), num_samples=RADIUS_SAMPLE_POINTS, random_generator=self.random_generator)
    
    def get_centroid(self):
        if self.centroid is None:
            self.centroid = find_centroid_3d(self.points)
        return self.centroid

    def get_area(self):
        if self.area is None:
            self.area = calculate_area_of_roi(self.points)
        return self.area

    def get_radius(self):
        if self.avg_radius is None:
            self.avg_radius = calculate_avg_radius(self.points, self.get_centroid(), num_samples=RADIUS_SAMPLE_POINTS, random_generator=self.random_generator)
        return self.avg_radius

    def get_original_index(self):
        return self.original_index

    def add_point(self, point):
        """Add a new point to the region and update the bounds."""
        self.points.append(point)
        self.check_point_values(point)

    def add_points(self, points):
        """Add multiple points to the region and update the bounds."""
        for point in points:
            self.add_point(point)
    
    def get_boundary_points(self):
        return self.points

    def check_point_values(self, point):
        """Update the min/max values based on the new point."""
        x, y, z = point
        # Update min values
        self.xmin = min(self.xmin, x)
        self.ymin = min(self.ymin, y)
        self.zmin = min(self.zmin, z)
        # Update max values
        self.xmax = max(self.xmax, x)
        self.ymax = max(self.ymax, y)
        self.zmax = max(self.zmax, z)

    # Determine if two regions have a chance at overlapping points
    def overlaps_with_region(self, other_region):
        return  self.xmin <= other_region.xmax and self.xmax >= other_region.xmin and \
                self.ymin <= other_region.ymax and self.ymax >= other_region.ymin and \
                self.zmin <= other_region.zmax and self.zmax >= other_region.zmin


class BoundaryRegion(Region):
    # Generate a number of points within this region
    def generate_region_points(self, num_points):
        # Determine static value in xyz
        prev_x = None
        prev_y = None
        prev_z = None
        total_x_diff = 0
        total_y_diff = 0
        total_z_diff = 0
        for x,y,z in self.points:
            if prev_x and prev_y and prev_z:
                total_x_diff += abs(prev_x - x)
                total_y_diff += abs(prev_y - y)
                total_z_diff += abs(prev_z - z)

            prev_x = x
            prev_y = y
            prev_z = z

        # TODO - worth as a hyper?
        threshold = 0.0001
        points_2d = []

        if total_x_diff / len(self.points) < threshold :
            points_2d = [(y,z) for x,y,z in self.points]
        elif total_y_diff / len(self.points) < threshold :
            points_2d = [(x,z) for x,y,z in self.points]
        elif total_z_diff / len(self.points) < threshold :
            points_2d = [(x,y) for x,y,z in self.points]
        else:
            return []

        ordered_points = order_points_convex_hull(np.array(points_2d))
        polygon = create_polygon_from_points(ordered_points)
        if not polygon:
            return []
        
        min_x, min_y, max_x, max_y = polygon.bounds
        points = []
    
        while len(points) < num_points:
            random_point = Point(self.random_generator.uniform(min_x, max_x), self.random_generator.uniform(min_y, max_y))
            if polygon.contains(random_point):
                points.append((random_point.x, random_point.y))
            
        
        if total_x_diff / len(self.points) < threshold :
            fixed_x, _, _ = self.points[0]
            points_3d = [(fixed_x, y,z) for y,z in points]
        elif total_y_diff / len(self.points) < threshold :
            _, fixed_y, _ = self.points[0]
            points_3d = [(x, fixed_y,z) for x,z in points]
        else:
            _, _, fixed_z = self.points[0]
            points_3d = [(x, y, fixed_z) for x,y in points]

        return points_3d

class ClusterRegion(Region):
    def __init__(self, points=None):
        super().__init__(points)
        self.hull = None
        if points:
            self.hull_updated = True
        else:
            self.hull_updated = False

    def get_boundary_points(self):
        if self.hull_updated and self.points:
            self.hull = ConvexHull(self.points)
            self.hull_updated = False

        # Extract the vertices of the convex hull
        if self.hull:
            vertices = self.hull.points[self.hull.vertices]
            # Convert the vertices to a list of (x, y) tuples
            boundary_points = [(x, y) for x, y in vertices]

            return boundary_points
        else:
            # No points in this object
            return None

    def add_point(self, point):
        super().add_point(point)
        self.hull_updated = True

    def add_points(self, points):
        for point in points:
            self.add_point(point)

# TESTING

def order_points_convex_hull(points):
    hull = ConvexHull(points)
    ordered_points = points[hull.vertices]
    if not np.array_equal(ordered_points[0], ordered_points[-1]):
        # If not, append the first point to the end
        ordered_points = np.vstack([ordered_points, ordered_points[0]])
    return ordered_points

def create_polygon_from_points(points):
    if len(points >= 4):
        polygon = Polygon(points)
        return polygon
    else: 
        return None

