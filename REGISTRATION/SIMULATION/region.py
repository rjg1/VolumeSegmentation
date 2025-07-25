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
    """Calculate the geometric center (bounding box midpoint) of a list of 3D points."""
    points = np.array(points)
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)
    centroid = (min_xyz + max_xyz) / 2
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
    def __init__(self, points=None, original_index = None, precalc_area = False, precalc_centroid = False, precalc_radius = False):
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
        if points is not None and len(points) > 0:
            # Attempt to order points using convexhull
            self.points = [(x,y,z) for x,y,z in order_points_convex_hull(points)]
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


    def compute_uoi_with(self, other_region):
        """
        Computes the Union of Intersection (UoI) between this region and another.
        This is done in 2D (x, y) by default.
        Returns a float between 0 and 1, where 1 means perfect overlap.
        """
        # Extract 2D projections of boundary points (default x, y)
        poly_a = Polygon([(x, y) for x, y, z in self.get_boundary_points()])
        poly_b = Polygon([(x, y) for x, y, z in other_region.get_boundary_points()])

        if not poly_a.is_valid or not poly_b.is_valid:
            return 0.0

        intersection = poly_a.intersection(poly_b)
        union = poly_a.union(poly_b)

        if union.area == 0:
            return 0.0
        return intersection.area / union.area
        

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
    if len(points) < 3:
        return points  # Can't form a polygon

    try:
        points = np.array(points)  # Ensure NumPy for indexing
        points_2d = points[:, :2]  # Take x and y
        zs = points[:, 2]          # Save z separately

        hull = ConvexHull(points_2d, qhull_options='QJ')
        ordered_xy = points_2d[hull.vertices]
        ordered_z = zs[hull.vertices]

        ordered = np.column_stack((ordered_xy, ordered_z))

        # Ensure closed polygon
        if not np.array_equal(ordered[0], ordered[-1]):
            ordered = np.vstack([ordered, ordered[0]])

        return ordered
    except Exception as e:
        print(f"[REGION - order_points_convex_hull][ConvexHull ordering failed]: {e}")
        return points

def create_polygon_from_points(points):
    if len(points >= 4):
        polygon = Polygon(points)
        return polygon
    else: 
        return None

