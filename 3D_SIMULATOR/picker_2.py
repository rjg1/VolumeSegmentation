import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import csv

class Picker:
    def __init__(self, plotter, meshes, z_planes, y_planes, xy_slice_meshes, xz_slice_meshes, xz_generated, dimensions):
        self.plotter = plotter
        self.meshes = meshes
        self._points = []
        self.z_planes = z_planes
        self.y_planes = y_planes
        self.xy_slice_meshes = xy_slice_meshes
        self.xz_slice_meshes = xz_slice_meshes
        self.xz_generated = xz_generated
        center, size = dimensions
        self.center = center
        self.size = size
        
    @property
    def points(self):
        """To access all the points when done."""
        return self._points
        
    def __call__(self, *args):
        picked_pt = np.array(self.plotter.pick_mouse_position())
        direction = picked_pt - self.plotter.camera_position[0]
        direction = direction / np.linalg.norm(direction)
        start = picked_pt - 10000 * direction
        end = picked_pt + 10000 * direction
        
        # Initialize variables to store the closest point and its distance
        closest_point = None
        closest_distance = float('inf')
        
        # Loop through each mesh and perform ray tracing
        for mesh in self.meshes:
            point, ix = mesh.ray_trace(start, end, first_point=True)
            if len(point) > 0:
                # Calculate distance from the picked point to the intersection point
                distance = np.linalg.norm(picked_pt - point)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_point = point
        
        # If a closest point is found, add it to the list of points
        if closest_point is not None:
            self._points.append(closest_point)
            # Add a sphere at the closest point
            z = self.find_nearest_z(closest_point[2])
            self.display_2d_plane(z, self.xy_slice_meshes, self.plotter, self.center, self.size)
            y = self.find_nearest_y(closest_point[1])
            self.display_2d_plane_xz(y, self.xz_slice_meshes, self.plotter, self.center, self.size)
            self.update_plot(y)

    def find_nearest_z(self, z):
        return min(self.z_planes, key=lambda x: abs(x - z))
    
    def find_nearest_y(self, y):
        return min(self.y_planes, key=lambda x: abs(x - y))

    def update_plot(self, y):
        xz_list = self.xz_generated[y]
        with open("points_out.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x', 'z'])
            writer.writerows(xz_list)


        x_vals, z_vals = zip(*xz_list) if xz_list else ([], [])
        plt.figure()  # Open a new window
        plt.scatter(x_vals, z_vals, color='blue')  # Create scatter plot
        plt.title('Scatter Plot of (x, z) Points')
        plt.xlabel('x')
        plt.ylabel('z')

        plt.xlim(-3,3)
        plt.ylim(-3,3)

        plt.grid(True)
        plt.show()

    def display_2d_plane(self, z, xy_slice_meshes, plotter, center, size):
        # Clear the plotter
        cmap = plt.cm.get_cmap('tab10')
        plotter.clear()

        # Add the plane to the left subplot
        left_plane = pv.Plane(center=(center[0], center[1], z), direction=(0, 0, 1), i_size=size[0], j_size=size[1])
        plotter.subplot(0, 0)
        plotter.add_mesh(left_plane, color='green', style='wireframe', opacity=0.2)

        # Add meshes to the left subplot
        for z_plane, objs in xy_slice_meshes.items():
            for obj_num, mesh in objs.items():
                plotter.subplot(0, 0)
                plotter.add_mesh(mesh, color=cmap(obj_num))

                # If the z_plane matches the current z value, add mesh to the right subplot
                if z_plane == z:
                    plotter.subplot(0, 1)
                    plotter.add_mesh(mesh, color=cmap(obj_num))
                    # plotter.subplot(0, 1)
                    # plotter.camera_position = [center[0], center[1], center[2] + 2 * size[2]]
            if z_plane == z:
                right_plane = pv.Plane(center=(center[0], center[1], z), direction=(0, 0, 1), i_size=size[0], j_size=size[1])
                plotter.subplot(0, 1)
                plotter.add_mesh(right_plane, color='green', style='wireframe', opacity=0.2)

        plotter.subplot(0, 1)
        plotter.camera_position = [center[0], center[1], center[2] + 2 * size[2]]


        # Set camera position to focus on the x,z plane
        # plotter.subplot(0, 0)
        # plotter.camera_position = [center[0] + 2 * size[0], center[1], center[2]]

    def display_2d_plane_xz(self, y, xz_slice_meshes, plotter, center, size):
        # Clear the plotter
        cmap = plt.cm.get_cmap('tab10')

        # Add the plane to the left subplot
        left_plane = pv.Plane(center=(center[0], y, center[2]), direction=(0, 1, 0), i_size=size[2], j_size=size[0])
        plotter.subplot(1, 0)
        plotter.add_mesh(left_plane, color='green', style='wireframe', opacity=0.2)

        # Add meshes to the left subplot
        for y_plane, objs in xz_slice_meshes.items():
            for obj_num, mesh in objs.items():
                plotter.subplot(1, 0)
                plotter.add_mesh(mesh, color=cmap(obj_num))

                # If the z_plane matches the current z value, add mesh to the right subplot
                if y_plane == y:
                    plotter.subplot(1, 1)
                    plotter.add_mesh(mesh, color=cmap(obj_num))
                    
            if y_plane == y:
                plane = pv.Plane(center=(center[0], y, center[2]), direction=(0, 1, 0), i_size=size[2], j_size=size[0])
                plotter.subplot(1, 1)
                plotter.add_mesh(plane, color='green', style='wireframe', opacity=0.2)

        plotter.subplot(1, 1)
        plotter.camera_position = [center[0], center[1] + 2 * size[1], center[2]]
         # Set camera position to focus on the x,z plane
        # plotter.subplot(1, 0)
        # plotter.camera_position = [center[0] + 2 * size[0], center[1], center[2]]