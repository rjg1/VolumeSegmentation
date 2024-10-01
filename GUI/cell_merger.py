import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import tifffile as tif
import numpy as np
import random
import colorsys
import os
import alphashape
from scipy.spatial import distance
from shapely.geometry import LineString



POINT_PRECISION = 2 # Lower = more precise clicks required, but can select ROIs in dense areas easier
NUM_INTERPOLATION_POINTS = 75 # Higher = hopefully better, but more cost at the end (helps prevent overlap in new hulls)

class VolumeSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cluster Merging GUI")
        random.seed(5)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Variables
        self.csv_path = ""
        self.image_stack = None
        self.current_z = 0
        self.roi_visibility = {}
        self.roi_colours = {}
        self.rois_per_z = {}  # Dictionary to store ROIs per z-plane
        self.data = None
        self.cluster_modified = {} # track whether a cluster has been modified
        self.current_roi = 1
        self.zoom_selector = None
        self.cl_to_roi = {}
        self.selected_clusters = set()  # Clusters selected for merging
        self.selected_cluster_id = None  # Track selected cluster
        self.roi_colours = {}  # ROI color map
        self.roi_colours[self.current_roi] = self.generate_roi_color()
        self.trashed_clusters = {}
        self.merge_mode = True  # Merge mode flag
        self.brightness_factor = 1.0  # Brightness adjustment
        self.zoom_enabled = False
        self.trash_mode = False
        self.stored_xlim = None
        self.stored_ylim = None
        self.current_scroll_position = (0, 0) 
        self.image_updated = False
        self.opened_listbox = None  # Track the currently opened listbox

        # Setup icons and GUI
        self.setup_gui()

        # Hotkey bindings
        self.root.bind("1", lambda event: self.prev_z())
        self.root.bind("2", lambda event: self.next_z())
        self.root.bind("m", lambda event: self.toggle_merge_mode())
        self.root.bind("n", lambda event: self.new_cluster())
        self.root.bind("z", lambda event: self.toggle_zoom())
        self.root.bind("t", lambda event: self.toggle_trash_mode())
        self.root.bind("<space>", lambda event: self.merge_selected_clusters())

    def setup_gui(self):
        # Create the menu bar
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open CSV", command=self.load_csv)
        file_menu.add_command(label="Open TIF", command=self.load_tif_file)

         # Export menu
        export_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Export", menu=export_menu)
        export_menu.add_command(label="Export ROI Data", command=self.export_roi_data)

        # Import menu
        import_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Import", menu=import_menu)
        import_menu.add_command(label="Import GUI State", command=self.import_gui_state)

        # Navigation frame
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.prev_button = tk.Button(control_frame, text="Previous Z", command=self.prev_z)
        self.prev_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.next_button = tk.Button(control_frame, text="Next Z", command=self.next_z)
        self.next_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Entry to jump to a specific z-plane
        self.z_entry = tk.Entry(control_frame, width=5)
        self.z_entry.pack(side=tk.LEFT, padx=5)
        self.z_entry.bind("<Return>", self.jump_to_z)  # Jump when Enter is pressed

        self.z_label = tk.Label(control_frame, text="Z: 0/0")
        self.z_label.pack(side=tk.LEFT, padx=5)

        # Add Merge button with toggle effect and mutual exclusivity with zoom
        self.merge_button = tk.Button(control_frame, text="Merge Mode (m)", bg='lightgreen', command=self.toggle_merge_mode)
        self.merge_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Zoom button with toggle effect
        self.zoom_button = tk.Button(control_frame, text="Zoom Mode (z)", command=self.toggle_zoom, bg='lightgray')
        self.zoom_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Trash Mode button
        self.trash_button = tk.Button(control_frame, text="Trash Mode (t)", bg='lightgray', command=self.toggle_trash_mode)
        self.trash_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Reset zoom button
        self.reset_zoom_button = tk.Button(control_frame, text="Reset Zoom", command=self.reset_zoom)
        self.reset_zoom_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Current ROI
        self.current_roi_label = tk.Label(control_frame, text=f"Current ROI: {self.current_roi}")
        self.current_roi_label.pack(side=tk.LEFT, padx=5)

        # Delete ROI
        self.delete_roi_button = tk.Button(control_frame, text="Delete ROI", command=self.delete_roi)
        self.delete_roi_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Brightness slider
        slider_frame = tk.Frame(self.root)
        slider_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.brightness_slider = tk.Scale(slider_frame, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, 
                                        label="Brightness")
        self.brightness_slider.set(1.0)  # Set default brightness to 1 (no change)
        self.brightness_slider.pack(fill=tk.X, padx=5, pady=5)
        self.brightness_slider.bind("<ButtonRelease-1>", self.on_slider_release)

        # Sidebar for ROI management
        sidebar_frame = tk.Frame(self.root, width=200)
        sidebar_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.sidebar_canvas = tk.Canvas(sidebar_frame, width=200)
        self.sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.sidebar_scrollbar = ttk.Scrollbar(sidebar_frame, orient="vertical", command=self.sidebar_canvas.yview)
        self.sidebar_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.sidebar_content = tk.Frame(self.sidebar_canvas)
        self.sidebar_content.bind("<Configure>", self.configure_scroll_region)
        self.sidebar_canvas.create_window((0, 0), window=self.sidebar_content, anchor="nw")
        self.sidebar_canvas.config(yscrollcommand=self.sidebar_scrollbar.set)

        # Canvas for image display
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)

    def import_gui_state(self):
        def find_most_recent_file(folder, file_prefix):
            """Find the most recent file in a folder matching the prefix."""
            files = [f for f in os.listdir(folder) if f.startswith(file_prefix) and f.endswith('.csv')]
            if not files:
                return None
            # Sort files based on the timestamp in their filenames (assuming format: file_prefix_YYYYMMDD_HHMMSS.csv)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True)
            return os.path.join(folder, files[0])  # Return the most recent file

        # Ask user whether to import the most recent files automatically or choose manually
        user_choice = tk.messagebox.askyesno("Import Data", "Would you like to import the most recent data from the debug folder?")

        debug_folder = './debug/'
        if user_choice:
            # Option 1: Automatically find and load the most recent files from the debug folder
            roi_merges_file = find_most_recent_file(debug_folder, 'roi_merges')
            trash_file = find_most_recent_file(debug_folder, 'trash')
            if not roi_merges_file or not trash_file:
                tk.messagebox.showerror("Error", "No recent roi_merges.csv or trash.csv files found in the debug folder.")
                return
        else:
            # Option 2: Allow user to manually select files
            roi_merges_file = filedialog.askopenfilename(title="Select ROI Merges CSV", filetypes=[("CSV Files", "*.csv")])
            trash_file = filedialog.askopenfilename(title="Select Trash CSV", filetypes=[("CSV Files", "*.csv")])
        
        # Ensure both files exist before proceeding
        if not roi_merges_file or not trash_file or not os.path.exists(roi_merges_file) or not os.path.exists(trash_file):
            tk.messagebox.showerror("Error", "roi_merges.csv or trash.csv file not found or not selected.")
            return

        # 1. Load the ROI merges
        try:
            roi_merges_df = pd.read_csv(roi_merges_file)
            print(f"Loaded roi_merges.csv from {roi_merges_file}")

            # Clear existing ROI data
            self.rois_per_z.clear()
            self.cl_to_roi.clear()

            # Group by z-plane and MERGE_ID
            for z, roi_group in roi_merges_df.groupby('z'):
                if z not in self.rois_per_z:
                    self.rois_per_z[z] = {}
                for merge_id, roi_data in roi_group.groupby('MERGE_ID'):
                    # The MERGE_ID is what will be used as the key for self.rois_per_z
                    cluster_ids = roi_data['ROI_ID'].unique()
                    self.rois_per_z[z][merge_id] = list(cluster_ids)
                    
                    # Generate a color for each ROI (MERGE_ID)
                    if merge_id not in self.roi_colours:
                        self.roi_colours[merge_id] = self.generate_roi_color()

                    # Map each cluster ID to the respective ROI (MERGE_ID)
                    if z not in self.cl_to_roi:
                        self.cl_to_roi[z] = {}
                    for cl_id in cluster_ids:
                        self.cl_to_roi[z][cl_id] = merge_id
                        self.cluster_modified[z][cl_id] = True

            print("ROI merges successfully restored.")

        except Exception as e:
            print(f"Error loading roi_merges.csv: {e}")
            tk.messagebox.showerror("Error", f"Error loading roi_merges.csv: {e}")
            return

        # 2. Load the trashed clusters
        try:
            trash_df = pd.read_csv(trash_file)
            print(f"Loaded trash.csv from {trash_file}")

            # Clear existing trashed clusters
            self.trashed_clusters.clear()

            # Group by z-plane and collect trashed clusters
            for z, group in trash_df.groupby('z'):
                self.trashed_clusters[z] = group['ROI_ID'].unique().tolist()

            print("Trashed clusters successfully restored.")

        except Exception as e:
            print(f"Error loading trash.csv: {e}")
            tk.messagebox.showerror("Error", f"Error loading trash.csv: {e}")
            return

        # Update the display and sidebar
        self.update_display()
        self.update_sidebar()

        tk.messagebox.showinfo("Import Complete", "ROI and trash data have been successfully imported.")

    def export_roi_data(self):
        if self.data is None or self.data.empty:
            tk.messagebox.showerror("Error", "No data loaded to export.")
            return

        # Make a copy of the original data
        export_data = self.data.copy()

        # Step 1: Remove any rows corresponding to trashed clusters
        for z, trashed_clusters in self.trashed_clusters.items():
            export_data = export_data[~((export_data['z'] == z) & (export_data['ROI_ID'].isin(trashed_clusters)))]

        # Helper function to snap points to nearest cluster points
        def snap_to_nearest_points(boundary_points, original_points):
            snapped_boundary = []
            for point in boundary_points:
                # Compute the distance between the boundary point and all original points
                dists = distance.cdist([point], original_points, 'euclidean')
                # Find the nearest original point
                nearest_point = original_points[dists.argmin()]
                snapped_boundary.append(nearest_point)
            return np.array(snapped_boundary)

        # Helper function to interpolate along the alpha shape boundary
        def interpolate_boundary(boundary_points, num_interpolation_points=10):
            line = LineString(boundary_points)
            interpolated_points = []
            for i in range(num_interpolation_points):
                interpolated_points.append(line.interpolate(i / (num_interpolation_points - 1), normalized=True).coords[0])
            return np.array(interpolated_points)

        # Step 2: Handle merged clusters (those in `rois_per_z`)
        for z, rois in self.rois_per_z.items():
            print(f"Processing merged clusters in z={z}")
            for roi_id, cluster_ids in rois.items():
                # Collect all points for the merged clusters
                roi_points = []
                for cl_id in cluster_ids:
                    cluster_points = export_data[(export_data['z'] == z) & (export_data['ROI_ID'] == cl_id)][['x', 'y']].values
                    if cluster_points.size > 0:
                        roi_points.extend(cluster_points)
                    # Remove the original rows for this cluster
                    export_data = export_data[~((export_data['z'] == z) & (export_data['ROI_ID'] == cl_id))]

                roi_points = np.array(roi_points)

                # If more than 2 points, use alpha shape to get the boundary
                if len(roi_points) > 2:
                    # Generate the alpha shape
                    alpha_shape = alphashape.alphashape(roi_points, 0.05)

                    # Extract the boundary points from the alpha shape if valid
                    if not alpha_shape.is_empty and alpha_shape.geom_type == 'Polygon':
                        boundary_points = np.array(alpha_shape.exterior.coords)

                        # Interpolate to increase the density of boundary points
                        boundary_points = interpolate_boundary(boundary_points, num_interpolation_points=NUM_INTERPOLATION_POINTS)
                    else:
                        boundary_points = roi_points

                    # Snap boundary points to nearest original points
                    snapped_boundary_points = snap_to_nearest_points(boundary_points, roi_points)

                    if len(snapped_boundary_points) > 0:
                        min_cl_id = min(cluster_ids)  # ROI_ID is the minimum cluster ID

                        # Add the snapped boundary points to the dataframe
                        new_rows = pd.DataFrame(snapped_boundary_points, columns=['x', 'y'])
                        new_rows['z'] = z
                        new_rows['ROI_ID'] = min_cl_id
                        export_data = pd.concat([export_data, new_rows], ignore_index=True)
                else:
                    # If not enough points for a boundary, retain the original points
                    if roi_points.size > 0:
                        new_rows = pd.DataFrame(roi_points, columns=['x', 'y'])
                        new_rows['z'] = z
                        new_rows['ROI_ID'] = min(cluster_ids)
                        export_data = pd.concat([export_data, new_rows], ignore_index=True)

        # Step 3: Save the final data to CSV
        export_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])

        if export_path:
            # Export the final data to CSV
            export_data.to_csv(export_path, index=False)
            tk.messagebox.showinfo("Export Complete", f"CSV has been exported successfully to {export_path}.")
        else:
            tk.messagebox.showerror("Error", "Export was cancelled or no file selected.")

        self.export_gui_state()



    def export_gui_state(self, closing = False):
        if self.data is None or self.data.empty and not closing:
            tk.messagebox.showerror("Error", "No data loaded to export.")
            return

        # Create debug directory if it doesn't exist
        debug_dir = "./debug"
        os.makedirs(debug_dir, exist_ok=True)

        # Get the current timestamp to append to the filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Paths for the CSVs with a timestamp
        trash_file = os.path.join(debug_dir, f"trash_{timestamp}.csv")
        roi_merges_file = os.path.join(debug_dir, f"roi_merges_{timestamp}.csv")

        # DataFrame for ROI clusters
        roi_export_data = pd.DataFrame(columns=['x', 'y', 'z', 'ROI_ID', 'MERGE_ID'])

        # DataFrame for trashed clusters
        trash_export_data = pd.DataFrame(columns=['x', 'y', 'z', 'ROI_ID'])

        # 1. Iterate through each z-plane
        for z in self.data['z'].unique():
            # Get all clusters in this z-plane
            all_clusters = self.data[self.data['z'] == z]['ROI_ID'].unique()

            # Process clusters associated with ROIs
            for roi_id, cluster_ids in self.rois_per_z.get(z, {}).items():
                for cl_id in cluster_ids:
                    # Get all points for this cluster
                    cluster_points = self.data[(self.data['z'] == z) & (self.data['ROI_ID'] == cl_id)][['x', 'y']]
                    cluster_points['z'] = z
                    cluster_points['ROI_ID'] = cl_id
                    cluster_points['MERGE_ID'] = roi_id
                    roi_export_data = pd.concat([roi_export_data, cluster_points], ignore_index=True)

            # Process trashed clusters
            trashed_clusters = self.trashed_clusters.get(z, [])
            for cl_id in trashed_clusters:
                # Get all points for this trashed cluster
                cluster_points = self.data[(self.data['z'] == z) & (self.data['ROI_ID'] == cl_id)][['x', 'y']]
                cluster_points['z'] = z
                cluster_points['ROI_ID'] = cl_id
                trash_export_data = pd.concat([trash_export_data, cluster_points], ignore_index=True)

        # Save ROI clusters to CSV
        roi_export_data.to_csv(roi_merges_file, index=False)

        # Save trashed clusters to CSV
        trash_export_data.to_csv(trash_file, index=False)
        # if not closing:
        #     tk.messagebox.showinfo("Export Complete", f"State has been successfully exported.\nROI Clusters CSV: {roi_csv_path}\nTrashed Clusters CSV: {trash_csv_path}")

    def toggle_trash_mode(self):
            """Enable trash mode, mutually exclusive with merge and zoom modes."""
            self.trash_mode = not self.trash_mode
            if self.merge_mode and self.trash_mode:
                self.toggle_merge_mode()
            if self.zoom_enabled and self.trash_mode:
                self.toggle_zoom()

            self.trash_button.config(bg='lightgreen' if self.trash_mode else 'lightgray')

            # If in trash mode, collapse all other tabs and expand trash tab
            if self.trash_mode:
                self.collapse_all_tabs_except_trash()
            else:
                self.update_sidebar()

            self.update_display()

    def collapse_all_tabs_except_trash(self):
        """Collapse all other tabs when trash mode is enabled and show the trash tab."""
        for widget in self.sidebar_content.winfo_children():
            widget.pack_forget()

        # Create the trash tab
        self.add_trash_tab()
    
    def add_trash_tab(self):
        """Add a dedicated tab for trashed clusters."""
        trash_frame = tk.Frame(self.sidebar_content, width=200)  # Fixed width for consistency
        trash_frame.pack(fill=tk.X, pady=2)

        trash_toggle_button = tk.Button(trash_frame, text=f"Trashed Clusters", bg="black", width=25)
        trash_toggle_button.pack(fill=tk.X)

        # Listbox for trashed clusters
        trash_listbox = tk.Listbox(trash_frame)
        for cl_id in self.trashed_clusters.get(self.current_z, []):
            trash_listbox.insert(tk.END, f"Cluster {cl_id}")

        trash_listbox.pack(fill=tk.X, padx=5, pady=5)
        trash_listbox.bind('<<ListboxSelect>>', self.on_trash_select)

    def on_trash_select(self, event):
        """Handle zooming to a selected trashed cluster from the trash list."""
        listbox = event.widget
        selection = listbox.curselection()
        if not selection:
            return

        selected_cluster = listbox.get(selection[0]).replace("Cluster ", "")
        selected_cluster_id = int(selected_cluster)

        clusters = self.data[(self.data['ROI_ID'] == selected_cluster_id) & (self.data['z'] == self.current_z)][['x', 'y']].values
        if clusters.size > 0:
            x_min, x_max = clusters[:, 0].min(), clusters[:, 0].max()
            y_min, y_max = clusters[:, 1].min(), clusters[:, 1].max()

            self.stored_xlim = (x_min - 50, x_max + 50)  # Zoom to [100, 100] dimensions
            self.stored_ylim = (y_max + 50, y_min - 50)

            self.ax.set_xlim(self.stored_xlim)
            self.ax.set_ylim(self.stored_ylim)
            self.canvas.draw_idle()

        self.update_display()

    def jump_to_z(self, event):
        if self.image_stack is not None:
            try:
                z_value = int(self.z_entry.get())
                if 0 <= z_value < self.image_stack.shape[0]:
                    old_current_z = self.current_z
                    self.current_z = z_value
                    if self.current_z != old_current_z:
                        # Re-draw clusters when this plane is re-visited
                        for cl_id in self.cluster_modified[old_current_z]:
                            self.cluster_modified[old_current_z][cl_id] = True
                        
                        # Update current ROI based on the new z-plane
                        self.current_roi = len(self.rois_per_z.get(self.current_z, {})) + 1
                        self.current_roi_label.config(text=f"Current ROI: {self.current_roi}")

                        # Update sidebar to reflect the new z-plane
                        self.update_display(z_changed = True)
                        self.update_sidebar()
                else:
                    tk.messagebox.showerror("Invalid Z-Plane", "Z-plane out of range.")
            except ValueError:
                tk.messagebox.showerror("Invalid Input", "Please enter a valid integer.")

    def delete_roi(self):
        # Check if the current ROI exists in the rois_per_z dictionary for the current z-plane
        if self.rois_per_z.get(self.current_z, {}).get(self.current_roi, None):
            # Get all clusters for the current ROI
            clusters_to_delete = self.rois_per_z[self.current_z][self.current_roi]

            # Dissociate the clusters from the ROI
            for cl_id in clusters_to_delete:
                self.cl_to_roi[self.current_z][cl_id] = None  # Dissociate the cluster
                self.selected_clusters.discard(cl_id)  # Discard from selected clusters if present
                self.cluster_modified[self.current_z][cl_id] = True  # Mark the cluster as modified

            # Remove the current ROI from the rois_per_z dictionary
            del self.rois_per_z[self.current_z][self.current_roi]

            # Re-number the remaining ROIs to maintain continuous numbering from 1 to n
            new_rois_per_z = {}
            roi_counter = 1
            for old_roi_id, clusters in sorted(self.rois_per_z[self.current_z].items()):
                new_rois_per_z[roi_counter] = clusters
                # Update cl_to_roi to reflect the new numbering
                for cl_id in clusters:
                    self.cl_to_roi[self.current_z][cl_id] = roi_counter
                    self.cluster_modified[self.current_z][cl_id] = True # Redraw new colours
                roi_counter += 1
            self.rois_per_z[self.current_z] = new_rois_per_z

            # Check if the current ROI was deleted, update current ROI to new num_rois + 1
            num_rois = len(self.rois_per_z[self.current_z])
            self.current_roi = num_rois + 1

            # Update the current ROI label
            self.current_roi_label.config(text=f"Current ROI: {self.current_roi}")

            # Update the display and sidebar
            self.update_sidebar()
            self.update_display()


    def onselect(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None not in (x1, y1, x2, y2):
            self.ax.set_xlim(min(x1, x2), max(x1, x2))
            self.ax.set_ylim(min(y1, y2), max(y1, y2))
            self.stored_xlim = self.ax.get_xlim()
            self.stored_ylim = self.ax.get_ylim()
            self.canvas.draw_idle()

            if not self.ax.yaxis_inverted():
                self.ax.invert_yaxis()

        if self.zoom_selector:
            self.zoom_selector.set_visible(False)
            self.ax.figure.canvas.draw_idle()

    def on_closing(self):
        self.export_gui_state(closing = True)
        self.root.quit()
        self.root.destroy()

    def configure_scroll_region(self, event=None):
        """Update the scroll region and keep the current scroll position."""
        self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))
        self.sidebar_canvas.yview_moveto(self.current_scroll_position[0])

    def load_csv(self):
        # Load CSV file
        self.csv_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if self.csv_path:
            self.data = pd.read_csv(self.csv_path)
            self.data.columns = self.data.columns
            self.current_z = self.data['z'].min()
            
             # Initialize the cluster_modified dictionary
            self.cluster_modified = {}
            for z_level in self.data['z'].unique():
                self.cluster_modified[z_level] = {}
                self.cl_to_roi[z_level] = {}
                self.trashed_clusters[z_level] = []
                cluster_ids = self.data[self.data['z'] == z_level]['ROI_ID'].unique()
                for cl_id in cluster_ids:
                    self.cluster_modified[z_level][cl_id] = True  # Set all clusters to True initially
                    self.cl_to_roi[z_level][cl_id] = None
            
            self.update_display()

    def load_tif_file(self):
        self.tif_path = filedialog.askopenfilename(filetypes=[("TIF Files", "*.tif")])
        if self.tif_path:
            self.image_stack = tif.imread(self.tif_path)
            self.current_z = 0
            self.image_updated = True
            self.update_display()


    def update_display(self, z_changed = False):
        if self.image_stack is not None and self.data is not None and 0 <= self.current_z < self.image_stack.shape[0]:
            if self.image_updated or z_changed:
                img = self.image_stack[self.current_z]
                img = np.clip(img * self.brightness_factor, 0, 255).astype(np.uint8)
                self.ax.clear()
                self.ax.imshow(img)
                self.image_updated = False

            # Get the clusters for the current z-plane
            clusters = self.data[self.data['z'] == self.current_z]

            unassociated_color_map = plt.colormaps['Reds']

            # Plot clusters
            for cl_id in clusters['ROI_ID'].unique():
                if self.cluster_modified[self.current_z].get(cl_id, True):  # Only redraw modified clusters
                    associated_roi = self.cl_to_roi.get(self.current_z,{}).get(cl_id, None)
                       
                    if associated_roi:
                        cl_points = clusters[clusters['ROI_ID'] == cl_id][['x', 'y']].values
                        roi_colour = self.roi_colours.get(associated_roi, 'lightgray')
                        self.ax.plot(cl_points[:, 0], cl_points[:, 1], label=f'ROI {associated_roi}', color=roi_colour)
                    elif cl_id in self.trashed_clusters.get(self.current_z, []):
                        cl_points = clusters[clusters['ROI_ID'] == cl_id][['x', 'y']].values
                        self.ax.plot(cl_points[:, 0], cl_points[:, 1], label=f'TRASHED CLUSTER {cl_id}', color='black')
                    else:
                        cl_points  = clusters[clusters['ROI_ID'] == cl_id][['x', 'y']].values
                        color_index = hash(cl_id) % 256
                        color = unassociated_color_map(color_index / 256)
                        self.ax.plot(cl_points[:, 0], cl_points[:, 1], label=f'CLUSTER {cl_id}', color=color)

                    self.cluster_modified[self.current_z][cl_id] = False


            # Reapply zoom limits if available
            if self.stored_xlim and self.stored_ylim:
                self.ax.set_xlim(self.stored_xlim)
                self.ax.set_ylim(self.stored_ylim)

            if not self.ax.yaxis_inverted():
                self.ax.invert_yaxis()
            self.canvas.draw()
            self.z_label.config(text=f"Z: {self.current_z}/{self.image_stack.shape[0] - 1}")
            self.z_entry.delete(0, tk.END)
            self.z_entry.insert(0, str(self.current_z))

    def toggle_merge_mode(self):
        self.merge_mode = not self.merge_mode
        if self.zoom_enabled and self.merge_mode:
            self.toggle_zoom()
        elif self.trash_mode and self.merge_mode:
            self.toggle_trash_mode()
        self.merge_button.config(bg='lightgreen' if self.merge_mode else 'lightgray')
        self.update_display()
        
    def new_cluster(self):   
        if not self.rois_per_z.get(self.current_z, None):
            return
        if self.current_roi != len(self.rois_per_z[self.current_z]) + 1:
            self.current_roi = len(self.rois_per_z[self.current_z]) + 1
            if not self.roi_colours.get(self.current_roi, None):
                self.roi_colours[self.current_roi] = self.generate_roi_color(previous_color=self.roi_colours[self.current_roi - 1])
            self.current_roi_label.config(text=f"Current ROI: {self.current_roi}") 
    
    def merge_selected_clusters(self):
        if self.selected_clusters and self.current_roi is not None:
            self.selected_clusters.clear()
            
            self.update_display()
            # Update current ROI after merging
            if self.current_roi == len(self.rois_per_z.get(self.current_z,0)):
                self.current_roi = 1 + max(self.rois_per_z.get(self.current_z, {}).keys(), default=0)
                print(f"Set current roi to: {self.current_roi} after merge")
                self.roi_colours[self.current_roi] = self.generate_roi_color(previous_color=self.roi_colours[self.current_roi - 1])
                self.current_roi_label.config(text=f"Current ROI: {self.current_roi}")

                # Close listbox for old selected roi if any
                if self.opened_listbox:
                    self.opened_listbox.pack_forget()
                    self.opened_listbox = None
            
            self.update_sidebar()

    def prev_z(self):
        if self.image_stack is not None:
            old_current_z = self.current_z
            self.current_z = max(0, self.current_z - 1)
            if self.current_z != old_current_z:

                # Re-draw clusters when this plane is re-visited
                for cl_id in self.cluster_modified[old_current_z]:
                    self.cluster_modified[old_current_z][cl_id] = True
                
                # Update current ROI based on the new z-plane
                self.current_roi = len(self.rois_per_z.get(self.current_z, {})) + 1
                self.current_roi_label.config(text=f"Current ROI: {self.current_roi}")

                # Update sidebar to reflect the new z-plane
                self.update_display(z_changed = True)
                self.update_sidebar()

    def next_z(self):
        if self.image_stack is not None:
            old_current_z = self.current_z
            self.current_z = min(self.image_stack.shape[0] - 1, self.current_z + 1)
            if self.current_z != old_current_z:
                
                # Re-draw clusters when this plane is re-visited
                for cl_id in self.cluster_modified[old_current_z]:
                    self.cluster_modified[old_current_z][cl_id] = True

                # Update current ROI based on the new z-plane
                self.current_roi = len(self.rois_per_z.get(self.current_z, {})) + 1
                self.current_roi_label.config(text=f"Current ROI: {self.current_roi}")

                # Update sidebar to reflect the new z-plane
                self.update_display(z_changed = True)
                self.update_sidebar()

    def toggle_zoom(self):
        self.zoom_enabled = not self.zoom_enabled
        # Disable merge if zoom is to be toggled on
        if self.merge_mode and self.zoom_enabled:
            self.toggle_merge_mode()
        elif self.trash_mode and self.zoom_enabled:
            self.toggle_trash_mode()
        self.zoom_button.config(bg='lightgreen' if self.zoom_enabled else 'lightgray')
        if self.zoom_enabled:
            self.enable_zoom()
        else:
            if self.zoom_selector:
                self.zoom_selector.set_active(False)
        print("calling update display - zoom toggle")
        self.update_display()

    def reset_zoom(self):
        # Reset zoom to the original state
        if self.image_stack is not None and self.image_stack.size > 0:
            self.ax.set_xlim(0, self.image_stack.shape[2])
            self.ax.set_ylim(self.image_stack.shape[1], 0)

        # Clear stored limits since zoom is reset
        self.stored_xlim = None
        self.stored_ylim = None
        self.canvas.draw_idle()

    def enable_zoom(self):
        if self.zoom_selector:
            self.zoom_selector.set_active(False)
            self.zoom_selector = None

        self.zoom_selector = RectangleSelector(
            self.ax,
            self.onselect,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True
        )
        self.canvas.draw_idle()


    def on_click(self, event):
        if self.data is None or self.image_stack is None:
            return

        x_click, y_click = event.xdata, event.ydata
        clusters = self.data[self.data['z'] == self.current_z]

        if not x_click or not y_click:
            return

        # Detect the clicked cluster by checking the distance from the click to cluster points
        for cl_id in clusters['ROI_ID'].unique():
            cl_points = clusters[clusters['ROI_ID'] == cl_id][['x', 'y']].values
            associated_roi = self.cl_to_roi.get(self.current_z,{}).get(cl_id, None)
            distances = np.sqrt((cl_points[:, 0] - x_click) ** 2 + (cl_points[:, 1] - y_click) ** 2)
            if (np.any(distances < POINT_PRECISION)):
                if self.merge_mode:
                    num_rois = len(self.rois_per_z.get(self.current_z, {}))
                    if not self.rois_per_z.get(self.current_z,None):
                        self.rois_per_z[self.current_z] = {}
                    if not self.rois_per_z[self.current_z].get(self.current_roi, None):
                        self.rois_per_z[self.current_z][self.current_roi] = []
                    if not associated_roi: # If not already associated with an ROI, associate it
                        self.rois_per_z[self.current_z][self.current_roi].append(cl_id)
                        self.cl_to_roi[self.current_z][cl_id] = self.current_roi
                        self.selected_clusters.add(cl_id)
                        self.cluster_modified[self.current_z][cl_id] = True
                        print(f"Cluster {cl_id} added to ROI {self.current_roi}")
                        self.update_display()
                        # Instantly merge in if editing an existing ROI
                        if self.current_roi < num_rois and num_rois > 0:
                            print("Updating sidebar for adding")
                            self.merge_selected_clusters()
                    elif associated_roi:
                        self.rois_per_z[self.current_z][associated_roi].remove(cl_id)
                        self.cl_to_roi[self.current_z][cl_id] = None
                        self.selected_clusters.discard(cl_id)
                        self.cluster_modified[self.current_z][cl_id] = True
                        print(f"Cluster {cl_id} removed from ROI {associated_roi}")
                        self.update_display()
                        # Instantly remove the ROI if editing an existing ROI
                        if self.current_roi < num_rois and num_rois > 0:
                            print("Updating sidebar for removal")
                            self.update_sidebar()
                elif self.trash_mode:
                    if cl_id in self.trashed_clusters.get(self.current_z,[]):
                        # Remove from trash
                        print(f"Removing cluster {cl_id} from trash")
                        self.trashed_clusters[self.current_z].remove(cl_id)
                        self.cluster_modified[self.current_z][cl_id] = True
                    else:
                        # Add to trash
                        print(f"Adding cluster {cl_id} to trash")
                        # Disassociate with any volumes
                        if associated_roi:
                            self.rois_per_z[self.current_z][associated_roi].remove(cl_id)
                            self.cl_to_roi[self.current_z][cl_id] = None
                            self.selected_clusters.discard(cl_id)
                        self.trashed_clusters[self.current_z].append(cl_id)
                        self.cluster_modified[self.current_z][cl_id] = True
                    
                    self.update_trash_sidebar()
                    self.update_display()
                else:
                    print(f"Cluster {cl_id} selected")
                break
        

    def update_trash_sidebar(self):
            """Update the trash tab in the sidebar to reflect the current trashed clusters."""
            self.current_scroll_position = self.sidebar_canvas.yview()  # Save scroll position

            # Remove existing sidebar content
            for widget in self.sidebar_content.winfo_children():
                widget.destroy()

            # Re-add trash tab with updated trashed clusters
            self.add_trash_tab()

            # Restore the scroll position
            self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))
            self.sidebar_canvas.yview_moveto(self.current_scroll_position[0])

    # def generate_roi_color(self):
    #     hue = random.random()
    #     lightness = 0.6
    #     saturation = 0.9
    #     rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    #     return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    def generate_roi_color(self, previous_color=None):
        min_distance = 100  # Minimum distance between colors to ensure distinctiveness

        def hex_to_rgb(hex_color):
            """Convert hex color string to an RGB tuple."""
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        def is_color_distinct(new_color_rgb, previous_rgb):
            """Calculate if the new RGB color is sufficiently distinct from the previous one."""
            if previous_rgb is None:
                return True
            distance = np.sqrt(sum((new - old) ** 2 for new, old in zip(new_color_rgb, previous_rgb)))
            return distance >= min_distance

        # Convert previous_color from hex to RGB if it exists
        if previous_color:
            previous_rgb = hex_to_rgb(previous_color)
        else:
            previous_rgb = None

        while True:
            # Generate a hue avoiding the red hue range (close to 0 and 1)
            hue = random.uniform(0.05, 0.95)  # Exclude values close to 0 (red) and 1 (red again)
            lightness = 0.6
            saturation = 0.9
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            rgb_255 = [int(x * 255) for x in rgb]

            # Check if the color is distinct from the previous color
            if is_color_distinct(rgb_255, previous_rgb):
                break

        return "#{:02x}{:02x}{:02x}".format(rgb_255[0], rgb_255[1], rgb_255[2])


    def toggle_roi_list(self, frame, roi_id):
        listbox = frame.winfo_children()[1]  # Find the listbox in the frame
        clusters = self.rois_per_z[self.current_z][roi_id]

        # Toggle visibility of the current listbox
        if listbox.winfo_viewable():
            listbox.pack_forget()  # Collapse the listbox
            self.opened_listbox = None
            # Reset current ROI to 1 greater than the total number of ROIs for this z-plane
            self.current_roi = len(self.rois_per_z.get(self.current_z, {})) + 1
            self.current_roi_label.config(text=f"Current ROI: {self.current_roi}")
        else:
            # Close any previously opened listbox
            if self.opened_listbox and self.opened_listbox != listbox:
                self.opened_listbox.pack_forget()

            # Adjust the height of the listbox based on the number of clusters
            listbox.config(height=min(len(clusters), 10))  # Cap at height 10
            listbox.pack(fill=tk.X, padx=5, pady=5)
            self.opened_listbox = listbox  # Track the currently opened listbox

            # Update the current ROI and top label to the selected ROI
            self.current_roi = roi_id
            self.current_roi_label.config(text=f"Current ROI: {self.current_roi}")

   
    def update_sidebar(self):
        # Remember the scroll position before updating the sidebar
        self.current_scroll_position = self.sidebar_canvas.yview()

        # Destroy the old sidebar content
        for widget in self.sidebar_content.winfo_children():
            widget.destroy()

        rois_for_z = self.rois_per_z.get(self.current_z, {})
        for roi_id, clusters in rois_for_z.items():
            color = self.roi_colours.get(roi_id, 'lightgray')

            # Create a fixed-width frame for consistency
            frame = tk.Frame(self.sidebar_content, width=200)  # Fixed width for consistency
            frame.pack(fill=tk.X, pady=2)

            # ROI Toggle button and cluster list for each ROI
            toggle_button = tk.Button(frame, text=f"ROI {roi_id}", bg=color, command=lambda roi=roi_id, f=frame: self.toggle_roi_list(f, roi), width=25)  # Fixed width
            toggle_button.pack(fill=tk.X)

            # Listbox for showing clusters associated with the ROI (initially hidden)
            listbox = tk.Listbox(frame, height=1)  # Default height 1, updated later
            for cl_id in clusters:
                listbox.insert(tk.END, f"Cluster {cl_id}")

            listbox.pack(fill=tk.X, padx=5, pady=5)
            listbox.bind('<<ListboxSelect>>', lambda event, roi=roi_id: self.on_cluster_select(event, roi))
            listbox.pack_forget()  # Start hidden

            # Reopen the listbox if the ROI is the current one
            if self.current_roi == roi_id:
                listbox.config(height=min(len(clusters), 10))  # Cap at height 10
                listbox.pack(fill=tk.X, padx=5, pady=5)

        # Restore the previous scroll position
        self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))
        self.sidebar_canvas.yview_moveto(self.current_scroll_position[0])

    def toggle_roi_visibility(self, roi_id):
        pass  # Handle ROI visibility toggling here

    def on_cluster_select(self, event, roi_id):
        listbox = event.widget
        selection = listbox.curselection()
        if not selection:
            return

        selected_cluster = listbox.get(selection[0]).replace("Cluster ", "")
        selected_cluster_id = int(selected_cluster)

        clusters = self.data[(self.data['ROI_ID'] == selected_cluster_id) & (self.data['z'] == self.current_z)][['x', 'y']].values
        if clusters.size > 0:
            x_min, x_max = clusters[:, 0].min(), clusters[:, 0].max()
            y_min, y_max = clusters[:, 1].min(), clusters[:, 1].max()

            self.stored_xlim = (x_min - 100, x_max + 100)  # Zoom to [100, 100] dimensions
            self.stored_ylim = (y_max + 100, y_min - 100)

            self.ax.set_xlim(self.stored_xlim)
            self.ax.set_ylim(self.stored_ylim)
            self.canvas.draw_idle()

        # Update selected cluster and set the ROI as current
        self.selected_cluster_id = selected_cluster_id
        self.current_roi = roi_id
        self.current_roi_label.config(text=f"Current ROI: {self.current_roi}")

    def on_slider_release(self, event):
        """Update brightness when the slider is released."""
        self.brightness_factor = self.brightness_slider.get()
        self.image_updated = True
        for cl_id in self.cluster_modified[self.current_z]:
                    self.cluster_modified[self.current_z][cl_id] = True
        self.update_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = VolumeSegmentationApp(root)
    root.mainloop()
