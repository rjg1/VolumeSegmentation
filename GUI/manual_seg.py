import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
from PIL import Image, ImageTk
import tifffile as tif
import numpy as np
import argparse
import colorsys
import json
import os
import random

POINT_PRECISION = 2
EYE_OPEN_IMG = "eye_open.png"
EYE_CLOSED_IMG = "eye_closed.png"
UP_BUTTON_IMG = "up_button.png"
DOWN_BUTTON_IMG = "down_button.png"


class VolumeSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Volume Segmentation GUI")
        random.seed(5)

        # Bind window close event to `on_closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Variables
        self.csv_path = ""
        self.image_folder = ""
        self.data = None
        self.current_z = 0
        self.volumes = {}  # Dictionary of volumes, each volume is a set of (ROI_ID, Z-level)
        self.image_stack = None
        self.current_volume = None
        self.selected_rois = {}  # Tracks ROIs associated with volumes
        self.roi_modified = {} # Tracks whether an ROI has been updated
        self.zoom_enabled = False
        self.image_updated = False
        self.zoom_selector = None
        self.volumes_changed = False
        self.stored_xlim = None  # To store x-axis limits for persistent zoom
        self.stored_ylim = None  # To store y-axis limits for persistent zoom
        self.volume_visibility = {}  # To track visibility of each volume
        self.roi_to_volume = {}  # Track ROI to volume assignments
        self.volume_colours = {}  # Track colours per volume
        self.scroll_manually_adjusted = False  # Track manual scrollbar adjustment
        self.current_scroll_position = (0, 0)  # Initialize scroll position tracker
        self.volume_elements = {}  # Track volume elements in sidebar
        self.brightness_factor = 1.0  # Initialize brightness factor
        self.plotted_rois = {} # Dictionary of ROIs in the plot
        
        # Load parameters if passed
        self.parser = argparse.ArgumentParser(description="Set parameters for cell merger.")
        self.parser.add_argument('--cell_data_path', help='Path to the cell data csv')
        self.parser.add_argument('--tif_file_path', help='Path to the 3D .tif image file')
        self.parser.add_argument('--scenarios_path', help = "Path to scenarios json file")
        self.parser.add_argument('--active_scenario', help = "Name of the currently active scenario")
        self.parser.add_argument('--autosave_exit_path', help='Path to export Validation csv to on exit')
        self.parser.add_argument('--autosave_name', help='Name of output csv')
        self.parser.add_argument('--roi_exit_path', help='Path to export ROI csv to on exit')
        self.parser.add_argument('--up_img_path', help='Path to up button image')
        self.parser.add_argument('--down_img_path', help='Path to down button image')
        self.parser.add_argument('--eye_open_img_path', help='Path to eye open image')
        self.parser.add_argument('--eye_closed_img_path', help='Path to eye closed image')
        # Parse command-line arguments
        self.input_args = self.parser.parse_args()

        up_button_path = self.input_args.up_img_path or UP_BUTTON_IMG
        down_button_path = self.input_args.down_img_path or DOWN_BUTTON_IMG
        eye_open_path = self.input_args.eye_open_img_path or EYE_OPEN_IMG
        eye_closed_path = self.input_args.eye_closed_img_path or EYE_CLOSED_IMG

        # Icons for visibility toggle
        self.eye_open_img = ImageTk.PhotoImage(Image.open(eye_open_path).resize((15, 15))) 
        self.eye_closed_img = ImageTk.PhotoImage(Image.open(eye_closed_path).resize((15, 15))) 

        # Icons for z navigation buttons
        self.up_button_img = ImageTk.PhotoImage(Image.open(up_button_path).resize((20, 20)))  
        self.down_button_img = ImageTk.PhotoImage(Image.open(down_button_path).resize((20, 20))) 
        
        # Setup GUI
        self.setup_gui()

        # Load cell data and tif file if specified:
        if self.input_args.cell_data_path and self.input_args.tif_file_path:

            with open(self.input_args.scenarios_path, 'r') as file:
                scenarios = json.load(file)
            if scenarios[self.input_args.active_scenario]["HAS_VALIDATION"]:
                # Importing past manual segmentation
                self.import_segmentation(path = self.input_args.cell_data_path)
            else:
                # Starting new manual segmentation
                print("Starting new manual volume segmentation")
                self.load_csv(path = self.input_args.cell_data_path)
            self.load_tif_file(path = self.input_args.tif_file_path)

        # Bind hotkeys
        self.root.bind("q", lambda event: self.prev_z())
        self.root.bind("w", lambda event: self.next_z())
        self.root.bind("z", lambda event: self.toggle_zoom())
        self.root.bind("n", lambda event: self.create_new_volume())
        self.root.bind("r", lambda event: self.reset_zoom())

    def setup_gui(self):
        # Create the menu bar
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        # Create the File menu
        if not self.input_args.cell_data_path and not self.input_args.tif_file_path:
            file_menu = tk.Menu(menu_bar, tearoff=0)
            menu_bar.add_cascade(label="File", menu=file_menu)
            file_menu.add_command(label="Open CSV", command=self.load_csv)
            file_menu.add_command(label="Open TIF", command=self.load_tif_file)

        if not self.input_args.autosave_exit_path:
            # Create the Import menu
            import_menu = tk.Menu(menu_bar, tearoff=0)
            menu_bar.add_cascade(label="Import", menu=import_menu)
            import_menu.add_command(label="Import Segmentation", command=self.import_segmentation)

            # Create the Export menu
            export_menu = tk.Menu(menu_bar, tearoff=0)
            menu_bar.add_cascade(label="Export", menu=export_menu)
            export_menu.add_command(label="Export CSV", command=self.export_csv)

        # Frame for controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Navigation buttons
        self.prev_button = tk.Button(control_frame, image=self.down_button_img, command=self.prev_z)
        self.prev_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.next_button = tk.Button(control_frame, image=self.up_button_img, command=self.next_z)
        self.next_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Entry to jump to a specific z-plane
        self.z_entry = tk.Entry(control_frame, width=5)
        self.z_entry.pack(side=tk.LEFT, padx=5)
        self.z_entry.bind("<Return>", self.jump_to_z)  # Jump when Enter is pressed

        # Label to display current z-plane and total planes
        self.z_label = tk.Label(control_frame, text="Z: 0/0")
        self.z_label.pack(side=tk.LEFT, padx=5, pady=5)

        # Zoom button with toggle effect
        self.zoom_button = tk.Button(control_frame, text="Zoom (z)", command=self.toggle_zoom, bg='lightgray')
        self.zoom_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Reset zoom button
        self.reset_zoom_button = tk.Button(control_frame, text="Reset Zoom (r)", command=self.reset_zoom)
        self.reset_zoom_button.pack(side=tk.LEFT, padx=5, pady=5)

        # New Volume button moved to control frame
        self.new_volume_button = tk.Button(control_frame, text="New Volume (n)", command=self.create_new_volume)
        self.new_volume_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Sidebar for volume management with scrollbar
        sidebar_frame = tk.Frame(self.root, width=200)
        sidebar_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.sidebar_canvas = tk.Canvas(sidebar_frame, highlightthickness=0, width=200)
        self.sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.sidebar_scrollbar = ttk.Scrollbar(sidebar_frame, orient="vertical", command=self.sidebar_scroll_event)
        self.sidebar_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.sidebar_content = tk.Frame(self.sidebar_canvas, width=200)
        self.sidebar_content.bind("<Configure>", self.configure_scroll_region)

        # Create a window in the canvas for sidebar content
        self.canvas_window = self.sidebar_canvas.create_window((0, 0), window=self.sidebar_content, anchor="nw")
        self.sidebar_canvas.config(yscrollcommand=self.sidebar_scrollbar.set)

        # Canvas for image display
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Create a frame for the brightness slider
        slider_frame = tk.Frame(self.root)
        slider_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Brightness slider
        self.brightness_slider = tk.Scale(slider_frame, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, 
                                        label="Brightness")
        self.brightness_slider.set(1.0)  # Set default brightness to 1 (no change)
        self.brightness_slider.pack(fill=tk.X, padx=5, pady=5)

        # Bind slider to run the update_brightness function when the mouse is released
        self.brightness_slider.bind("<ButtonRelease-1>", self.on_slider_release)


    def on_slider_release(self, event):
        # Update brightness only when the slider is released
        self.brightness_factor = float(self.brightness_slider.get())
        self.image_updated = True
        if self.roi_modified.get(self.current_z, None):
            for roi_id in self.roi_modified[self.current_z]:
                self.roi_modified[self.current_z][roi_id] = True

        self.update_display()  # Redraw the image with the new brightness


    def load_csv(self, path = None):
        # Load CSV file
        if not path:
            self.csv_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        else:
            self.csv_path = path
        if self.csv_path:
            self.data = pd.read_csv(self.csv_path)
            self.current_z = self.data['z'].min()

            for z_level in self.data['z'].unique():
                self.roi_modified[z_level] = {}
                roi_ids = self.data[self.data['z'] == z_level]['ROI_ID'].unique()
                for roi_id in roi_ids:
                    self.roi_modified[z_level][roi_id] = True
            self.update_display()

    def configure_scroll_region(self, event=None):
        """Update the scroll region and keep the current scroll position."""
        self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))
        self.sidebar_canvas.yview_moveto(self.current_scroll_position[0])

    def sidebar_scroll_event(self, *args):
        """Track manual scrollbar adjustments."""
        self.scroll_manually_adjusted = True
        self.sidebar_canvas.yview(*args)

    def load_tif_file(self, path = None):
        # Load .tif file
        if not path:
            self.tif_path = filedialog.askopenfilename(filetypes=[("TIF Files", "*.tif"), ("TIFF Files", "*.tiff")])
        else:
            self.tif_path = path
        if self.tif_path:
            # Reset zoom settings
            if self.image_stack is not None and self.image_stack.size > 0:
                self.reset_zoom()
                self.zoom_enabled = False
                self.zoom_button.config(bg='lightgray')
                if self.zoom_selector:
                    self.zoom_selector.set_active(False)

            self.image_updated = True
            self.image_stack = tif.imread(self.tif_path)
            self.current_z = 0  # Start at the first z-plane
            self.update_display()


    def update_display(self, z_changed = False):
        if self.image_stack is not None and self.data is not None:
            # Get the current z-plane image
            if 0 <= self.current_z < self.image_stack.shape[0]:
                if self.image_updated or z_changed:
                    img = self.image_stack[self.current_z]
                    # Apply brightness adjustment
                    img = np.clip(img * self.brightness_factor, 0, 255).astype(np.uint8)
                    self.plotted_rois = {}
                    self.ax.clear()
                    self.ax.imshow(img)
                    self.image_updated = False

                # Overlay ROIs with color coding based on volume association and visibility
                rois = self.data[self.data['z'] == self.current_z]
                associated_rois = {roi for volume in self.volumes.values() for roi in volume}

                # Plot ROIs without a volume
                for roi_id in rois['ROI_ID'].unique():
                    if (roi_id, self.current_z) not in associated_rois and self.roi_modified[self.current_z][roi_id]:  # Plot only if not already plotted
                         # Remove existing plot value if it exists
                        if (roi_id, self.current_z) in self.plotted_rois: 
                            self.plotted_rois[(roi_id, self.current_z)].remove()  # Remove the plot line
                            del self.plotted_rois[(roi_id, self.current_z)]  # Delete the reference
                        roi_points = rois[rois['ROI_ID'] == roi_id][['x', 'y']].values
                        color = 'white'
                        line, = self.ax.plot(roi_points[:, 0], roi_points[:, 1], label=f'ROI {roi_id}', color=color)
                        self.plotted_rois[(roi_id, self.current_z)] = line
                        self.roi_modified[self.current_z][roi_id] = False
                # Plot ROIs with a volume
                for volume_id, rois_set in self.volumes.items():
                    visible = self.volume_visibility.get(volume_id, True)
                    for roi_id, z_level in sorted(rois_set, key=lambda x: x[1]):
                        if z_level == self.current_z:
                            if self.roi_modified[self.current_z][roi_id] and visible:  # Plot only if modified
                                # Remove existing plot value if it exists
                                if (roi_id, self.current_z) in self.plotted_rois: 
                                    # print(f"Removing previous ref to: {roi_id}")
                                    self.plotted_rois[(roi_id, self.current_z)].remove()  # Remove the plot line
                                    del self.plotted_rois[(roi_id, self.current_z)]  # Delete the reference
                                # print(f"Drawing roi: {roi_id}")
                                roi_points = rois[rois['ROI_ID'] == roi_id][['x', 'y']].values
                                vol_colour = self.volume_colours.get(volume_id, 'lightgray')
                                # Store new value
                                line, = self.ax.plot(roi_points[:, 0], roi_points[:, 1], label=f'ROI {roi_id}', color=vol_colour)
                                self.plotted_rois[(roi_id, self.current_z)] = line 
                                self.roi_modified[self.current_z][roi_id] = False # Don't replot until next update
                            elif not visible and (roi_id, self.current_z) in self.plotted_rois:
                                # print(f"Turning off roi: {roi_id}")
                                self.plotted_rois[(roi_id, self.current_z)].remove()  # Remove the plot line
                                del self.plotted_rois[(roi_id, self.current_z)]  # Delete the reference
                # Reapply the stored zoom limits if available
                if self.stored_xlim and self.stored_ylim:
                    self.ax.set_xlim(self.stored_xlim)
                    self.ax.set_ylim(self.stored_ylim)

                if not self.ax.yaxis_inverted():
                    self.ax.invert_yaxis()

                self.canvas.draw()

            # Update the entry box and label with the current Z-level (0-indexed)
            self.z_entry.delete(0, tk.END)
            self.z_entry.insert(0, str(self.current_z))
            self.z_label.config(text=f"Z: {self.current_z}/{self.image_stack.shape[0] - 1}")

    def prev_z(self):
        if self.image_stack is not None:
            old_current_z = self.current_z
            self.current_z = max(0, self.current_z - 1)
            if self.current_z != old_current_z:
                # Re-draw rois when this plane is re-visited
                if self.roi_modified.get(old_current_z, None):
                    for roi_id in self.roi_modified[old_current_z]:
                        self.roi_modified[old_current_z][roi_id] = True

                self.update_display(z_changed = True)

    def next_z(self):
        if self.image_stack is not None:
            old_current_z = self.current_z
            self.current_z = min(self.image_stack.shape[0] - 1, self.current_z + 1)
            if self.current_z != old_current_z:
                # Re-draw rois when this plane is re-visited
                if self.roi_modified.get(old_current_z, None):
                    for roi_id in self.roi_modified[old_current_z]:
                        self.roi_modified[old_current_z][roi_id] = True

                self.update_display(z_changed = True)

    def on_roi_select(self, event, volume_id):
        listbox = event.widget
        selection = listbox.curselection()
        if not selection:
            return

        selected_roi = listbox.get(selection[0])
        roi_id, z_level = map(int, selected_roi.replace("ROI ", "").split(", Z: "))

        old_z = self.current_z
        self.current_z = z_level
        roi_points = self.data[(self.data['ROI_ID'] == roi_id) & (self.data['z'] == z_level)][['x', 'y']].values
        if roi_points.size > 0:
                x_min, x_max = roi_points[:, 0].min(), roi_points[:, 0].max()
                y_min, y_max = roi_points[:, 1].min(), roi_points[:, 1].max()

                self.stored_xlim = (x_min - 100, x_max + 100)
                self.stored_ylim = (y_max + 100, y_min - 100)

                self.ax.set_xlim(self.stored_xlim)
                self.ax.set_ylim(self.stored_ylim)
                self.canvas.draw_idle()
        if old_z != self.current_z:
            if self.roi_modified.get(old_z, None):
                for roi_id in self.roi_modified[old_z]:
                    self.roi_modified[old_z][roi_id] = True
            self.update_display(z_changed=True)
        else:
            self.update_display()
            

    def on_closing(self):
        # Volumes exist, perform export
        if len(self.volumes) > 0 and self.volumes_changed and self.input_args.autosave_exit_path:
            result = self.export_csv(export_path=self.input_args.autosave_exit_path, roi_export_path=self.input_args.roi_exit_path)
            # Get scenarios dict
            with open(self.input_args.scenarios_path, 'r') as file:
                scenarios = json.load(file)
            # Update path to validation file
            scenarios[self.input_args.active_scenario]["V_GT_CSV"] = self.input_args.autosave_name
            scenarios[self.input_args.active_scenario]["HAS_VALIDATION"] = True
            if self.input_args.roi_exit_path: # Generate custom csv for this scenario
                scenarios[self.input_args.active_scenario]["SEG_IN_FILE"] = self.input_args.roi_exit_path
            # Determine if any ROIs are not in a volume
            if result:
                print("dataset is not reduced")
                scenarios[self.input_args.active_scenario]["HAS_REDUCED_DATASET"] = False
            else:
                print("dataset is reduced")
                scenarios[self.input_args.active_scenario]["HAS_REDUCED_DATASET"] = True
            # Export result
            with open(self.input_args.scenarios_path, 'w') as file:
                json.dump(scenarios, file, indent=4)

        self.root.quit()
        self.root.destroy()

    def jump_to_z(self, event):
        if self.image_stack is not None:
            try:
                z_value = int(self.z_entry.get())
                if 0 <= z_value < self.image_stack.shape[0]:
                    # Re-draw rois when this plane is re-visited
                    if self.roi_modified.get(self.current_z, None):
                        for roi_id in self.roi_modified[self.current_z]:
                            self.roi_modified[self.current_z][roi_id] = True
                    self.current_z = z_value
                    self.update_display(z_changed = True)
                else:
                    tk.messagebox.showerror("Invalid Z-Plane", "Z-plane out of range.")
            except ValueError:
                tk.messagebox.showerror("Invalid Input", "Please enter a valid integer.")

    def toggle_zoom(self):
        self.zoom_enabled = not self.zoom_enabled
        self.zoom_button.config(bg='lightgreen' if self.zoom_enabled else 'lightgray')
        if self.zoom_enabled:
            self.enable_zoom()
            self.deactivate_current_volume()  # Deactivate any selected volume
        else:
            if self.zoom_selector:
                self.zoom_selector.set_active(False)
        self.update_volume_sidebar(preserve_scroll=True)

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
            self.toggle_zoom()
        

    def reset_zoom(self):
        # Check if the image stack is loaded and has a valid shape
        if self.image_stack is not None and self.image_stack.size > 0:
            # Set axis limits based on the image stack dimensions
            self.ax.set_xlim(0, self.image_stack.shape[2])
            self.ax.set_ylim(self.image_stack.shape[1], 0)
        else:
            # Set axis limits to default [0, 1] if no image stack is loaded
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)

        # Clear stored limits since zoom is reset
        self.stored_xlim = None
        self.stored_ylim = None
        self.canvas.draw_idle()

    def generate_volume_color(self, volume_id):
        hue = random.random()
        lightness = 0.6
        saturation = 0.9
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

    def create_new_volume(self):
        volume_id = f"Volume {len(self.volumes) + 1}"
        self.volumes[volume_id] = set()
        self.volume_visibility[volume_id] = True
        self.deactivate_current_volume()  # Deactivate the previous volume if any
        self.current_volume = volume_id

        self.volume_colours[volume_id] = self.generate_volume_color(volume_id)

        if self.zoom_enabled:
            self.zoom_enabled = False
            self.zoom_button.config(bg='lightgray')
            if self.zoom_selector:
                self.zoom_selector.set_active(False)

        self.update_volume_sidebar(preserve_scroll=True)
        self.sidebar_canvas.yview_moveto(self.current_scroll_position[0])

    def deactivate_current_volume(self):
        if self.current_volume:
            current_elements = self.volume_elements.get(self.current_volume, [])
            if current_elements:
                listbox, stats_label = current_elements[3], current_elements[4]
                listbox.pack_forget()
                stats_label.pack_forget()
            self.current_volume = None

    def update_volume_sidebar(self, preserve_scroll=True):
        """Update the sidebar with volume information, maintaining scroll position if specified."""
        # Preserve the current scroll position of the sidebar
        self.current_scroll_position = self.sidebar_canvas.yview()

        # Remove elements of volumes no longer in the current data
        for volume_id, elements in list(self.volume_elements.items()):
            if volume_id not in self.volumes:
                # Destroy widgets associated with removed volumes
                for widget in elements:
                    widget.destroy()
                del self.volume_elements[volume_id]

        # Add new volumes to the sidebar if they do not already exist
        for volume_id, rois_set in self.volumes.items():
            if volume_id not in self.volume_elements:
                frame = tk.Frame(self.sidebar_content, bd=1, relief=tk.RAISED, width=200)
                frame.pack(fill=tk.X, pady=2)

                header = tk.Frame(frame, width=200)
                header.pack(fill=tk.X)

                toggle_btn = tk.Button(
                    header,
                    image=self.eye_open_img if self.volume_visibility[volume_id] else self.eye_closed_img,
                    command=lambda v=volume_id: self.toggle_visibility(v)
                )
                toggle_btn.pack(side=tk.LEFT)

                vol_color = self.volume_colours.get(volume_id, 'lightgray')
                label_font = ("Arial", 10, "bold") if self.current_volume == volume_id else ("Arial", 10)
                font_color = "white" if self.current_volume != volume_id else "black"
                vol_label = tk.Label(
                    header,
                    text=volume_id,
                    anchor='w',
                    bg=vol_color if self.current_volume == volume_id else self.fade_color(vol_color),
                    fg=font_color,
                    font=label_font,
                    width=200
                )
                vol_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
                vol_label.bind("<Button-1>", lambda e, v=volume_id: self.select_volume(v))

                # Listbox and stats label for each volume
                roi_listbox = tk.Listbox(frame, selectmode=tk.SINGLE)
                roi_listbox.bind("<<ListboxSelect>>", lambda e, v=volume_id: self.on_roi_select(e, v))

                roi_stats_label = tk.Label(frame, text="", anchor="w")

                # Store these elements in the volume_elements dictionary
                self.volume_elements[volume_id] = [frame, toggle_btn, vol_label, roi_listbox, roi_stats_label]

            # Update existing volume elements
            elements = self.volume_elements[volume_id]
            frame, toggle_btn, vol_label, roi_listbox, roi_stats_label = elements

            toggle_btn.config(image=self.eye_open_img if self.volume_visibility[volume_id] else self.eye_closed_img)

            vol_color = self.volume_colours.get(volume_id, 'lightgray')
            label_font = ("Arial", 10, "bold") if self.current_volume == volume_id else ("Arial", 10)
            font_color = "white" if self.current_volume != volume_id else "black"
            vol_label.config(
                bg=vol_color if self.current_volume == volume_id else self.fade_color(vol_color),
                fg=font_color,
                font=label_font
            )

            # Manage listboxes and stats labels based on volume activation
            if self.current_volume == volume_id:
                # Dynamically set listbox height based on the number of ROIs (min height 1, max height 100)
                sorted_rois = sorted(rois_set, key=lambda x: x[1])
                listbox_height = min(max(len(sorted_rois), 1), 100)

                roi_listbox.delete(0, tk.END)  # Clear existing entries
                for roi_id, z_level in sorted_rois:
                    roi_listbox.insert(tk.END, f"ROI {roi_id}, Z: {z_level}")

                z_levels = [z_level for _, z_level in sorted_rois]
                z_min = min(z_levels) if z_levels else 0
                z_max = max(z_levels) if z_levels else 0

                roi_listbox.config(height=listbox_height)
                roi_listbox.pack(fill=tk.X, pady=2)
                
                # Ensure the stats label is correctly updated and shown
                roi_stats_label.config(text=f"Total ROIs: {len(sorted_rois)} | Z Range: [{z_min}, {z_max}]")
                roi_stats_label.pack(fill=tk.X)  # Ensure the stats label is packed correctly
            else:
                # Hide listboxes and stats if no volume is selected
                roi_listbox.pack_forget()
                roi_stats_label.pack_forget()

        # Adjust the scroll region and restore the scroll position
        self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))
        if preserve_scroll and self.scroll_manually_adjusted:
            self.sidebar_canvas.yview_moveto(self.current_scroll_position[0])


    def fade_color(self, color):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        faded_rgb = (int(r * 0.7), int(g * 0.7), int(b * 0.7))
        return "#{:02x}{:02x}{:02x}".format(*faded_rgb)

    def toggle_visibility(self, volume_id):
        self.volume_visibility[volume_id] = not self.volume_visibility[volume_id]
        if self.roi_modified.get(self.current_z, None):
            for roi_id in self.roi_modified[self.current_z]:
                self.roi_modified[self.current_z][roi_id] = True # Redraw when visible
        self.update_volume_sidebar(preserve_scroll=True)
        self.update_display()

    def select_volume(self, volume_id):
        prev_volume = self.current_volume
        self.deactivate_current_volume()  # Deactivate the currently selected volume
        if not prev_volume == volume_id:
            self.current_volume = volume_id
            self.zoom_enabled = False
            self.zoom_button.config(bg='lightgray')
            if self.zoom_selector:
                self.zoom_selector.set_active(False)
        self.update_volume_sidebar(preserve_scroll=True)
        self.update_display()

    def on_click(self, event):
        # Detect clicks on ROI points and associate/disassociate with the current volume
        if self.data is None or self.image_stack is None:
            print("No data loaded, ignoring click")
            return

        if event.inaxes == self.ax:
            x_click, y_click = event.xdata, event.ydata
            rois = self.data[self.data['z'] == self.current_z]

            for roi_id in rois['ROI_ID'].unique():
                roi_points = rois[rois['ROI_ID'] == roi_id][['x', 'y']].values
                distances = np.sqrt((roi_points[:, 0] - x_click) ** 2 + (roi_points[:, 1] - y_click) ** 2)
                existing_rois = self.volumes.get(self.current_volume, set())
                if np.any(distances < POINT_PRECISION):  # Tolerance for point selection
                    roi = (roi_id, self.current_z)
                    associated_volume = self.roi_to_volume.get(roi)
                    if associated_volume == None and self.current_volume == None: ## TODO test
                        continue
                    if associated_volume == self.current_volume:
                        # Disassociate the ROI
                        self.volumes[associated_volume].remove(roi)
                        del self.roi_to_volume[roi]
                        print(f"ROI {roi_id} disassociated from {associated_volume}")
                        self.roi_modified[self.current_z][roi_id] = True
                        self.volumes_changed = True
                        # tk.messagebox.showinfo("Info", f"ROI {roi_id} disassociated from {associated_volume}.")
                    elif roi in existing_rois:
                        print(f"ROI is already in {self.current_volume}")
                    elif associated_volume:
                        print(f"ROI is associated with a different volume than that which is currently selected")
                    # elif any(roi[1] == z for r, z in existing_rois):
                    #     print(f"Only one ROI allowed per z-level for each volume")
                    # elif not associated_volume and existing_rois and (self.current_z - 1) not in [z for _, z in existing_rois] and (self.current_z + 1) not in [z for _, z in existing_rois]:
                    #     print("Requires a ROI in an adjacent z-plane to add to this volume")
                    elif not associated_volume:
                        # Associate the ROI with the current volume
                        self.volumes[self.current_volume].add(roi)
                        self.roi_to_volume[roi] = self.current_volume
                        self.roi_modified[self.current_z][roi_id] = True
                        self.volumes_changed = True
                        # tk.messagebox.showinfo("Info", f"ROI {roi_id} associated with {self.current_volume}.")

                    # Update display and sidebar
                    self.update_display()
                    self.update_volume_sidebar()
                    break
    
    def import_segmentation(self, path = None):
        """Import a segmentation CSV file and update the volume dictionary."""
        segmentation_path = path or filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if segmentation_path:
            # Read the CSV file containing segmentation data
            print(f"Importing from : {segmentation_path}")
            segmentation_data = pd.read_csv(segmentation_path)
            # Check if the CSV has the required columns
            required_columns = {'x', 'y', 'z', 'ROI_ID', 'VOLUME_ID'}
            if not required_columns.issubset(segmentation_data.columns):
                tk.messagebox.showerror("Error", "The segmentation file is missing required columns.")
                return
            # Clear existing sidebar content before loading new data
            for volume_id, elements in list(self.volume_elements.items()):
                for widget in elements:
                    widget.destroy()
            self.volume_elements.clear()

            # Reset data structures to avoid conflicts with previous data
            self.volumes = {}
            self.plotted_rois = {}
            self.image_updated = True
            self.roi_modified = {}
            self.roi_to_volume = {}
            self.volume_visibility = {}
            self.volume_colours = {}
            self.selected_rois = {}
            random.seed(5) # Re-seed to ensure consistent colouring
            # Group by VOLUME_ID to reconstruct the volumes and associated data
            for volume_id, group in segmentation_data.groupby('VOLUME_ID'):
                if volume_id == -1:
                    continue  # Skip ROIs that are not associated with any volume

                # Create a volume name and initialize its data

                volume_name = f"Volume {int(volume_id)}"
                self.volumes[volume_name] = set()
                self.volume_visibility[volume_name] = True
                self.volume_colours[volume_name] = self.generate_volume_color(volume_name)

                # Track each ROI and associate it with the volume
                for _, row in group.iterrows():
                    roi = (int(row['ROI_ID']), int(row['z']))
                    self.volumes[volume_name].add(roi)
                    self.roi_to_volume[roi] = volume_name

            # Reset selected ROIs
            self.selected_rois = {volume_name: sorted(self.volumes[volume_name]) for volume_name in self.volumes}

            # Load the original data without the VOLUME_ID column to maintain visualization
            self.data = segmentation_data.drop(columns=['VOLUME_ID'])
            self.current_z = self.data['z'].min()  # Set current Z to the lowest value in the data

            # Reset roi modified dict
            for z_level in self.data['z'].unique():
                self.roi_modified[z_level] = {}
                roi_ids = self.data[self.data['z'] == z_level]['ROI_ID'].unique()
                for roi_id in roi_ids:
                    self.roi_modified[z_level][roi_id] = True

            # Reset zoom settings
            if self.image_stack is not None and self.image_stack.size > 0:
                self.reset_zoom()
                self.zoom_enabled = False
                self.zoom_button.config(bg='lightgray')
                if self.zoom_selector:
                    self.zoom_selector.set_active(False)

            # Deselect any currently selected volume and update the sidebar and display
            self.current_volume = None
            self.update_volume_sidebar()
            self.update_display()
            if not path:
                tk.messagebox.showinfo("Import Complete", "Segmentation data imported successfully.")
        else:
            tk.messagebox.showerror("Error", "No file selected.")


    def export_csv(self, export_path = None, roi_export_path = None):
        # Export the modified CSV with numerical VOLUME_IDs
        if self.data is not None:
            # Create a mapping of volume names to numerical IDs
            volume_name_to_id = {volume_name: idx + 1 for idx, volume_name in enumerate(self.volumes.keys())}

            # Create a copy of the original data and add the "VOLUME_ID" column with numerical values
            export_data = self.data.copy()
            export_data['VOLUME_ID'] = export_data.apply(
                lambda row: volume_name_to_id.get(self.roi_to_volume.get((row['ROI_ID'], row['z']), ''), -1), axis=1
            )
            if not export_path:
                # Prompt the user to select the save location for the CSV
                export_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                    filetypes=[("CSV Files", "*.csv")])
            if export_path:
                print(f"Exporting to: {export_path}")
                export_data.to_csv(export_path, index=False)
                print("Volume Data Exported Successfully")

            # Export ROI base data if provided
            if roi_export_path:
                self.data.to_csv(roi_export_path, index=False)

            # Determine if any ROIs are not in a volume
            if (export_data['VOLUME_ID'] == -1).any():
                return 1
            else: 
                return 0
        else:
            tk.messagebox.showerror("Error", "No CSV file has been loaded.")
            return -1



if __name__ == "__main__":
    root = tk.Tk()
    app = VolumeSegmentationApp(root)
    root.mainloop()
