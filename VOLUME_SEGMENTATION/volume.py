import numpy as np

class Volume:
    def __init__(self, id, track_avg_area = False):
        self.id = id
        self.xmin = np.inf
        self.ymin = np.inf
        self.zmin = np.inf
        self.xmax = -np.inf
        self.ymax = -np.inf
        self.zmax = -np.inf
        self.xz_rois = {}
        self.xy_rois = {}
        self.xy_rois_per_z = {}
        self.track_avg_area = track_avg_area
        self.roi_areas = {}
        self.running_average_area = []

    # Add an XZ ROI to this volume
    def add_xz_roi(self, xz_id, xz_roi):
        self.xz_rois[xz_id] = xz_roi

        # Update bounds of volume
        self.xmin = min(self.xmin, xz_roi.xmin)
        self.xmax = max(self.xmax, xz_roi.xmax)

        self.ymin = min(self.ymin, xz_roi.ymin)
        self.ymax = max(self.ymax, xz_roi.ymax)

        self.zmin = min(self.zmin, xz_roi.zmin)
        self.zmax = max(self.zmax, xz_roi.zmax)
    
    # Add an XY ROI to this volume
    def add_xy_roi(self, xy_id, xy_roi):
        self.xy_rois[xy_id] = xy_roi
        
        # Update bounds of volume
        self.xmin = min(self.xmin, xy_roi.xmin)
        self.xmax = max(self.xmax, xy_roi.xmax)

        self.ymin = min(self.ymin, xy_roi.ymin)
        self.ymax = max(self.ymax, xy_roi.ymax)

        self.zmin = min(self.zmin, xy_roi.zmin)
        self.zmax = max(self.zmax, xy_roi.zmax)

        # Cache rois on each z level
        if not xy_roi.zmin in self.xy_rois_per_z:
            self.xy_rois_per_z[xy_roi.zmin] = []
        self.xy_rois_per_z[xy_roi.zmin].append((xy_id, xy_roi))

        if self.track_avg_area:
            # Calc volume for this xy roi
            self.roi_areas[xy_id] = xy_roi.get_area()
             # Update the running average
            self.update_running_average(self.roi_areas[xy_id])

    def get_roi_areas(self):
        if not self.track_avg_area:
            self.track_avg_area = True
            # Calculate volumes and running average
            for z, roi_list in self.xy_rois_per_z.items():
                for roi_id, roi in roi_list:
                    self.roi_areas[roi_id] = roi.get_area()
                    self.update_running_average(self.roi_areas[roi_id])
        return self.roi_areas, self.running_average_area

    # Calculates the volume of an xy roi
    def calculate_roi_volume(self, roi):
        current_z = roi.zmin
        previous_z = None
        # This approach assumes xys are added in ascending z - which they are, but could be a pitfall in future
        for z in sorted(self.xy_rois_per_z.keys()):
            if z < current_z:
                previous_z = z
            else:
                break
        current_area = roi.get_area()
        if previous_z is not None:
            z_gap = current_z - previous_z
            # Volume of the current ROI
            return current_area * z_gap
        else:
            return current_area # No previous z, assume height of 1

    def update_running_average(self, new_area):
        # Calculate new running average
        if len(self.running_average_area) == 0:
            # First ROI, the average is just the first volume
            self.running_average_area.append(new_area)
        else:
            # Compute new average based on previous value
            total_xy_rois = len(self.xy_rois)
            previous_avg = self.running_average_area[-1]
            new_avg = ((previous_avg * (total_xy_rois - 1)) + new_area) / total_xy_rois
            self.running_average_area.append(new_avg)

    # Get the XZ ROI dict
    def get_xz_rois(self):
        return self.xz_rois

    # Get the XY ROI dict
    def get_xy_rois(self):
        return self.xy_rois    
    
    # Merge another volume into this volume
    def volume_merge(self, other_volume):
        other_xz_rois = other_volume.get_xz_rois()
        other_xy_rois = other_volume.get_xy_rois()

        self.xz_rois.update(other_xz_rois)
        self.xy_rois.update(other_xy_rois)