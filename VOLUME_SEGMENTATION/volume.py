import numpy as np

class Volume:
    def __init__(self, id):
        self.id = id
        self.xmin = np.inf
        self.ymin = np.inf
        self.zmin = np.inf
        self.xmax = -np.inf
        self.ymax = -np.inf
        self.zmax = -np.inf
        self.xz_rois = {}
        self.xy_rois = {}
    
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