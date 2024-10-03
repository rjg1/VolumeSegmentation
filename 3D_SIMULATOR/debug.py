import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from picker import Picker
from shapely.geometry import Polygon
import math
import pickle
import os
from actions import ActionSet, AddSliceAction, MergeAction, ProjectionAction

ACTION_FILE = 'DEBUG_ACTIONS_s42_o7.pickle'
SEGMENTATION_FILE = 'DEBUG_SEG_OBJ_s42_o7.pickle'
PROJECTIONS_FILE = 'DEBUG_PROJECTIONS_s42_o7.pickle'
CMAP = plt.cm.get_cmap('tab10')
STARTING_ACTION = 0
NUM_ACTIONS_PER_PRESS = 1

def display_action_set(actions, segmented_objects, projections):
    def callback(event=None):  # Modify callback function signature
        nonlocal current_index
        if current_index < len(actions.actions):
            # Update the plot based on the current action
            update_plot(actions.actions[current_index])
        else:
            print("No more actions.")

    def proj_callback(event = None):
        nonlocal proj_index
        nonlocal proj_plots
        nonlocal included_projections
        nonlocal unincluded_projections
        nonlocal prev_xy_slice
        nonlocal displayed_slices
        nonlocal displayed_actors
        print(f"Projection event, Index: {proj_index}")
        if proj_index < len(projections.actions):
            projection = projections.actions[proj_index]
            # Check if current xy slice changed
            changed_xy = False
            if (projection.xy_slice != prev_xy_slice):
                changed_xy = True
                if prev_xy_slice is None:
                    prev_xy_slice = projection.xy_slice
                elif prev_xy_slice in included_projections:
                    print("colouring old xy slice")
                    # remove old xy slice
                    prev_action = projections.actions[proj_index - 1]
                    actor = displayed_actors[displayed_slices.index(prev_xy_slice)]
                    plotter.remove_actor(actor)
                    # readd old xy slice recoloured
                    proj_plots[prev_action][0] = plotter.add_mesh(prev_xy_slice, color="green", opacity=0.25)
                    displayed_actors[displayed_slices.index(prev_xy_slice)] = proj_plots[prev_action][0]
                    # set new xy slice as old one
                    prev_xy_slice = projection.xy_slice
                else:
                    # Remove old xy entirely
                    prev_action = projections.actions[proj_index - 1]
                    actor = displayed_actors[displayed_slices.index(prev_xy_slice)]
                    plotter.remove_actor(actor)
                    prev_xy_slice = projection.xy_slice
            # Check if previous action existed
            if (proj_index > 0):
                print(f"Examining action at index: {proj_index - 1}")
                prev_action = projections.actions[proj_index - 1]
                if proj_plots.get(prev_action, None):
                    # xz tings
                    if not prev_action.outcome:
                        if not prev_action.xz_slice in included_projections:
                            print(f"Removing xz slice at index: {proj_index - 1}")
                            actor = proj_plots[prev_action][1] #xz slice remove
                            plotter.remove_actor(actor)
                            displayed_actors.pop(displayed_slices.index(prev_action.xz_slice))
                            displayed_slices.pop(displayed_slices.index(prev_action.xz_slice))
                        else: # colour green    
                            actor = displayed_actors[displayed_slices.index(prev_action.xz_slice)]
                            plotter.remove_actor(actor)
                            # readd old xy slice recoloured
                            proj_plots[prev_action][1] = plotter.add_mesh(prev_action.xz_slice, color="green", opacity=0.25)
                            displayed_actors[displayed_slices.index(prev_action.xz_slice)] = proj_plots[prev_action][1]
                    else: # colour green    
                        actor = displayed_actors[displayed_slices.index(prev_action.xz_slice)]
                        plotter.remove_actor(actor)
                        # readd old xy slice recoloured
                        proj_plots[prev_action][1] = plotter.add_mesh(prev_action.xz_slice, color="green", opacity=0.25)
                        displayed_actors[displayed_slices.index(prev_action.xz_slice)] = proj_plots[prev_action][1]

                    # xy tings
                    if not changed_xy:
                        if prev_action.xy_slice not in included_projections:
                            print(f"Removing xy slice at index: {proj_index - 1}")
                            actor = proj_plots[prev_action][0]  # xy slice remove
                            plotter.remove_actor(actor)
                            displayed_actors.pop(displayed_slices.index(prev_action.xy_slice))
                            displayed_slices.pop(displayed_slices.index(prev_action.xy_slice))
                        else: #recolour as green
                            actor = displayed_actors[displayed_slices.index(prev_action.xy_slice)]
                            plotter.remove_actor(actor)
                            # readd old xy slice recoloured
                            proj_plots[prev_action][0] = plotter.add_mesh(prev_action.xy_slice, color="green", opacity=0.25)
                            displayed_actors[displayed_slices.index(prev_action.xy_slice)] = proj_plots[prev_action][0]

            if isinstance(projection, ProjectionAction):
                # plot projection and outcome
                if projection.outcome:
                    if projection.xy_slice not in included_projections:
                        included_projections.append(projection.xy_slice)
                    if projection.xz_slice not in included_projections:
                        included_projections.append(projection.xz_slice)
                    
                    if projection.xy_slice in unincluded_projections:
                        unincluded_projections.pop(unincluded_projections.index(projection.xy_slice))
                    if projection.xz_slice in unincluded_projections:
                        unincluded_projections.pop(unincluded_projections.index(projection.xz_slice))
                else:
                    if projection.xy_slice not in included_projections and projection.xy_slice not in unincluded_projections:
                        unincluded_projections.append(projection.xy_slice)
                    if projection.xz_slice not in included_projections and projection.xz_slice not in unincluded_projections:
                        unincluded_projections.append(projection.xz_slice)
                
                proj_plots[projection] = []
                if (projection.xy_slice in displayed_slices): # set existing mesh colour
                    xy_actor = displayed_actors[displayed_slices.index(projection.xy_slice)]
                    plotter.remove_actor(xy_actor)
                    xy_actor = plotter.add_mesh(projection.xy_slice, color="purple", opacity=0.25)
                    displayed_actors[displayed_slices.index(projection.xy_slice)] = xy_actor
                else:
                    print("adding xy slice to mesh") # add new xy mesh
                    displayed_slices.append(projection.xy_slice) # set as displayed
                    xy_actor = plotter.add_mesh(projection.xy_slice, color="purple", opacity=0.25)
                    displayed_actors.append(xy_actor)

                if (projection.xz_slice in displayed_slices): # set existing mesh colour
                    xz_actor = displayed_actors[displayed_slices.index(projection.xz_slice)] 
                    plotter.remove_actor(xz_actor) #remove
                    xz_actor = plotter.add_mesh(projection.xz_slice, color="purple", opacity=0.25) # recolour
                    displayed_actors[displayed_slices.index(projection.xz_slice)] = xz_actor # put back
                else:
                    print("adding xz slice to mesh") # add new xz mesh
                    displayed_slices.append(projection.xz_slice) # set as displayed
                    xz_actor = plotter.add_mesh(projection.xz_slice, color="purple", opacity=0.25)
                    displayed_actors.append(xz_actor)
                
                # Add in actors for this projection
                proj_plots[projection].append(xy_actor)
                proj_plots[projection].append(xz_actor) # add mesh to proj plots

                # recolour actors

            proj_index += 1
        else:
            for slice in unincluded_projections:
                plotter.add_mesh(slice, color="red", opacity=0.25)
            print("No more projections")

    def prev_action_callback(event=None):
        nonlocal current_index
        nonlocal action_plots
        nonlocal modified_actions
        if current_index > 0:
            current_index -= 1
            # Undo current action
            print(f"Undoing action: {current_index}")
            action = actions.actions[current_index]
            if isinstance(action, AddSliceAction):
                print(f"Removing slice from {action.object}")
                plotter.remove_actor(action_plots[current_index])
            elif isinstance(action, MergeAction):
                print(f"Unmerging object: {action.discard_object} from {action.keep_object}")
                modified_actions_list = modified_actions[current_index]
                for (index, mod_action) in modified_actions_list:
                    # Remove all reworked actors
                    plotter.remove_actor(action_plots[index])
                    # Remake action list
                    mod_action.object = action.discard_object
                    actions.actions[index] = mod_action
                    action_plots[index] = plotter.add_mesh(mod_action.slice, color=CMAP(mod_action.object), opacity=0.25)
            
        else:
            print("Already at the beginning.")

    def update_plot(action):
        nonlocal current_index 
        nonlocal action_plots
        nonlocal modified_actions
        num_actions_to_process = NUM_ACTIONS_PER_PRESS if current_index + NUM_ACTIONS_PER_PRESS < len(actions.actions) else len(actions.actions) - current_index
        for _ in range(num_actions_to_process):
            # Draw the next action
            print(f"Drawing action: {current_index}")
            action = actions.actions[current_index]
            # print(type(action))
            if isinstance(action, AddSliceAction):
                print(f"Adding slice to {action.object}")
                action_plots[current_index] = plotter.add_mesh(action.slice, color=CMAP(action.object), opacity=0.25)
            elif isinstance(action, MergeAction):
                print(f"Merging object: {action.discard_object} into {action.keep_object}")
                modified_actions[current_index] = merge_action_set(actions, action.keep_object, action.discard_object, current_index)
                for (index, mod_action) in modified_actions[current_index]:
                    plotter.remove_actor(action_plots[index])
                    action_plots[index] = plotter.add_mesh(mod_action.slice, color=CMAP(mod_action.object), opacity=0.25)
            
            current_index += 1

    
    # Create PyVista plotter
    plotter = pv.Plotter()

    # Add key event to handle next action
    plotter.add_key_event("n", callback)
    plotter.add_key_event("b", prev_action_callback)
    plotter.add_key_event("p", proj_callback)

    action_plots = {}
    proj_plots = {}
    modified_actions = {}
    included_projections = []
    unincluded_projections = []
    displayed_slices = []
    displayed_actors = []
    # Display initial action
    current_index = STARTING_ACTION
    proj_index = 0
    prev_xy_slice = None
    plotter.show()


def merge_action_set(actions, keep_obj, discard_obj, current_index):
    modified_actions = []
    for index, action in enumerate(actions.actions):
        if index > current_index:
            break
        # modify all slice actions
        if isinstance(action, AddSliceAction) and action.object == discard_obj:
            action.object = keep_obj
            modified_actions.append((index, action))
    return modified_actions


def main():
    # Attempt to load a prior segmentation
    segmented_objects = None
    actions = None
    projections = None
    if os.path.exists(SEGMENTATION_FILE):
        with open(SEGMENTATION_FILE, 'rb') as f:
            segmented_objects = pickle.load(f)
        print("Loaded saved segmentation")
    else:
        print("No saved segmentation found")
        return
    
    if os.path.exists(ACTION_FILE):
        with open(ACTION_FILE, 'rb') as f:
            actions = pickle.load(f)
        print("Loaded saved action set")
    else:
        print("No saved action set found")
        return
    
    if os.path.exists(PROJECTIONS_FILE):
        with open(PROJECTIONS_FILE, 'rb') as f:
            projections = pickle.load(f)
        print("Loaded saved projections set")
    else:
        print("No saved projections set found")
        return

    # # Display the action set
    # # Action 60 is sus lets explore
    # import matplotlib.pyplot as plt
    # from shapely import Point, LineString
    # from shapely.ops import unary_union
    # projection = projections.actions[60]
    # projection = projections.actions[67]
    # plotter = pv.Plotter()
    # plotter.add_mesh(projection.xy_slice, color="red", opacity=0.25)
    # plotter.add_mesh(projection.xz_slice, color="red", opacity=0.25)
    
    # xy_list = projection.xy_list
    # xz_list = projection.xz_list
    # xz_points = [(x,z) for x,y,z in xz_list]

    # #IDEA: need to make a set of points between all vertices of projected region
    # _,_, z = xy_list[0] # extract z value
    # if len(xy_list) < 4:  
    #     print("too small list!")
    
    # poly2 = Polygon(xy_list)

    # print(f"z = {z}")
    # x,y = poly2.exterior.xy
    # plt.figure()
    # plt.scatter(x, y, alpha=0.5)
    # plt.plot(x, y, 'o', color='black')  # plot the points
    # plt.title("Polygon Plot")
    # plt.xlabel("X coordinate")
    # plt.ylabel("Y coordinate")
    # plt.grid(True)
    # plt.show()

    # point_data = projection.xz_slice.points
    # # Convert 3D points to 2D by discarding the y-coordinate (assuming xz-plane)
    # points_2d = [(x,z) for x,y,z in point_data]
    # # Calculate centroid of the points
    # cx, cz = centroid(points_2d)
    # # Sort points
    # sorted_points = sort_points_by_angle(points_2d, (cx, cz))

    # # Create a Shapely polygon
    # polygon = Polygon(sorted_points)

    # sorted_points.append(sorted_points[0])
    # line = LineString(sorted_points)
    # n = 300
    # distances = np.linspace(0, line.length, n)
    # points = [line.interpolate(distance) for distance in distances]# + [line.boundary[1]]
    # x_coords = [point.x for point in points]  # List comprehension to extract x coordinates
    # y_coords = [point.y for point in points]

    # # Plotting the polygon
    # x, z = polygon.exterior.xy
    # plt.figure()
    # plt.scatter(x_coords, y_coords, alpha=0.5, ec='r')
    # plt.title("Ordered Polygon")
    # plt.xlabel("X coordinate")
    # plt.ylabel("Z coordinate")
    # plt.grid(True)
    # plt.show()


    #plotter.show()
    display_action_set(actions, segmented_objects, projections)



# Function to sort points by angle from centroid
def sort_points_by_angle(points, center):
    cx, cz = center
    def angle(point):
        return np.arctan2(point[1] - cz, point[0] - cx)
    return sorted(points, key=angle)


# Function to calculate centroid of the points
def centroid(points):
    x, z = zip(*points)
    cx = sum(x) / len(points)
    cz = sum(z) / len(points)
    return cx, cz

if __name__ == "__main__":
    main()
