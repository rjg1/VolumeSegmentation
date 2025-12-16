import json
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

debugFile = "./debug.json"
errorThreshold = 0.5
maxIterations = 500 # Maximum number of iterations to run while loop to prevent infinite loops
verbose = True

# Plots the 2D points passed in overlayed
def plot_points(planeA2D, planeB2D, planeAIdx, planeBIdx, debugEntry, title =  None):
    fig, ax =  plt.subplots(figsize = (6,6))
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Overlay of 2D planes")
    # Plane A pts
    for idx, point in enumerate(planeA2D):
        if idx == 0:
            # Special handling to label anchor points
            ax.scatter(point[0], point[1], color = "red", s=60, label = f"Anchor_A [{planeAIdx[idx]}]")
            ax.text(point[0] + 1.5, point[1] + 1.5, f"Anchor_A [{planeAIdx[idx]}]", color='red')
        else:
            ax.scatter(point[0], point[1], color = "red", s=30)
            ax.text(point[0] + 1.5, point[1] + 1.5, f"{planeAIdx[idx]}", color='red')
    # Plane B pts
    for idx, point in enumerate(planeB2D):
        if idx == 0:
            # Special handling to label anchor points
            ax.scatter(point[0], point[1], color = "blue", s=60, label = f"Anchor_B [{planeBIdx[idx]}]")
            ax.text(point[0] + 1.5, point[1] + 1.5, f"Anchor_B [{planeBIdx[idx]}]", color='blue')
        else:
            ax.scatter(point[0], point[1], color = "blue", s=30)
            ax.text(point[0] + 1.5, point[1] + 1.5, f"{planeBIdx[idx]}", color='blue')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.grid(True)
    plt.show()


def plot_zstacks(zstacks, z, ptlist = None,  idlist = None, colstacks = None, colpts = None):
    fig, ax = plt.subplots(figsize = (10,10))
    ax.set_title(f"Plot of {len(zstacks)} z-stacks overlaid at z={z}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.grid(True)

    cmap = mpl.colormaps['viridis'] 
    indices = np.linspace(0,1, len(zstacks))
    colours = cmap(indices)


    for idx, zstack in enumerate(zstacks):
        z_level = zstack.z_planes[z] # go to a specific z-plane
        col = cmap(indices[idx]) if not colstacks else colstacks[idx]
        for roi_id in z_level: # iterate over all rois o this z-plane
            for (x,y) in z_level[roi_id]['coords']:
                # Plot all xy points of this roi
                plt.scatter(x, y, c=col)

    # plot centroid pts if any - this is deranged dw about it
    if ptlist and idlist:
        for iidx, points in enumerate(ptlist):
            colp = cmap(indices[iidx]) if not colpts else colpts[iidx]
            planeIdx = idlist[iidx]
            for idx, point in enumerate(points):
                if idx == 0:
                    # Special handling to label anchor points
                    ax.scatter(point[0], point[1], color = colp, s=60, label = f"Anchor [{planeIdx[idx]}]")
                    ax.text(point[0] + 1.5, point[1] + 1.5, f"Anchor [{planeIdx[idx]}]", color=colp)
                else:
                    ax.scatter(point[0], point[1], color = colp, s=30)
                    ax.text(point[0] + 1.5, point[1] + 1.5, f"{planeIdx[idx]}", color=colp)

        plt.show()

def overlay_pts_on_zstack(plane2D, planeIdx, zstack, z, colz = "cyan", colp = "blue"):
    fig, ax = plt.subplots(figsize = (10,10))
    ax.set_title(f"Plot of plane roi pts and zstack rois overlaid at z={z}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.grid(True)

    # zstack points
    z_level = zstack.z_planes[z] # go to a specific z-plane
    for roi_id in z_level: # iterate over all rois o this z-plane
        for (x,y) in z_level[roi_id]['coords']:
            # Plot all xy points of this roi
            plt.scatter(x, y, c=colz)

    # Plane roi pts
    for idx, point in enumerate(plane2D):
        if idx == 0:
            # Special handling to label anchor points
            ax.scatter(point[0], point[1], color = colp, s=60, label = f"Anchor [{planeIdx[idx]}]")
            ax.text(point[0] + 1.5, point[1] + 1.5, f"Anchor [{planeIdx[idx]}]", color=colp)
        else:
            ax.scatter(point[0], point[1], color = colp, s=30)
            ax.text(point[0] + 1.5, point[1] + 1.5, f"{planeIdx[idx]}", color=colp)

    plt.show()


def main():
    # Load debug file - [{<debugEntry}, {}]
    try:
        with open(debugFile) as file:
            debug = json.load(file)
    except:
        debug = None



    # Select an erroneous entry
    score = np.inf
    idx = 0
    debugEntry = None
    iterations = 0

    # while score > errorThreshold or iterations < maxIterations:
    #     # Select a random entry in debug list
    #     idx = random.randint(0, len(debug) - 1)
    #     debugEntry = debug[idx]

    #     # Extract score
    #     score = sum(list(debugEntry["score_contributions"].values()))

    #     iterations += 1
    
    # Example correct
    idx = 20
    debugEntry = debug[idx]

    # # # Example incorrect
    # idx = 19
    # debugEntry = debug[idx]

    if verbose:
        print(f"Selected debugEntry: {debugEntry} \n Debug index: {idx}")
    
    # Plot plane points raw
    planeA2D = [item[1][0:2] for item in debugEntry["A_points"]]
    planeAIdx = [item[0][1] for item in debugEntry["A_points"]]
    planeB2D = [item[1][0:2] for item in debugEntry["B_points"]]
    planeBIdx = [item[0][1] for item in debugEntry["B_points"]]



    # giga debug
    from zstack import ZStack
    pretrans_b = ZStack("pretrans_z_stack_b.csv")
    posttrans_b = ZStack("z_stack_b.csv")
    stack_a = ZStack("real_data_filtered_algo_VOLUMES_g.csv")
    # plot_zstacks([pretrans_b, posttrans_b], 30, colstacks=["maroon", "cyan"])

    # # remove translation
    posttrans_b.affine_transform(0, (-10, -10), 1)
    # plot_zstacks([pretrans_b, posttrans_b], 30)
    # # remove rotation
    #posttrans_b.affine_transform(310, (0, 0), 1)
    # plot_zstacks([pretrans_b, posttrans_b], 30)

    # Plot b points overlaid on post-trans rois
    overlay_pts_on_zstack(planeB2D, planeBIdx, posttrans_b, 30)
    overlay_pts_on_zstack(planeA2D, planeAIdx, stack_a, 30, colz = "maroon", colp = "red")

    plot_zstacks([pretrans_b, posttrans_b], 30, [planeA2D, planeB2D], [planeAIdx, planeBIdx], colstacks=["maroon", "cyan"], colpts=["red", "blue"])
    return
    # end giga debug


    plot_points(planeA2D, planeB2D, planeAIdx, planeBIdx, debugEntry, title = "Overlay of 2D planes pre transformation")

    # Plot plane points with rotation fixed
    # angleRadians = np.deg2rad(debugEntry["offset"]) # Could manually set to 310
    angleRadians = np.deg2rad(310) # Manually set to 310
    rotationMatrix = np.array([[np.cos(angleRadians), -np.sin(angleRadians)],
                            [np.sin(angleRadians), np.cos(angleRadians)]])

    rotationPoint = np.array(debugEntry["B_anch"][0:2]) # Extract 2D points of B anchor position

    rotatedB2D = []

    for idx, point in enumerate(planeB2D):
        # discount anchor points
        if idx == 0:
            rotatedB2D.append(point)
            continue
        
        # Translate to "origin"
        translatedPoint = np.array(point) - np.array(rotationPoint)
        rotatedTranslated = np.dot(rotationMatrix, translatedPoint)
        rotatedPoint = np.array(rotatedTranslated + rotationPoint)

        rotatedB2D.append(rotatedPoint)

    # plot_points(planeA2D, rotatedB2D, planeAIdx, planeBIdx, debugEntry, title = "Overlay of 2D planes post rotation")

    # Plot plane points with translation fixed
    translationPoint = debugEntry["A_anch"][0:2] # Translation to be applied to all b points
    
    rotatedTranslatedB2D = []

    translation = np.array(translationPoint) - np.array(rotationPoint)

    for point in rotatedB2D:
        rotatedTranslatedPoint = np.array(point) + np.array(translation)
        rotatedTranslatedB2D.append(rotatedTranslatedPoint)

    print(rotatedTranslatedB2D)

    plot_points(planeA2D, rotatedTranslatedB2D, planeAIdx, planeBIdx, debugEntry, title = "Overlay of 2D planes post rotation and translation")



if __name__ == "__main__":
    main()