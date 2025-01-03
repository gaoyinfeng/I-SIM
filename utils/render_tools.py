import matplotlib
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection

from interaction_gym import geometry

import yaml

def set_get_visible_area(laneletmap, axes):
    min_x = 10e9
    min_y = 10e9
    max_x = -10e9
    max_y = -10e9

    for point in laneletmap.pointLayer:
        min_x = min(point.x, min_x)
        min_y = min(point.y, min_y)
        max_x = max(point.x, max_x)
        max_y = max(point.y, max_y)
    
    axes.set_aspect('equal', adjustable='box')
    axes.set_xlim([min_x - 10, max_x + 10])
    axes.set_ylim([min_y - 10, max_y + 10])
    # x_lim() return same value as get_xbound()
    map_x_bound = axes.get_xbound()
    map_y_bound = axes.get_ybound()
    
    return map_x_bound, map_y_bound

def draw_grid_map(laneletmap, axes):
    assert isinstance(axes, matplotlib.axes.Axes)

    map_x_bound, map_y_bound = set_get_visible_area(laneletmap, axes)
    # print('viaible area: ', map_x_bound, map_y_bound)

    unknown_linestring_types = list()

    for ls in laneletmap.lineStringLayer:

        if "type" not in ls.attributes.keys():
            raise RuntimeError("ID " + str(ls.id) + ": Linestring type must be specified")
        elif ls.attributes["type"] == "curbstone":
            # type_dict = dict(color="white", linewidth=1, zorder=10)
            continue
        elif ls.attributes["type"] == "line_thin":
            if "subtype" in ls.attributes.keys() and ls.attributes["subtype"] == "dashed":
                # type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[10, 10])
                continue
            else:
                # type_dict = dict(color="white", linewidth=1, zorder=10)
                continue
        elif ls.attributes["type"] == "line_thick":
            if "subtype" in ls.attributes.keys() and ls.attributes["subtype"] == "dashed":
                # type_dict = dict(color="white", linewidth=2, zorder=10, dashes=[10, 10])
                continue
            else:
                # type_dict = dict(color="white", linewidth=2, zorder=10)
                continue
        elif ls.attributes["type"] == "pedestrian_marking":
            # type_dict = dict(color="black", linewidth=1, zorder=10, dashes=[5, 10])
            continue
        elif ls.attributes["type"] == "bike_marking":
            # type_dict = dict(color="black", linewidth=1, zorder=10, dashes=[5, 10])
            continue
        elif ls.attributes["type"] == "stop_line":
            # type_dict = dict(color="white", linewidth=3, zorder=10)
            continue
        elif ls.attributes["type"] == "virtual":
            # type_dict = dict(color="blue", linewidth=1, zorder=10, dashes=[2, 5])
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif ls.attributes["type"] == "road_border":
            # type_dict = dict(color="black", linewidth=1, zorder=10)
            # type_dict = dict(color="white", linewidth=1, zorder=10)
            continue
        elif ls.attributes["type"] == "guard_rail":
            # type_dict = dict(color="black", linewidth=1, zorder=10)
            continue
        elif ls.attributes["type"] == "traffic_sign":
            continue
        elif ls.attributes["type"] == "building":
            type_dict = dict(color="pink", zorder=1, linewidth=5)
        elif ls.attributes["type"] == "spawnline":
            if ls.attributes["spawn_type"] == "start":
                type_dict = dict(color="green", zorder=11, linewidth=2)
            elif ls.attributes["spawn_type"] == "end":
                type_dict = dict(color="red", zorder=11, linewidth=2)

        else:
            if ls.attributes["type"] not in unknown_linestring_types:
                unknown_linestring_types.append(ls.attributes["type"])
            continue

        ls_points_x = [pt.x for pt in ls]
        ls_points_y = [pt.y for pt in ls]

        axes.plot(ls_points_x, ls_points_y, **type_dict)

    if len(unknown_linestring_types) != 0:
        print("Found the following unknown types, did not plot them: " + str(unknown_linestring_types))

    lanelets = []
    for ll in laneletmap.laneletLayer:
        points = [[pt.x, pt.y] for pt in ll.polygon2d()]
        polygon = Polygon(points, True)
        lanelets.append(polygon)

    ll_patches = PatchCollection(lanelets, facecolors="lightgray", edgecolors="black", zorder=5)
    axes.add_collection(ll_patches)

    if len(laneletmap.laneletLayer) == 0:
        axes.patch.set_facecolor('lightgrey')

    axes.set_xticks([]) 
    axes.set_yticks([])
    axes.set_title('Grid Representation')


def draw_vector_map(laneletmap, road_centerline_list, axes, draw_type='arrow'):

    assert isinstance(axes, matplotlib.axes.Axes)
    map_x_bound, map_y_bound = set_get_visible_area(laneletmap, axes)

    # draw road centerline points
    for centerline in road_centerline_list:
        # draw map
        ls_points_x = [pt[0] for pt in centerline]
        ls_points_y = [pt[1] for pt in centerline]

        if draw_type == 'dashline':
            type_dict = dict(color="grey", linewidth=1, zorder=10)
            axes.plot(ls_points_x, ls_points_y, **type_dict)
        elif draw_type == 'arrow':
            for index in range(len(ls_points_x)-1):
                type_dict = dict(length_includes_head=True, head_width=1, head_length=0.3, color="grey", lw=1, zorder=10)
                axes.arrow(ls_points_x[index], ls_points_y[index], ls_points_x[index+1]-ls_points_x[index], ls_points_y[index+1]-ls_points_y[index], **type_dict)
 
    axes.set_xticks([]) 
    axes.set_yticks([]) 
    axes.set_title('Vector Representation')


def draw_controlled_vehicles(motionstate_dict, polygon_dict, patch_dict, text_dict, grid_axes, vector_axes, surrounding_vehicle_id_list=None):
    for v_id in motionstate_dict.keys():
        # ego render as red, react npc render as violet when in surrounding_vehicle_id_list, otherwise blue
        if surrounding_vehicle_id_list is None:
            color = 'r' 
        else:
            color = 'violet' if v_id in surrounding_vehicle_id_list else 'thistle'
        # draw controlled vehicles
        if v_id not in patch_dict.keys(): # first time draw
            # alpha is the depth of color, rect for grid map, circle and path for vector map
            rect = matplotlib.patches.Polygon(polygon_dict[v_id], closed=True, facecolor=color, edgecolor='black', alpha=1, zorder=20)
            circle = matplotlib.patches.CirclePolygon((motionstate_dict[v_id].x, motionstate_dict[v_id].y), radius=1, facecolor=color, edgecolor='black', alpha=1, zorder=20)
            patch_dict[v_id] = [rect, circle]
            # set them in different maps
            grid_axes.add_patch(patch_dict[v_id][0])
            vector_axes.add_patch(patch_dict[v_id][1])
            # vehicle id in grid map
            text_dict[v_id] = grid_axes.text(motionstate_dict[v_id].x, motionstate_dict[v_id].y + 2, str(v_id), horizontalalignment='center', zorder=30, color='white')
        else:
            # change their location in different maps
            patch_dict[v_id][0].set_xy(polygon_dict[v_id])
            patch_dict[v_id][0].set_facecolor(color)
            patch_dict[v_id][1].set_visible(False)
            patch_dict[v_id][1] = matplotlib.patches.CirclePolygon((motionstate_dict[v_id].x, motionstate_dict[v_id].y), radius=1, facecolor=color, edgecolor='black', alpha=1, zorder=20)
            vector_axes.add_patch(patch_dict[v_id][1])
            # vehicle id in grid map
            text_dict[v_id].set_position((motionstate_dict[v_id].x, motionstate_dict[v_id].y + 2))


def draw_uncontrolled_vehicle(motionstate_dict, polygon_dict, patch_dict, text_dict, grid_axes, vector_axes, surrounding_vehicle_id_list, settings):
    
    vdi_num = settings['vdi_num']

    for v_id in motionstate_dict.keys():
        if v_id in surrounding_vehicle_id_list:
            color = 'purple' if v_id in surrounding_vehicle_id_list[:vdi_num] else 'cyan'
        else:
            color = 'b'
        # draw uncotrolled vehicles
        if v_id not in patch_dict.keys(): # first time draw
            rect = matplotlib.patches.Polygon(polygon_dict[v_id], closed=True, facecolor=color, edgecolor='black', zorder=20)
            circle = matplotlib.patches.CirclePolygon((motionstate_dict[v_id].x, motionstate_dict[v_id].y), radius=1, facecolor=color, edgecolor='black', zorder=20)
            patch_dict[v_id] = [rect, circle]
            # set them in d            python3 -m pip install ipythonifferent maps
            grid_axes.add_patch(patch_dict[v_id][0])
            vector_axes.add_patch(patch_dict[v_id][1])
            # vehicle id in grid map
            text_dict[v_id] = grid_axes.text(motionstate_dict[v_id].x, motionstate_dict[v_id].y + 2, str(v_id), horizontalalignment='center', zorder=30,color='white')
        else: # already appeared
            # change their location in different maps
            patch_dict[v_id][0].set_xy(polygon_dict[v_id]) # grid
            patch_dict[v_id][0].set_facecolor(color)
            patch_dict[v_id][1].set_visible(False) # vector
            patch_dict[v_id][1] = matplotlib.patches.CirclePolygon((motionstate_dict[v_id].x, motionstate_dict[v_id].y), radius=1, facecolor=color, edgecolor='black', zorder=20)
            vector_axes.add_patch(patch_dict[v_id][1])
            # vehicle id in grid map
            text_dict[v_id].set_position((motionstate_dict[v_id].x, motionstate_dict[v_id].y + 2))

    # remove out of map object
    for v_id in patch_dict.keys():
        if v_id not in motionstate_dict.keys():
            # remove them from different maps
            patch_dict[v_id][0].remove() # grid
            patch_dict[v_id][1].set_visible(False) # vector
            # remove from env parameters
            patch_dict.pop(v_id)
            text_dict[v_id].remove()
            text_dict.pop(v_id)


def draw_ghost_vehicle(motionstate_dict, polygon_dict, patch_dict, text_dict, grid_axes, render_as_ego):
    for v_id in motionstate_dict:
        # set ghost color
        if render_as_ego:
            facecolor, edgecolor, alpha = 'r', 'black', 1 # (0.41, 0.35, 0.80)
        else:
            facecolor, edgecolor, alpha = 'whitesmoke', 'white', 0.6
        # render ghost
        if v_id not in patch_dict:
            patch_dict[v_id] = matplotlib.patches.Polygon(polygon_dict[v_id], closed=True, zorder=20, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
            grid_axes.add_patch(patch_dict[v_id])
            if render_as_ego:
                text_dict[v_id] = grid_axes.text(motionstate_dict[v_id].x, motionstate_dict[v_id].y + 2, str(v_id), horizontalalignment='center', zorder=30, color='white')
        else:
            patch_dict[v_id].set_xy(polygon_dict[v_id])
            if render_as_ego:
                text_dict[v_id].set_position((motionstate_dict[v_id].x, motionstate_dict[v_id].y + 2))


def draw_static_line(line_dict, axes):
    for line in line_dict.values():
        line_point_list = [[point[0], point[1]] for point in line]
        # render all points in green
        ls_points_x = [pt[0] for pt in line_point_list]
        ls_points_y = [pt[1] for pt in line_point_list]
        # dashline
        arg_dict = dict(color="green", linewidth=3, zorder=10)
        axes.plot(ls_points_x, ls_points_y, **arg_dict)


def draw_dynamic_line(line_dict, patch_dict, axes, axes_type, arg_dict):
    for v_id, v_line in line_dict.items():
        line_point_list = [[pt[0], pt[1]] for pt in v_line]
        future_path = matplotlib.path.Path(line_point_list)
        if v_id not in patch_dict.keys():
            future_path_patch = matplotlib.patches.PathPatch(future_path, **arg_dict)
            patch_dict[v_id] = future_path_patch
            axes.add_patch(patch_dict[v_id])
        else:
            patch_dict[v_id].set_visible(False) # future trajectory
            patch_dict[v_id] = matplotlib.patches.PathPatch(future_path, **arg_dict)
            axes.add_patch(patch_dict[v_id])

    # remove out of map object
    for v_id in patch_dict.keys():
        if v_id not in line_dict.keys():
            if axes_type == 'grid':
                patch_dict[v_id].remove()
            elif axes_type == 'vector':
                patch_dict[v_id].set_visible(False)
            patch_dict.pop(v_id)



def draw_route_bounds(route_bounds, axes):
    # render all bounds points in green
    ls_points_x = [pt[0] for pt in route_bounds] # ls_points_x = [pt.x for pt in route_bounds]
    ls_points_y = [pt[1] for pt in route_bounds] # ls_points_y = [pt.y for pt in route_bounds]

    # dashline
    type_dict = dict(color="magenta", linewidth=1.5, zorder=10)
    axes.plot(ls_points_x, ls_points_y, **type_dict)

def draw_closet_bound_point(previous_closet_points, current_closet_points, axes):
    if previous_closet_points:
        ls_points_x = [pt[0] for pt in previous_closet_points]
        ls_points_y = [pt[1] for pt in previous_closet_points]

        # circle
        centerline_circle = []
        for i in range(len(ls_points_x)):
            cirl = Circle(xy = (ls_points_x[i], ls_points_y[i]), radius=0.45, alpha=0.5)
            centerline_circle.append(cirl)
        centerline_circle_patches = PatchCollection(centerline_circle, facecolors="magenta", zorder=5)
        axes.add_collection(centerline_circle_patches)
        

    # render current future route in green
    ls_points_x = [pt[0] for pt in current_closet_points]
    ls_points_y = [pt[1] for pt in current_closet_points]

    # green circle
    centerline_circle = []
    for i in range(len(ls_points_x)):
        cirl = Circle(xy = (ls_points_x[i], ls_points_y[i]), radius=0.45, alpha=0.5)
        centerline_circle.append(cirl)
    centerline_circle_patches = PatchCollection(centerline_circle, facecolors="green", zorder=5)
    axes.add_collection(centerline_circle_patches)