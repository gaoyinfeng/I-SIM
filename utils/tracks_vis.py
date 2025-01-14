#!/usr/bin/env python
import matplotlib
import matplotlib.patches
import matplotlib.transforms
import math
import numpy as np

from .dataset_types import Track, MotionState
# from interaction_interface.interface import interface 
# print('1:',__file__)
from shapely.geometry import Point, Polygon

def polygon_intersect(poly1_corner_np,poly2_corner_np):
    poly1 = Polygon([tuple(item) for item in poly1_corner_np])
    poly2 = Polygon([tuple(item) for item in poly2_corner_np])

    is_intersect = poly1.intersects(poly2)

    return is_intersect


def rotate_around_center(pts, center, yaw):
    return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center


def polygon_xy_from_motionstate(ms, width, length):
    assert isinstance(ms, MotionState)
    lowleft = (ms.x - length / 2., ms.y - width / 2.)
    lowright = (ms.x + length / 2., ms.y - width / 2.)
    upright = (ms.x + length / 2., ms.y + width / 2.)
    upleft = (ms.x - length / 2., ms.y + width / 2.)
    return rotate_around_center(np.array([lowleft, lowright, upright, upleft]), np.array([ms.x, ms.y]), yaw=ms.psi_rad)


def polygon_xy_from_motionstate_pedest(ms, width, length):
    assert isinstance(ms, MotionState)
    lowleft = (ms.x - length / 2., ms.y - width / 2.)
    lowright = (ms.x + length / 2., ms.y - width / 2.)
    upright = (ms.x + length / 2., ms.y + width / 2.)
    upleft = (ms.x - length / 2., ms.y + width / 2.)
    return np.array([lowleft, lowright, upright, upleft])


def update_objects_plot(timestamp, patches_dict, text_dict, axes, track_dict=None, pedest_dict=None):

    if track_dict is not None:

        for key, value in track_dict.items():
            assert isinstance(value, Track)
            if value.time_stamp_ms_first <= timestamp <= value.time_stamp_ms_last:
                # object is visible
                ms = value.motion_states[timestamp]
                assert isinstance(ms, MotionState)

                if key not in patches_dict:
                    width = value.width
                    length = value.length

                    rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate(ms, width, length), closed=True,
                                                      zorder=20)
                    patches_dict[key] = rect
                    axes.add_patch(rect)
                    text_dict[key] = axes.text(ms.x, ms.y + 2, str(key), horizontalalignment='center', zorder=30)
                else:
                    width = value.width
                    length = value.length
                    patches_dict[key].set_xy(polygon_xy_from_motionstate(ms, width, length))
                    text_dict[key].set_position((ms.x, ms.y + 2))
            else:
                if key in patches_dict:
                    patches_dict[key].remove()
                    patches_dict.pop(key)
                    text_dict[key].remove()
                    text_dict.pop(key)

    if pedest_dict is not None:

        for key, value in pedest_dict.items():
            assert isinstance(value, Track)
            if value.time_stamp_ms_first <= timestamp <= value.time_stamp_ms_last:
                # object is visible
                ms = value.motion_states[timestamp]
                assert isinstance(ms, MotionState)

                if key not in patches_dict:
                    width = 1.5
                    length = 1.5

                    rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate_pedest(ms, width, length), closed=True,
                                                      zorder=20, color='red')
                    patches_dict[key] = rect
                    axes.add_patch(rect)
                    text_dict[key] = axes.text(ms.x, ms.y + 2, str(key), horizontalalignment='center', zorder=30,color='white')
                else:
                    width = 1.5
                    length = 1.5
                    patches_dict[key].set_xy(polygon_xy_from_motionstate_pedest(ms, width, length))
                    text_dict[key].set_position((ms.x, ms.y + 2))
            else:
                if key in patches_dict:
                    patches_dict[key].remove()
                    patches_dict.pop(key)
                    text_dict[key].remove()
                    text_dict.pop(key)


def update_objects_plot_without_ego(timestamp, other_patches_dict, other_vehicle_polygon, text_dict, axes, other_track_dict=None, pedest_dict=None):
    """
    plot other objects without ego vehicles 
    """
    if other_track_dict is not None:

        for key, value in other_track_dict.items():
            assert isinstance(value, Track)
            if value.time_stamp_ms_first <= timestamp <= value.time_stamp_ms_last:
                # object is visible
                ms = value.motion_states[timestamp]
                assert isinstance(ms, MotionState)

                if key not in other_patches_dict:
                    width = value.width
                    length = value.length

                    other_vehicle_polygon[key] = polygon_xy_from_motionstate(ms, width, length)

                    rect = matplotlib.patches.Polygon(other_vehicle_polygon[key], closed=True,
                                                      zorder=20)
                    other_patches_dict[key] = rect
                    axes.add_patch(rect)
                    text_dict[key] = axes.text(ms.x, ms.y + 2, str(key), horizontalalignment='center', zorder=30,color='white')
                else:
                    width = value.width
                    length = value.length

                    other_vehicle_polygon[key] = polygon_xy_from_motionstate(ms, width, length)
                    other_patches_dict[key].set_xy(other_vehicle_polygon[key])
                    text_dict[key].set_position((ms.x, ms.y + 2))
            else:
                if key in other_patches_dict:
                    # print('other_vehicle_polygon:',type(other_vehicle_polygon[key]))
                    # other_vehicle_polygon[key].remove()
                    other_vehicle_polygon.pop(key)
                    other_patches_dict[key].remove()
                    other_patches_dict.pop(key)
                    text_dict[key].remove()
                    text_dict.pop(key)

    if pedest_dict is not None:

        for key, value in pedest_dict.items():
            assert isinstance(value, Track)
            if value.time_stamp_ms_first <= timestamp <= value.time_stamp_ms_last:
                # object is visible
                ms = value.motion_states[timestamp]
                assert isinstance(ms, MotionState)

                if key not in other_patches_dict:
                    width = 1.5
                    length = 1.5

                    rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate_pedest(ms, width, length), closed=True,
                                                      zorder=20, color='red')
                    other_patches_dict[key] = rect
                    axes.add_patch(rect)
                    text_dict[key] = axes.text(ms.x, ms.y + 2, str(key), horizontalalignment='center', zorder=30)
                else:
                    width = 1.5
                    length = 1.5
                    other_patches_dict[key].set_xy(polygon_xy_from_motionstate_pedest(ms, width, length))
                    text_dict[key].set_position((ms.x, ms.y + 2))
            else:
                if key in other_patches_dict:
                    other_patches_dict[key].remove()
                    other_patches_dict.pop(key)
                    text_dict[key].remove()
                    text_dict.pop(key)

def update_objects_without_ego_and_conflict(timestamp, other_vehicle_polygon, other_vehicle_motionstate, ego_vehicle_polygon, other_track_dict=None, pedest_dict=None):
    """
    plot other objects without ego vehicles also without position conflicf with ego vehicle
    """
    if other_track_dict is not None:

        for key, value in other_track_dict.items():
            assert isinstance(value, Track)
            if value.time_stamp_ms_first <= timestamp <= value.time_stamp_ms_last:
                # object is visible
                ms = value.motion_states[timestamp]
                assert isinstance(ms, MotionState)
                # first time appear
                if key not in other_vehicle_polygon:
                    width = value.width
                    length = value.length

                    other_vehicle_polygon[key] = polygon_xy_from_motionstate(ms, width, length)
                    other_vehicle_motionstate[key] = ms

                    conflict = False
                    # slove born postion conflict
                    for v in ego_vehicle_polygon.values():
                        if polygon_intersect(v,other_vehicle_polygon[key]):
                            conflict = True
                    if conflict:
                        other_vehicle_polygon.pop(key)

                else:
                    width = value.width
                    length = value.length

                    other_vehicle_polygon[key] = polygon_xy_from_motionstate(ms, width, length)
                    other_vehicle_motionstate[key] = ms

            else:
                # object is invisible
                if key in other_vehicle_polygon:
                    other_vehicle_polygon.pop(key)
                    other_vehicle_motionstate.pop(key)
       

    if pedest_dict is not None:

        for key, value in pedest_dict.items():
            assert isinstance(value, Track)
            if value.time_stamp_ms_first <= timestamp <= value.time_stamp_ms_last:
                # object is visible
                ms = value.motion_states[timestamp]
                assert isinstance(ms, MotionState)

                if key not in other_patches_dict:
                    width = 1.5
                    length = 1.5

                    rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate_pedest(ms, width, length), closed=True,
                                                      zorder=20, color='red')
                    other_patches_dict[key] = rect
                    axes.add_patch(rect)
                    text_dict[key] = axes.text(ms.x, ms.y + 2, str(key), horizontalalignment='center', zorder=30)
                else:
                    width = 1.5
                    length = 1.5
                    other_patches_dict[key].set_xy(polygon_xy_from_motionstate_pedest(ms, width, length))
                    text_dict[key].set_position((ms.x, ms.y + 2))
            else:
                if key in other_patches_dict:
                    other_patches_dict[key].remove()
                    other_patches_dict.pop(key)
                    text_dict[key].remove()
                    text_dict.pop(key)

def render_objects_without_ego_and_conflict(other_patches_dict, other_vehicle_polygon, other_vehicle_motionstate,text_dict,axes):
    for key in other_vehicle_polygon:
        ms = other_vehicle_motionstate[key]
        
        if key not in other_patches_dict:
            # first time appear
            rect = matplotlib.patches.Polygon(other_vehicle_polygon[key], closed=True,
                                                            zorder=20)
            other_patches_dict[key] = rect
            axes.add_patch(rect)
            text_dict[key] = axes.text(ms.x, ms.y + 2, str(key), horizontalalignment='center', zorder=30,color='white')
        else:
            # already appear
            other_patches_dict[key].set_xy(other_vehicle_polygon[key])
            text_dict[key].set_position((ms.x, ms.y + 2))

    # remove invisible object
    for key,value in other_patches_dict.items():
        if key not in other_vehicle_polygon:
            other_patches_dict[key].remove()
            other_patches_dict.pop(key)
            text_dict[key].remove()
            text_dict.pop(key)

def render_objects_without_ego_and_conflict_with_highlight(other_patches_dict, other_vehicle_polygon, other_vehicle_motionstate,surrounding_vehicle_id_list,text_dict,axes):
    for key in other_vehicle_polygon:
        ms = other_vehicle_motionstate[key]
        if key not in other_patches_dict:
            # first time appear
            if key in surrounding_vehicle_id_list:
                # surrounding
                rect = matplotlib.patches.Polygon(other_vehicle_polygon[key], closed=True,
                                                            zorder=20, facecolor='y', edgecolor='black')
                # rect = matplotlib.patches.Polygon(other_vehicle_polygon[key], closed=True,
                #                                             zorder=20, facecolor='b', edgecolor='black')
            else:
                # not surrounding
                rect = matplotlib.patches.Polygon(other_vehicle_polygon[key], closed=True,
                                                            zorder=20, facecolor='b', edgecolor='black')
            other_patches_dict[key] = rect
            axes.add_patch(rect)
            text_dict[key] = axes.text(ms.x, ms.y + 2, str(key), horizontalalignment='center', zorder=30,color='white')
        else:
            # already appear but is not in surrounding before 
            if key not in surrounding_vehicle_id_list:
                # not surrounding 
                # change origin color to blue
                other_patches_dict[key].remove()
                other_patches_dict.pop(key)
                rect = matplotlib.patches.Polygon(other_vehicle_polygon[key], closed=True,
                                                            zorder=20, facecolor='b', edgecolor='black')
                other_patches_dict[key] = rect
                axes.add_patch(rect)
            else:
                # surrounding
                # change origin color to yellow
                other_patches_dict[key].remove()
                other_patches_dict.pop(key)
                rect = matplotlib.patches.Polygon(other_vehicle_polygon[key], closed=True,
                                                            zorder=20, facecolor='y', edgecolor='black')
                # rect = matplotlib.patches.Polygon(other_vehicle_polygon[key], closed=True,
                #                                             zorder=20, facecolor='b', edgecolor='black')
                other_patches_dict[key] = rect
                axes.add_patch(rect)
            other_patches_dict[key].set_xy(other_vehicle_polygon[key])
            text_dict[key].set_position((ms.x, ms.y + 2))

    # remove invisible object
    for key,value in other_patches_dict.items():
        if key not in other_vehicle_polygon:
            other_patches_dict[key].remove()
            other_patches_dict.pop(key)
            text_dict[key].remove()
            text_dict.pop(key)

def update_objects_ego(ego_patches_dict, ego_vehicle_polygon, ego_shape_dict=None, ego_motionstate_dict=None):
    """
    plot ego vehicles
    param: patches_dict is used for saving the visible vehicles till now
    """
    # ego vehicles are different from others as they do not disappear until reach termination condition
    # for simplicity sake, we think if one of the ego vehicles reach the termination condition ,all other vehicles should be reset
   
    if (ego_motionstate_dict is not None) and (ego_shape_dict is not None):

        for key, value in ego_motionstate_dict.items():
            assert isinstance(value, MotionState)
            
            if key not in ego_vehicle_polygon:
                length = ego_shape_dict[key][0]
                width = ego_shape_dict[key][1]

                ego_vehicle_polygon[key] = polygon_xy_from_motionstate(value, width, length)

            else:
                length = ego_shape_dict[key][0]
                width = ego_shape_dict[key][1]
                ego_vehicle_polygon[key] = polygon_xy_from_motionstate(value, width, length)

            # if ego_patches_dict:
            #     # ego_speed_value = min(10, math.sqrt(value.vx**2 + value.vy**2))
            #     # speed_color = (10 - ego_speed_value)/10
            #     # speed_color *= 0.5
            #     speed_color = 0
            #     ego_patches_dict[key].set_facecolor((1, speed_color, speed_color))


def update_objects_ghost(timestamp, ghost_vehicle_polygon, ghost_motionstate_dict, ghost_track_dict=None):
    if ghost_track_dict is not None:

        for key, value in ghost_track_dict.items():
            assert isinstance(value, Track)
            if value.time_stamp_ms_first <= timestamp <= value.time_stamp_ms_last:
                # object is visible
                ms = value.motion_states[timestamp]
                assert isinstance(ms, MotionState)
                # first time appear
                if key not in ghost_vehicle_polygon:
                    width = value.width
                    length = value.length
                    ghost_vehicle_polygon[key] = polygon_xy_from_motionstate(ms, width, length)
                    ghost_motionstate_dict[key] = ms
                 
                else:
                    width = value.width
                    length = value.length
                    ghost_vehicle_polygon[key] = polygon_xy_from_motionstate(ms, width, length)
                    ghost_motionstate_dict[key] = ms
            else:
                # object is invisible
                if key in ghost_vehicle_polygon:
                    ghost_vehicle_polygon.pop(key)
                    ghost_motionstate_dict.pop(key)
               
def render_objects_ego(ego_patches_dict, ego_vehicle_polygon, ego_motionstate_dict, text_dict, axes):
    for key in ego_vehicle_polygon:
        if key not in ego_patches_dict:
            alpha = 1
            # ego_speed_value = min(10, math.sqrt(value.vx**2 + value.vy**2))
            # alpha = max(0.5, ego_speed_value/10)

            rect = matplotlib.patches.Polygon(ego_vehicle_polygon[key], closed=True, zorder=20, facecolor=(1, 0, 0), edgecolor='black', alpha=1)
            ego_patches_dict[key] = rect
            axes.add_patch(rect)
            text_dict[key] = axes.text(ego_motionstate_dict[key].x, ego_motionstate_dict[key].y + 2, str(key), horizontalalignment='center', zorder=30, color='white')
        else:
            ego_patches_dict[key].set_xy(ego_vehicle_polygon[key])
            text_dict[key].set_position((ego_motionstate_dict[key].x, ego_motionstate_dict[key].y + 2))

def render_objects_ghost(ghost_patches_dict, ghost_vehicle_polygon, ghost_motionstate_dict, text_dict, axes, color_str, is_ego_render):
    for key in ghost_vehicle_polygon:
        if key not in ghost_patches_dict:
            if is_ego_render:
                rect = matplotlib.patches.Polygon(ghost_vehicle_polygon[key], closed=True, zorder=20, facecolor='b', edgecolor='black')
            else:
                rect = matplotlib.patches.Polygon(ghost_vehicle_polygon[key], closed=True,zorder=20,color=color_str,alpha=0.6)

            ghost_patches_dict[key] = rect
            axes.add_patch(rect)

            if is_ego_render:
                text_dict[key] = axes.text(ghost_motionstate_dict[key].x, ghost_motionstate_dict[key].y + 2, str(key), horizontalalignment='center', zorder=30,color='white')
        else:
            ghost_patches_dict[key].set_xy(ghost_vehicle_polygon[key])
            if is_ego_render:
                text_dict[key].set_position((ghost_motionstate_dict[key].x, ghost_motionstate_dict[key].y + 2))