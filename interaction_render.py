
try:
    import lanelet2
    print("Using Lanelet2 visualization")
except:
    import warnings
    string = "Could not import lanelet2. It must be built and sourced, " + \
             "see https://github.com/fzi-forschungszentrum-informatik/Lanelet2 for details."
    warnings.warn(string)

import matplotlib.pyplot as plt

from utils.render_tools import *

class InteractionRender:
    def __init__(self, settings):
        
        # create figures
        self._fig, self._axes = plt.subplots(1, 2, facecolor = 'lightgray', figsize=(15, 5)) # figure backcolor (figure size > map render size)
        self._fig.canvas.set_window_title("I-SIM Visualization " + str(settings['port']))

        self._grid_axes = self._axes[0]
        self._vector_axes = self._axes[1]
        self._grid_axes.set_facecolor('white')
        self._vector_axes.set_facecolor('white')

        plt.subplots_adjust(top=0.95, bottom=0, left=0.05, right=0.95, hspace=0.2, wspace=0)

        # vehicle positions
        self._ego_state_dict = dict()
        self._react_vdi_state_dict = dict()
        self._record_vdi_state_dict = dict()
        self._ghost_state_dict = dict()

        # vehicle polygons positions
        self._ego_polygon_dict = dict()
        self._react_vdi_polygon_dict = dict()
        self._record_vdi_polygon_dict = dict()
        self._ghost_polygon_dict = dict()

        # vehicles patches
        self._ego_patches_dict = dict()
        self._react_vdi_patches_dict = dict()
        self._record_vdi_patches_dict = dict()
        self._ghost_patches_dict = dict()
        # lane patches
        self._future_route_patches_dict = dict()
        self._prediction_patches_dict = dict()
        # use for vehicle id visualization
        self._text_dict = dict()

    def reset(self, laneletmap, road_centerline_list):
        # clear
        self._grid_axes.clear()
        self._vector_axes.clear()

        self._ego_patches_dict.clear()
        self._react_vdi_patches_dict.clear()
        self._record_vdi_patches_dict.clear()
        self._ghost_patches_dict.clear()

        self._future_route_patches_dict.clear()
        self._prediction_patches_dict.clear()
        self._text_dict.clear()
        
        # initialize grid map and vector map
        draw_grid_map(laneletmap, self._grid_axes)
        draw_vector_map(laneletmap, road_centerline_list, self._vector_axes, draw_type='dashline') # arrow or dashline

    def render(self):
        self._fig.canvas.draw()

    def update_param(self, map):
        # state positions
        self._ego_state_dict = map.ego_motion_state_dict.copy()
        self._react_vdi_state_dict = map.react_motion_state_dict.copy()
        self._record_vdi_state_dict = map.record_vdi_motion_state_dict.copy()
        self._ghost_state_dict = map.ghost_motion_state_dict.copy()

        # polygon postions
        self._ego_polygon_dict = map.ego_polygon_dict.copy()
        self._react_vdi_polygon_dict = map.react_vdi_polygon_dict.copy()
        self._record_vdi_polygon_dict = map.record_vdi_polygon_dict.copy()
        self._ghost_polygon_dict = map.ghost_polygon_dict.copy()
    
    def save_images(self, ego_id, current_time):
        self._fig.savefig('saved_images/' + 'ego_'+ str(ego_id) + '_' + str(current_time) +'.png')


    def render_vehicles(self, surrounding_vehicle_id_list, ghost_vis=True):
        plt.ion()
        
        draw_controlled_vehicles(self._ego_state_dict, self._ego_polygon_dict, self._ego_patches_dict, self._text_dict, self._grid_axes, self._vector_axes)
        draw_controlled_vehicles(self._react_vdi_state_dict, self._react_vdi_polygon_dict, self._react_vdi_patches_dict, self._text_dict, self._grid_axes, self._vector_axes, surrounding_vehicle_id_list)
        draw_uncontrolled_vehicle(self._record_vdi_state_dict, self._record_vdi_polygon_dict, self._record_vdi_patches_dict, self._text_dict, self._grid_axes, self._vector_axes, surrounding_vehicle_id_list)
        if ghost_vis:
            draw_ghost_vehicle(self._ghost_state_dict, self._ghost_polygon_dict, self._ghost_patches_dict, self._text_dict, self._grid_axes, render_as_ego = False)

        plt.show()
        plt.ioff()


    def render_static_route(self, line_dict, axes_type):
        plt.ion()

        if axes_type == 'grid':
            axes = self._grid_axes
        elif axes_type == 'vector':
            axes = self._vector_axes

        # render in map
        draw_static_line(line_dict, axes)

        plt.show()
        plt.ioff()


    def render_future_route(self, line_dict, axes_type):
        plt.ion()

        if axes_type == 'grid':
            axes = self._grid_axes
        elif axes_type == 'vector':
            axes = self._vector_axes

        # render in map
        arg_dict = dict(color='red', linewidth=4, zorder=15)
        draw_dynamic_line(line_dict, self._future_route_patches_dict, axes, axes_type, arg_dict)

        plt.show()
        plt.ioff()


    def render_prediction(self, line_dict, axes_type):
        plt.ion()

        if axes_type == 'grid':
            axes = self._grid_axes
        elif axes_type == 'vector':
            axes = self._vector_axes

        # render in map
        arg_dict = dict(color='yellow', linewidth=1.8, zorder=35)
        draw_dynamic_line(line_dict, self._prediction_patches_dict, axes, axes_type, arg_dict)

        plt.show()
        plt.ioff()


    # render route bounds
    def render_route_bounds(self, route_left_bounds, route_right_bounds, axes_type='grid'):
        plt.ion()

        if axes_type == 'grid':
            axes = self._grid_axes
        elif axes_type == 'vector':
            axes = self._vector_axes

        for k, v in route_left_bounds.items():
            draw_route_bounds(route_left_bounds[k], axes)
            draw_route_bounds(route_right_bounds[k], axes)

        plt.show()
        plt.ioff()

    def render_closet_bound_point(self, previous_closet_points, current_closet_points, axes_type='grid'):
        plt.ion()

        if axes_type == 'grid':
            axes = self._grid_axes
        elif axes_type == 'vector':
            axes = self._vector_axes

        draw_closet_bound_point(previous_closet_points, current_closet_points, axes)
        
        plt.show()
        plt.ioff()