
try:
    import lanelet2
    from lanelet2.core import AttributeMap, getId, Point3d, Polygon2d
    from lanelet2.geometry import distance, intersects2d
except:
    import warnings
    string = "Could not import lanelet2. It must be built and sourced, " + \
             "see https://github.com/fzi-forschungszentrum-informatik/Lanelet2 for details."
    warnings.warn(string)

import random
import argparse
import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import geometry

from collections import defaultdict

from utils.dataset_reader import read_trajectory, read_tracks, read_pedestrian

# the map data in intersection dataset
# x axis direction is from left to right
# y axis direction is from top to bottom

class InteractionMap:

    def __init__(self, map_file_path, settings):
        
        print("Loading map...")
        # load lanelet2 map data
        self._lanelet_map_file = map_file_path
        lat_origin = 0.  # origin is necessary to correctly project the lat lon values in the osm file to the local
        lon_origin = 0.  # coordinates in which the tracks are provided; we decided to use (0|0) for every scenario
        self._rules_map = {"vehicle": lanelet2.traffic_rules.Participants.Vehicle}
        self._projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(lat_origin, lon_origin))
        self._max_steps = settings['max_steps']
        
        # initialize vehicles' id list
        self._ego_id_list = list()
        self._react_vdi_id_list = list()
        self._record_vdi_id_list = list()

        # tracks which the vehicles will follow
        self._vdi_type = settings['vdi_type']
        # self._load_mode = settings['load_mode']
        self.track_dict = dict()
        self.ego_track_dict = dict()
        self._record_vdi_track_dict = dict()
        self._pedestrian_dict = dict()

        # polygon used for collison detection and distance calculation
        self.ego_polygon_dict = dict()
        self.react_vdi_polygon_dict = dict()
        self.record_vdi_polygon_dict = dict()
        self.ghost_polygon_dict = dict()
        
        # (current) motion states
        self.ego_motion_state_dict = dict()
        self.react_motion_state_dict = dict()
        self.record_vdi_motion_state_dict = dict()
        self.ghost_motion_state_dict = dict()

        # record past location for vector representation
        self.past_ego_vehicle_state = defaultdict(list)
        self.past_react_vdi_vehicle_state = defaultdict(list)
        self.past_record_vdi_vehicle_state = defaultdict(list)

        # including ego and react vdi
        self.controlled_vehicle_start_end_state_dict = dict()
        # shapes
        self.vehicle_shape = dict()
                
    def _reset_vehicle_dict(self, ego_id_list, ego_start_timestamp=None):
        self.controlled_vehicle_start_end_state_dict['ego'] = dict()
        self.controlled_vehicle_start_end_state_dict['vdi'] = dict()
        # set ego first
        self._ego_id_list = ego_id_list
        self.ego_track_dict = {key:value for key,value in self.track_dict.items() if key in self._ego_id_list}
        # then vdis
        if self._vdi_type == 'react':
            self._react_vdi_id_list = [30]
            self._record_vdi_id_list = []
            # controlled vehicle includes ego and react vdi, we set their start and end timestamp and state manully in react vdi scen
            for vehicle_id in self._ego_id_list:
                ego_info = self.track_dict[vehicle_id]
                length, width = ego_info.length, ego_info.width
                ego_start_time_delay, ego_end_time_front = 3000, 3000
                ego_start_timestamp_ms = ego_info.time_stamp_ms_first + ego_start_time_delay
                ego_end_timestamp_ms = ego_start_timestamp_ms + 100 * self._max_steps - 100 if self._max_steps else ego_info.time_stamp_ms_last - ego_end_time_front
                self.controlled_vehicle_start_end_state_dict['ego'][vehicle_id] = [ego_start_timestamp_ms, ego_end_timestamp_ms, length, width, ego_info.motion_states[ego_start_timestamp_ms], ego_info.motion_states[ego_end_timestamp_ms]]
            for vehicle_id in self._react_vdi_id_list:
                vdi_info = self.track_dict[vehicle_id]
                length, width = vdi_info.length, vdi_info.width
                vdi_start_time_delay = 10000
                vdi_start_timestamp_ms = vdi_info.time_stamp_ms_first + vdi_start_time_delay
                vdi_end_timestamp_ms = vdi_start_timestamp_ms + 100 * self._max_steps - 100 if self._max_steps else vdi_info.time_stamp_ms_last
                self.controlled_vehicle_start_end_state_dict['vdi'][vehicle_id] = [vdi_start_timestamp_ms, vdi_end_timestamp_ms, length, width, vdi_info.motion_states[vdi_start_timestamp_ms], vdi_info.motion_states[vdi_end_timestamp_ms]]
        elif self._vdi_type == 'record':
            self._record_vdi_id_list = list(set(self.track_dict.keys()) - set(self._ego_id_list))
            for ego_id in self._ego_id_list:
                ego_info = self.track_dict[ego_id]
                length, width = ego_info.length, ego_info.width
                if ego_start_timestamp and self._max_steps:
                    ego_timestamp_ms_first, ego_timestamp_ms_last = int(ego_start_timestamp[0]), ego_timestamp_ms_first + 100 * self._max_steps - 100
                else:
                    ego_timestamp_ms_first, ego_timestamp_ms_last = ego_info.time_stamp_ms_first, ego_info.time_stamp_ms_last
                self.controlled_vehicle_start_end_state_dict['ego'][ego_id] = [ego_timestamp_ms_first, ego_timestamp_ms_last, length, width, ego_info.motion_states[ego_timestamp_ms_first], ego_info.motion_states[ego_timestamp_ms_last]]
        else:
            print('Please check if vdi type setting is right')

    def reset(self, track_dict, ego_id_list, ego_start_timestamp=None):
        # clear
        self.ego_track_dict.clear()
        self._record_vdi_track_dict.clear()
        self.track_dict.clear()
        # self._pedestrian_dict.clear()

        self.ego_polygon_dict.clear()
        self.react_vdi_polygon_dict.clear()
        self.record_vdi_polygon_dict.clear()
        self.ghost_polygon_dict.clear()

        self.ego_motion_state_dict.clear()
        self.react_motion_state_dict.clear()
        self.record_vdi_motion_state_dict.clear()
        self.ghost_motion_state_dict.clear()

        self.past_ego_vehicle_state.clear()
        self.past_react_vdi_vehicle_state.clear()
        self.past_record_vdi_vehicle_state.clear()

        self.controlled_vehicle_start_end_state_dict.clear()
        self.vehicle_shape.clear()
        
        # load vehicles tracks
        self.track_dict = track_dict
        # seperate different kinds of vehicles to different dicts
        self._reset_vehicle_dict(ego_id_list, ego_start_timestamp=ego_start_timestamp)

    def get_lanelet_configs(self):

        # initialize map
        laneletmap = lanelet2.io.load(self._lanelet_map_file, self._projector)
        # route graph is used for lanelet relationship searching
        routing_cost = lanelet2.routing.RoutingCostDistance(0.) # zero cost for lane changes
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, self._rules_map['vehicle'])
        routing_graph = lanelet2.routing.RoutingGraph(laneletmap, traffic_rules, [routing_cost])

        return laneletmap, traffic_rules, routing_graph
    
    def get_road_centerline_list(self, laneletmap):
        
        # get all lanelets from the map
        lanelet_list = []
        for lanelet in laneletmap.laneletLayer:
            lanelet_list.append(lanelet)

        # get all centerline points of all the lanelets
        centerline_list = []
        for lanelet in lanelet_list:
            centerpoint_list = []
            for index in range(len(lanelet.centerline)):
                centerpoint_list.append([lanelet.centerline[index].x, lanelet.centerline[index].y])
            centerline_list.append(centerpoint_list)

        # preprocess centerline points to a uniformed intervel
        # centerline_list = geometry.preprocess_centerlines_by_distance(centerline_list, interval_distance=1)
        # preprocess centerline points to a uniformed number
        centerline_list = geometry.preprocess_centerlines_by_number(centerline_list, vector_num=10)

        return centerline_list

    # update all env parameters like vehicles states etc and history motion states for vector representation
    # for controlled vehicles like ego and react vdi, their current states are calculated by vehicle dynamic model;
    # for uncontrolled vehicles, their current states are read from track file
    def update_param(self, timestamp, controlled_state_dict, time_horizen=20, ghost_vis=True):

        def polygon_xy_from_motionstate(ms, length, width):
            lowleft = (ms.x - length / 2., ms.y - width / 2.)
            lowright = (ms.x + length / 2., ms.y - width / 2.)
            upright = (ms.x + length / 2., ms.y + width / 2.)
            upleft = (ms.x - length / 2., ms.y + width / 2.)
            polygons = np.array([lowleft, lowright, upright, upleft])
            center = np.array([ms.x, ms.y])
            yaw = ms.psi_rad
            return np.dot(polygons - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center
        
        def polygon_xy_from_motionstate_pedest(ms, length, width):
            lowleft = (ms.x - length / 2., ms.y - width / 2.)
            lowright = (ms.x + length / 2., ms.y - width / 2.)
            upright = (ms.x + length / 2., ms.y + width / 2.)
            upleft = (ms.x - length / 2., ms.y + width / 2.)
            return np.array([lowleft, lowright, upright, upleft])
        
        def polygon_intersect(poly1_corner_np, poly2_corner_np):
            getAttributes = lambda: AttributeMap({"key": "value"})
            ego_polypoint_np = poly1_corner_np
            ego_polyPoint3d = [Point3d(getId(), p[0], p[1], 0, getAttributes()) for p in ego_polypoint_np]
            poly1 = Polygon2d(getId(), [ego_polyPoint3d[0], ego_polyPoint3d[1], ego_polyPoint3d[2], ego_polyPoint3d[3]], getAttributes())

            vpi_polypoint_np = poly2_corner_np
            vpi_polyPoint3d = [Point3d(getId(), p[0], p[1], 0, getAttributes()) for p in vpi_polypoint_np]
            poly2 = Polygon2d(getId(), [vpi_polyPoint3d[0], vpi_polyPoint3d[1], vpi_polyPoint3d[2], vpi_polyPoint3d[3]], getAttributes())
            return intersects2d(poly1, poly2)
            
        # controlled vehicles shapes
        for ego_id, ego_info in self.controlled_vehicle_start_end_state_dict['ego'].items():
            if ego_id not in self.vehicle_shape.keys():
                self.vehicle_shape[ego_id] = (ego_info[2], ego_info[3]) # length and width
        for vehicle_id, vehicle_info in self.controlled_vehicle_start_end_state_dict['vdi'].items():
            if vehicle_id not in self.vehicle_shape.keys():
                self.vehicle_shape[vehicle_id] = (vehicle_info[2], vehicle_info[3]) # length and width

        # update controlled vehicles(ego + react) polygons, their motion states are provided as function inputs
        for vehicle_id in self.controlled_vehicle_start_end_state_dict['ego'].keys():
            length, width, vehicle_ms = self.vehicle_shape[vehicle_id][0], self.vehicle_shape[vehicle_id][1], controlled_state_dict[vehicle_id]
            self.ego_polygon_dict[vehicle_id] = polygon_xy_from_motionstate(vehicle_ms, length, width)
            self.ego_motion_state_dict[vehicle_id] = vehicle_ms
        for vehicle_id in self.controlled_vehicle_start_end_state_dict['vdi'].keys():
            length, width, vehicle_ms = self.vehicle_shape[vehicle_id][0], self.vehicle_shape[vehicle_id][1], controlled_state_dict[vehicle_id]
            self.react_vdi_polygon_dict[vehicle_id] = polygon_xy_from_motionstate(vehicle_ms, length, width)
            self.react_motion_state_dict[vehicle_id] = vehicle_ms
        # update uncontrolled vehicles(record) polygons and motion states
        for vehicle_id in self._record_vdi_id_list:
            vehicle_track = self.track_dict[vehicle_id]
            if vehicle_track.time_stamp_ms_first <= timestamp <= vehicle_track.time_stamp_ms_last:
                length, width, vehicle_ms = vehicle_track.length, vehicle_track.width, vehicle_track.motion_states[timestamp]
                polygon = polygon_xy_from_motionstate(vehicle_ms, length, width)
                if vehicle_id not in self.record_vdi_polygon_dict.keys(): # check if it starts in conflict position with ego
                    conflict_with_ego = []
                    for ego_id in self._ego_id_list:
                        conflict = polygon_intersect(polygon, self.ego_polygon_dict[ego_id])
                        conflict_with_ego.append(conflict)
                    if any(conflict_with_ego):
                        continue
                self.record_vdi_polygon_dict[vehicle_id] = polygon
                self.record_vdi_motion_state_dict[vehicle_id] = vehicle_ms
            else:
                if vehicle_id in self.record_vdi_polygon_dict.keys():
                    self.record_vdi_polygon_dict.pop(vehicle_id)
                    self.record_vdi_motion_state_dict.pop(vehicle_id)
        # update ghost (record ego) polygons and motion states
        if ghost_vis:
            for ghost_id, ghost_track in self.ego_track_dict.items():
                if ghost_track.time_stamp_ms_first <= timestamp <= ghost_track.time_stamp_ms_last:
                    length, width, ghost_ms = ghost_track.length, ghost_track.width, ghost_track.motion_states[timestamp]
                    self.ghost_polygon_dict[ghost_id] = polygon_xy_from_motionstate(ghost_ms, length, width)
                    self.ghost_motion_state_dict[ghost_id] = ghost_ms
                else:
                    if ghost_id in self.ghost_polygon_dict.keys():
                        self.ghost_polygon_dict.pop(ghost_id)
                        self.ghost_motion_state_dict.pop(ghost_id)

        # update history motion states for vector representation
        for ego_id in self._ego_id_list:
            ego_x, ego_y, ego_heading = controlled_state_dict[ego_id].x, controlled_state_dict[ego_id].y, controlled_state_dict[ego_id].psi_rad
            self.past_ego_vehicle_state[ego_id].append([ego_x, ego_y, ego_heading])
            if len(self.past_ego_vehicle_state[ego_id]) > time_horizen:
                self.past_ego_vehicle_state[ego_id] = self.past_ego_vehicle_state[ego_id][-time_horizen:]
            elif len(self.past_ego_vehicle_state[ego_id]) < time_horizen: # NOTE: this only happened once when the vehicle is first added!
                while len(self.past_ego_vehicle_state[ego_id]) < time_horizen:
                    self.past_ego_vehicle_state[ego_id].append(self.past_ego_vehicle_state[ego_id][0])
        for react_id in self._react_vdi_id_list:
            react_vdi_x, react_vdi_y, react_vdi_heading = controlled_state_dict[react_id].x, controlled_state_dict[react_id].y, controlled_state_dict[react_id].psi_rad
            self.past_react_vdi_vehicle_state[react_id].append([react_vdi_x, react_vdi_y, react_vdi_heading])
            if len(self.past_react_vdi_vehicle_state[react_id]) > time_horizen:
                self.past_react_vdi_vehicle_state[react_id] = self.past_react_vdi_vehicle_state[react_id][-time_horizen:]
            elif len(self.past_react_vdi_vehicle_state[react_id]) < time_horizen:
                while len(self.past_react_vdi_vehicle_state[react_id]) < time_horizen: # NOTE: this only happened once when the vehicle is first added!
                    self.past_react_vdi_vehicle_state[react_id].append(self.past_react_vdi_vehicle_state[react_id][0])
        for record_id in self.record_vdi_motion_state_dict.keys():
            record_vdi_x, record_vdi_y, record_vdi_heading = self.record_vdi_motion_state_dict[record_id].x, self.record_vdi_motion_state_dict[record_id].y, self.record_vdi_motion_state_dict[record_id].psi_rad
            self.past_record_vdi_vehicle_state[record_id].append([record_vdi_x, record_vdi_y, record_vdi_heading])
            if len(self.past_record_vdi_vehicle_state[record_id]) > time_horizen:
                self.past_record_vdi_vehicle_state[record_id] = self.past_record_vdi_vehicle_state[record_id][-time_horizen:]
            elif len(self.past_record_vdi_vehicle_state[record_id]) < time_horizen: # NOTE: this only happened once when the vehicle is first added!
                while len(self.past_record_vdi_vehicle_state[record_id]) < time_horizen:
                    self.past_record_vdi_vehicle_state[record_id].append(self.past_record_vdi_vehicle_state[record_id][0])

        
    """ relate to Lane Vector Representation """ 
    # def get_all_centerline(self):
    #     lanelet_list = []
    #     for lanelet in self.laneletmap.laneletLayer:
    #         lanelet_list.append(lanelet)

    #     centerline_list = []
    #     for lanelet in lanelet_list:
    #         centerpoint_list = []
    #         for index in range(len(lanelet.centerline)):
    #             centerpoint_list.append([lanelet.centerline[index].x, lanelet.centerline[index].y])
    #         centerline_list.append(centerpoint_list)

    #     return centerline_list
