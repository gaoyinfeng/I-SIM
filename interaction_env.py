#-*- coding: UTF-8 -*- 
import sys
sys.path.append("..")

import os
import glob
import copy
import numpy as np
import math
import argparse
import zmq
import copy
import time

import lanelet2
import lanelet2_matching

import geometry
import reward

from interaction_map import InteractionMap
from interaction_render import InteractionRender
from track_loader import Large_scale_dataset_loader, Small_scale_dataset_loader
from vehicle_model import ControlledVehicle
from vehicle_policy import NpcPolicy
from observation import Observation

class InteractionEnv:
    def __init__(self, settings):
        if settings['max_steps'] == 'None':
            settings['max_steps'] = None

        # settings
        self._settings = settings
        # env settings
        self._map_name = settings['map_name']
        self._max_steps = settings['max_steps'] # max duration of an episode, if None, equals to the actual lifetime
        self._vdi_type = settings['vdi_type'] # reactive or non-reactive vdi
        self._loader_type = settings['loader_type'] # vdi routes from INTERACTIVE dataset, or from in-house prediction dataset
        self._route_type = settings['route_type'] # ground truth or centerline(derived from ground truth) or in-house prediction
        # ego settings
        self._drive_as_record = settings['drive_as_record'] # ego drive as record, in other words, 'replaying'
        self._control_steering = settings['control_steering'] # control both acc and delta_yaw or only target speed
        self._continous_action = settings['continous_action'] # continuous or discret action space
        # vis settings
        self._visualization = settings['visualization']
        self._ghost_visualization = settings['ghost_visualization']
        self._route_visualization = settings['route_visualization']
        self._route_bound_visualization = settings['route_bound_visualization']
        # eval mode
        self._eval = settings['eval']
        
        # maps and tracks dirs
        root_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(root_dir, "dataset")
        groundtruth_tracks_dir = os.path.join(dataset_dir, "recorded_trackfiles", self._map_name)
        # load map
        map_file_path = os.path.join(dataset_dir, "maps", self._map_name + '.osm')
        self._map = InteractionMap(map_file_path, settings)
        
        # load vehicle tracks
        if self._loader_type == 'small_scale':
            self._track_loader = Small_scale_dataset_loader(groundtruth_tracks_dir, eval=self._eval)
        elif self._loader_type == 'large_scale': 
            self._track_loader = Large_scale_dataset_loader(groundtruth_tracks_dir, eval=self._eval)
        
        else:
            print('Please check if str args is right')

        # visualization render
        self._render = InteractionRender(settings)

        # some configurations
        self._delta_time = 100 # 100 ms = 0.1s
        self._max_speed = 9.0 # m/s
        self._discrete_action_num = 4 # number of discrete action

        # env variables 
        self._ego_vehicle_dict = dict()
        self._ego_route_dict = dict()
        self._ego_previous_route_points_dict = dict()
        self._ego_trajectory_record = dict()
        self.ego_future_route_points_dict = dict()
        # self._ego_route_lanelet_dict = dict()

        self._react_vdi_vehicle_dict = dict()
        self._react_vdi_policy_dict = dict()
        self._react_vdi_route_dict = dict()
        self._react_vdi_previous_route_points_dict = dict()
        self.react_vdi_future_route_points_dict = dict()

        self._stepnum = 0
        self._current_time = None
        self._scenario_start_time = None
        self._scenario_end_time = None # select the earliest end time
        self._start_end_state = None  # ego vehicle start & end state (start_time, end_time, length, width, start motion_state, end motion_state)
        self._observation = None

        self.save_images = True

    # def __del__(self):
    #     self._map.__del__()
    
    def change_track(self):
        
        if self._loader_type == 'small_scale':
            gt_csv_index = self._track_loader.change_track_file()

        elif self._loader_type == 'large_scale': 
            gt_csv_index = self._track_loader.change_track_file()
        else:
            print('Please check if str args is right')
        return gt_csv_index

    def scenario_init(self):

        # clear variables
        self._ego_vehicle_dict.clear()
        self._ego_route_dict.clear()
        # self._ego_route_lanelet_dict.clear()
        self._react_vdi_vehicle_dict.clear()
        self._react_vdi_policy_dict.clear()
        self._react_vdi_route_dict.clear()

        # read tracks and ego informations
        track_dict = self._track_loader.read_track_file(vdi_type=self._vdi_type, route_type=self._route_type)
        ego_id_list = self._track_loader.select_ego()
        start_timestamp_list = self._track_loader.get_start_timestamp()

        # setting map and render
        self._map.reset(track_dict, ego_id_list, start_timestamp_list)
        self._laneletmap, self._traffic_rules, self._routing_graph = self._map.get_lanelet_configs()
        self._road_centerline_list = self._map.get_road_centerline_list(self._laneletmap)
        self._render.reset(self._laneletmap, self._road_centerline_list)

        # initialize controlled vehicles (ego and reactive vdis), and their routes to be followed
        self._start_end_state = self._map.controlled_vehicle_start_end_state_dict  # controlled vehicle start & end state dict, key = ego_id, value = (start_time,end_time,length,width,start motion_state,end motion_state)

        if self._route_type == 'ground_truth':
            route_dict = self.get_ground_truth_route()
        elif self._route_type == 'centerline':
            route_dict = self.get_centerline_route()
        else:
            print('Please check if str args is right')
        for ego_id, ego_start_end_state in self._start_end_state['ego'].items():
            self._ego_vehicle_dict[ego_id] = ControlledVehicle(ego_start_end_state, self._delta_time, discrete_action_num=self._discrete_action_num, max_speed=self._max_speed) # delta_time means tick-time length
            self._ego_route_dict[ego_id] = route_dict[ego_id]
        for vehicle_id, vehicle_start_end_state in self._start_end_state['vdi'].items():
            self._react_vdi_vehicle_dict[vehicle_id] = ControlledVehicle(vehicle_start_end_state, self._delta_time, discrete_action_num=self._discrete_action_num, max_speed=self._max_speed)
            self._react_vdi_route_dict[vehicle_id] = route_dict[vehicle_id] # react vdi usually runs in ground truth route
            self._react_vdi_policy_dict[vehicle_id] = NpcPolicy(agg_prec=0.5)

        # controlled vehicle (ego + reactive vdis) observation manager init
        controlled_vehicle_dict = {'ego': self._ego_vehicle_dict, 'vdi': self._react_vdi_vehicle_dict}
        self._observation = Observation(self._settings, self._map, self._road_centerline_list, controlled_vehicle_dict)

        # time setting
        self._scenario_start_time = max([self._start_end_state['ego'][i][0] for i in self._start_end_state['ego']]) # select the latest start time among all ego vehicle
        if self._max_steps:
            self._scenario_end_time = self._scenario_start_time + 100 * self._max_steps # 10s = 100 * 0.1s
        else:
            self._scenario_end_time = min([self._start_end_state['ego'][i][1] for i in self._start_end_state['ego']]) # select the earliest end time among all ego vehicle

        if self._scenario_start_time > self._scenario_end_time:
            print('start time > end time?')
            return False

        # return self._map._ego_id_list
        return ego_id_list

    def reset(self):
        # clear previous route points record
        self._ego_previous_route_points_dict.clear()
        self._react_vdi_previous_route_points_dict.clear()
        # clear trajectory record
        self._ego_trajectory_record.clear()
        # for jerk and steer reward calculation if they exist
        self._ego_previous_acc = 0
        self._ego_previous_steer = 0

        # initialize controlled vehicle (ego + reactive vdi) state
        ego_state_dict = dict()
        for ego_id, ego_state in self._start_end_state['ego'].items():
            ego_state_dict[ego_id] = ego_state[4] # NOTE: now "state" only contains start motion_state: (time_stamp_ms, x, y, vx, vy, psi_rad)
            self._ego_trajectory_record[ego_id] = []
            for ego_id, ego_state in self._ego_vehicle_dict.items():
                ego_state.reset_state(self._start_end_state['ego'][ego_id][4])
                self._ego_trajectory_record[ego_id].append([ego_state._current_state.x, ego_state._current_state.y, 
                                                            ego_state._current_state.vx, ego_state._current_state.vy, ego_state._current_state.psi_rad])
        react_vdi_state_dict = dict()
        if self._vdi_type == 'react':
            for react_vdi_id, react_vdi_state in self._start_end_state['vdi'].items():
                react_vdi_state_dict[react_vdi_id] = react_vdi_state[4]        
            for react_vdi_id, react_vdi_state in self._react_vdi_vehicle_dict.items():
                react_vdi_state.reset_state(self._start_end_state['vdi'][react_vdi_id][4])

        # initialize global environment time
        self._current_time = self._scenario_start_time

        # reset episode ticker
        self._stepnum = 0
        self._total_stepnum = (self._scenario_end_time - self._scenario_start_time) / self._delta_time

        controlled_state_dict = {**ego_state_dict, **react_vdi_state_dict}
        self._map.update_param(self._current_time, controlled_state_dict, ghost_vis=self._ghost_visualization)
        
        # reset/clear observation and get initial observation
        self._observation.reset(self._route_type, self._ego_route_dict, self._react_vdi_route_dict)
        self.observation_dict = self._observation.get_scalar_observation(self._current_time)

        # target point for PID control
        self.ego_future_route_points_dict, self.react_vdi_future_route_points_dict = self._observation.get_future_route_points(self.observation_dict)
        
        # visualize map, vehicles and ego's planned future route
        if self._visualization:
            self._render.update_param(self._map)
            # specified ego(with/without ghost ego) and selected vehicle highlight
            surrounding_vehicle_id_list = self._observation.get_surrounding_vehicle_id(self.observation_dict)
            self._render.render_vehicles(surrounding_vehicle_id_list, self._ghost_visualization)
            if self._route_visualization:
                static_route_dict = {**self._ego_route_dict, **self._react_vdi_route_dict}
                self._render.render_static_route(static_route_dict, axes_type='grid')
                future_route_dict = {**self.ego_future_route_points_dict, **self.react_vdi_future_route_points_dict}
                self._render.render_future_route(future_route_dict, axes_type='grid')
            if self._route_bound_visualization:
                self._render.render_route_bound(self._observation.ego_route_left_bound_points, self._observation.ego_route_right_bound_points)
                self._previous_bound_points_list = []
                current_bound_points_list = self._observation.get_current_bound_points(self.observation_dict)
                self._render.render_closet_bound_point(self._previous_bound_points_list, current_bound_points_list)
            self._render.render()
            # save images
            if self.save_images:
                self._render.save_images(ego_id, self._current_time)
        
        return self.observation_dict

    def step(self, action_dict, prediction=None):

        # episode step and current time count
        self._stepnum += 1
        self._current_time += self._delta_time

        # disposable variables
        ego_state_dict = dict()
        ego_action_dict = dict()
        reward_dict = dict()
        aux_info_dict = dict()
        react_vdi_state_dict = dict()

        # update ego state
        if self._drive_as_record: # in this mode, ego action is useless
            for ego_id in action_dict.keys():
                ego_state = self._map.track_dict[ego_id].motion_states[self._current_time]
                self._ego_vehicle_dict[ego_id].update_motion_state(ego_state.x, ego_state.y, ego_state.vx, ego_state.vy, ego_state.psi_rad) # directly update state without using action
                ego_state_dict[ego_id] = ego_state
                ego_action_dict[ego_id] = None
        else:
            if self._control_steering:
                for ego_id, action_list in action_dict.items():
                    ego_state, ego_action = self._ego_vehicle_dict[ego_id].step_continuous_action(action_list)
                    ego_state_dict[ego_id] = ego_state
                    ego_action_dict[ego_id] = ego_action
            else: # action is target speed, using PID to do the bottom controls
                for ego_id, action_list in action_dict.items():
                    future_route_points_list = self.ego_future_route_points_dict[ego_id]
                    index = int(len(future_route_points_list) / 2) # middle point of future 5 points
                    target_point = [future_route_points_list[index][0], future_route_points_list[index][1]]
                    if self._continous_action:
                        ego_state, ego_action = self._ego_vehicle_dict[ego_id].step_continuous_action(action_list, target_point)
                    else:
                        ego_state, ego_action = self._ego_vehicle_dict[ego_id].step_discret_action(action_list, target_point)
                    ego_state_dict[ego_id] = ego_state
                    ego_action_dict[ego_id] = ego_action
                    
        # update react vdi's action and state
        if self._vdi_type == 'react':
            for react_vdi_id in self._react_vdi_vehicle_dict.keys():
                # get react vdi's action, only consider target speed
                if self.observation_dict['reach_goal'][react_vdi_id]:
                    react_action = -100.
                else:
                    react_action = self._react_vdi_policy_dict[react_vdi_id].run_step(self._react_vdi_vehicle_dict[react_vdi_id], self._ego_vehicle_dict[ego_id])
                    behavior_type = self._react_vdi_policy_dict[react_vdi_id].get_behavior_type()
                react_vdi_action = [react_action]
                # get route points
                future_route_points_list = self.react_vdi_future_route_points_dict[react_vdi_id]
                index = int(len(future_route_points_list)/2)
                target_point = [future_route_points_list[index][0], future_route_points_list[index][1]]
                # update react vdi's state based on action and predefined route
                react_vdi_state, _ = self._react_vdi_vehicle_dict[react_vdi_id].step_continuous_action(react_vdi_action, target_point)
                react_vdi_state_dict[react_vdi_id] = react_vdi_state

        # update vehicls' states (if visualize ghost ego, then update its state too)
        controlled_state_dict = {**ego_state_dict, **react_vdi_state_dict}
        self._map.update_param(self._current_time, controlled_state_dict, ghost_vis=self._ghost_visualization)

        # get new observation, calculate rewards and results
        self.observation_dict = self._observation.get_scalar_observation(self._current_time)
        done_dict, result_dict = self.reach_terminate_condition(self._current_time, self.observation_dict)
        for ego_id in result_dict.keys():

            ego_state = ego_state_dict[ego_id]

            self._ego_trajectory_record[ego_id].append([ego_state.x, ego_state.y, ego_state.vx, ego_state.vy, ego_state.psi_rad]) # actual trajcetory
            ego_speed = math.sqrt(ego_state.vx ** 2 + ego_state.vy ** 2) # speed value
            route_length = geometry.get_trajectory_length(self._ego_route_dict[ego_id])
            completed_length = geometry.get_trajectory_length(self._ego_trajectory_record[ego_id])
            completion_rate = (completed_length + 2) / route_length # TODO: 2 is the detect range of goal

            # aux infos
            aux_info_dict[ego_id] = dict()
            aux_info_dict[ego_id]['result'] = result_dict[ego_id]
            aux_info_dict[ego_id]['trajectory'] = self._ego_trajectory_record[ego_id]
            aux_info_dict[ego_id]['speed'] = ego_speed
            aux_info_dict[ego_id]['completion_rate'] = completion_rate if completion_rate <= 1.0 else 1.0
            aux_info_dict[ego_id]['distance_to_gt'] = self.observation_dict['trajectory_distance'][ego_id][0]

            # terminal reward
            if aux_info_dict[ego_id]['result'] == 'success':
                terminal_reward = 0 # 10
            elif aux_info_dict[ego_id]['result'] == 'time_exceed':
                terminal_reward = 0 # -10
            elif aux_info_dict[ego_id]['result'] == 'collision':
                current_speed_norm = self.observation_dict['current_speed'][ego_id][0]/self._max_speed
                terminal_reward = -30 * (1 + current_speed_norm)
            elif aux_info_dict[ego_id]['result'] == 'deflection':
                terminal_reward = -100
            else:
                terminal_reward = 0
            # step reward
            step_reward = -0.3
            if self._control_steering:
                position_reward = reward.calculate_lane_keeping_reward(self.observation_dict, ego_id)
                speed_reward = reward.calculate_speed_reward(self.observation_dict, ego_id, ego_max_speed=self._max_speed, control_steering=True)
                steer_reward = reward.calculate_steer_reward(self._ego_previous_steer, ego_action_dict[ego_id].steering)
                jerk_reward = 0
                self._ego_previous_steer = ego_action_dict[ego_id].steering
            else:
                position_reward = 0 # reward.calculate_trajectory_location_reward(self.observation_dict, ego_id)
                speed_reward = -step_reward * reward.calculate_speed_reward(self.observation_dict, ego_id, ego_max_speed=self._max_speed, control_steering=False)
                steer_reward = 0
                jerk_reward = 0 # -0.6 * (abs(ego_action_dict[ego_id].acc - self._ego_previous_acc) / 2) # acc is a normlized value, i.e. [-1, 1]
                self._ego_previous_acc = ego_action_dict[ego_id].acc
            
            reward_dict[ego_id] = terminal_reward + position_reward + speed_reward + jerk_reward + steer_reward + step_reward
        
        # transfer prediction from ego frame to global frame
        if prediction is not None:
            render_prediction_dict = dict()
            record_prediction_dict = dict()
            prediciton = np.array(prediction)
            for i in range(prediciton.shape[0]):
                prediction = prediciton[i]
                if np.any(prediction): # if this prediction is existent
                    prediction, _ = geometry.delocalize_transform_list((ego_state.x, ego_state.y), ego_state.psi_rad, prediction)
                    render_prediction_dict[i] = prediction
                    record_prediction_dict[i] = prediction
                else:
                    record_prediction_dict[i] = None
            # predictions are conditional on ego's action, thus we record it from the second step of the episode
            aux_info_dict[ego_id]['prediction_global'] = {self._current_time: record_prediction_dict}
            aux_info_dict[ego_id]['ego_loc'] = (ego_state.x, ego_state.y)
            aux_info_dict[ego_id]['track_id'] = self._track_loader.track_id

        # update target point for PID control
        self.ego_future_route_points_dict, self.react_vdi_future_route_points_dict = self._observation.get_future_route_points(self.observation_dict)

        # visualize map, vehicles and ego's planed route
        if self._visualization:
            self._render.update_param(self._map)
            # specified ego(with/without ghost ego) and selected vehicle highlight
            surrounding_vehicle_id_list = self._observation.get_surrounding_vehicle_id(self.observation_dict)
            self._render.render_vehicles(surrounding_vehicle_id_list, self._ghost_visualization)
            # render log defined routes
            if self._route_visualization:
                static_route_dict = self._ego_route_dict # dict(self._ego_route_dict.items() + self._react_vdi_route_dict.items())
                self._render.render_static_route(static_route_dict, axes_type='grid')
                future_route_dict = self.ego_future_route_points_dict # dict(self.ego_future_route_points_dict.items() + self.react_vdi_future_route_points_dict.items())
                self._render.render_future_route(future_route_dict, axes_type='grid')
            # render route bounderies
            if self._route_bound_visualization:
                self._render.render_route_bound(self._observation.ego_route_left_bound_points, self._observation.ego_route_right_bound_points)
                self._previous_bound_points_list = []
                current_bound_points_list = self._observation.get_current_bound_points(self.observation_dict)
                self._render.render_closet_bound_point(self._previous_bound_points_list, current_bound_points_list)
            # render predictions
            if prediction is not None:
                # additionally save images with no prediction lines
                for i in self._render._prediction_patches_dict.values():
                    i.set_visible(False)
                self._render.save_images(ego_id, str(self._current_time) + '_no_prediction') 
                # visualize the predictions
                self._render.render_prediction(render_prediction_dict, axes_type='grid')

            self._render.render()
            # save images
            if self.save_images:
                self._render.save_images(ego_id, self._current_time)

        # print split line when episode ends
        all_done = False not in done_dict.values()
        if all_done:
            print('----------' * 2)

        return self.observation_dict, reward_dict, done_dict, aux_info_dict

    def reach_terminate_condition(self, current_time, observation_dict):

        ego_id_list = observation_dict['ego_shape'].keys()

        # none by default
        done_dict = dict()
        result_dict = dict()
        for ego_id in ego_id_list:
            done_dict[ego_id] = False
            result_dict[ego_id] = 'none'

        # reach end time
        if (current_time + self._delta_time >= self._scenario_end_time):
            print('Scenario Ends: reach end time')
            for ego_id in ego_id_list:
                done_dict[ego_id] = True
                result_dict[ego_id] = 'time_exceed'
        # success, collision or deflection 
        else:
            for term_condition in ['collision', 'deflection', 'reach_goal']:
                # collision ego vehicles
                if term_condition == 'collision':
                    for ego_id in ego_id_list:
                        if observation_dict[term_condition][ego_id] is True:
                            if self._eval:
                                done_dict[ego_id] = True
                                # done_dict[ego_id] = False
                            else:
                                done_dict[ego_id] = False
                            result_dict[ego_id] = 'collision'
                            print(ego_id, 'Fail: collision')
                # deflection ego vehicles
                elif term_condition == 'deflection':
                    for ego_id in ego_id_list:
                        if observation_dict[term_condition][ego_id] is True:
                            done_dict[ego_id] = True
                            result_dict[ego_id] = 'deflection'
                            print(ego_id, 'Fail: deflection (distance)')
                # successfully reach end point ego vehicles
                elif term_condition == 'reach_goal':
                    for ego_id in ego_id_list:
                        if observation_dict[term_condition][ego_id] is True:
                            done_dict[ego_id] = True
                            # done_dict[ego_id] = False
                            result_dict[ego_id] = 'success'
                            print(ego_id, 'Success: reach goal point')
                            break # success has the highest priority

        return done_dict, result_dict

    # generate centerline routes
    def get_centerline_route(self, start_timestamp_list=None):
        centerline_route_dict = dict()
        track_dict = self._map.track_dict
        for vehicle_id in self._start_end_state['ego'].keys():
            vehicle_dict = track_dict[vehicle_id]
            # time horizen
            if start_timestamp_list:
                start_timestamp = int(start_timestamp_list[0])
                end_timestamp = start_timestamp + 100 * self._max_steps - 100
            else:
                start_timestamp = vehicle_dict.time_stamp_ms_first
                end_timestamp = vehicle_dict.time_stamp_ms_last
            # in order to get all of the lanelets
            initial_timestamp = vehicle_dict.time_stamp_ms_first
            terminal_timestamp = vehicle_dict.time_stamp_ms_last

            # get vehicle's whole lanelet
            ms_dict = vehicle_dict.motion_states
            start_lanelet, end_lanelet = self.get_start_end_lanelet_from_ms_dict_with_min_heading(ms_dict, initial_timestamp, terminal_timestamp)

            # if cant find proper start and end lanelet, then:
            if not start_lanelet or not end_lanelet or not self._routing_graph.getRoute(start_lanelet, end_lanelet, 0):
                initial_lanelet, terminal_lanelet = start_lanelet, end_lanelet
                print('can\'t find route, try to use start time instead of initial time')
                start_lanelet, end_lanelet = self.get_start_end_lanelet_from_ms_dict_with_min_heading(ms_dict, start_timestamp, end_timestamp)
                if not start_lanelet or not end_lanelet or not self._routing_graph.getRoute(start_lanelet, end_lanelet, 0):
                    print('still can\'t find route, try to mix them up')
                    start_lanelet, end_lanelet = self.try_to_find_practicable_start_end_lanelet(start_lanelet, initial_lanelet, end_lanelet, terminal_lanelet)
                    if not start_lanelet or not end_lanelet or not self._routing_graph.getRoute(start_lanelet, end_lanelet, 0):
                        print('the centerline route doesn\'t exist, using ground truth route')
                        ground_truth_route_dict = self.get_ground_truth_route()
                        centerline_route_dict[vehicle_id] = ground_truth_route_dict[vehicle_id]
                        continue

            # if proper start and end lanelet exist
            route_lanelet = self.get_route_lanelet(start_lanelet, end_lanelet)

            # get vehicle's route based on whole route lanelet and a specific time horizen
            vehicle_start_pos = [ms_dict[start_timestamp].x, ms_dict[start_timestamp].y]
            vehicle_end_pos = [ms_dict[end_timestamp].x, ms_dict[end_timestamp].y]
            vehicle_route_list = self.get_route_from_lanelet(route_lanelet, vehicle_start_pos, vehicle_end_pos)
            # if this route's start point away from ego's start position
            start_point = [vehicle_route_list[0][0], vehicle_route_list[0][1]]
            if math.sqrt((start_point[0] - vehicle_start_pos[0])**2 + (start_point[1] - vehicle_start_pos[1])**2) > 3:
                print('the centerline route doesn\'t reliable, using ground truth route')
                ground_truth_route_dict = self.get_ground_truth_route()
                centerline_route_dict[vehicle_id] = ground_truth_route_dict[vehicle_id]
            else:
                centerline_route_dict[vehicle_id] = vehicle_route_list

        return centerline_route_dict
    
    def try_to_find_practicable_start_end_lanelet(self, start_lanelet_1, start_lanelet_2, end_lanelet_1, end_lanelet_2):
        start_list = []
        end_list = []
        if start_lanelet_1:
            start_list.append(start_lanelet_1)
            start_lanelet_3_list = self._routing_graph.previous(start_lanelet_1)
            if start_lanelet_3_list:
                start_list.append(start_lanelet_3_list[0])
        if start_lanelet_2:
            start_list.append(start_lanelet_1)
            start_lanelet_4_list = self._routing_graph.previous(start_lanelet_2)
            if start_lanelet_4_list:
                start_list.append(start_lanelet_4_list[0])
        if end_lanelet_1:
            end_list.append(end_lanelet_1)
            end_lanelet_3_list = self._routing_graph.following(end_lanelet_1)
            if end_lanelet_3_list:
                end_list.append(end_lanelet_3_list[0])
        if end_lanelet_2:
            end_list.append(end_lanelet_2)
            end_lanelet_4_list = self._routing_graph.following(end_lanelet_2)
            if end_lanelet_4_list:
                end_list.append(end_lanelet_4_list[0])


        for start_lanelet in start_list:
            for end_lanelet in end_list:
                if self._routing_graph.getRoute(start_lanelet, end_lanelet, 0):
                    return start_lanelet, end_lanelet
        return start_lanelet, end_lanelet

    def get_start_end_lanelet_from_ms_dict(self, ms_dict, start_timestamp, end_timestamp):
        # get the start and end lanelet set of ego vehicles
        
        # start lanelet
        ms_initial = ms_dict[start_timestamp]
        vehicle_initial_pos = (ms_initial.x, ms_initial.y)

        obj_start = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_initial_pos[0], vehicle_initial_pos[1], 0), [])
        obj_start_matches = lanelet2_matching.getDeterministicMatches(self._laneletmap, obj_start, 0.2)
        obj_start_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(obj_start_matches, self._traffic_rules)

        if len(obj_start_matches_rule_compliant) > 0:
            # first matching principle
            start_lanelet = obj_start_matches_rule_compliant[0].lanelet

        # end lanelet
        ms_terminal = ms_dict[end_timestamp]
        vehicle_terminal_pos = (ms_terminal.x, ms_terminal.y)
        
        obj_end = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_terminal_pos[0], vehicle_terminal_pos[1], 0), [])
        obj_end_matches = lanelet2_matching.getDeterministicMatches(self._laneletmap, obj_end, 0.2)
        obj_end_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(obj_end_matches, self._traffic_rules)

        if obj_end_matches_rule_compliant:
            end_lanelet = obj_end_matches_rule_compliant[0].lanelet

        return start_lanelet, end_lanelet

    def get_start_end_lanelet_from_ms_dict_with_min_heading(self, ms_dict, start_timestamp, end_timestamp):
        start_lanelet = None
        end_lanelet = None
        
        # start lanelet
        ms_initial = ms_dict[start_timestamp]
        vehicle_initial_pos = (ms_initial.x, ms_initial.y)
        vehicle_initial_velocity = (ms_initial.vx, ms_initial.vy)

        obj_start = lanelet2_matching.Object2d(1, lanelet2_matching.Pose2d(vehicle_initial_pos[0], vehicle_initial_pos[1], 0), [])
        obj_start_matches = lanelet2_matching.getDeterministicMatches(self._laneletmap, obj_start, 0.2)
        obj_start_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(obj_start_matches, self._traffic_rules)
        if len(obj_start_matches_rule_compliant) > 0:
            # similar min heading error matching principle
            min_heading_error = 90
            start_lanelet_index = 0

            for index, match in enumerate(obj_start_matches_rule_compliant):
                match_lanelet = match.lanelet
                heading_error = geometry.get_vehicle_and_lanelet_heading_error(vehicle_initial_pos, vehicle_initial_velocity, match_lanelet, 2)
                if min_heading_error > heading_error:
                    min_heading_error = heading_error
                    start_lanelet_index = index
            start_lanelet = obj_start_matches_rule_compliant[start_lanelet_index].lanelet
        
        # end lanelet
        ms_terminal = ms_dict[end_timestamp]
        vehicle_terminal_pos = (ms_terminal.x, ms_terminal.y)
        vehicle_terminal_velocity = (ms_terminal.vx, ms_terminal.vy)
        
        obj_end = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_terminal_pos[0], vehicle_terminal_pos[1], 0), [])
        obj_end_matches = lanelet2_matching.getDeterministicMatches(self._laneletmap, obj_end, 0.2)
        obj_end_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(obj_end_matches, self._traffic_rules)
        if len(obj_end_matches_rule_compliant) > 0:
            # similar min heading error matching principle
            min_heading_error = 90
            end_lanelet_index = 0

            for index,match in enumerate(obj_end_matches_rule_compliant):
                match_lanelet = match.lanelet
                heading_error = geometry.get_vehicle_and_lanelet_heading_error(vehicle_terminal_pos, vehicle_terminal_velocity, match_lanelet, 2)
                if min_heading_error > heading_error:
                    min_heading_error = heading_error
                    end_lanelet_index = index
            end_lanelet = obj_end_matches_rule_compliant[end_lanelet_index].lanelet

        return start_lanelet, end_lanelet

    def get_route_lanelet(self, start_lanelet, end_lanelet):
        lanelet_list = []
        if start_lanelet.id == end_lanelet.id:
            lanelet_list.append(start_lanelet)
        else:
            # print(start_lanelet.id, end_lanelet.id)
            lanelet_route = self._routing_graph.getRoute(start_lanelet, end_lanelet, 0)
            # print(lanelet_route)
            all_following_lanelet = lanelet_route.fullLane(start_lanelet)
            for lanelet in all_following_lanelet:
                lanelet_list.append(lanelet)
            if lanelet_list[0].id != start_lanelet.id:
                print('error route do not match start lanelet')
            if lanelet_list[-1].id != end_lanelet.id:
                print('error route do not match end lanelet')
                lanelet_list.append(end_lanelet)
        return lanelet_list

    def get_route_from_lanelet(self, route_lanelet, vehicle_start_pos, vehicle_end_pos):
        # we set the max speed of the vehicle as the recommand speed
        recommand_speed = 10 # m/s
        yaw_by_default = 0
        # all centerline points on the whole route
        centerline_point_list = []
        for lanelet in route_lanelet:
            if lanelet is route_lanelet[-1]:
                for index in range(len(lanelet.centerline)):
                    centerline_point_list.append([lanelet.centerline[index].x, lanelet.centerline[index].y, yaw_by_default, recommand_speed, 0]) # recommand_speed = sqrt(recommand_speed**2 + 0**2)
            else:
                for index in range(len(lanelet.centerline)-1):
                    centerline_point_list.append([lanelet.centerline[index].x, lanelet.centerline[index].y, yaw_by_default, recommand_speed, 0])

        # we just need a part of it
        condensed_centerline_point_list = []
        min_distance_with_start = 100
        min_distance_with_end = 100
        for index, point in enumerate(centerline_point_list):
            # find start centerline point's index
            distance_with_start = math.sqrt((point[0] - vehicle_start_pos[0])**2 + (point[1] - vehicle_start_pos[1])**2)
            if distance_with_start < min_distance_with_start:
                min_distance_with_start = distance_with_start
                start_index = index
            # find end centerline point's index
            distance_with_end = math.sqrt((point[0] - vehicle_end_pos[0])**2 + (point[1] - vehicle_end_pos[1])**2)
            if distance_with_end < min_distance_with_end:
                min_distance_with_end = distance_with_end
                end_index = index
        # make sure there are at least two points
        if start_index == end_index:
            end_index += 1

        for index in range(start_index, end_index + 1):
            condensed_centerline_point_list.append(centerline_point_list[index])
        
        # get route from the condensed centerline point list
        route = self.get_route_from_trajectory(trajectory_list=condensed_centerline_point_list)

        return route

    # generate ground truth routes
    def get_ground_truth_route(self, interval_distance=2.):
        ground_truth_route_dict = dict()
        # route_lanelet_dict = dict()

        track_dict = self._map.track_dict
        for vehicle_type in ['ego', 'vdi']:
            for vehicle_id, vehicle_start_end_state in self._start_end_state[vehicle_type].items():
                vehicle_dict = track_dict[vehicle_id]
                # time horizen
                start_timestamp = vehicle_start_end_state[0] # int(start_timestamp_list[0])
                end_timestamp = vehicle_start_end_state[1]
                # get vehicle's trajectory according to start and end timestep
                ms_dict = vehicle_dict.motion_states
                vehicle_trajectory_list = self.get_trajectory_from_ms_dict(ms_dict, start_timestamp, end_timestamp)
                if vehicle_trajectory_list:
                    ms_end = ms_dict[end_timestamp]
                    vehicle_route_list = self.get_route_from_trajectory(trajectory_list=vehicle_trajectory_list, interval_distance=interval_distance, ms_end=ms_end)
                    ground_truth_route_dict[vehicle_id] = vehicle_route_list

        return ground_truth_route_dict #, route_lanelet_dict
        
    def get_route_from_trajectory(self, trajectory_list, interval_distance=2., ms_end=None):
        # a list [[x, y, point_recommend_speed]]

        # first make them equal distance
        average_trajectory_list = []
        for index, point in enumerate(trajectory_list):
            # first point
            if index == 0:
                average_trajectory_list.append([point[0], point[1]])
            # middle points
            elif index != (len(trajectory_list) - 1):
                point_previous = average_trajectory_list[-1]
                distance_to_previous = math.sqrt((point[0] - point_previous[0])**2 + (point[1] - point_previous[1])**2)
                # distance is fine, just add it to the list
                if distance_to_previous >= 0.75 * interval_distance and distance_to_previous <= 1.25 * interval_distance:
                    average_trajectory_list.append([point[0], point[1]])
                # distace is too small, pass
                elif distance_to_previous < 0.75 * interval_distance:
                    continue
                # distance is too big, make it fine
                elif distance_to_previous > 1.25 * interval_distance:
                    ratio = 1.25 * interval_distance / distance_to_previous
                    insert_point_x = point_previous[0] + ratio * (point[0] - point_previous[0])
                    insert_point_y = point_previous[1] + ratio * (point[1] - point_previous[1])
                    average_trajectory_list.append([insert_point_x, insert_point_y])
            # last point
            else:
                point_previous = average_trajectory_list[-1]
                distance_to_previous = math.sqrt((point[0] - point_previous[0])**2 + (point[1] - point_previous[1])**2)
                if point[0:2] == point_previous:
                    if len(average_trajectory_list) > 1:
                        continue
                    else:
                        direction = ms_end.psi_rad
                        point_x =  point_previous[0] + interval_distance * np.cos(direction)
                        point_y =  point_previous[1] + interval_distance * np.sin(direction)
                        point = [point_x, point_y]
                        average_trajectory_list.append([point[0], point[1]])
                else:
                    # if distance too big, make it fine
                    while distance_to_previous > 1.25 * interval_distance:
                        ratio = 1.25 * interval_distance / distance_to_previous
                        insert_point_x = point_previous[0] + ratio * (point[0] - point_previous[0])
                        insert_point_y = point_previous[1] + ratio * (point[1] - point_previous[1])
                        average_trajectory_list.append([insert_point_x, insert_point_y])

                        point_previous = average_trajectory_list[-1]
                        distance_to_previous = math.sqrt((point[0] - point_previous[0])**2 + (point[1] - point_previous[1])**2)

                    average_trajectory_list.append([point[0], point[1]])

        # then the recommend speed value is the nearest trajectory point's speed value
        average_trajectory_with_speed_list = []
        for point in average_trajectory_list:
            min_distance = 100
            min_distance_point = None
            # get closest point in trajectory
            for point_with_speed in trajectory_list:
                distance = math.sqrt((point[0] - point_with_speed[0])**2 + (point[1] - point_with_speed[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    min_distance_point = point_with_speed

            # calculate speed value
            point_speed = math.sqrt(min_distance_point[3] ** 2 + min_distance_point[4] ** 2)
            average_trajectory_with_speed_list.append([point[0], point[1], point_speed])

        return average_trajectory_with_speed_list

    def get_trajectory_from_ms_dict(self, ms_dict, start_timestamp, end_timestamp):
        # a list [[x, y, vehicle_yaw, vehicle_vx, vehicle_vy]...]
        trajectory_list = []
        # sort mc_dict based on time
        sorted_time = sorted(ms_dict)
        for time in sorted_time:
            if time >= start_timestamp and time <= end_timestamp:
                ms = ms_dict[time]
                trajectory_list.append([ms.x, ms.y, ms.psi_rad, ms.vx, ms.vy])
        # make sure the end point and start point's interval distance is long enough
        if trajectory_list: # if vehicle exist in the time horizen
            start_point = [trajectory_list[0][0], trajectory_list[0][1]]
            end_point = [trajectory_list[-1][0], trajectory_list[-1][1]]
            if math.sqrt((start_point[0] - end_point[0])**2 + (start_point[1] - end_point[1])**2) < 2:
                ms = ms_dict[list(ms_dict.keys())[-1]]
                trajectory_list.append([ms.x, ms.y, ms.psi_rad, ms.vx, ms.vy])

        return trajectory_list

class SeverInterface:
    def __init__(self, port):
        # communication related
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.port = port
        url = ':'.join(["tcp://*", str(self.port)])
        self.socket.bind(url)
        self.gym_env = None

        # env statue flag
        self.can_change_track_file_flag = False
        self.env_init_flag = False
        self.scen_init_flag = False
        self.reset_flag = False
        print('docker server established!')
                
    def start_communication(self):
        while not self.socket.closed:
            message = self.socket.recv()
            str_message = bytes.decode(message)
            if str_message == 'close':
                self.socket.close()
                return
            message = eval(str_message)

            # env init
            if message['command'] == 'env_init':
                print('I-SIM initialize...')
                print()
                self.gym_env = InteractionEnv(message['content'])
                self.socket.send_string('env_init_done')
                self.env_init_flag = True

            # change track file
            elif message['command'] == 'track_init':
                print('Vehicles tracks initialize...')
                gt_csv_index = self.gym_env.change_track()
                self.socket.send_string(str(gt_csv_index))
                self.can_change_track_file_flag = False    

            # choose ego & initialize map  
            elif message['command'] == 'scen_init':
                print('Scenario initialize...')
                ego_id_list = self.gym_env.scenario_init()
                self.socket.send_string(str(ego_id_list))
                self.scen_init_flag = True

            # reset
            elif message['command'] == 'reset':
                print('Env reset...')
                observation_dict = self.gym_env.reset()
                if observation_dict is not None:
                    self.reset_flag = True
                    # remove some unuseable item
                    condensed_observation_dict = copy.deepcopy(observation_dict)
                    condensed_observation_dict = self.pop_useless_item(condensed_observation_dict)
                    reset_message = {'observation': condensed_observation_dict, 'reward': 0, 'done': False}
                    self.socket.send_string(str(reset_message)) # 
                else: 
                    self.scen_init_flag = False
                    self.socket.send_string(str(self.reset_flag))

            # step
            elif message['command'] == 'step': 
                # receiving action
                action_dict = dict()
                for ego_id in self.gym_env._ego_vehicle_dict.keys():
                    action_dict[ego_id] = message['content'][ego_id]
                    # vx,vy,x,y,psi_rad
                prediction = message['content']['prediction']
                observation_dict, reward_dict, done_dict, aux_info_dict = self.gym_env.step(action_dict, prediction=prediction)

                if False not in done_dict.values(): # all egos are done
                    self.can_change_track_file_flag = True
                    self.scen_init_flag = False
                    self.reset_flag = False

                if observation_dict is not None:
                    condensed_observation_dict = copy.deepcopy(observation_dict)
                    condensed_observation_dict = self.pop_useless_item(condensed_observation_dict)
                    step_message = {'observation': condensed_observation_dict, 
                                    'reward': reward_dict, 
                                    'done': done_dict, 
                                    'aux_info': aux_info_dict}
                    self.socket.send_string(str(step_message))

            else:
                print('env_init:', self.env_init_flag)
                print('scen_init:', self.scen_init_flag)
                print('can_change_track_file', self.can_change_track_file_flag)
                print('env_reset:', self.reset_flag)
                self.socket.send_string('null type')

    def pop_useless_item(self, observation):
        # remove some useless item from raw observation, to reduce communication costs
        useless_key = ['reach_goal', 'collision', 'deflection', 'current_bound_points', 
                       'trajectory_location', 'trajectory_distance', 'trajectory_speed', 'ego_next_loc', 'target_speed','future_route_points', 
                       'interaction_vehicles_observation', 'distance_from_bound', 'current_speed', 'attention_mask', 'lane_observation']
        observation_key = observation.keys()
        for item in useless_key:
            if item in observation_key:
                observation.pop(item)
        return observation

if __name__ == "__main__":

    # for docker external communication test
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, help="Number of the port (int)", default=None, nargs="?")
    args = parser.parse_args()
    sever = SeverInterface(args.port)
    sever.start_communication()