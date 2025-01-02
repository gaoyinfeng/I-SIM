
import numpy as np
import math
import time
from collections import deque

import geometry
from utils import tracks_vis
from utils.dataset_types import Track, MotionState, Action

class ControlledVehicle:
    def __init__(self, start_end_state, delta_time, discrete_action_num, max_speed=9., consider_jerk_threshold=False):
        # initialize motion state and control
        self._current_state = MotionState(start_end_state[4].time_stamp_ms)  # MotionState type (time_stamp_ms, x, y, vx, vy, psi_rad)
        self._action_for_current_state = Action(self._current_state.time_stamp_ms)

        # vehicle shape
        self._length = start_end_state[2]
        self._width = start_end_state[3]
        # control frequency
        self._dt_in_millisecond = delta_time # 100 ms
        self._dt_in_second = self._dt_in_millisecond / 1000.0
        # discrete create_action space
        self._discrete_action_num = discrete_action_num
        # bicycle model motion parameters
        self._acc_norm_scale = 3 # m/s^2
        self._dec_normalize_scale = 3 # m/s^2
        self._steering_norm_scale = 30 # degree
        self._max_speed = max_speed


        # PID controller
        args_lateral_dict = {'K_P': 1.4,
                              'K_D': 0.05,
                              'K_I': 0.25,
                              'dt': self._dt_in_second}
        args_longitudinal_dict = {'K_P': 1.0,
                                  'K_D': 0,
                                  'K_I': 0.05,
                                  'dt': self._dt_in_second}
        self._pid_controller = VehiclePIDController(args_lateral=args_lateral_dict, 
                                                   args_longitudinal=args_longitudinal_dict)

    @property
    def width(self):
        return self._width

    @property
    def length(self):
        return self._length

    def reset_state(self, start_state):
        self._current_state.time_stamp_ms = start_state.time_stamp_ms
        self._current_state.x = start_state.x
        self._current_state.y = start_state.y
        self._current_state.vx = start_state.vx
        self._current_state.vy = start_state.vy
        self._current_state.psi_rad = start_state.psi_rad

    def step_discret_action(self, action_list, next_waypoint_position):
        if action_list[0] == -100.:
            acc_norm = 'stop'
            steering_norm = 0
        else:
            speed_value_intervel = self._max_speed / (self._discrete_action_num - 1)
            target_speed_value = action_list[0] * speed_value_intervel
            acc_norm = self._pid_controller.run_lon_step(math.sqrt(self._current_state.vx**2 + self._current_state.vy**2), target_speed_value)
            steering_norm = self._pid_controller.run_lat_step(current_position=[self._current_state.x,self._current_state.y], 
                                                                    waypoint_position=next_waypoint_position, current_direction=[math.cos(self._current_state.psi_rad), math.sin(self._current_state.psi_rad)])
        return self.update_state(acc_norm, steering_norm)

    def step_continuous_action(self, action_list, next_waypoint_position=None):
        if next_waypoint_position: # using fixed pid controller controls steering
            if action_list[0] == -100.: # stop mode
                acc_norm = 'stop'
                steering_norm = 0
            else:
                target_speed_value = action_list[0] * self._max_speed
                acc_norm = self._pid_controller.run_lon_step(math.sqrt(self._current_state.vx**2 + self._current_state.vy**2), target_speed_value)
                steering_norm = self._pid_controller.run_lat_step(current_position=[self._current_state.x,self._current_state.y], 
                                                                        waypoint_position=next_waypoint_position, current_direction=[math.cos(self._current_state.psi_rad), math.sin(self._current_state.psi_rad)])
        else: # controls steering by algorithm
            acc_norm, steering_norm = action_list
        return self.update_state(acc_norm, steering_norm)

    def update_motion_state(self, x, y, vx, vy, psi):
        self._current_state.x, self._current_state.y = x, y
        self._current_state.vx, self._current_state.vy = vx, vy
        self._current_state.psi_rad = psi
        self._current_state.time_stamp_ms += self._dt_in_millisecond

    def update_action(self, acc_norm, steering_norm):
        self._action_for_current_state.acc = acc_norm
        self._action_for_current_state.steering = steering_norm
        self._action_for_current_state.time_stamp_ms += self._dt_in_millisecond

    def update_state(self, acc_norm, steering_norm):
        # scale acceleration
        if acc_norm == 'stop':
            acc = -10000
        else:
            if acc_norm >= 0:
                acc = acc_norm * self._acc_norm_scale
            else:
                acc = acc_norm * self._dec_normalize_scale
        
        # scale steering angle, and transform it to rad
        steering = steering_norm * self._steering_norm_scale
        steering_rad = math.radians(steering)

        current_pos = np.array([self._current_state.x,self._current_state.y])
        current_speed_value = math.sqrt(self._current_state.vx**2 + self._current_state.vy**2)
        # current_direction = np.array([math.cos(self._current_state.psi_rad), math.sin(self._current_state.psi_rad)])

        # using bicycle model to calculate states
        wheelbase_scale = 0.6
        wheelbase = self._length * wheelbase_scale
        gravity_core_scale = 0.4
        f_len = wheelbase * gravity_core_scale
        r_len = wheelbase - f_len

        beta = math.atan((r_len / (r_len + f_len)) * math.tan(steering_rad))
        new_pos_x = current_pos[0] + current_speed_value * math.cos(self._current_state.psi_rad + beta) * self._dt_in_second
        new_pos_y = current_pos[1] + current_speed_value * math.sin(self._current_state.psi_rad + beta) * self._dt_in_second
        new_pos = np.array([new_pos_x, new_pos_y])

        new_psi_rad = self._current_state.psi_rad + (current_speed_value / r_len) * math.sin(beta) * self._dt_in_second
        if acc > 0 and current_speed_value >= 15:
            new_velocity = [self._current_state.vx, self._current_state.vy]
        else:
            new_speed_value = current_speed_value + acc * self._dt_in_second
            if new_speed_value < 0:
                new_speed_value = 0
            new_velocity = [new_speed_value * math.cos(new_psi_rad), new_speed_value * math.sin(new_psi_rad)]
        
        # update global variable
        self.update_motion_state(new_pos[0], new_pos[1], new_velocity[0], new_velocity[1], new_psi_rad)
        self.update_action(acc_norm, steering_norm)

        return self._current_state, self._action_for_current_state
        
class VehiclePIDController(object):
    def __init__(self, args_lateral, args_longitudinal, jerk_threshold=None, offset=0):
        self._lon_controller = PIDLongitudinalController(**args_longitudinal)
        self._lat_controller = PIDLateralController(offset, **args_lateral)

    def run_lon_step(self, current_speed, target_speed):
        acceleration = self._lon_controller.run_step(current_speed, target_speed)
        return acceleration
    
    def run_lat_step(self, current_position, waypoint_position, current_direction):
        current_steering = self._lat_controller.run_step(current_position, waypoint_position, current_direction)
        return current_steering
    

class PIDLongitudinalController():

    def __init__(self, K_P, K_D, K_I, dt):

        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, current_speed, target_speed, debug=False):

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):

        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

class PIDLateralController():

    def __init__(self, offset, K_P, K_D, K_I, dt):

        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._offset = offset
        self._e_buffer = deque(maxlen=10)

    def run_step(self, current_position, waypoint_position, current_direction):

        return self._pid_control(waypoint_position, current_position, current_direction)

    def _pid_control(self, waypoint_position, current_position, current_direction):

        # Get the ego's location and forward vector
        ego_loc_x = current_position[0]
        ego_loc_y = current_position[1]

        v_vec = np.array([current_direction[0], current_direction[1], 0.0])

        # Get the vector vehicle-target_wp
        w_loc_x = waypoint_position[0]
        w_loc_y = waypoint_position[1]

        w_vec = np.array([w_loc_x - ego_loc_x,
                          w_loc_y - ego_loc_y,
                          0.0])

        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)
