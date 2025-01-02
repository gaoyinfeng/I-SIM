import math
import random
import numpy as np
from enum import Enum

import shapely.geometry
import shapely.affinity

def detect_lane_obstacle(vdi_info, ego_info, extension_factor=3.5, margin=1.02):
    """
    This function identifies if an obstacle is present in front of the reference actor (only consider ego car now)
    """
    # ego
    ego_location = [ego_info._current_state.x, ego_info._current_state.y]
    ego_shape = [ego_info._length, ego_info._width]
    ego_psi_rad = ego_info._current_state.psi_rad
    ego_bbox = [(ego_shape[0]*math.cos(ego_psi_rad) + ego_shape[1]*math.sin(ego_psi_rad))/2, (ego_shape[0]*math.sin(ego_psi_rad) + ego_shape[1]*math.cos(ego_psi_rad))/2] # polygon extent in x and y axis from center vehicle point (like Carla)

    # vdi
    vdi_location = [vdi_info._current_state.x, vdi_info._current_state.y]
    vdi_shape = [vdi_info._length, vdi_info._width]
    vdi_psi_rad = vdi_info._current_state.psi_rad
    vdi_bbox = [(vdi_shape[0]*math.cos(vdi_psi_rad) + vdi_shape[1]*math.sin(vdi_psi_rad))/2, (vdi_shape[0]*math.sin(vdi_psi_rad) + vdi_shape[1]*math.cos(vdi_psi_rad))/2] # polygon extent in x and y axis from center vehicle point (like Carla)
    
    vdi_vector = np.array([math.cos(vdi_psi_rad), math.sin(vdi_psi_rad)])
    vdi_vector = vdi_vector / np.linalg.norm(vdi_vector) # normlization, maybe surplus
    vdi_vector = vdi_vector * (extension_factor - 1) * vdi_bbox[0]
    fake_vdi_location = vdi_location - vdi_vector # the virtual "head" location of the vehicle as the true location of vdichaoshen 

    # decide whether stop the vdi car
    is_hazard = False
    distance = math.sqrt((fake_vdi_location[0] - ego_location[0])**2 + (fake_vdi_location[1] - ego_location[1])**2)
    if distance < 50:
        overlap_ego = RotatedRectangle(
            ego_location[0], ego_location[1],
            2 * margin * ego_bbox[0], 2 * margin * ego_bbox[1], ego_psi_rad)
        overlap_vdi = RotatedRectangle(
            vdi_location[0], vdi_location[1],
            2 * margin * vdi_bbox[0] * extension_factor, 2 * margin * vdi_bbox[1], vdi_psi_rad)
        overlap_area = overlap_ego.intersection(overlap_vdi).area
        # print("overlap_area: ", overlap_area)
        if overlap_area > 2:
            is_hazard = True

    return is_hazard

class RotatedRectangle(object):

    """
    This class contains method to draw rectangle and find intersection point.
    """

    def __init__(self, c_x, c_y, width, height, angle):
        self.c_x = c_x
        self.c_y = c_y
        self.w = width      # pylint: disable=invalid-name
        self.h = height     # pylint: disable=invalid-name
        self.angle = angle

    def get_contour(self):
        """
        create contour
        """
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w / 2.0, -h / 2.0, w / 2.0, h / 2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.c_x, self.c_y)

    def intersection(self, vpi):
        """
        Obtain a intersection point between two contour.
        """
        return self.get_contour().intersection(vpi.get_contour())


class NpcState(Enum):
    """
    Represents the possible states of a scenario agent
    """
    STARTING = -1
    STOPPING = 0
    GOING = 1

class NpcPolicy(object):

    def __init__(self, agg_prec=0.5):
        self.behavior_state = NpcState.STARTING
        self.actor_behavior = 'go'

        self.stop_count = 0
        self.stop_thresh = 90 # stop steps limitation, about 90*0.1=9s

        self.agg_prec = agg_prec
    
    def get_behavior_type(self):
        return self.actor_behavior
    
    def run_step(self, vdi_info, ego_info):
        vdi_x = vdi_info._current_state.x
        vdi_speed = math.sqrt(vdi_info._current_state.vx**2 + vdi_info._current_state.vy**2)
        vdi_psi_rad = vdi_info._current_state.psi_rad
        
        ego_y = ego_info._current_state.y
        ego_speed = math.sqrt(ego_info._current_state.vx**2 + ego_info._current_state.vy**2)
        ego_psi_rad = ego_info._current_state.psi_rad

        def at_intersection():
            return (1020 > vdi_x > 980)

        def angle_between(v1, v2):
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v1 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / 3.1415926

        # calculat angle between vehicle and ego
        ego_vehicle_vector = [math.cos(ego_psi_rad), math.sin(ego_psi_rad)] # ego.get_transform().get_forward_vector()
        vdi_vehicle_vector = [math.cos(vdi_psi_rad), math.sin(vdi_psi_rad)]
        angle = angle_between(ego_vehicle_vector, vdi_vehicle_vector)

        # detect ego + ego has speed (+ ego location is away from start) 
        if detect_lane_obstacle(vdi_info, ego_info, extension_factor = 1.0, margin = 2) and ego_speed > 1:
            target_speed = 0
            return target_speed

        # change vdi behavior
        if at_intersection() and self.behavior_state == NpcState.STARTING:
            rand = random.uniform(0, 1)
            if angle < 175 and ego_y < 1010:
                if rand < self.agg_prec:
                    self.behavior_state = NpcState.GOING 
                    self.actor_behavior = 'go' 
                else:
                    self.behavior_state = NpcState.STOPPING
                    self.actor_behavior = 'stop'
            else:
                self.behavior_state = NpcState.GOING
                self.actor_behavior = 'go'

        if self.behavior_state == NpcState.STOPPING:
            # ego and target vehicle are all wait at interaction
            if ego_speed < 0.01 and vdi_speed < 0.01:
                self.stop_count += 1
            # going if ego vehicle passes intersection or target vehicle waits too long
            if  ego_y < 986 or (self.stop_count >= self.stop_thresh): 
                self.behavior_state = NpcState.GOING 
                self.actor_behavior = 'go'
            else:
                target_speed = 0
                return target_speed

        # starting and going are the same
        if self.behavior_state == NpcState.STARTING or self.behavior_state == NpcState.GOING:
            target_speed = random.uniform(0.5, 0.8) # 0.7
            return target_speed
        
class IDMPolicy(object):
    def __init__(self, env, target_speed= 8.94,  t_future=[0., 1., 2., 3.],  half_angle=60.):

        self._env = env
        self.t_future = t_future
        self.half_angle = half_angle

        # Default IDM parameters
        assert target_speed>0, 'negative target speed'
        self.v_max = target_speed
        self.a_max = np.array([3.]) # nominal acceleration
        self.tau = 0.5 # desired time headway
        self.b_pref = 2.5 # preferred deceleration
        self.d_min = 3 #minimum spacing
        self.max_pos_error = 2 # m, for matching vehicles to ego path
        self.max_deg_error = 30 # degree, for matching vehicles to ego path

    def forward(self, *args, **kwargs):
        """
        Generate action from underlying environment

        Returns
            action (np.ndarray): action for controlled agent to take
        """
        agent = self._env._agent
        full_state = self._env._env.projected_state.numpy() #(nv, 5)
        ego_state = full_state[agent] # (5,)
        v_ego = ego_state[2]
        v = full_state[:,2:3] # (nv, 1)

        length = 20
        step = 0.1
        x, y = self._env._env._generate_paths(delta=step, n=length/step, is_distance=True)
        heading = to_circle(np.arctan2(np.diff(y), np.diff(x)))

        paths = np.stack([x[:,:-1],y[:,:-1], heading], axis=1) # (nv, 3, (path_length-1))
        ego_path = paths[agent:agent+1] # (1, 3, path_length-1)

        # (x,y,phi) of all vehicles
        poses = np.expand_dims(full_state[:, [0,1,3]], 2) # (nv, 3, 1)

        diff = ego_path - poses
        diff[:, 2, :] = to_circle(diff[:, 2, :])

        # Test if position and heading angle are close for some point on the future vehicle track
        pos_close = np.sum(diff[:, 0:2, :]**2, 1) <= self.max_pos_error**2 # (nv, path_length-1)
        heading_close = np.abs(diff[:, 2, :]) <= self.max_deg_error * np.pi / 180 # (nv, path_length-1)
        # For all vehicles get the path points where they are close to the ego path
        close = np.logical_and(pos_close, heading_close) # (nv, path_length-1)
        close[agent, :] = False # exclude ego agent

        leader = agent
        min_idx = np.Inf
        # Determine vehicle that is closest to ego in terms of path coordinate
        for veh_id in range(len(close)):
            path_idx = np.nonzero(close[veh_id])[0]
            # veh_id is never close to agent
            if len(path_idx) == 0:
                continue
            # first path index where veh_id is close to agent
            elif path_idx[0] < min_idx:
                leader = veh_id
                min_idx = path_idx[0]

        if leader != agent:
            # distance along ego path to point with closest distance
            d = step * min_idx

            # Update environment interaction graph with leader
            self._env._env._graph._neighbor_dict={agent:[leader]}
            self._env._update_graph = True

            delta_v = v_ego - v[leader, 0]
            d_des = self.d_min + self.tau * v_ego + v_ego * delta_v / (2* (self.a_max*self.b_pref)**0.5 )
            d_des = max(d_des, self.d_min)
        else:
            d = np.Inf
            d_des = self.d_min
            self._env._env._graph._neighbor_dict={}

        assert (d_des>= self.d_min)
        action = self.a_max*(1 - (v_ego/self.v_max)**4 - (d_des/d)**2)

        # normalize action to range if env is a NormalizedActionSpace
        if isinstance(self._env, NormalizedActionSpace):
            action = self._env._normalize(action)

        return action

def to_circle(x):
    """
    Casts x (in rad) to [-pi, pi)

    Args:
        x (np.ndarray): (*) input angle (radians)

    Returns:
        y (np.ndarray): (*) x cast to [-pi, pi)
    """
    y = np.remainder(x + np.pi, 2*np.pi) - np.pi
    return y
