import numpy as np

def calculate_trajectory_location_reward(observation_dict, ego_id):
    ego_trajectory_distance = observation_dict['trajectory_distance'][ego_id][0]

    trajectory_location_reward = 1 - 0.2*ego_trajectory_distance
    
    return trajectory_location_reward

def calculate_speed_reward(observation_dict, ego_id, ego_max_speed, control_steering):
    # ego_target_speed = observation_dict['target_speed'][ego_id][0]
    if control_steering:
        ego_speed = observation_dict['lane_observation'][ego_id][2] # along route direction
    else:
        ego_speed = observation_dict['current_speed'][ego_id][0]
    # print('ego_speed', ego_speed)

    if ego_max_speed != 0:
        if ego_speed <= ego_max_speed:
            speed_reward = ego_speed / ego_max_speed
        else:
            speed_reward = 1 - ((ego_speed - ego_max_speed) / ego_max_speed)
    else:
        speed_reward = -ego_speed
    
    return speed_reward

def calculate_lane_keeping_reward(observation_dict, ego_id):
    ego_x_in_point_axis = observation_dict['lane_observation'][ego_id][0]
    
    ego_speed_along_lane = observation_dict['lane_observation'][ego_id][2]
    current_heading_error = observation_dict['lane_observation'][ego_id][3]
    future_heading_errors = observation_dict['lane_observation'][ego_id][4:]
    
    l_1 = 0.75
    l_2 = 0.75
    lk_reward_current = ego_speed_along_lane * (np.cos(current_heading_error) - l_1*(np.sin(abs(current_heading_error)))) - l_2*(abs(ego_x_in_point_axis))
    # lk_reward_future = np.sum(np.cos(future_heading_errors))*0.25 - l_1*np.sum(np.sin(np.abs(future_heading_errors)))*0.25

    lk_reward = lk_reward_current
    return lk_reward


def calculate_steer_reward(previous_steer, current_steer):

    l_3 = 1
    # steer_reward = -l_4*abs(current_steer - previous_steer)
    steer_reward = -l_3*abs(current_steer)
    
    return steer_reward