import lanelet2
from lanelet2.core import AttributeMap, getId, BasicPoint2d, LineString3d, LineString2d, Point3d, Polygon2d
from lanelet2.geometry import area, inside, distance, intersects2d, length2d, intersectCenterlines2d, follows

import math
import numpy as np

def getAttributes():
    return AttributeMap({"key": "value"})

def is_equal_point(point1,point2):
    if point1.x == point2.x and point1.y == point2.y:
        return True
    else:
        return False

def lanelet_length(lanelet):
    # return centerline length
    length = length2d(lanelet)
    return length

def is_following_lanelet(previous_lanelet, next_lanelet):
    # check whether the next lanelet is the following lanelet of previous lanelet
    return follows(previous_lanelet, next_lanelet)

def localize_transform_list(origin_location, origin_heading, global_location_list, global_heading_list=None):
    localized_location_list = []
    localized_heading_list = []
    for index in range(len(global_location_list)):
        # NOTE: 0.0 is possible, so we use 'None' to check
        if global_heading_list is not None:
            localized_location, localized_heading = localize_transform(origin_location, origin_heading, global_location_list[index], global_heading_list[index])
            localized_heading_list.append(localized_heading)
        else:
            localized_location, _ = localize_transform(origin_location, origin_heading, global_location_list[index])
        localized_location_list.append(localized_location)
    
    return localized_location_list, localized_heading_list

def localize_transform(origin_location, origin_heading, global_location, global_heading=None):
    localized_x = (global_location[1] - origin_location[1])*np.cos(origin_heading) - (global_location[0] - origin_location[0])*np.sin(origin_heading)
    localized_y = (global_location[1] - origin_location[1])*np.sin(origin_heading) + (global_location[0] - origin_location[0])*np.cos(origin_heading)
    localized_location = [localized_x, localized_y]
    # NOTE: 0.0 is possible, so we use 'None' to check
    if global_heading is not None:   
        localized_heading = global_heading - origin_heading
    else:
        localized_heading = None

    return localized_location, localized_heading

def delocalize_transform_list(origin_location, origin_heading, localized_location_list, localized_heading_list=None):
    global_location_list = []
    global_heading_list = []
    for index in range(len(localized_location_list)):
        if localized_heading_list:
            global_location, global_heading = delocalize_transform(origin_location, origin_heading, localized_location_list[index], localized_heading_list[index])
            global_heading_list.append(global_heading)
        else:
            global_location, _ = delocalize_transform(origin_location, origin_heading, localized_location_list[index])
        global_location_list.append(global_location)
    return global_location_list, global_heading_list

def delocalize_transform(origin_location, origin_heading, localized_location, localized_heading=None):
    global_x = localized_location[1] * np.cos(origin_heading) - localized_location[0] * np.sin(origin_heading) + origin_location[0]
    global_y = localized_location[1] * np.sin(origin_heading) + localized_location[0] * np.cos(origin_heading) + origin_location[1]
    global_location = [global_x, global_y]
    if localized_heading:    
        global_heading = localized_heading + origin_heading
    else:
        global_heading = None

    return global_location, global_heading

def vectorize_point_list(point_list, has_heading):
    # make point list become vector list
    vector_list = []
    for index in range(len(point_list) - 1):
        vector = [point_list[index][0], point_list[index][1], point_list[index+1][0], point_list[index+1][1]]
        if has_heading:
            vector.append(point_list[index+1][2])
        vector_list.append(vector)
    return vector_list


def preprocess_centerlines_by_number(centerline_list, vector_num):
    preprocessed_centerline_list = []
    for centerline in centerline_list:
        # get lane length
        lane_length = 0
        for index in range(len(centerline)-1):
            lane_length += math.sqrt((centerline[index][0] - centerline[index+1][0])**2 + (centerline[index][1] - centerline[index+1][1])**2)
        interval_distance = lane_length / vector_num
        # 
        preprocessed_centerline = []
        for index in range(len(centerline)):
            if index == 0: # if is the first point
                preprocessed_centerline.append(centerline[index])
            else:
                # calculate distance between current point and last point
                distance_x = (centerline[index][0] - preprocessed_centerline[-1][0])
                distance_y = (centerline[index][1] - preprocessed_centerline[-1][1])
                distance = math.sqrt(distance_x**2 + distance_y**2)
                # interval distance is almost disired
                # if abs(distance - interval_distance) < 0.1:
                #     preprocessed_centerline.append(centerline[index])
                # interval distance is larger than disired
                if distance > interval_distance: # insert new points
                    newpoints_num = int(math.ceil(distance / interval_distance))
                    for i in range(newpoints_num):
                        pt_x = preprocessed_centerline[-1][0] + distance_x / newpoints_num
                        pt_y = preprocessed_centerline[-1][1] + distance_y / newpoints_num
                        preprocessed_centerline.append([pt_x, pt_y])
                # interval distance is smaller than disired
                elif distance <= interval_distance:
                    pass
        # make the point number of centerline is equal to 11 (10 vector)
        while len(preprocessed_centerline) > 11:
            min_distance = 1e4
            min_index = None
            for index in range(len(preprocessed_centerline)-1):
                distance = math.sqrt((preprocessed_centerline[index][0] - preprocessed_centerline[index+1][0])**2 + (preprocessed_centerline[index][1] - preprocessed_centerline[index+1][1])**2)
                if distance < min_distance:
                    min_distance = distance
                    min_index = index
            new_point = [(preprocessed_centerline[min_index][0]+preprocessed_centerline[min_index+1][0])/2, (preprocessed_centerline[min_index][1]+preprocessed_centerline[min_index+1][1])/2]
            for _ in range(2):
                preprocessed_centerline.pop(min_index)
            preprocessed_centerline.insert(min_index, new_point)
        preprocessed_centerline_list.append(preprocessed_centerline)

    return preprocessed_centerline_list


def preprocess_centerlines_by_distance(centerline_list, interval_distance):
    preprocessed_centerline_list = []
    for centerline in centerline_list:
        preprocessed_centerline = []
        for index in range(len(centerline)):
            if index == 0: # if is the first point
                preprocessed_centerline.append(centerline[index])
            else:
                # calculate distance between current point and last point
                distance_x = (centerline[index][0] - preprocessed_centerline[-1][0])
                distance_y = (centerline[index][1] - preprocessed_centerline[-1][1])
                distance = math.sqrt(distance_x**2 + distance_y**2)
                # interval distance is almost disired
                if abs(distance - interval_distance) < 0.1:
                    preprocessed_centerline.append(centerline[index])
                # interval distance is larger than disired
                elif distance > interval_distance: # insert new points
                    newpoints_num = math.ceil(distance / interval_distance)
                    for i in range(int(newpoints_num)):
                        pt_x = preprocessed_centerline[-1][0] + distance_x / newpoints_num
                        pt_y = preprocessed_centerline[-1][1] + distance_y / newpoints_num
                        preprocessed_centerline.append([pt_x, pt_y])
                # interval distance is smaller than disired
                elif distance <= interval_distance:
                    pass
        preprocessed_centerline_list.append(preprocessed_centerline)

    return preprocessed_centerline_list


def insert_node_to_meet_min_interval(centerline_point_list, min_interval):
    # convert point form
    point_list = []
    if not isinstance(centerline_point_list[0], (list, tuple)):
        for point in centerline_point_list:
            point_list.append([point.x, point.y])
    else:
        for point in centerline_point_list:
            point_list.append([point[0], point[1]])        
    # uniform insert node to meet the minmum interval distance requirement
    extend_centerline_point_list = [] 
    for index in range(len(point_list)-1):
        extend_centerline_point_list.append(point_list[index])
        # print('origin point type:',type(centerline_point_list[index]))
        current_interval_distance =  math.sqrt((point_list[index][0] - point_list[index+1][0])**2 + (point_list[index][1] -point_list[index+1][1])**2)
        if current_interval_distance > min_interval:
            interval_num = math.ceil(current_interval_distance / min_interval)
            interval_point_num = interval_num - 1
            # print('interval_point_num:',interval_point_num)
            for i in range(int(interval_point_num)):
                pt_x = point_list[index][0] + (i+1) * (point_list[index+1][0] - point_list[index][0]) / interval_num
                pt_y = point_list[index][1] + (i+1) * (point_list[index+1][1] - point_list[index][1]) / interval_num
                # interval_point = Point3d(getId(), pt_x, pt_y, 0, getAttributes())
                interval_point = [pt_x, pt_y]
                extend_centerline_point_list.append(interval_point)
        
    extend_centerline_point_list.append(point_list[-1])

    return extend_centerline_point_list


def sat(rect1, rect2):
    # python implement of collision detection
    
    def intervals_overlap(interval1, interval2):
        return interval1[0] <= interval2[1] and interval2[0] <= interval1[1]
    
    def project_polygon(polygon, axis):
        min_projection = np.dot(polygon[0], axis)
        max_projection = min_projection
        for point in polygon[1:]:
            projection = np.dot(point, axis)
            if projection < min_projection:
                min_projection = projection
            elif projection > max_projection:
                max_projection = projection
        return min_projection, max_projection
    
    edges = [rect1[1] - rect1[0], rect1[2] - rect1[1],
             rect2[1] - rect2[0], rect2[2] - rect2[1]]

    axes = [np.array([-edge[1], edge[0]]) for edge in edges]

    for axis in axes:
        min_proj_rect1, max_proj_rect1 = project_polygon(rect1, axis)
        min_proj_rect2, max_proj_rect2 = project_polygon(rect2, axis)

        if not intervals_overlap((min_proj_rect1, max_proj_rect1),
                                 (min_proj_rect2, max_proj_rect2)):
            return False

    return True



def ego_vpi_distance_and_collision(ego_state_dict, vpi_state_dict):
    # calculte the minmum distance between two polygon2d
    ego_polypoint_np = ego_state_dict['polygon']
    ego_polyPoint3d = [Point3d(getId(), p[0], p[1], 0, getAttributes()) for p in ego_polypoint_np]
    ego_poly = Polygon2d(getId(), [ego_polyPoint3d[0], ego_polyPoint3d[1], ego_polyPoint3d[2], ego_polyPoint3d[3]], getAttributes())
    
    vpi_polypoint_np = vpi_state_dict['polygon']
    vpi_polyPoint3d = [Point3d(getId(), p[0], p[1], 0, getAttributes()) for p in vpi_polypoint_np]
    vpi_poly = Polygon2d(getId(), [vpi_polyPoint3d[0], vpi_polyPoint3d[1], vpi_polyPoint3d[2], vpi_polyPoint3d[3]], getAttributes())
    
    # a = intersects2d(ego_poly, vpi_poly)
    # b = sat(ego_polypoint_np, vpi_polypoint_np)
    # if a != b:
    #     print(ego_polypoint_np, vpi_polypoint_np)

    if intersects2d(ego_poly, vpi_poly):
        # print('shared area:', area(ego_poly, vpi_poly))
        return 0, True
    else:   
        poly_distance = distance(ego_poly, vpi_poly)
        return poly_distance, False



""" functions about trajectory """

def get_trajectory_distance(loc, trajectory_location):
    trajectory_distance = math.sqrt((loc[0] - trajectory_location[0]) ** 2 + (loc[1] - trajectory_location[1]) ** 2)
    return trajectory_distance

def get_trajectory_location(state_dict, trajectory_location):
    loc = state_dict['loc']
    heading = state_dict['heading']
    x_in_loc_frame = (trajectory_location[1] - loc[1])*np.cos(heading) - (trajectory_location[0] - loc[0])*np.sin(heading)
    y_in_loc_frame = (trajectory_location[1] - loc[1])*np.sin(heading) + (trajectory_location[0] - loc[0])*np.cos(heading)
    return [x_in_loc_frame, y_in_loc_frame]

def get_trajectory_speed(trajectory_velocity):
    # print('ego_trajectory_velocity:', trajectory_velocity)
    trajectory_vx = trajectory_velocity[0]
    trajectory_vy = trajectory_velocity[1]

    trajectory_speed = math.sqrt(trajectory_vx**2 + trajectory_vy**2)

    return [trajectory_speed]

def get_trajectory_length(trajectory):
    length = 0
    for i in range(len(trajectory) - 1):
        point_prev = trajectory[i][:2]
        point = trajectory[i+1][:2]
        distance = math.sqrt((point[0]-point_prev[0]) ** 2 + (point[1]-point_prev[1]) ** 2)
        length += distance
    return length

def get_track_length(track):
    length = 0
    track_time = sorted(track.motion_states)
    track = track.motion_states
    # print(track)
    for i in range(len(track_time) - 1):
        ms_prev, ms = track[track_time[i]], track[track_time[i+1]]
        point_prev, point = (ms_prev.x, ms_prev.y), (ms.x, ms.y)
        distance = math.sqrt((point[0]-point_prev[0]) ** 2 + (point[1]-point_prev[1]) ** 2)
        length += distance
    return length


""" functions about boundraies """

def get_closet_bound_point(vehicle_loc, left_point_list, right_point_list):
    min_dist_l = 100
    min_dist_r = 100
    closet_left_point_index = 0
    closet_right_point_index = 0
    for index, point in enumerate(left_point_list):
        if isinstance(point, (list, tuple)):
            vehicle_to_point_dist = math.sqrt((point[0] - vehicle_loc[0])**2 + (point[1] - vehicle_loc[1])**2)
        else:
            vehicle_to_point_dist = math.sqrt((point.x - vehicle_loc[0])**2 + (point.y - vehicle_loc[1])**2)
        if min_dist_l > vehicle_to_point_dist:
            min_dist_l = vehicle_to_point_dist
            closet_left_point_index = index
    # closet_left_point = [left_point_list[closet_left_point_index].x, left_point_list[closet_left_point_index].y]
    closet_left_point = [left_point_list[closet_left_point_index][0], left_point_list[closet_left_point_index][1]]

    for index, point in enumerate(right_point_list):
        if isinstance(point, (list, tuple)):
            vehicle_to_point_dist = math.sqrt((point[0] - vehicle_loc[0])**2 + (point[1] - vehicle_loc[1])**2)
        else:
            vehicle_to_point_dist = math.sqrt((point.x - vehicle_loc[0])**2 + (point.y - vehicle_loc[1])**2)
        if min_dist_r > vehicle_to_point_dist:
            min_dist_r = vehicle_to_point_dist
            closet_right_point_index = index
    # closet_right_point = [right_point_list[closet_right_point_index].x, right_point_list[closet_right_point_index].y]
    closet_right_point = [right_point_list[closet_right_point_index][0], right_point_list[closet_right_point_index][1]]

    road_width = math.sqrt((closet_right_point[0] - closet_left_point[0])**2 + (closet_right_point[1] - closet_left_point[1])**2)
    if min_dist_l > road_width:
        min_dist_r = 0
    elif  min_dist_r > road_width:
        min_dist_l = 0

    return [closet_left_point, closet_right_point], [min_dist_l, min_dist_r]

def get_route_bounds_points(route_lanelet, min_interval_distance):
    min_interval_distance = min_interval_distance/2
    left_bound_points = []
    right_bound_points = []
    for lanelet in route_lanelet:
        left_bound = lanelet.leftBound
        right_bound = lanelet.rightBound
        for i in range(len(left_bound)):
            left_bound_points.append(left_bound[i])
        for j in range(len(right_bound)):
            right_bound_points.append(right_bound[j])
    left_bound_points = insert_node_to_meet_min_interval(left_bound_points, min_interval_distance)
    right_bound_points = insert_node_to_meet_min_interval(right_bound_points, min_interval_distance)

    return left_bound_points, right_bound_points

""" functions about lanelet """

def get_vehicle_and_lanelet_heading_error(vehicle_loc, vehicle_heading, current_lanelet, min_interval_distance):
    # first find the closet centerline point
    # this function may need repair as the centerline points do not have uniform distance

    centerline_point_list = current_lanelet.centerline
    extend_centerline_point_list = insert_node_to_meet_min_interval(centerline_point_list, min_interval_distance)
    closet_point_index = get_closet_centerline_point(vehicle_loc, extend_centerline_point_list)

    # calculate the heading along the lanelet
    if closet_point_index < len(extend_centerline_point_list) - 1:
        # lanelet_heading_vector = np.array([extend_centerline_point_list[closet_point_index+1].x - extend_centerline_point_list[closet_point_index].x, extend_centerline_point_list[closet_point_index+1].y - extend_centerline_point_list[closet_point_index].y])
        lanelet_heading_vector = np.array([extend_centerline_point_list[closet_point_index+1][0] - extend_centerline_point_list[closet_point_index][0], extend_centerline_point_list[closet_point_index+1][1] - extend_centerline_point_list[closet_point_index][1]])
    else:
        # lanelet_heading_vector = np.array([extend_centerline_point_list[closet_point_index].x - extend_centerline_point_list[closet_point_index-1].x, extend_centerline_point_list[closet_point_index].y - extend_centerline_point_list[closet_point_index-1].y])
        lanelet_heading_vector = np.array([extend_centerline_point_list[closet_point_index][0] - extend_centerline_point_list[closet_point_index-1][0], extend_centerline_point_list[closet_point_index][1] - extend_centerline_point_list[closet_point_index-1][1]])

    vehicle_heading_vector = np.array([vehicle_heading[0],vehicle_heading[1]])

    L_lanelet_heading = np.sqrt(lanelet_heading_vector.dot(lanelet_heading_vector))
    # print('L_lanelet_heading:',L_lanelet_heading)
    L_vehicle_heading = np.sqrt(vehicle_heading_vector.dot(vehicle_heading_vector))
    # print('L_vehicle_heading:',L_vehicle_heading)
    cos_angle = vehicle_heading_vector.dot(lanelet_heading_vector)/(L_lanelet_heading*L_vehicle_heading)
    # print('cos_angle:',cos_angle)
    cos_angle = np.clip(cos_angle,-1,1)
    radian = np.arccos(cos_angle)
    heading_error =  radian * 180 / np.pi

    return heading_error

""" functions about points """

def get_centerline_point_list_with_heading_and_average_interval(centerline_point_list, min_interval_distance):
    # calculate each point's heading
    previous_point_yaw = None
    centerline_point_list_with_heading = []

    for index, point in enumerate(centerline_point_list):
        if index == (len(centerline_point_list) - 1): # last centerlane point
            point_yaw = centerline_point_list_with_heading[-1][-1]
        else:
            point = centerline_point_list[index]
            point_next = centerline_point_list[index + 1]
            point_vector = np.array((point_next[0] - point[0], point_next[1] - point[1]))

            point_vector_length =  np.sqrt(point_vector.dot(point_vector))
            cos_angle = point_vector.dot(np.array(([1,0])))/(point_vector_length*1) # angle with x locitive (same with carla)
            point_yaw = np.arccos(cos_angle) # rad
            if point_vector[1] < 0: # in the upper part of the axis, yaw is a locitive value
                point_yaw = - point_yaw
            if previous_point_yaw:
                if (abs(point_yaw - previous_point_yaw) > np.pi/2 and abs(point_yaw - previous_point_yaw) < np.pi* (3/2)):
                    continue
                else:
                    previous_point_yaw = point_yaw
                   
            else:
                previous_point_yaw = point_yaw

        centerline_point_list_with_heading.append((point[0], point[1], point_yaw))

    return centerline_point_list_with_heading

def get_route_point_with_heading_from_point_list(predict_route, min_interval_distance):
    point_list = []
    point_previous = None
    for route_speedpoint in predict_route:
        point_x = route_speedpoint[0]
        point_y = route_speedpoint[1]
        if [point_x, point_y] == point_previous:
            continue
        else:
            point_previous = [point_x, point_y]
        point_list.append([point_x, point_y])

    route_point_with_heading = get_centerline_point_list_with_heading_and_average_interval(point_list, min_interval_distance)

    return route_point_with_heading

def get_target_speed_from_point_list(predict_route):
    route_point_speed_list = []
    for route_point in predict_route:
        point_speed = route_point[2]
        route_point_speed_list.append(point_speed)
    
    return route_point_speed_list
    
def get_closet_centerline_point(vehicle_loc, centerline_point_list):
    min_dist = 100
    closet_point_index = 0
    for index, point in enumerate(centerline_point_list):
        if isinstance(point, (list, tuple)):
            vehicle_to_point_dist = math.sqrt((point[0] - vehicle_loc[0])**2 + (point[1] - vehicle_loc[1])**2)
        else:
            vehicle_to_point_dist = math.sqrt((point.x - vehicle_loc[0])**2 + (point.y - vehicle_loc[1])**2)
        if min_dist > vehicle_to_point_dist:
            min_dist = vehicle_to_point_dist
            closet_point_index = index

    return closet_point_index

def get_closet_front_centerline_point(vehicle_loc, centerline_point_list_with_heading):
    min_dist = 100
    closet_point_index = 0
    for index, point in enumerate(centerline_point_list_with_heading):
        vehicle_to_point_dist = math.sqrt((point[0] - vehicle_loc[0])**2 + (point[1] - vehicle_loc[1])**2)
        vehicle_y_in_point_axis = (vehicle_loc[1] - point[1])*np.sin(point[2]) + (vehicle_loc[0] - point[0])*np.cos(point[2])
        if min_dist > vehicle_to_point_dist and vehicle_y_in_point_axis < 0:
            min_dist = vehicle_to_point_dist
            closet_point_index = index

    return closet_point_index

def get_ego_next_loc(ego_state_dict):
    next_point_loc_x = ego_state_dict['loc'][0] + ego_state_dict['speed'] * np.cos(ego_state_dict['heading'])
    next_point_loc_y = ego_state_dict['loc'][1] + ego_state_dict['speed'] * np.sin(ego_state_dict['heading'])
    next_point_loc = (next_point_loc_x, next_point_loc_y)

    next_x_in_ego_frame = (next_point_loc[1] - ego_state_dict['loc'][1])*np.cos(ego_state_dict['heading']) - (next_point_loc[0] - ego_state_dict['loc'][0])*np.sin(ego_state_dict['heading'])
    next_y_in_ego_frame = (next_point_loc[1] - ego_state_dict['loc'][1])*np.sin(ego_state_dict['heading']) + (next_point_loc[0] - ego_state_dict['loc'][0])*np.cos(ego_state_dict['heading'])
    # print('next_loc_from_ego', [next_x_in_ego_frame, next_y_in_ego_frame])
    return [next_x_in_ego_frame, next_y_in_ego_frame]

def get_target_speed_and_future_route_points(vehicle_state, vehicle_route, vehicle_route_target_speed):
    vehicle_loc = vehicle_state['loc']
    vehicle_heading = vehicle_state['heading']
    vehicle_speed = vehicle_state['speed']

    # gather next 5 route points
    future_points = []
    # first find the closet route point, and corresponding target speed
    closet_point_index = get_closet_front_centerline_point(vehicle_loc, vehicle_route)
    target_speed = vehicle_route_target_speed[closet_point_index]
    future_points.append(vehicle_route[closet_point_index])
    # get next 4 points
    require_num = 4
    remain_point_num = len(vehicle_route) - 1 - closet_point_index
    if remain_point_num < require_num:
        for i in range(closet_point_index + 1, len(vehicle_route)):
            future_points.append(vehicle_route[i])
    else:
        for i in range(closet_point_index + 1, closet_point_index + require_num + 1):
            future_points.append(vehicle_route[i])
    
    return target_speed, future_points

def get_lane_observation(state_dict, future_points, control_steering, normalization):
    vehicle_loc = state_dict['loc']
    vehicle_heading = state_dict['heading']
    vehicle_speed = state_dict['speed']
    
    closet_point = future_points[0]

    # get heading errors from next 5 points
    require_num = 5
    ego_heading_error_next_list = []
    for i in range(len(future_points)):
        point_heading = future_points[i][2]
        ego_heading_error = point_heading - vehicle_heading
        ego_heading_error_next_list.append(ego_heading_error)
    if len(ego_heading_error_next_list) < require_num:
        while len(ego_heading_error_next_list) < require_num:
            ego_heading_error_next_list.append(ego_heading_error_next_list[-1])

    if control_steering:
        ego_heading_error_closet = ego_heading_error_next_list[0]
        ego_x_in_point_axis = (vehicle_loc[1] - future_points[0][1])*np.cos(future_points[0][2]) - (vehicle_loc[0] - future_points[0][0])*np.sin(future_points[0][2])
        ego_y_in_point_axis = (vehicle_loc[1] - future_points[0][1])*np.sin(future_points[0][2]) + (vehicle_loc[0] - future_points[0][0])*np.cos(future_points[0][2])
        ego_speed_x_in_point_axis = vehicle_speed * np.sin(ego_heading_error_closet)
        ego_speed_y_in_point_axis = vehicle_speed * np.cos(ego_heading_error_closet)
        if normalization:
            ego_x_in_point_axis /= 2
            ego_speed_x_in_point_axis /= 10
            ego_speed_y_in_point_axis /= 10
            ego_heading_error_closet /= np.pi
            ego_heading_error_next_list = [i/np.pi for i in ego_heading_error_next_list]
        lane_observation = [ego_x_in_point_axis, ego_y_in_point_axis, ego_speed_x_in_point_axis, ego_speed_y_in_point_axis] + ego_heading_error_next_list
    else:
        lane_observation = ego_heading_error_next_list

    return lane_observation

