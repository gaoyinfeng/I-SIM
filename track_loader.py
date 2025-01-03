#-*- coding: UTF-8 -*- 
import os
import pickle
import random
import time
from copy import copy

from geometry import get_track_length
from utils.dataset_reader import read_trajectory, read_tracks, read_pedestrian

# large_scale dataset
class Large_scale_dataset_loader():
    # get all files in the directory according to the scenario setting
    def __init__(self, gt_tracks_dir, ego_max_length=5.5, ego_num=1, eval=False):
        self._gt_tracks_dir = gt_tracks_dir
        self._gt_tracks = os.listdir(self._gt_tracks_dir)
        self._gt_csv_index = None
        self._eval = eval
        self._ego_max_length = ego_max_length
        self._ego_num = ego_num
        self._possible_ego_dict = dict()
        self._possible_ego_id = []

        # remove troubled vehicles which can not be properly controlled
        self._cont_be_used_as_ego_dict = {0: [], 1: [], 2: [11], 3: [81], 4:[], 5: [57], 6:[], 7:[15, 67]}
        # remove small set data from big set data
        small_set_ego_dict = {5: [29, 30, 33, 36, 37, 39, 40, 41]}
        for csv, value in small_set_ego_dict.items():
            self._cont_be_used_as_ego_dict[csv].extend(value)
        # collect tracks that can be used as ego
        self._set_possible_ego_id_dict()
    
    @property
    def track_id(self):
        return self._gt_csv_index

    def _set_possible_ego_id_dict(self):
        tracks_num = len(self._gt_tracks) 
        for csv_index in range(tracks_num):
            track_file_path = os.path.join(self._gt_tracks_dir, "vehicle_tracks_" + str(csv_index).zfill(3) + ".csv")
            track_dict = read_tracks(track_file_path)
            vehicle_id_list = track_dict.keys()
            possible_ego_id = []
            # some vehicles are not appropriate to use as ego
            for vehicle_id in vehicle_id_list:
                length_condition = track_dict[vehicle_id].length <= self._ego_max_length # ego should have a proper vehicle length
                time_condition = (track_dict[vehicle_id].time_stamp_ms_last - track_dict[vehicle_id].time_stamp_ms_first)/1000 >= 5 # ego should exist at least 5 seconds
                track_condition = get_track_length(track_dict[vehicle_id]) >= 20 # ego route should be longer than 20 meters
                if length_condition and time_condition and track_condition:
                    if vehicle_id not in self._cont_be_used_as_ego_dict[csv_index]:
                        possible_ego_id.append(vehicle_id)
            print('csv {} possible ego num: {}'.format(csv_index, len(possible_ego_id)))
            self._possible_ego_dict[csv_index] = possible_ego_id

    # if mode is train, then use the former 3/4 files, else the latter 1/4 files
    def _extract_data_from_file(self, random=False):
        tracks_num = len(self._gt_tracks)
        train_num = int(tracks_num * 0.75)
        # eval_num = tracks_num - train_num

        if not self._eval:
            # choose randomly from all tracks
            if random:
                self._gt_csv_index = random.randint(0, train_num - 1)
            # choose in turn
            else:
                if self._gt_csv_index is None:
                    self._gt_csv_index = 0 
                else:
                    if not self._possible_ego_id: 
                        self._gt_csv_index = 0 if self._gt_csv_index == train_num - 1 else int(self._gt_csv_index) + 1
        else:
            if self._gt_csv_index is None:
                self._gt_csv_index = train_num
            else:
                if not self._possible_ego_id:
                    self._gt_csv_index = train_num if self._gt_csv_index == tracks_num - 1 else int(self._gt_csv_index) + 1

    def _get_possible_ego_id(self):
        return copy(self._possible_ego_dict[self._gt_csv_index])
            
    # change track file randomly in train mode, and in turn in eval mode
    def change_track_file(self):
        self._extract_data_from_file()
        return self._gt_csv_index

    def read_track_file(self, vdi_type, route_type):
        if vdi_type == 'react':
            self._gt_csv_index = '5'
            track_file_path = os.path.join(self._gt_tracks_dir, "vehicle_tracks_" + str(self._gt_csv_index).zfill(3) + ".csv")
            track_dict = read_tracks(track_file_path)
            self._possible_ego_id = [41]
        elif vdi_type == 'record' and route_type == 'ground_truth': # 
            track_file_path = os.path.join(self._gt_tracks_dir, "vehicle_tracks_" + str(self._gt_csv_index).zfill(3) + ".csv")
            track_dict = read_tracks(track_file_path)
            if not self._eval:
                if not self._possible_ego_id:
                    self._possible_ego_id = self._get_possible_ego_id()
            else:
                if not self._possible_ego_id:
                    self._possible_ego_id = self._get_possible_ego_id()
        else:
            print('please check if the vdi_type and route_type are correct')

        return track_dict
    
    def select_ego(self):
        ego_id_list = []
        if not self._eval:
            print('csv {} remain {} train ego id: {}'.format(self._gt_csv_index, len(self._possible_ego_id), self._possible_ego_id))
            ego_id_list.append(self._possible_ego_id[0])
            self._possible_ego_id.pop(0)
        else:
            for _ in range(self._ego_num):
                print('csv {} remain {} eval ego id: {}'.format(self._gt_csv_index, len(self._possible_ego_id), self._possible_ego_id))
                ego_id_list.append(self._possible_ego_id[0])
                self._possible_ego_id.pop(0)

        print('csv:', self._gt_csv_index, 'ego:', ego_id_list)
        return ego_id_list

    def get_start_timestamp(self):
        return None

        
# small_sclae dataset     
class Small_scale_dataset_loader():
    def __init__(self, gt_tracks_dir, ego_max_length=5.5, ego_num=1, eval=False):
        self._gt_tracks_dir = gt_tracks_dir
        self._gt_csv_index = None
        self._eval = eval
        self._ego_max_length = ego_max_length
        self._ego_num = ego_num
        self._possible_ego_dict = dict()
        self._possible_ego_id = []
        self._data = None
        self.data_file = None
        self.small_scale_set_dict = {5: [29, 30, 33, 36, 37, 39, 40, 41]}
        self._current_ego_index = 0
        self._used_ego_ids = set()
        self._set_possible_ego_id_dict()

    @property
    def track_id(self):
        return self._gt_csv_index
    
    def _set_possible_ego_id_dict(self):
        for csv_index, vehicle_ids in self.small_scale_set_dict.items():
            possible_ego_id = []

            track_file_path = os.path.join(self._gt_tracks_dir, "vehicle_tracks_" + str(csv_index).zfill(3) + ".csv")
            track_dict = read_tracks(track_file_path)
            
            for vehicle_id in vehicle_ids:
                if vehicle_id in track_dict:
                    length_condition = track_dict[vehicle_id].length <= self._ego_max_length
                    time_condition = (track_dict[vehicle_id].time_stamp_ms_last - track_dict[vehicle_id].time_stamp_ms_first)/1000 >= 5
                    track_condition = get_track_length(track_dict[vehicle_id]) >= 20
                    
                    if length_condition and time_condition and track_condition:
                        possible_ego_id.append(vehicle_id)
                        
            self._possible_ego_dict[csv_index] = possible_ego_id

            if self._gt_csv_index is None:
                self._gt_csv_index = csv_index
    
    def change_track_file(self):
        self._current_ego_index = 0
        self._used_ego_ids.clear()
        
        if not self._eval:
            if self._gt_csv_index is None:
                self._gt_csv_index = list(self.small_scale_set_dict.keys())[0]
            else:
                self._gt_csv_index = list(self.small_scale_set_dict.keys())[0]
        else:
            if self._gt_csv_index is None:
                self._gt_csv_index = list(self.small_scale_set_dict.keys())[0]
            else:
                self._gt_csv_index = list(self.small_scale_set_dict.keys())[0]
                
        return self._gt_csv_index

    def read_track_file(self, vdi_type, route_type):
        if vdi_type == 'react':
            self._gt_csv_index = '5'
            track_file_path = os.path.join(self._gt_tracks_dir, "vehicle_tracks_" + str(self._gt_csv_index).zfill(3) + ".csv") # 
            track_dict = read_tracks(track_file_path)
            self._possible_ego_id = [41] # force it to be 41
        
        elif vdi_type == 'record' and route_type == 'ground_truth':
            track_file_path = os.path.join(self._gt_tracks_dir, "vehicle_tracks_" + str(self._gt_csv_index).zfill(3) + ".csv")
            track_dict = read_tracks(track_file_path)

        return track_dict

    def select_ego(self):
        ego_id_list = []
        if not self._eval:
            possible_egos = self._possible_ego_dict[self._gt_csv_index]

            if len(self._used_ego_ids) >= len(possible_egos):
                self._used_ego_ids.clear()

            available_egos = [ego for ego in possible_egos if ego not in self._used_ego_ids]
            if available_egos:
                selected_ego = random.choice(available_egos)
                ego_id_list.append(selected_ego)
                self._used_ego_ids.add(selected_ego)
                print("Selected new ego: {}".format(selected_ego))
        else:
            possible_egos = self._possible_ego_dict[self._gt_csv_index]
            selected_ego = possible_egos[self._current_ego_index]
            ego_id_list.append(selected_ego)
            self._current_ego_index = (self._current_ego_index + 1) % len(possible_egos)
            
        return ego_id_list

    def get_start_timestamp(self):
        return None

    def get_ego_routes(self):
        ego_route_dict = dict()
        for ego_id in self.ego_id_list:
            ego_info = self._data['egos_track'][ego_id]
            ego_route_dict[ego_id] = ego_info[1:]
            print("self._data : ",self._data)
        return ego_route_dict
