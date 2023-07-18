import os
import sys
import glob
import argparse
import numpy as np
from vehicle import Vehicle as vehicle
from vru import VRU as vru

from loguru import logger
from tracks_import import read_from_csv
from track_visualizer import TrackVisualizer
import matplotlib.pyplot as plt


def create_args():
    config_specification = argparse.ArgumentParser(description="ParameterOptimizer")
    # --- Input paths ---
    config_specification.add_argument('--input_path', default="../data/",
                                      help="Dir with track files", type=str)
    config_specification.add_argument('--recording_name', default="00",
                                      help="Choose recording name.", type=str)

    # --- Settings ---
    config_specification.add_argument('--scale_down_factor', default=12,
                                      help="Factor by which the tracks are scaled down to match a scaled down image.",
                                      type=float)
    # --- Visualization settings ---
    config_specification.add_argument('--skip_n_frames', default=5,
                                      help="Skip n frames when using the second skip button.",
                                      type=int)
    config_specification.add_argument('--plotLaneIntersectionPoints', default=False,
                                      help="Optional: decide whether to plot the direction triangle or not.",
                                      type=bool)
    config_specification.add_argument('--plotBoundingBoxes', default=True,
                                      help="Optional: decide whether to plot the bounding boxes or not.",
                                      type=bool)
    config_specification.add_argument('--plotDirectionTriangle', default=True,
                                      help="Optional: decide whether to plot the direction triangle or not.",
                                      type=bool)
    config_specification.add_argument('--plotTrackingLines', default=True,
                                      help="Optional: decide whether to plot the direction lane intersection points or not.",
                                      type=bool)
    config_specification.add_argument('--plotFutureTrackingLines', default=True,
                                      help="Optional: decide whether to plot the tracking lines or not.",
                                      type=bool)
    config_specification.add_argument('--showTextAnnotation', default=True,
                                      help="Optional: decide whether to plot the text annotation or not.",
                                      type=bool)
    config_specification.add_argument('--showClassLabel', default=True,
                                      help="Optional: decide whether to show the class in the text annotation.",
                                      type=bool)
    config_specification.add_argument('--showVelocityLabel', default=True,
                                      help="Optional: decide whether to show the velocity in the text annotation.",
                                      type=bool)
    config_specification.add_argument('--showRotationsLabel', default=False,
                                      help="Optional: decide whether to show the rotation in the text annotation.",
                                      type=bool)
    config_specification.add_argument('--showAgeLabel', default=False,
                                      help="Optional: decide whether to show the current age of the track the text annotation.",
                                      type=bool)

    parsed_config_specification = vars(config_specification.parse_args())
    return parsed_config_specification


if __name__ == '__main__':
    config = create_args()

    input_root_path = config["input_path"]
    recording_name = config["recording_name"]

    if recording_name is None:
        logger.error("Please specify a recording!")
        sys.exit(1)

    # Search csv files
    tracks_files = glob.glob(input_root_path + recording_name + "*_tracks.csv")
    static_tracks_files = glob.glob(input_root_path + recording_name + "*_tracksMeta.csv")
    recording_meta_files = glob.glob(input_root_path + recording_name + "*_recordingMeta.csv")
    if len(tracks_files) == 0 or len(static_tracks_files) == 0 or len(recording_meta_files) == 0:
        logger.error("Could not find csv files for recording {} in {}. Please check parameters and path!",
                     recording_name, input_root_path)
        sys.exit(1)

    # Load csv files
    logger.info("Loading csv files {}, {} and {}", tracks_files[0], static_tracks_files[0], recording_meta_files[0])
    tracks, static_info, meta_info = read_from_csv(tracks_files[0], static_tracks_files[0], recording_meta_files[0])
    if tracks is None:
        logger.error("Could not load csv files!")
        sys.exit(1)

    # Load background image for visualization
    background_image_path = input_root_path + recording_name + "_background.png"
    if not os.path.exists(background_image_path):
        logger.warning("No background image {} found. Fallback using a black background.".format(background_image_path))
        background_image_path = None
    config["background_image_path"] = background_image_path

    #visualization_plot = TrackVisualizer(config, tracks, static_info, meta_info)
    #visualization_plot.show()

    frame_limit = 5400
    start_frame = 5100
    i = 0
    ego_veh_list = []
    #TODO
    vru_list = []
    risk_in_frame = []

    min_x = min([track['xCenter'].min() for track in tracks])
    min_y = min([track['yCenter'].min() for track in tracks])
    max_x = max([track['xCenter'].min() for track in tracks])
    max_y = max([track['yCenter'].min() for track in tracks])
    
    parking_vehicles = list(filter(lambda x: x['xCenter'].min() == x['xCenter'].max() and x['yCenter'].min() == x['yCenter'].max(), tracks))
    px_m = meta_info['orthoPxToMeter']
    plt.show()
    for i in np.arange(start_frame, frame_limit):
        print("Current Frame: " + str(i))

        tracks_in_frame = list(filter(lambda x: i in x['frame'], tracks))
        for track in tracks_in_frame:
            object_class = static_info[track['trackId']]['class']
            is_vehicle = object_class in ["car", "truck_bus", "motorcycle"]
            is_vru = object_class in ["bicycle", "pedestrian"]
            
            track_frame = i - track['frame'][0]

            if is_vehicle and track not in parking_vehicles:
                veh = list(filter(lambda x: x.trackId == track['trackId'], ego_veh_list))
                if veh != []:
                    if(len(veh) > 1):
                        print("Warning: more than 1 Vehicle is associated as Ego")
                    veh = list(veh)[0]
                    veh.update_postion(track['xCenter'][track_frame], track['yCenter'][track_frame], track['heading'][track_frame], track['xVelocity'][track_frame], track['yVelocity'][track_frame], track['xAcceleration'][track_frame], track['yAcceleration'][track_frame])
                    veh.clear_other_objects()
                else:
                    veh = vehicle(track['xCenter'][track_frame], track['yCenter'][track_frame], track['heading'][track_frame], track['xVelocity'][track_frame], track['yVelocity'][track_frame], track['xAcceleration'][track_frame], track['yAcceleration'][track_frame], track['length'][track_frame], track['width'][track_frame], track['trackId'], px_m)
                    ego_veh_list.append(veh)
                
                other_tracks = list(filter(lambda x: x['trackId'] != veh.trackId, tracks_in_frame))

                for other_track in other_tracks:
                    other_class = static_info[other_track['trackId']]['class']
                    other_is_vru = other_class in ["bicycle", "pedestrian"]

                    other_track_frame_index = i - other_track['frame'][0]
                    if(other_is_vru):
                        other_vru = list(filter(lambda x: x.trackId == other_track['trackId'], vru_list))
                        if other_vru == []:
                            other_vru = vru(other_track['xCenter'][other_track_frame_index], other_track['yCenter'][other_track_frame_index], other_class, other_track['trackId'], px_m)
                            vru_list.append(other_vru)
                        else:
                            other_vru = other_vru[0]
                            other_vru.update_position(other_track['xCenter'][other_track_frame_index], other_track['yCenter'][other_track_frame_index])
                        veh.add_other_vru(other_vru)
                    else:
                        other_veh = vehicle(other_track['xCenter'][other_track_frame_index], other_track['yCenter'][other_track_frame_index], other_track['heading'][other_track_frame_index], other_track['xVelocity'][other_track_frame_index], other_track['yVelocity'][other_track_frame_index], other_track['xAcceleration'][other_track_frame_index], other_track['yAcceleration'][other_track_frame_index], other_track['length'][other_track_frame_index], other_track['width'][other_track_frame_index], other_track['trackId'], px_m)
                        veh.add_other_veh(other_veh)

                veh.calculate_risk()
        
        ego_veh_list[1].visualize(min_x, max_x, min_y, max_y)
    print("--- Vehicle Statistics ---")
    for veh in ego_veh_list:
        print("Statistics for Veh #" + str(veh.trackId))
        print("VRUs in critical Range: " + str(veh.in_range_counter))
        print("From those, VRUs not in LOS: " + str(veh.risk_counter))
    print("--- VRU Statistics ---")
    for vru in vru_list:
        print("Statistics for VRU #" + str(vru.trackId))
        print("In critical Range: " + str(vru.in_critical_range))
        print("Not in LOS: " + str(vru.risk_counter))
    #         veh = list(filter(lambda x: x.trackId == track['trackId'] and is_vehicle, veh_list))
    #         other_veh = list(filter(lambda x: x.trackId != track['trackId'] and is_vehicle, veh_list))
    #         other_vru = list(filter(lambda x: x.trackId != track['trackId'] and is_vru, veh_list))
    #         if is_vehicle:
    #             if veh != []:
    #                 print("Veh already exists")
    #                 veh = list(veh)[0]
    #                 # check index!!!
    #                 veh.update_postion(track['xCenter'][track_frame], track['yCenter'][track_frame], track['heading'][track_frame], track['xVelocity'][track_frame], track['yVelocity'][track_frame], track['xAcceleration'][track_frame], track['yAcceleration'][track_frame])
    #             else:
    #                 print("New vehicle")
    #                 veh = vehicle(track['xCenter'][track_frame], track['yCenter'][track_frame], track['heading'][track_frame], track['xVelocity'][track_frame], track['yVelocity'][track_frame], track['xAcceleration'][track_frame], track['yAcceleration'][track_frame], track['length'][track_frame], track['width'][track_frame], track['trackId'], px_m)
    #                 veh_list.append(veh)

    #             if other_veh != []:
    #                 veh.update_objects(other_veh)

    #         veh.calculate_risk()
            
    #         print("first track frame: " + str(track_frame))
    #         print("Track " + str(track['trackId']) + " first is in frame " + str(i))
        
    # for veh in veh_list:
    #     print("Total risk: " + str(veh.risk_counter) + " In Range: " + str(veh.in_range_counter))

    #plt.show()
    plt.show()