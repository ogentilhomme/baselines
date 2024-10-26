import os
import shutil
import yaml
import copy
import numpy as np

from projectaria_tools.core import mps, data_provider

def represent_matrix(dumper, data):
    """Represent large matrix lists (e.g., T_imu_cam) in multi-line format."""
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

def represent_single_line_list(dumper, data):
    """Represent smaller lists like intrinsics and distortion_coeffs on a single line."""
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

def custom_string_presenter(dumper, data):
    """Custom presenter for strings to handle strings with spaces properly."""
    if len(data.split()) > 1:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

def numpy_array_representer(dumper, data):
    """Convert NumPy arrays to lists before representation."""
    return dumper.represent_list(data.tolist())

def numpy_array_to_list(data):
    """Convert NumPy arrays to Python lists."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data

def get_aria_imu_calibs():
    gravity = 9.80600
    acc_nd = float(
            0.8e-4 * gravity
        )
    gyro_nd = float(1e-2 * (np.pi / 180.0))
    acc_rw = float(3.5e-5 * gravity * np.sqrt(353))
    gyro_rw = float(1.3e-3 * (np.pi / 180.0) * np.sqrt(116))
    sat_acc = float(8 * gravity)
    sat_gyr = float(1000)
    
    return acc_nd, gyro_nd, acc_rw, gyro_rw, sat_acc, sat_gyr

def get_params(mps_slam_path, 
                     imu_stream_label='imu-right',
                     left_cam_label='camera-slam-left', 
                     right_cam_label='camera-slam-right'):
    
    data_paths = mps.MpsDataPathsProvider(mps_slam_path).get_data_paths()

    online_calibrations = mps.read_online_calibration(data_paths.slam.online_calibrations)
    first_calib = online_calibrations[0]

    imu_calibs = first_calib.imu_calibs
    imu0_calib = None

    for imu_calib in imu_calibs:
        if imu_calib.get_label() == imu_stream_label:
            imu0_calib = imu_calib
            break

    if imu0_calib is None:
        raise ValueError(f'Could not find imu calibration with label {imu_stream_label}')

    cam_calibs = first_calib.camera_calibs
    cam0_calib = None
    cam1_calib = None

    for cam_calib in cam_calibs:
        if cam_calib.get_label() == left_cam_label:
            cam0_calib = cam_calib
        elif cam_calib.get_label() == right_cam_label:
            cam1_calib = cam_calib
                
    if cam0_calib is None or cam1_calib is None:
        raise ValueError(f'Could not find camera calibration with label {left_cam_label} or {right_cam_label}')
    
    T_device_imu = imu0_calib.get_transform_device_imu()
    T_device_cam0 = cam0_calib.get_transform_device_camera()
    T_device_cam1 = cam1_calib.get_transform_device_camera()

    T_imu_device = T_device_imu.inverse()
    T_imu_cam0 = T_device_imu.inverse() @ T_device_cam0
    T_imu_cam1 = T_device_imu.inverse() @ T_device_cam1

    T_imu_device_list = T_imu_device.to_matrix().tolist()
    T_device_imu_list = T_device_imu.to_matrix().tolist()
    T_device_cam0_list = T_device_cam0.to_matrix().tolist()
    T_device_cam1_list = T_device_cam1.to_matrix().tolist()
    T_imu_cam0_list = T_imu_cam0.to_matrix().tolist()
    T_imu_cam1_list = T_imu_cam1.to_matrix().tolist()

    acc_nd, gyro_nd, acc_rw, gyro_rw, sat_acc, sat_gyr = get_aria_imu_calibs()

    camera_params = cam0_calib.projection_params()
    f_x_cam0, f_y_cam0, c_x_cam0, c_y_cam0 = (
        camera_params[0],
        camera_params[0],
        camera_params[1],
        camera_params[2],
    )
    p2_cam0, p1_cam0 = camera_params[-5], camera_params[-6]
    k1_cam0, k2_cam0, _, _, _, _ = camera_params[3:9]

    camera_params = cam1_calib.projection_params()
    f_x_cam1, f_y_cam1, c_x_cam1, c_y_cam1 = (
        camera_params[0],
        camera_params[0],
        camera_params[1],
        camera_params[2],
    )
    p2_cam1, p1_cam1 = camera_params[-5], camera_params[-6]
    k1_cam1, k2_cam1, _, _, _, _ = camera_params[3:9]


    sensors = [
        {
            'id': '',
            'topic': '/imu0',
            'description': "VI-Sensor IMU",
            'sensor_type': 'IMU',
            'sigmas':{
                'acc_noise_density': acc_nd,
                'acc_bias_random_walk_noise_density': acc_rw,
                'gyro_noise_density': gyro_nd,
                'gyro_bias_random_walk_noise_density': gyro_rw,
                'saturation_accel_max_mps2': sat_acc,
                'saturation_gyro_max_radps': sat_gyr,
                'gravity_magnitude_mps2': 9.81
            }
        },
        {
            'id': 'aabb23479caf7592b35518478a2fe08f',#TOCHECK
            'sensor_type': 'ODOMETRY_6DOF',#TOCHECK
            'description': "ROVIOLI in Odometry mode",#TOCHECK
            'topic': '/odometry/maplab_odom_T_M_I', #TOCHECK
            'T_St_Stp1_fixed_covariance': {
                'rows': 6,
                'cols': 6,
                'data':[
                    [0.001,0,0,0,0,0],
                    [0,0.001,0,0,0,0],
                    [0,0,0.001,0,0,0],
                    [0,0,0,0.0001,0,0],
                    [0,0,0,0,0.0001,0],
                    [0,0,0,0,0,0.0001],#TOCHECK
                ]
            }
        },
        {
            'id': '',
            'sensor_type': 'NCAMERA',
            'description': "VISensor - Aria - original_calibration",
            'topic': "",
            'T_G_B_fixed_covariance':{
                'rows': 6,
                'cols': 6,
                'data':[
                    [0.1,0,0,0,0,0],
                    [0,0.1,0,0,0,0],
                    [0,0,0.1,0,0,0],
                    [0,0,0,0.01,0,0],
                    [0,0,0,0,0.01,0],
                    [0,0,0,0,0,0.01]#TOCHECK
                ]
            },  
            'cameras':[
                {
                    'camera':{
                        'id': cam0_calib.get_serial_number(),
                        'sensor_type': 'CAMERA',
                        'description': "Sensor cam0",
                        'topic': '/cam0',
                        'line-delay-nanoseconds': 0,#TOCHECK
                        'image_height': 480,
                        'image_width': 640,
                        'type': 'pinhole',
                        'intrinsics':{
                            'cols': 1,
                            'rows': 4,
                            'data': copy.deepcopy(numpy_array_to_list(np.array([f_x_cam0, f_y_cam0, c_x_cam0, c_y_cam0])))
                        },
                        'distortion':{
                            'type': 'equidistant',#TOCHECK
                            'parameters':{
                                'cols': 1,
                                'rows': 4,
                                'data': copy.deepcopy(numpy_array_to_list(np.array([k1_cam0, k2_cam0, p1_cam0, p2_cam0])))
                            }
                        }
                    },
                    'T_B_C':{
                        'cols': 4,
                        'rows': 4,
                        'data': T_imu_cam0_list
                    }
                },
                {
                    'camera':{
                        'id': cam1_calib.get_serial_number(),
                        'sensor_type': 'CAMERA',
                        'description': "Sensor cam1",
                        'topic': '/cam1',
                        'line-delay-nanoseconds': 0,#TOCHECK
                        'image_height': 480,
                        'image_width': 640,
                        'type': 'pinhole',#TOCHECK
                        'intrinsics':{
                            'cols': 1,
                            'rows': 4,
                            'data': copy.deepcopy(numpy_array_to_list(np.array([f_x_cam1, f_y_cam1, c_x_cam1, c_y_cam1])))
                        },
                        'distortion':{
                            'type': 'equidistant',#TOCHECK
                            'parameters':{
                                'cols': 1,
                                'rows': 4,
                                'data': copy.deepcopy(numpy_array_to_list(np.array([k1_cam1, k2_cam1, p1_cam1, p2_cam1])))
                            }
                        },
                        'T_B_C':{
                            'cols': 4,
                            'rows': 4,
                            'data': T_imu_cam1_list
                        }
                    }
                }
            ]
        }       
    ]

    identity = np.eye(4)
    extrinsics = [
        {
            'sensor_id': '',#NCAMERA
            'base_sensor_id': '',#IMU
            'T_B_S':{
                'cols': 4,
                'rows': 4,
                'data':copy.deepcopy(identity)
            }
        },
        {
            'sensor_id': '',#IMU
            'base_sensor_id': '',#IMU
            'T_B_S':{
                'rows': 4,
                'cols': 4,
                'data':T_device_imu_list
            }
        },
        {
            'sensor_id': '',#OBOMETRY
            'base_sensor_id': '',#IMU
            'T_B_S':{
                'rows': 4,
                'cols': 4,
                'data':copy.deepcopy(identity)
            }
        }
    ]

    data = {
        'sensors': sensors,
        'extrinsics': extrinsics
    }

    return data

def write_aria_yaml(data, filepath):
    
    with open(filepath, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)   

if __name__ == "__main__":

    seq_name = "3cp_eth"

    home = f'/local/home/ogentilhomme/Documents/baselines/datasets/{seq_name}'
    slam_folder = f'mps_{seq_name}_vrs/'
    mps_folder = os.path.join(home, slam_folder)
    print("Mps folder",mps_folder)

    yaml.add_representer(list, represent_matrix)  # For multi-line matrices
    yaml.add_representer(str, custom_string_presenter)  # For strings
    yaml.add_representer(np.ndarray, numpy_array_representer)  # For NumPy arrays
    # yaml.add_representer(list, represent_single_line_list) 

    configs_folder = f'/local/home/ogentilhomme/Documents/baselines/aria_configs/maplab'
    seq_configs_folder = os.path.join(configs_folder, seq_name)
    os.makedirs(seq_configs_folder, exist_ok=True)

    data = get_params(mps_folder)
    output_file = os.path.join(seq_configs_folder, 'output_aria.yaml')
    write_aria_yaml(data, output_file)
