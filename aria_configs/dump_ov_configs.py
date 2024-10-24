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
    
    return acc_nd, gyro_nd, acc_rw, gyro_rw

def save_yaml_file(data, filepath):
    with open(filepath, 'w') as outfile:
        outfile.write("%YAML:1.0\n")
    
        yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)


def get_kalibr_imu_params(mps_slam_path, imu_stream_label='imu-right'):
    data_paths = mps.MpsDataPathsProvider(mps_slam_path).get_data_paths()

    kalibr_imu_chain_params = {}
    kalibr_imu_chain_params['imu0'] = {} # working with one imu

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
    
    T_device_imu = imu0_calib.get_transform_device_imu()
    T_imu_device = T_device_imu.inverse()
    T_imu_device_list = T_imu_device.to_matrix().tolist()
    kalibr_imu_chain_params['imu0']['T_i_b'] = T_imu_device_list

    acc_nd, gyro_nd, acc_rw, gyro_rw = get_aria_imu_calibs()
    kalibr_imu_chain_params['imu0']['accelerometer_noise_density'] = acc_nd
    kalibr_imu_chain_params['imu0']['gyroscope_noise_density'] = gyro_nd
    kalibr_imu_chain_params['imu0']['accelerometer_random_walk'] = acc_rw
    kalibr_imu_chain_params['imu0']['gyroscope_random_walk'] = gyro_rw
    kalibr_imu_chain_params['imu0']['rostopic'] = '/imu0'
    kalibr_imu_chain_params['imu0']['update_rate'] = float(1000.0)
    kalibr_imu_chain_params['imu0']['time_offset'] = float(0.0)
    kalibr_imu_chain_params['imu0']['model'] = 'kalibr'

    identity = np.eye(3)
    kalibr_imu_chain_params['imu0']['Tw'] = copy.deepcopy(identity)
    kalibr_imu_chain_params['imu0']['R_IMUtoGYRO'] = copy.deepcopy(identity)
    kalibr_imu_chain_params['imu0']['Ta'] = copy.deepcopy(identity)
    kalibr_imu_chain_params['imu0']['R_IMUtoACC'] = copy.deepcopy(identity)

    zeroes = np.zeros((3, 3))
    kalibr_imu_chain_params['imu0']['Tg'] = zeroes
    
    return kalibr_imu_chain_params



def get_kalibr_cam_params(mps_slam_path, left_cam_label='camera-slam-left', right_cam_label='camera-slam-right', imu_stream_label='imu-right'):
    data_paths = mps.MpsDataPathsProvider(mps_slam_path).get_data_paths()

    kalibr_cam_chain_params = {}
    kalibr_cam_chain_params['cam0'] = {}
    kalibr_cam_chain_params['cam1'] = {}

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
    
    T_device_imu = imu0_calib.get_transform_device_imu()

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
    
    T_device_cam0 = cam0_calib.get_transform_device_camera()
    T_device_cam1 = cam1_calib.get_transform_device_camera()

    T_imu_cam0 = T_device_imu.inverse() @ T_device_cam0
    T_imu_cam1 = T_device_imu.inverse() @ T_device_cam1

    T_imu_cam0_list = T_imu_cam0.to_matrix().tolist()
    T_imu_cam1_list = T_imu_cam1.to_matrix().tolist()

    kalibr_cam_chain_params['cam0']['T_imu_cam'] = T_imu_cam0_list
    kalibr_cam_chain_params['cam1']['T_imu_cam'] = T_imu_cam1_list

    kalibr_cam_chain_params['cam0']['rostopic'] = '/cam0/image_raw'
    kalibr_cam_chain_params['cam1']['rostopic'] = '/cam1/image_raw'

    kalibr_cam_chain_params['cam0']['camera_model'] = 'pinhole'
    kalibr_cam_chain_params['cam1']['camera_model'] = 'pinhole'

    kalibr_cam_chain_params['cam0']['distortion_model'] = 'radtan'
    kalibr_cam_chain_params['cam1']['distortion_model'] = 'radtan'

    camera_params = cam0_calib.projection_params()
    f_x, f_y, c_x, c_y = (
        camera_params[0],
        camera_params[0],
        camera_params[1],
        camera_params[2],
    )
    p2, p1 = camera_params[-5], camera_params[-6]
    k1, k2, _, _, _, _ = camera_params[3:9]

    kalibr_cam_chain_params['cam0']['intrinsics'] = copy.deepcopy(numpy_array_to_list(np.array([f_x, f_y, c_x, c_y])))
    kalibr_cam_chain_params['cam0']['distortion_coeffs'] = copy.deepcopy(numpy_array_to_list(np.array([k1, k2, p1, p2])))

    camera_params = cam1_calib.projection_params()
    f_x, f_y, c_x, c_y = (
        camera_params[0],
        camera_params[0],
        camera_params[1],
        camera_params[2],
    )
    p2, p1 = camera_params[-5], camera_params[-6]
    k1, k2, _, _, _, _ = camera_params[3:9]

    kalibr_cam_chain_params['cam1']['intrinsics'] = copy.deepcopy(numpy_array_to_list(np.array([f_x, f_y, c_x, c_y])))
    kalibr_cam_chain_params['cam1']['distortion_coeffs'] = copy.deepcopy(numpy_array_to_list(np.array([k1, k2, p1, p2])))

    resoln = cam0_calib.get_image_size()
    kalibr_cam_chain_params['cam0']['resolution'] = copy.deepcopy(numpy_array_to_list(resoln))
    kalibr_cam_chain_params['cam1']['resolution'] = copy.deepcopy(numpy_array_to_list(resoln))

    kalibr_cam_chain_params['cam0']['cam_overlaps'] = []
    kalibr_cam_chain_params['cam1']['cam_overlaps'] = []

    return kalibr_cam_chain_params

if __name__ == "__main__":
    # seq_type = 'test'
    # seq_name = 'openvins_test'
    seq_name = "3cp_eth"

    home = f'/local/home/ogentilhomme/Documents/baselines/datasets/{seq_name}/'
    slam_folder = f'mps_{seq_name}_vrs/'
    mps_folder = os.path.join(home, slam_folder)
    print(os.path.join(home, slam_folder))

    yaml.add_representer(list, represent_matrix)  # For multi-line matrices
    yaml.add_representer(str, custom_string_presenter)  # For strings
    yaml.add_representer(np.ndarray, numpy_array_representer)  # For NumPy arrays
    yaml.add_representer(list, represent_single_line_list) 

    configs_folder = f'/local/home/ogentilhomme/Documents/baselines/aria_configs/openvins'
    seq_configs_folder = os.path.join(configs_folder, seq_name)
    os.makedirs(seq_configs_folder, exist_ok=True)
    shutil.copy(os.path.join(configs_folder, 'estimator_config.yaml'), os.path.join(seq_configs_folder, 'estimator_config.yaml'))  
    
    kalibr_imu_file = os.path.join(seq_configs_folder, 'kalibr_imu_chain.yaml')
    kalibr_imu_params = get_kalibr_imu_params(mps_folder)
    save_yaml_file(kalibr_imu_params, kalibr_imu_file)

    kalibr_cam_file = os.path.join(seq_configs_folder, 'kalibr_imucam_chain.yaml')
    kalibr_cam_params = get_kalibr_cam_params(mps_folder)
    save_yaml_file(kalibr_cam_params, kalibr_cam_file)

    


