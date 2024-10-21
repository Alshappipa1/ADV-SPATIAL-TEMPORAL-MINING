#Reading skeleton data from NTU RGB+D .skeleton files and saving the processed data into .npy files for faster and more efficient loading of the skeleton data for tasks such as action recognition and machine learning tasks.
#the orginal code at: https://github.com/shahroudy/NTURGB-D/tree/master


import numpy as np
import os

def read_skeleton_file(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    
    cursor = 0
    num_frames = int(data[cursor].strip())  # First line is the number of frames
    cursor += 1
    skeleton_data = {'frames': []}
    
    for _ in range(num_frames):
        frame_data = {}
        num_bodies = int(data[cursor].strip())  # Number of bodies in this frame
        cursor += 1
        frame_data['bodies'] = []
        
        for _ in range(num_bodies):
            body_data = {}
            body_info = data[cursor].strip().split()  # Body information line
            body_data['body_id'] = body_info[0]
            cursor += 1
            
            num_joints = int(data[cursor].strip())  # Number of joints in this body
            cursor += 1
            joints = []
            
            for _ in range(num_joints):
                joint_info = list(map(float, data[cursor].strip().split()))
                joint = {
                    'x': joint_info[0],  # 3D coordinates of the joint
                    'y': joint_info[1],
                    'z': joint_info[2],
                    'depth_x': joint_info[3],  # 2D coordinates in depth map
                    'depth_y': joint_info[4],
                    'color_x': joint_info[5],  # 2D coordinates in RGB image
                    'color_y': joint_info[6],
                    'orientation_w': joint_info[7],  # Joint orientation (quaternion)
                    'orientation_x': joint_info[8],
                    'orientation_y': joint_info[9],
                    'orientation_z': joint_info[10],
                    'tracking_state': joint_info[11]  # Tracking state (tracked, inferred, etc.)
                }
                joints.append(joint)
                cursor += 1
            
            body_data['joints'] = joints
            frame_data['bodies'].append(body_data)
        
        skeleton_data['frames'].append(frame_data)
    
    return skeleton_data

def process_skeleton_files(data_dir, save_npy_path):
    if not os.path.exists(save_npy_path):
        os.makedirs(save_npy_path)

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.skeleton'):
            file_path = os.path.join(data_dir, file_name)
            skeleton_data = read_skeleton_file(file_path)
            
            # Save each skeleton file as a numpy array
            npy_file_path = os.path.join(save_npy_path, file_name.replace('.skeleton', '.npy'))
            np.save(npy_file_path, skeleton_data)
            print(f"Processed and saved: {npy_file_path}")


if __name__ == "__main__":
    data_dir = '/home/adel-h/Downloads/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons'  # Replace with the actual path
    save_npy_path = 'file.mpy'  # Replace with the path to save processed .npy files
    
    # Process skeleton files and save them one by one to avoid memory overflow
    process_skeleton_files(data_dir, save_npy_path)

