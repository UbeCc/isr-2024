import os
# import h5py
import torch
import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader

from training.policy import ACTPolicy, CNNMLPPolicy
import mediapy
from decord import VideoReader, cpu

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, idxs, dataset_dir, norm_stats):
        super(EpisodicDataset).__init__()
        self.dataset_dir = dataset_dir
        self.trajs = os.listdir(dataset_dir)
        self.trajs = [os.path.join(dataset_dir, traj) for traj in self.trajs]
        self.trajs = [self.trajs[i] for i in idxs]
        self.norm_stats = norm_stats
        #self.__getitem__(0) # initialize self.is_sim

    def _load_video(self,video_path, frame_ids):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all()
        assert (np.array(frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()  # (frame, h, w, c)
        return frame_data

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, index):
        traj = self.trajs[index]

        # load data
        max_length = 200
        state = np.load(f'{traj}/2.npy')
        episode_len = state.shape[0]
        # rgb0 = mediapy.read_video(f'{traj}/0.mp4')
        rgb0 = self._load_video(f'{traj}/0.mp4', list(range(episode_len)))
        # rgb1 = mediapy.read_video(f'{traj}/1.mp4')
        rgb1 = self._load_video(f'{traj}/1.mp4', list(range(episode_len)))
        
        start_ts = np.random.choice(episode_len)
        image_dict = dict()
        image_dict['rgb0'] = rgb0[start_ts]
        image_dict['rgb1'] = rgb1[start_ts]

        # action
        skip = 1
        action = state[start_ts+skip:]
        action_len = episode_len - start_ts - skip
        padded_action = np.zeros((max_length, 4), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(max_length)
        is_pad[action_len:] = 1

        # state
        qpos = state[start_ts]

        # images
        all_cam_images = []
        for cam_name in image_dict.keys():
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.tensor(qpos,dtype=torch.float32)
        action_data = torch.tensor(padded_action, dtype=torch.float32)
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad



def load_data(dataset_dir, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    trajs = os.listdir(dataset_dir)
    trajs = [os.path.join(dataset_dir, traj) for traj in trajs]
    num_episodes = len(trajs)
    # obtain train test split
    train_ratio = 0.9
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    all_states = []
    for traj in trajs:
        a = np.load(f'{traj}/2.npy')
        all_states.append(a)
    all_states = np.concatenate(all_states, axis=0)
    print("all_state_shape", all_states.shape)
    mean = np.mean(all_states, axis=0)
    std = np.std(all_states, axis=0)
    print("state mean", mean)
    print("state std", std)
    norm_stats = {'action_mean':mean, 'action_std':std, 'qpos_mean':mean, 'qpos_std':std}

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=2, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=2, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats

# def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
#     print(f'\nData from: {dataset_dir}\n')
#     # obtain train test split
#     train_ratio = 0.8
#     shuffled_indices = np.random.permutation(num_episodes)
#     train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
#     val_indices = shuffled_indices[int(train_ratio * num_episodes):]

#     # obtain normalization stats for qpos and action
#     norm_stats = get_norm_stats(dataset_dir, num_episodes)

#     # construct dataset and dataloader
#     train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
#     val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

#     return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise ValueError(f"Unknown policy class: {policy_class}")
    return policy

def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise ValueError(f"Unknown policy class: {policy_class}")
    return optimizer

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def get_image(images, camera_names, device='cpu'):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(images[cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)
    return curr_image

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def pos2pwm(pos:np.ndarray) -> np.ndarray:
    """
    :param pos: numpy array of joint positions in range [-pi, pi]
    :return: numpy array of pwm values in range [0, 4096]
    """ 
    return (pos / 3.14 + 1.) * 2048
    
def pwm2pos(pwm:np.ndarray) -> np.ndarray:
    """
    :param pwm: numpy array of pwm values in range [0, 4096]
    :return: numpy array of joint positions in range [-pi, pi]
    """
    return (pwm / 2048 - 1) * 3.14

def pwm2vel(pwm:np.ndarray) -> np.ndarray:
    """
    :param pwm: numpy array of pwm/s joint velocities
    :return: numpy array of rad/s joint velocities 
    """
    return pwm * 3.14 / 2048

def vel2pwm(vel:np.ndarray) -> np.ndarray:
    """
    :param vel: numpy array of rad/s joint velocities
    :return: numpy array of pwm/s joint velocities
    """
    return vel * 2048 / 3.14
    
def pwm2norm(x:np.ndarray) -> np.ndarray:
    """
    :param x: numpy array of pwm values in range [0, 4096]
    :return: numpy array of values in range [0, 1]
    """
    return x / 4096
    
def norm2pwm(x:np.ndarray) -> np.ndarray:
    """
    :param x: numpy array of values in range [0, 1]
    :return: numpy array of pwm values in range [0, 4096]
    """
    return x * 4096