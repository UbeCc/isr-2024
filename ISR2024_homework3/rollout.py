from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG, ROBOT_PORTS # must import first

import os
import cv2
import torch
import pickle
import argparse
from time import time
from training.utils import *

from homework1.gym_env import PandaGymEnv



# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='task1')
args = parser.parse_args()
task = args.task

# config
cfg = TASK_CONFIG
policy_config = POLICY_CONFIG
train_cfg = TRAIN_CONFIG
device = os.environ['DEVICE']

if __name__ == "__main__":
    # make env
    panda_env = PandaGymEnv()

    # load the policy
    class PolicyConfig:
        def __init__(self, policy_config):
            for k, v in policy_config.items():
                setattr(self, k, v)
    policy_config = PolicyConfig(policy_config)
    ckpt_path = os.path.join(train_cfg['checkpoint_dir'], train_cfg['eval_ckpt_name'])
    policy = make_policy(policy_config.policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    print(loading_status)
    policy.to(device)
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(train_cfg['checkpoint_dir'], f'Lego_0/dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # action normalization
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    query_frequency = policy_config.num_queries
    if policy_config.temporal_agg:
        query_frequency = 1
        num_queries = policy_config.num_queries #100 or 50


    # start rollouts
    n_rollouts = 1
    for i in range(n_rollouts):
        ### evaluation loop
        if policy_config.temporal_agg:
            all_time_actions = torch.zeros([cfg['episode_len'], cfg['episode_len']+num_queries, cfg['state_dim']]).to(device)
        qpos_history = torch.zeros((1, cfg['episode_len'], cfg['state_dim'])).to(device)
        with torch.inference_mode():
             # init buffers
            obs_replay = [[], []]
            action_replay = []
            rgb0, rgb1, qpos = panda_env.reset()
            obs_replay[0].append(rgb0)
            obs_replay[1].append(rgb1)

            for t in range(200):
                
                # curr qpos
                qpos_numpy = np.array(qpos)
                print('qpos', qpos)
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                qpos_history[:, t] = qpos

                # curr_image
                rgb0 = rearrange(rgb0, 'h w c -> c h w')
                rgb1 = rearrange(rgb1, 'h w c -> c h w')
                curr_image = np.stack([rgb0, rgb1], axis=0)
                curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)

                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)
                if policy_config.temporal_agg:                    
                    # TODO: implement action chunking along the time dimension
                    # action is a weighted sum of all output actions in the past with exponential decay
                    # 
                    # action = SUM( exp(-k * t) * action_t ) / SUM( exp(-k * t) ) with k=0.01
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                print("action", action)
                rgb0, rgb1, qpos = panda_env.step(action)
                obs_replay[0].append(rgb0)
                obs_replay[1].append(rgb1)

            mediapy.write_video(f'test/{i}_rgb0.mp4',obs_replay[0])
            mediapy.write_video(f'test/{i}_rgb1.mp4', obs_replay[1])