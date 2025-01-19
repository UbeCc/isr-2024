import os
# fallback to cpu if mps is not available for specific operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
import torch

# data directory
DATA_DIR = 'demo'

# checkpoint directory
CHECKPOINT_DIR = 'checkpoints/panda'

# device
device = 'cpu'
if torch.cuda.is_available(): device = 'cuda'
#if torch.backends.mps.is_available(): device = 'mps'
os.environ['DEVICE'] = device

# robot port names
ROBOT_PORTS = {
    'leader': '/dev/tty.usbmodem57380045221',
    'follower': '/dev/tty.usbmodem57380046991'
}


# task config (you can add new tasks)
TASK_CONFIG = {
    'dataset_dir': DATA_DIR,
    'episode_len': 204,
    'state_dim': 4,
    'action_dim': 4,
    'cam_width': 256, #640,
    'cam_height': 256, #480,
    'camera_names': ['static','wrist'],
    'camera_port': 0
}


# policy config
POLICY_CONFIG = {
    'lr': 1e-5,
    'device': device,
    'num_queries': 40,
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['front'],
    'policy_class': 'ACT',
    'temporal_agg': True,

    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'position_embedding': 'sine',
    'masks': False,
    'dilation': False,
    'dropout': 0.1,
    'pre_norm': False,
    'weight_decay': 1e-4,
}

# training config
TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 2000,
    'batch_size_val': 8,
    'batch_size_train': 8,
    'eval_ckpt_name': 'Lego_0/policy_epoch_400_seed_42.ckpt',
    'checkpoint_dir': CHECKPOINT_DIR
}