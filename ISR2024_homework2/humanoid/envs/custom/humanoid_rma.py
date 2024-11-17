from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
import torch

from humanoid.envs.custom.humanoid_env import XBotLFreeEnv
from collections import deque


class XBotLRMAEnv(XBotLFreeEnv):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.num_rma_obs = cfg.env.num_rma_obs

    def _init_buffers(self):
        super()._init_buffers()
        self.rma_obs_history = deque(maxlen=self.cfg.env.r_frame_stack)
        for _ in range(self.cfg.env.r_frame_stack):
            self.rma_obs_history.append(torch.zeros(
                self.num_envs, self.cfg.env.single_num_rma_obs, dtype=torch.float, device=self.device))

    def compute_observations(self):
        super().compute_observations()

        # TODO --------------------------------------------------------
        # Please design self.rma_obs_buf, which is a stack of several single rma_obs
        # self.rma_obs_history is a designed deque for rma_obs, you can use it to update self.rma_obs_buf
        # Each single rma_obs should contain anything in privileged_obs but not in obs, and be scaled same in privileged_obs
        # 1. dof difference to reference, 12-dim
        # 2. base linear velocity, 3-dim
        # 3. push force, 2-dim
        # 4. push torque, 3-dim
        # 5. environment friction, 1-dim
        # 6. body mass, 1-dim
        # 7. stance mask, 2-dim
        # 8. contact mask, 2-dim
        # 9. heights, if self.cfg.terrain.measure_heights

        # Please refer to self.privileged_obs_buf, the implementation is basically the same.

        self.rma_obs_buf = torch.cat((
            self.dof_pos - self.ref_dof_pos, # 12-dim
            self.base_lin_vel * self.obs_scales.lin_vel, # 3-dim
            self.rand_push_force[:, :2], # 2-dim
            self.rand_push_torque, # 3-dim
            self.env_frictions, # 1-dim
            self.body_mass / 30, # 1-dim (scaled)
            self._get_gait_phase(),
            (self.contact_forces[:, self.feet_indices, 2] > 5.),
        ), dim=-1)
        
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.rma_obs_buf = torch.cat((self.rma_obs_buf, heights), dim=-1)
        
        self.rma_obs_history.append(self.rma_obs_buf)
        rma_obs_all = torch.stack([self.rma_obs_history[i] # N, T, K
                                   for i in range(self.rma_obs_history.maxlen)], dim=1)
        
        self.rma_obs_buf = rma_obs_all.reshape(self.num_envs, -1) # N, T*K
        # ------------------------------------------------------------

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        super().step(actions)
        clip_obs = self.cfg.normalization.clip_observations
        self.rma_obs_buf = torch.clip(self.rma_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rma_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # TODO -------------------------------------------------------
        # Please reset the rma obs of envs with env_ids
        # Refer to the base method
        for i in range(self.rma_obs_history.maxlen):
            self.rma_obs_history[i][env_ids] *= 0
        # You only need to change self.rma_obs_history
        # ------------------------------------------------------------

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def get_rma_observations(self):
        return self.rma_obs_buf