import os, time, copy, math
import numpy as np
import pybullet_data
# import pybullet as p
import sys
sys.path.append("homework1")
from core import *
from robot_env import *
import mediapy

class PandaGymEnv():

    def __init__(self, controlMode="position"):
        physicsClientId = p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=physicsClientId)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=physicsClientId)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=physicsClientId)

        self.robot = pandaEnv(physicsClientId, controlMode=controlMode)
        p.setGravity(0, 0, -9.8, physicsClientId=physicsClientId)
        self.physicsClientId = physicsClientId

        self.robot.debug_gui()
        self.robot.pre_grasp()
        for _ in range(120):
            p.stepSimulation(physicsClientId=physicsClientId)
            time.sleep(1 / 240)
        
        self.robot.reset()
        self.robot.pre_grasp()
        self.M = self.robot.get_end_eff_home_trans()
        self.grasped = False
        self.R = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((3.1415926, 0, 0)))).reshape(3, 3)
        # initial theta
        self.thetalist0 = [0] * 7

        self.info_list = [[],[],[]]
        self.sim_step_num = 400


    def update_obs(self):
        rgb = self.robot.static_camera()
        rgb2 = self.robot.panda_camera()
        # rgb = robot.panda_camera()
        # robot_theta, _, _ = self.robot.getJointStates()
        # screw_axis = self.robot.get_screw_axis()
        # T_sb = FK_in_space(self.M, np.array(screw_axis).T, robot_theta)
        # # print("T_sb",T_sb)
        # xyz = T_sb[:3, 3]
        # com_p, com_o, _, _, _, _ = p.getLinkState(self.robot.robot_id, 9, computeForwardKinematics=True)
        # print("9, com_p",com_p)
        # com_p, com_o, _, _, _, _ = p.getLinkState(self.robot.robot_id, 10, computeForwardKinematics=True)
        # print("10, com_p",com_p)
        com_p, com_o, _, _, _, _ = p.getLinkState(self.robot.robot_id, 11, computeForwardKinematics=True)
        # print("11, com_p",com_p)
        xyz = np.array(com_p)
        
        state = np.concatenate([xyz, np.array([self.grasped])])
        # print("action",action)
        self.info_list[0].append(rgb)
        self.info_list[1].append(rgb2)
        self.info_list[2].append(state)


    def gradual_IK(self,thetalist0, P, R):
        # do gradual IK
        T = rp_to_trans(R, P)
        print("target xyz", P)
        screw_axis = self.robot.get_screw_axis()
        thetalist, state = InverseKinematics_in_space(np.array(screw_axis).T, self.M, T, thetalist0, 0.001, 0.0005)

        act = thetalist
        # print(act)
        self.robot.apply_action(act, max_vel=1.0)
        for i in range(self.sim_step_num):
            p.stepSimulation(physicsClientId=self.physicsClientId)
            if i%10 == 0:
                com_p, com_o, _, _, _, _ = p.getLinkState(self.robot.robot_id, 11, computeForwardKinematics=True)
                xyz = np.array(com_p)
                if np.linalg.norm(np.array(P)-np.array(xyz)) < 0.002:
                    print("already reach the target pose")
                    break
        self.update_obs()
        # raise NotImplementedError
            # time.sleep(1 / 240)

        # rgb, _, _ = render(physicsClientId=physicsClientId)

        self.update_obs()

        M_new = self.robot.cal_eff_trans()
        return M_new, act
    
    def step(self,action):
        xyz, grasp = action[:3], action[3]
        _, self.thetalist0 = self.gradual_IK(self.thetalist0, xyz, self.R)

        if grasp>0.8 and self.grasped is  False:
            self.robot.pre_grasp()
            step_sim(0.5, physicsClientId=self.physicsClientId)
            print("---Grasping!---")
            self.robot.grasp()
            step_sim(0.5, physicsClientId=self.physicsClientId)
            self.grasped = True
        
        self.update_obs()

        
        return self.info_list[0][-1], self.info_list[1][-1], self.info_list[2][-1]

    def reset(self):
        self.robot.reset()
        self.robot.pre_grasp()
        self.M = self.robot.get_end_eff_home_trans()
        self.grasped = False
        self.R = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((3.1415926, 0, 0)))).reshape(3, 3)
        # initial theta
        self.thetalist0 = [0] * 7
        self.info_list = [[],[],[]]

        self.update_obs()

        return self.info_list[0][-1], self.info_list[1][-1], self.info_list[2][-1]

    def save_video(self, dir):
        os.makedirs(dir, exist_ok=True)
        mediapy.write_video(f"{dir}/0.mp4", self.info_list[0], fps=20)
        mediapy.write_video(f"{dir}/1.mp4", self.info_list[1], fps=20)




if __name__ == "__main__":
    # test
    
    env = PandaGymEnv()
    x = env.robot._obj_init_pose[0]
    y = env.robot._obj_init_pose[1]
    
    actions =  [[x/4, y/4, 0.4, 0],
        [x/4*2, y/4*2, 0.4, 0],
        [x/4*3, y/4*3, 0.4, 0],
        [x, y, 0.4, 0],
        [x, y, 0.3, 0],
        [x, y, 0.2, 0],
        [x, y, 0.1, 0],
        [x, y, 0.05, 1],
        [x, y, 0.1, 1],
        [x, y, 0.2, 1],
        [0.4, -0.2, 0.2, 1],
        [0.4, -0.1, 0.2, 1],
        [0.4, -0.0, 0.2, 1],
        [0.4, 0.1, 0.2, 1],
        [0.4, 0.2, 0.2, 1],
        [0.4, 0.3, 0.2, 1],
        [0.4, 0.3, 0.07, 1]]
    
    for action in actions:
        env.step(action)
    
    env.save_video("test")



