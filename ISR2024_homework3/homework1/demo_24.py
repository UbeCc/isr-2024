# -*- coding: utf-8 -*-
# @Author : zhux
# @Email : zhuxiang24@mails.tsinghua.edu.cn

import os, time, copy, math
import numpy as np
import pybullet_data
# import pybullet as p
from core import *
from robot_env import *


# for pick and place demo, use PICK_PLACE
# TESTING = "IK"
TESTING = "PICK_PLACE"


def IK_test(robot, physicsClientId):
    screw_axis = robot.get_screw_axis()
    M = robot.get_end_eff_home_trans()

    R = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((3.1415926, 0, 0)))).reshape(3, 3)
    P = np.array([0.3, 0.3, 0.4])

    theta_0 = np.array([0.0] * 7)

    def gradual_IK(thetalist0, P, R):
        T = rp_to_trans(R, P)
        thetalist, state = InverseKinematics_in_space(np.array(screw_axis).T, M, T, thetalist0, 0.001, 0.0005)

        act = thetalist

        robot.apply_action(act, max_vel=1)
        for _ in range(240):
            p.stepSimulation(physicsClientId=physicsClientId)
            time.sleep(1 / 240)
        M_new = robot.cal_eff_trans()
        return M_new, act

    print("---IK testing---")
    M_new, last_act = gradual_IK(theta_0, P, R)
    print("---Robot eff pose---")
    print(M_new)
    M_old = rp_to_trans(R, P)
    print(np.linalg.norm(M_new-M_old))
    print("\nyou should find that M_new is the same as your desired pose")
    print("---finish IK test---")


def pick_and_place_demo(robot, physicsClientId,demo_idx=0):
    print("---start pick_and_place_demo()---")
    # robot.load_env()
    robot.reset()
    robot.pre_grasp()
    screw_axis = robot.get_screw_axis()
    M = robot.get_end_eff_home_trans()
    grasped = 0

    # eff orientation
    R = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((3.1415926, 0, 0)))).reshape(3, 3)
    # initial theta
    thetalist0 = [0] * 7

    x = robot._obj_init_pose[0]
    y = robot._obj_init_pose[1]

    # define the via points
    # via_point_pick = [
    #     [x, y, 0.4],
    #     [x, y, 0.1],
    #     [x, y, 0.05]
    # ]
    # via_point_place = [
    #     [x, y, 0.1],
    #     [x, y, 0.4],
    #     [0.4, 0.3, 0.4],
    #     [0.4, 0.3, 0.07]
    # ]
    # via_point_back = [
    #     [0.4, 0.3, 0.4],
    #     [0.4, 0, 0.4]
    # ]

    via_point_pick = [
        [x/4, y/4, 0.4],
        [x/4*2, y/4*2, 0.4],
        [x/4*3, y/4*3, 0.4],
        [x, y, 0.4],
        [x, y, 0.3],
        [x, y, 0.2],
        [x, y, 0.1],
        [x, y, 0.05]
    ]
    via_point_place = [
        [x, y, 0.1],
        [x, y, 0.2],
        [0.4, -0.2, 0.2],
        [0.4, -0.1, 0.2],
        [0.4, -0.0, 0.2],
        [0.4, 0.1, 0.2],
        [0.4, 0.2, 0.2],
        [0.4, 0.3, 0.2],
        [0.4, 0.3, 0.07]
    ]
    # via_point_back = [
    #     [0.4, 0.3, 0.4],
    #     [0.4, 0, 0.4]
    # ]

    def gradual_IK(thetalist0, P, R,  info_list=[[],[],[]]):
        # do gradual IK
        T = rp_to_trans(R, P)
        screw_axis = robot.get_screw_axis()
        thetalist, state = InverseKinematics_in_space(np.array(screw_axis).T, M, T, thetalist0, 0.001, 0.0005)

        act = thetalist
        print(act)
        robot.apply_action(act, max_vel=1.0)
        for i in range(200):
            p.stepSimulation(physicsClientId=physicsClientId)
            time.sleep(1 / 240)

            if i%10 == 0:
                # rgb, _, _ = render(physicsClientId=physicsClientId)
                rgb = robot.static_camera()
                rgb2 = robot.panda_camera()
                # rgb = robot.panda_camera()
                # robot_theta, _, _ = robot.getJointStates()
                # T_sb = FK_in_space(M, np.array(screw_axis).T, robot_theta)
                # print("T_sb",T_sb)
                # xyz = T_sb[:3, 3]
                com_p, com_o, _, _, _, _ = p.getLinkState(robot.robot_id, 11, computeForwardKinematics=True)
                xyz = np.array(com_p)
                
                action = np.concatenate([xyz, np.array([grasped])])
                # print("action",action)
                info_list[0].append(rgb)
                info_list[1].append(rgb2)
                info_list[2].append(action)

                # if the distance between the current pose and the target pose is smaller than 0.1, break
                if np.linalg.norm(np.array(P)-np.array(xyz)) < 0.002:
                    print("already reach the target pose")
                    break

        M_new = robot.cal_eff_trans()
        return M_new, act, info_list
    
    info_list = [[],[],[]]
    print("len_pick", len(via_point_pick))
    for wp in via_point_pick:
        _, thetalist0, info_list = gradual_IK(thetalist0, np.array(wp), R, info_list)
    # info_list[2][-1][-1] = 1 # hard code the last action to be grasped

    info_list[0]+=[info_list[0][-1]]*10
    info_list[1]+=[info_list[1][-1]]*10

    xyz_open = info_list[2][-1][:]
    xyz_close = info_list[2][-1][:]
    xyz_close[-1] = 1
    info_list[2]+= [xyz_open]*5
    info_list[2]+= [xyz_close]*5

    grasped = 1
    robot.pre_grasp()
    step_sim(0.5, physicsClientId=physicsClientId)
    print("---Grasping!---")
    robot.grasp()
    step_sim(0.5, physicsClientId=physicsClientId)

    print("len_place", len(via_point_place))
    for wp in via_point_place:
        _, thetalist0, info_list = gradual_IK(thetalist0, np.array(wp), R, info_list)
    robot.pre_grasp()
    step_sim(0.5, physicsClientId=physicsClientId)
    print("---Placing!---")

    # print("len_back", len(via_point_back))
    # for wp in via_point_back:
    #     _, thetalist0, info_list = gradual_IK(thetalist0, np.array(wp), R, info_list)
    # print("---Back to home!---")
    print("---Finish pick_and_place_demo---")

    # save info_list
    import mediapy
    print("length", len(info_list[0]))
    os.makedirs(f"demo2/{demo_idx}", exist_ok=True)
    mediapy.write_video(f"demo2/{demo_idx}/0.mp4", info_list[0], fps=20)
    mediapy.write_video(f"demo2/{demo_idx}/1.mp4", info_list[1], fps=20)
    np.save(f"demo2/{demo_idx}/2.npy", info_list[2])
    print("video saved as pick_and_place_demo.mp4")


def test_panda(physicsClientId, controlMode="position", pb_client=None):
    # create robot
    robot = pandaEnv(physicsClientId, controlMode=controlMode)
    p.setGravity(0, 0, -9.8, physicsClientId=physicsClientId)

    robot.debug_gui()
    robot.pre_grasp()

    for _ in range(120):
        p.stepSimulation(physicsClientId=physicsClientId)
        time.sleep(1 / 240)
    if TESTING == "IK":
        IK_test(robot, physicsClientId)
    elif TESTING == "PICK_PLACE":
        for demo_idx in range(0,50):
            print(demo_idx)
            pick_and_place_demo(robot, physicsClientId,demo_idx=demo_idx)

    # while (1):
    #     p.stepSimulation(physicsClientId)
    #     time.sleep(1 / 240)

if __name__ == '__main__':
    # from pybullet_utils import bullet_client
    # import pybullet as pb
    # pb_client = bullet_client.BulletClient(connection_mode=pb.DIRECT)
    
    # _physics_client_id = p._client
    # p = bullet_client.BulletClient(connection_mode=pb.SHARED_MEMORY)
    # physicsClientId = pb_client._client


    physicsClientId = p.connect(p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=physicsClientId)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=physicsClientId)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=physicsClientId)
    test_panda(physicsClientId, controlMode="position")

