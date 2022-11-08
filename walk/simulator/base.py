# coding:utf-8

import os
from functools import partial

import numpy as np
import pybullet as bullet

######################################################
class Base:
    def __init__(self, urdf, ees, origin=[0.0, 0.0, 0.0], joints=None, grav=9.81, dt=0.01, realtime=False, verbose=True, **physics_params):
        # config
        base = os.path.dirname(os.path.abspath(__file__))
        world_path = os.path.normpath(os.path.join(base, "data/plane.urdf"))
        self.urdf_path = os.path.normpath(os.path.join(base, urdf))

        self.end_effectors = ees
        self.grav = grav
        self.dt = dt

        # initialize pybullet
        bullet.connect(bullet.GUI)
        bullet.setGravity(0.0, 0.0, -grav)
        self.world = bullet.loadURDF(world_path)
        self.body = bullet.loadURDF(self.urdf_path, useFixedBase=False)

        print(bullet.getPhysicsEngineParameters())
        bullet.setPhysicsEngineParameter(fixedTimeStep=dt, **physics_params)
        bullet.changeDynamics(self.body, -1, linearDamping=0, angularDamping=0)
        print(bullet.getPhysicsEngineParameters())

        self.ids_ee = {}
        self.initial_root = origin
        self.ids = []
        self.joint_names = []
        for id in range (bullet.getNumJoints(self.body)):
            bullet.changeDynamics(self.body, id, linearDamping=0, angularDamping=0)
            info = bullet.getJointInfo(self.body, id)
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if (joint_type == bullet.JOINT_PRISMATIC or joint_type == bullet.JOINT_REVOLUTE):
                self.ids.append(id)
                self.joint_names.append(joint_name)
                if verbose:
                    print("joint id {}: {}".format(id, info))
                if joint_name in ees.values():
                    self.ids_ee.update({key: id for key, name in ees.items() if name == joint_name})
                    if verbose:
                        print("end effector {}: {}".format(id, joint_name))
        assert joints is None or len(joints) == len(self.ids), "#joint seems to be wrong... given: {}, urdf: {}".format(len(joints), len(self.ids))
        self.initial_joints = np.zeros(len(self.ids)) if joints is None else joints
        self.commands = self.initial_joints.copy()

        # reset once
        self.realtime = realtime
        self.reset()
        print("simulator is ready!")

    def command(self, joint_angles):
        bullet.setJointMotorControlArray(self.body, self.ids, controlMode=bullet.POSITION_CONTROL, targetPositions=joint_angles)

    def observe(self):
        if not self.realtime:
            bullet.stepSimulation()
        self.root_pose, self.root_vel = self.get_root()
        self.joint_pos, self.joint_vel = self.get_joints()
        self.ee_contact = self.get_contact()
        bullet.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=50, cameraPitch=-30, cameraTargetPosition=self.root_pose[:3])

    def get_root(self):
        pos, ori = bullet.getBasePositionAndOrientation(self.body)
        vel, omega = bullet.getBaseVelocity(self.body)
        return np.array(pos + ori), np.array(vel + omega)

    def get_joints(self):
        js = np.array(bullet.getJointStates(self.body, self.ids), dtype=object)
        return js[:, 0].astype(np.float), js[:, 1].astype(np.float)

    def get_contact(self):
        return {key: len(bullet.getContactPoints(bodyA=self.body, linkIndexA=id)) != 0 for key, id in self.ids_ee.items()}

    def reset(self, realtime=None):
        bullet.setRealTimeSimulation(False)
        bullet.resetBaseVelocity(self.body, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        bullet.resetBasePositionAndOrientation(self.body, self.initial_root, [0. ,0. ,0. , 1.0])
        [bullet.resetJointState(self.body, id, q0) for id, q0 in zip(self.ids, self.initial_joints)]
        self.command(self.initial_joints)
        for _ in range(int(1.0 / self.dt)):
            bullet.stepSimulation()
        if realtime is not None:
            self.realtime = realtime
        bullet.setRealTimeSimulation(self.realtime)
        self.observe()

    def close(self):
        bullet.disconnect()

######################################################
ji = np.zeros(30)
ji[4] = -np.pi / 3.0
ji[12] = np.pi / 3.0
ji[18:24] = [0.0, -0.0, -0.75, 1.5, -0.75, 0.0]
ji[24:] = [0.0, -0.0, -0.75, 1.5, -0.75, 0.0]
Atlas = partial(Base,
                urdf = "data/atlas/atlas_v4_with_multisense.urdf",
                ees = {"left_leg": "l_leg_akx", "right_leg": "r_leg_akx", "body": "neck_ry"},
                origin=np.array([0.0, 0.0, 0.8]),
                joints=ji,
                )

ji = np.zeros(32)
ji[5] = 0.5
ji[13] = -0.5
ji[20:26] = [0.0, -0.0, -0.5, 1.0, -0.5, 0.0]
ji[26:] = [0.0, -0.0, -0.5, 1.0, -0.5, 0.0]
Talos = partial(Base,
                urdf = "data/talos/talos_reduced.urdf",
                ees = {"left_leg": "leg_left_6_joint", "right_leg": "leg_right_6_joint", "body": "head_2_joint"},
                origin=np.array([0.0, 0.0, 1.0]),
                joints=ji)

######################################################
def test_simulator(simulator, t_max, random_motion=True):
    import time
    print("start test...")
    ts = time.time()
    while time.time() - ts < t_max:
        time.sleep(simulator.dt)
        simulator.observe()
        print(simulator.root_pose, simulator.ee_contact)
        if random_motion:
            simulator.command(np.random.randn(*simulator.joint_pos.shape))
    print("end test!")
