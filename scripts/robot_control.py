import pybullet as p
import numpy as np

class PandaRobot:
    """
    A class to encapsulate the control and state management of a Franka Panda robot in PyBullet.
    """
    def __init__(self, physicsClient_id, robot_id):
        """
        Initializes the PandaRobot instance.

        Args:
            physicsClient_id (int): The ID of the PyBullet physics client.
            robot_id (int): The unique ID of the loaded robot model in PyBullet.
        """
        self.physicsClient = physicsClient_id # Store the physics client ID
        self.robot_id = robot_id

        # Get the total number of joints in the robot model
        self.num_joints = p.getNumJoints(self.robot_id)
        print(f"DEBUG: Robot ID: {self.robot_id}, Number of joints detected: {self.num_joints}")

        # Initialize lists to store joint information
        self.joint_names = []
        self.joint_indices = [] # Indices of the main arm joints (controllable revolute/prismatic)
        self.gripper_joints = [] # Indices of the gripper finger joints
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        self.joint_ranges = []
        self.rest_poses = []
        self.joint_types = []

        # Iterate through all joints to categorize and store their properties
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_index = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            joint_lower_limit = info[8]
            joint_upper_limit = info[9]
            joint_range = info[10]
            rest_pose = info[6] # Initial position from URDF

            self.joint_names.append(joint_name)
            self.joint_types.append(joint_type)

            # Categorize joints: arm joints vs. gripper joints
            # Panda arm joints are typically 0-6. Gripper joints are usually 9 and 10.
            # We check joint type to ensure we only control revolute/prismatic joints for the arm.
            if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                if "finger_joint" not in joint_name: # Heuristic to identify arm joints
                    self.joint_indices.append(joint_index)
                    self.joint_limits_lower.append(joint_lower_limit)
                    self.joint_limits_upper.append(joint_upper_limit)
                    self.joint_ranges.append(joint_range)
                    self.rest_poses.append(rest_pose)
                else: # Assume it's a gripper joint
                    self.gripper_joints.append(joint_index)
            elif "finger_joint" in joint_name: # Catch any other gripper joint types if needed
                self.gripper_joints.append(joint_index)

        # Define the end-effector link index for the Panda robot
        # This is typically 'panda_link8' or the last link before the gripper base.
        # For the default franka_panda/panda.urdf, link 11 (panda_link8) is usually suitable.
        self.ee_link_index = 11

        # Gripper control parameters (common values for Panda)
        self.gripper_open_pos = 0.04
        self.gripper_close_pos = 0.00

        # Set joint damping for stability during control
        for joint_index in self.joint_indices:
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                p.VELOCITY_CONTROL, # Set to velocity control with zero force initially for damping
                force=0,
                velocityGain=0.01,
                positionGain=0.01,
                maxVelocity=0.5
            )
            p.changeDynamics(
                self.robot_id,
                joint_index,
                jointDamping=0.1
            )

        # Call reset_robot *after* all joint information is populated
        self.reset_robot()
        print("PandaRobot initialized and reset to home position.")

    def get_joint_states(self):
        """
        Returns current joint positions, velocities, and applied torques for the arm joints.
        """
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        positions = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]
        torques = [state[3] for state in joint_states] # Measured motor torque
        return positions, velocities, torques

    def get_end_effector_pose(self):
        """
        Returns the current Cartesian pose (position and orientation quaternion) of the end-effector.
        """
        link_state = p.getLinkState(self.robot_id, self.ee_link_index)
        ee_pos = link_state[0]
        ee_orn = link_state[1] # As quaternion (x, y, z, w)
        return ee_pos, ee_orn

    def move_to_joint_position(self, target_joint_positions, max_force=500):
        """
        Moves the robot's arm joints to specified target positions using position control.

        Args:
            target_joint_positions (list): List of target angles for each arm joint.
            max_force (float): Maximum force to apply to reach the target.
        """
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.joint_indices, # Control only the main arm joints
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_joint_positions,
            forces=[max_force] * len(self.joint_indices)
        )

    def move_to_cartesian_pose(self, target_position, target_orientation, max_iterations=100):
        """
        Moves the end-effector to a target Cartesian pose (position and orientation) using IK.

        Args:
            target_position (list): [x, y, z] coordinates for the end-effector.
            target_orientation (list): [qx, qy, qz, qw] quaternion for the end-effector.
            max_iterations (int): Maximum iterations for the IK solver.
        """
        # Calculate target joint positions using Inverse Kinematics (IK)
        # Using Damped Least Squares (DLS) solver (solver=0)
        # Providing joint limits and rest poses helps guide the IK solution
        joint_target_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.ee_link_index,
            targetPosition=target_position,
            targetOrientation=target_orientation,
            # These limits and ranges are typical for a Franka Panda.
            # Adjust if your specific URDF has different constraints.
            lowerLimits=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            upperLimits=[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            jointRanges=[5.7946, 3.5256, 5.7946, 2.9999, 5.7946, 3.7699, 5.7946],
            restPoses=[0, -0.785, 0, -2.356, 0, 1.571, 0.785],
            solver=0,
            maxNumIterations=max_iterations,
            residualThreshold=1e-4
        )

        # Apply commands to move joints to target positions
        # Ensure the number of target positions matches the number of controllable joints
        if len(joint_target_positions) >= len(self.joint_indices):
            self.move_to_joint_position(joint_target_positions[:len(self.joint_indices)])
        else:
            print("WARNING: IK solution returned fewer joint positions than expected.")
            self.move_to_joint_position(joint_target_positions) # Use what IK returned

        # Give time for the robot to move (simulate steps)
        for _ in range(50):
            p.stepSimulation()

    def open_gripper(self):
        """Opens the robot gripper to its maximum extent."""
        for gripper_joint in self.gripper_joints:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=gripper_joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=self.gripper_open_pos,
                force=50 # Gripper force
            )

    def close_gripper(self):
        """Closes the robot gripper to a grasping position."""
        for gripper_joint in self.gripper_joints:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=gripper_joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=self.gripper_close_pos,
                force=100 # Gripper force
            )

    def reset_robot(self):
        """
        Resets the robot to a default home pose and opens the gripper.
        This is called during initialization and can be called to reset between episodes.
        """
        # Common initial joint positions for a Franka Panda robot
        initial_arm_joint_positions = [
            0.0,            # Joint 1
            -0.785,         # Joint 2 (-45 degrees)
            0.0,            # Joint 3
            -2.356,         # Joint 4 (-135 degrees)
            0.0,            # Joint 5
            1.571,          # Joint 6 (90 degrees)
            0.785           # Joint 7 (45 degrees)
        ]

        # Reset arm joints
        for i, joint_idx in enumerate(self.joint_indices):
            if i < len(initial_arm_joint_positions):
                p.resetJointState(self.robot_id, joint_idx, initial_arm_joint_positions[i])

        # Open the gripper
        self.open_gripper()

        # Give it a moment to settle after reset
        for _ in range(100):
            p.stepSimulation()
