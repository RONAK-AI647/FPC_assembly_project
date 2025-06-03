import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import shutil # For clearing dataset directory (though not used in current version, kept for future)
import json
import cv2 # <--- ADDED: Import OpenCV for image saving

from scripts.robot_control import PandaRobot
# Removed unused imports from scripts.data_utils as you're using cv2 and json directly
# from scripts.data_utils import save_rgb_image, save_depth_image, save_json_data, create_episode_dir

# --- Configuration Parameters ---
GUI_MODE = False # Set to True temporarily for visual debugging, False for data generation
TIME_STEP = 1./240. # Simulation time step

# Define paths relative to the current script's location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
URDF_PATH = os.path.join(PROJECT_ROOT, "urdf_models")
# Corrected: Use DATASET_OUTPUT_PATH consistently
DATASET_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "dataset")

# Dataset Generation Parameters
# Corrected: Removed duplicate NUM_EPISODES definition
NUM_EPISODES = 5 # Let's start with a small number of episodes for testing
NUM_SIM_STEPS_PER_EPISODE = 200 # Increased steps to allow for full pick-and-place sequence

# Randomization Ranges (in meters for position, radians for orientation)
# These values will define how much the initial placement can vary.
# You might need to adjust these based on your robot's workspace and desired difficulty.

# Motherboard Randomization
MB_POS_RANGE_X = (-0.05, 0.05) # +/- 5 cm along X from center
MB_POS_RANGE_Y = (-0.05, 0.05) # +/- 5 cm along Y from center
MB_ORIENTATION_RANGE_Z = (-np.pi/8, np.pi/8) # +/- 22.5 degrees yaw (Z-axis rotation)

# FPC Randomization (relative to motherboard)
FPC_REL_POS_RANGE_X = (-0.01, 0.01) # +/- 1 cm
FPC_REL_POS_RANGE_Y = (-0.01, 0.01) # +/- 1 cm
FPC_REL_POS_OFFSET_Z = 0.015 # Initial Z-offset above the motherboard surface
FPC_ORIENTATION_RANGE_Z = (-np.pi/4, np.pi/4) # +/- 45 degrees yaw (Z-axis rotation)

# Define relative positions of connectors (assuming FPC and Motherboard URDF origins)
# You might need to inspect your actual URDFs or use PyBullet's debug features to get these accurately.
# Example: FPC connector is at [0.0, 0.02, 0.0] relative to FPC base link
# Motherboard connector is at [0.0, 0.05, 0.005] relative to Motherboard base link
FPC_CONNECTOR_OFFSET_LOCAL = [0.0, 0.02, 0.0] # Relative to FPC base
MB_CONNECTOR_OFFSET_LOCAL = [0.0, 0.05, 0.005] # Relative to Motherboard base (where FPC should go)

# Pre-grasp/Insertion offsets relative to connector
PRE_GRASP_HEIGHT = 0.10 # 10 cm above FPC
GRASP_HEIGHT_OFFSET = 0.002 # 0.2 cm above FPC (or slight penetration for stable grasp)
INSERTION_HEIGHT = -0.003 # 0.3 cm below motherboard surface for proper insertion
RETREAT_HEIGHT = 0.15 # Height to retreat to after insertion

# --- IMPORTANT: Print these paths to verify they are correct ---
print(f"DEBUG: Script is running from: {os.path.dirname(__file__)}")
print(f"DEBUG: Detected Project Root: {PROJECT_ROOT}")
print(f"DEBUG: Custom URDF Models Path: {URDF_PATH}")
print(f"DEBUG: Dataset Output Path: {DATASET_OUTPUT_PATH}") # Corrected variable name for print
# --- End of DEBUG prints ---


# --- Camera Setup ---
CAMERA_PARAMS = {
    "eye": [0.3, 0.3, 0.7],
    "target": [0.0, 0.3, 0.0],
    "up": [0, 1, 0],
    "fov": 60,
    "aspect": 1.0,
    "near": 0.01,
    "far": 1.5,
    "width": 640,
    "height": 480
}

# --- PyBullet Setup ---
def setup_pybullet_env():
    """Sets up the PyBullet simulation environment."""
    if GUI_MODE:
        physicsClient = p.connect(p.DIRECT)
    else:
        physicsClient = p.connect(p.DIRECT) # Connect in headless mode

    try:
        pybullet_data_path = pybullet_data.getDataPath()
        p.setAdditionalSearchPath(pybullet_data_path)
        print(f"DEBUG: PyBullet data path added: {pybullet_data_path}")
    except Exception as e:
        print(f"ERROR: Could not get PyBullet data path: {e}")
        print("Please ensure 'pybullet_data' is installed in your virtual environment.")

    p.setAdditionalSearchPath(URDF_PATH)
    print(f"DEBUG: Added custom URDF path to search: {URDF_PATH}")

    p.setGravity(0, 0, -9.81)
    p.setTimeStep(TIME_STEP)
    return physicsClient

# Modify load_scene_objects function
def load_scene_objects(
    mb_pos_offset=(0,0,0), mb_orn_offset=(0,0,0,1), # Motherboard absolute offset
    fpc_rel_pos_offset=(0,0,0), fpc_rel_orn_offset=(0,0,0,1) # FPC relative to MB
):
    """
    Loads the robot, plane, motherboard, and FPC into the PyBullet simulation.
    Now includes randomization parameters for initial placement.

    Args:
        mb_pos_offset (tuple): (x, y, z) offset for motherboard's initial position.
        mb_orn_offset (tuple): (qx, qy, qz, qw) offset for motherboard's initial orientation.
        fpc_rel_pos_offset (tuple): (x, y, z) relative offset for FPC from motherboard's center.
        fpc_rel_orn_offset (tuple): (qx, qy, qz, qw) relative offset for FPC's orientation.

    Returns:
        tuple: (robotId, motherboardId, fpcId)
    """
    # Define pyd_path globally or pass it as an argument if setup_pybullet_env isn't used
    # Assuming setup_pybullet_env will be called, but load_scene_objects might be called
    # independently. To ensure pyd_path is available here:
    pyd_path = pybullet_data.getDataPath() # <--- Ensuring pyd_path is defined here

    # Load Plane
    absolute_plane_urdf_path = os.path.join(pyd_path, "plane.urdf")
    planeId = p.loadURDF(absolute_plane_urdf_path, [0, 0, 0])
    p.changeVisualShape(planeId, -1, rgbaColor=[0.7, 0.7, 0.7, 1]) # Make plane grey

    # Load Robot
    absolute_robot_urdf_path = os.path.join(pyd_path, "franka_panda/panda.urdf")
    robotId = p.loadURDF(absolute_robot_urdf_path, [0, 0, 0], useFixedBase=True)
    # Give the robot a moment to settle
    for _ in range(50): p.stepSimulation()

    # Load Motherboard with randomization
    absolute_motherboard_urdf_path = os.path.join(URDF_PATH, "motherboard.urdf")
    # Base position for motherboard (e.g., in front of the robot)
    # You might need to adjust this base_mb_pos based on your robot's setup
    base_mb_pos = [0.5, 0, 0.0025] # Slightly above the plane (half of its thickness)
    final_mb_pos = [base_mb_pos[0] + mb_pos_offset[0],
                    base_mb_pos[1] + mb_pos_offset[1],
                    base_mb_pos[2] + mb_pos_offset[2]] # Z-offset from randomization is 0 as per ranges

    # Apply motherboard orientation offset
    base_mb_orn = p.getQuaternionFromEuler([0, 0, 0]) # Default flat orientation
    final_mb_orn = p.multiplyTransforms([0,0,0], base_mb_orn, [0,0,0], mb_orn_offset)[1]

    motherboardId = p.loadURDF(absolute_motherboard_urdf_path, final_mb_pos, final_mb_orn)

    # Load FPC with randomization (relative to motherboard's position and orientation)
    absolute_fpc_urdf_path = os.path.join(URDF_PATH, "fcp.urdf")

    # Get motherboard's current pose
    mb_state = p.getBasePositionAndOrientation(motherboardId)
    mb_pos = mb_state[0]
    mb_orn = mb_state[1] # Motherboard's current orientation (quaternion)

    # Calculate FPC's absolute position and orientation based on motherboard's pose
    # and the relative offsets.
    fpc_offset_pos_local = [fpc_rel_pos_offset[0], fpc_rel_pos_offset[1], fpc_rel_pos_offset[2]]
    fpc_offset_orn_local = fpc_rel_orn_offset # Already a quaternion

    # Transform FPC's relative pose to world coordinates
    fpc_world_pos, fpc_world_orn = p.multiplyTransforms(
        mb_pos, mb_orn, # Parent frame (motherboard)
        fpc_offset_pos_local, fpc_offset_orn_local # Child frame (FPC relative to parent)
    )

    fpcId = p.loadURDF(absolute_fpc_urdf_path, fpc_world_pos, fpc_world_orn)

    print("All scene objects loaded.")
    return robotId, motherboardId, fpcId

def capture_data(frame_idx, episode_dir, robot, motherboard_id, fpc_id):
    """
    Captures RGB, Depth, Robot State, and Ground Truth data for a single frame.
    """
    # Using computeViewMatrixFromYawPitchRoll for simplicity, as it's common.
    # The camera target and orientation are fixed for now.
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.5, 0, 0.1], # Example target, adjust if needed
        distance=0.8,
        yaw=90,
        pitch=-20,
        roll=0,
        upAxisIndex=2
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=CAMERA_PARAMS["fov"],
        aspect=CAMERA_PARAMS["aspect"],
        nearVal=CAMERA_PARAMS["near"],
        farVal=CAMERA_PARAMS["far"]
    )

    width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
        width=CAMERA_PARAMS["width"],
        height=CAMERA_PARAMS["height"],
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER# This renderer might still be problematic if no GUI context
    )
    # --- START OF THE CRITICAL CORRECTION ---
    # Convert rgb_img to a NumPy array and ensure it's uint8 before removing alpha
    # PyBullet's getCameraImage usually returns 8-bit unsigned integers.
    # The .reshape might cause a type inference to int32/int64, so explicit cast is needed.
    rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(height, width, 4)[:, :, :3] # Remove alpha channel and ensure uint8 type
    # --- END OF THE CRITICAL CORRECTION ---
    
    # For depth, PyBullet returns float, typically scaled to 0-1 for normalized depth or actual distance.
    # To save as uint16, you usually scale it to the max value of uint16 (65535).
    # Ensure depth_array is float before scaling, then convert to uint16.
    depth_array = np.array(depth_img).reshape(height, width)
    # The original depth saving line was: cv2.imwrite(os.path.join(episode_dir, f"{frame_prefix}_depth.png"), (depth_array * 65535).astype(np.uint16))
    # This is generally correct for converting normalized depth to 16-bit PNG.
    
    # 2. Get Robot State Data
    joint_positions, joint_velocities, joint_torques = robot.get_joint_states()
    ee_pos, ee_orn = robot.get_end_effector_pose()

    robot_state_data = {
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
        "joint_torques": joint_torques,
        "end_effector_position": list(ee_pos),
        "end_effector_orientation_quaternion": list(ee_orn)
    }

    # 3. Get Ground Truth Data
    fpc_pos, fpc_orn = p.getBasePositionAndOrientation(fpc_id)
    mb_pos, mb_orn = p.getBasePositionAndOrientation(motherboard_id)

    # Calculate current FPC connector pose
    current_fpc_connector_pos_world, _ = p.multiplyTransforms(
        fpc_pos, fpc_orn,
        FPC_CONNECTOR_OFFSET_LOCAL, p.getQuaternionFromEuler([0,0,0])
    )

    # Calculate Motherboard connector pose
    mb_connector_pos_world, _ = p.multiplyTransforms(
        mb_pos, mb_orn,
        MB_CONNECTOR_OFFSET_LOCAL, p.getQuaternionFromEuler([0,0,0])
    )

    ground_truth_data = {
        "fpc_pose_world_position": list(fpc_pos),
        "fpc_pose_world_orientation_quaternion": list(fpc_orn),
        "motherboard_pose_world_position": list(mb_pos),
        "motherboard_pose_world_orientation_quaternion": list(mb_orn),
        "fpc_connector_world_position": list(current_fpc_connector_pos_world), # Added
        "motherboard_connector_world_position": list(mb_connector_pos_world), # Added
        # Calculate relative pose of FPC connector to MB connector
        "relative_fpc_to_mb_connector_position": list(np.array(current_fpc_connector_pos_world) - np.array(mb_connector_pos_world)),
        "assembly_stage": "in_progress" # This will be updated by the control logic
    }

    # 4. Save Data
    frame_prefix = f"frame_{frame_idx:04d}"
    # Use cv2.imwrite for images as it's directly used.
    # For JSON, direct json.dump is fine.
    cv2.imwrite(os.path.join(episode_dir, f"{frame_prefix}_rgb.png"), cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)) # OpenCV saves as BGR
    cv2.imwrite(os.path.join(episode_dir, f"{frame_prefix}_depth.png"), (depth_array * 65535).astype(np.uint16))
    
    with open(os.path.join(episode_dir, f"{frame_prefix}_robot_state.json"), 'w') as f:
        json.dump(robot_state_data, f, indent=4)
    with open(os.path.join(episode_dir, f"{frame_prefix}_ground_truth.json"), 'w') as f:
        json.dump(ground_truth_data, f, indent=4)


# --- Main Data Generation Loop ---
def generate_dataset():
    # Setup PyBullet physics client
    physicsClient = setup_pybullet_env() # Use the setup function

    # Corrected: Use DATASET_OUTPUT_PATH consistently
    # Create dataset output directory if it doesn't exist
    os.makedirs(DATASET_OUTPUT_PATH, exist_ok=True)

    print("\nStarting dataset generation...")

    for episode_idx in range(NUM_EPISODES):
        print(f"\n--- Generating Episode {episode_idx + 1}/{NUM_EPISODES} ---")

        # --- Randomization for this episode ---
        # Motherboard randomization
        mb_x_offset = np.random.uniform(MB_POS_RANGE_X[0], MB_POS_RANGE_X[1])
        mb_y_offset = np.random.uniform(MB_POS_RANGE_Y[0], MB_POS_RANGE_Y[1])
        mb_z_rot = np.random.uniform(MB_ORIENTATION_RANGE_Z[0], MB_ORIENTATION_RANGE_Z[1])
        mb_orn_quat = p.getQuaternionFromEuler([0, 0, mb_z_rot])
        mb_pos_offset = (mb_x_offset, mb_y_offset, 0) # Only x, y variation for position

        # FPC randomization (relative to motherboard)
        fpc_rel_x_offset = np.random.uniform(FPC_REL_POS_RANGE_X[0], FPC_REL_POS_RANGE_X[1])
        fpc_rel_y_offset = np.random.uniform(FPC_REL_POS_RANGE_Y[0], FPC_REL_POS_RANGE_Y[1])
        fpc_rel_z_rot = np.random.uniform(FPC_ORIENTATION_RANGE_Z[0], FPC_ORIENTATION_RANGE_Z[1])
        fpc_rel_orn_quat = p.getQuaternionFromEuler([0, 0, fpc_rel_z_rot])
        # FPC_REL_POS_OFFSET_Z ensures it starts slightly above the surface
        fpc_rel_pos_offset = (fpc_rel_x_offset, fpc_rel_y_offset, FPC_REL_POS_OFFSET_Z)
        # --- End Randomization ---

        # Reset simulation and load objects for the new episode
        p.resetSimulation() # <--- CORRECTED: No argument needed for resetSimulation
        robotId, motherboardId, fpcId = load_scene_objects(
            mb_pos_offset=mb_pos_offset, mb_orn_offset=mb_orn_quat,
            fpc_rel_pos_offset=fpc_rel_pos_offset, fpc_rel_orn_offset=fpc_rel_orn_quat
        )

        # Initialize robot controller
        robot = PandaRobot(physicsClient, robotId) # Robot is reset inside its __init__

        # Create episode directory
        episode_dir = os.path.join(DATASET_OUTPUT_PATH, f"episode_{episode_idx:05d}")
        os.makedirs(episode_dir, exist_ok=True)

        # --- Get object connector poses in world frame for this episode ---
        # Get Motherboard's current pose
        mb_pos_world, mb_orn_world = p.getBasePositionAndOrientation(motherboardId)
        # Calculate Motherboard Connector's world pose
        mb_connector_pos_world, mb_connector_orn_world = p.multiplyTransforms(
            mb_pos_world, mb_orn_world,
            MB_CONNECTOR_OFFSET_LOCAL, p.getQuaternionFromEuler([0,0,0]) # Assuming connector is axis-aligned
        )

        # Get FPC's current pose (this will change after grasp, so we get initial one)
        fpc_pos_world_initial, fpc_orn_world_initial = p.getBasePositionAndOrientation(fpcId)
        # Calculate FPC Connector's initial world pose
        fpc_connector_pos_world_initial, fpc_connector_orn_world_initial = p.multiplyTransforms(
            fpc_pos_world_initial, fpc_orn_world_initial,
            FPC_CONNECTOR_OFFSET_LOCAL, p.getQuaternionFromEuler([0,0,0])
        )

        # --- Define fixed grasp orientation for the FPC ---
        # This needs to be an orientation that allows the gripper to grab the FPC properly.
        # For a top-down grasp with Panda, usually pitch is -PI/2 (gripper points down).
        # Roll and Yaw depend on how you want to align the gripper to the FPC.
        # Let's try an orientation where gripper is pointing down (pitch=-90deg)
        # and aligned with FPC's current yaw for initial grasp.
        # You might need to adjust the roll/yaw if your FPC model's gripper area is not aligned.
        # Current FPC yaw is fpc_rel_z_rot from randomization.
        # Grasp orientation: roll=PI/2, pitch=-PI/2, yaw adjusted for FPC's initial yaw.
        # A common gripper orientation to point downwards for Panda is:
        # p.getQuaternionFromEuler([np.pi, 0, np.pi/2]) or similar for top-down
        # For a more robust top-down grasp, ensure gripper z-axis points down (roll=pi, pitch=0 or roll=0, pitch=pi)
        # and adjust yaw based on the object's orientation.
        # Let's use a standard vertical downward orientation, and assume the robot's IK
        # can find a valid solution.
        grasp_orientation_quat = p.getQuaternionFromEuler([np.pi, 0, fpc_rel_z_rot + np.pi/2]) # Roll 180, Pitch 0, Yaw relative to FPC initial yaw

        # --- Robot Actions (Pick and Place Sequence) ---
        current_step_in_sequence = 0
        
        # Total frames used for each stage (adjust these for smoother/faster motions)
        steps_per_stage = 25 
        # Total steps must be >= NUM_SIM_STEPS_PER_EPISODE if all stages are to complete

        # Loop through simulation steps for the entire episode
        for frame_idx in range(NUM_SIM_STEPS_PER_EPISODE):
            # Take a simulation step
            p.stepSimulation()

            # --- Robot Control Logic ---
            if current_step_in_sequence == 0: # Move to pre-grasp pose
                target_pos = [fpc_connector_pos_world_initial[0], fpc_connector_pos_world_initial[1], fpc_connector_pos_world_initial[2] + PRE_GRASP_HEIGHT]
                robot.move_to_cartesian_pose(target_pos, grasp_orientation_quat)
                if frame_idx >= steps_per_stage * 1: 
                    current_step_in_sequence = 1
                    print(f"DEBUG: Moving to grasp pose (step {frame_idx})")

            elif current_step_in_sequence == 1: # Move to grasp pose (descend to FPC)
                target_pos = [fpc_connector_pos_world_initial[0], fpc_connector_pos_world_initial[1], fpc_connector_pos_world_initial[2] + GRASP_HEIGHT_OFFSET]
                robot.move_to_cartesian_pose(target_pos, grasp_orientation_quat)
                if frame_idx >= steps_per_stage * 2:
                    current_step_in_sequence = 2
                    print(f"DEBUG: Closing gripper (step {frame_idx})")

            elif current_step_in_sequence == 2: # Close gripper
                robot.close_gripper()
                # Give gripper time to close and grip FPC
                if frame_idx >= steps_per_stage * 3:
                    current_step_in_sequence = 3
                    print(f"DEBUG: Lifting FPC (step {frame_idx})")

            elif current_step_in_sequence == 3: # Lift FPC (post-grasp)
                # Recalculate FPC's position as it's now attached to gripper
                current_fpc_pos, current_fpc_orn = p.getBasePositionAndOrientation(fpcId)
                target_pos = [current_fpc_pos[0], current_fpc_pos[1], current_fpc_pos[2] + PRE_GRASP_HEIGHT]
                robot.move_to_cartesian_pose(target_pos, grasp_orientation_quat)
                if frame_idx >= steps_per_stage * 4:
                    current_step_in_sequence = 4
                    print(f"DEBUG: Moving to pre-insertion pose (step {frame_idx})")

            elif current_step_in_sequence == 4: # Move to pre-insertion pose (above MB connector)
                # The target position for the gripper is relative to the MB connector.
                # So the gripper's target position should be MB_connector_pos_world + GRASP_HEIGHT_OFFSET_Z
                target_pos = [mb_connector_pos_world[0], mb_connector_pos_world[1], mb_connector_pos_world[2] + PRE_GRASP_HEIGHT]
                # Maintain the same grasp orientation, or align with MB connector if needed
                robot.move_to_cartesian_pose(target_pos, grasp_orientation_quat)
                if frame_idx >= steps_per_stage * 5:
                    current_step_in_sequence = 5
                    print(f"DEBUG: Moving to insertion pose (step {frame_idx})")

            elif current_step_in_sequence == 5: # Move to insertion pose (descend for insertion)
                target_pos = [mb_connector_pos_world[0], mb_connector_pos_world[1], mb_connector_pos_world[2] + INSERTION_HEIGHT]
                robot.move_to_cartesian_pose(target_pos, grasp_orientation_quat)
                if frame_idx >= steps_per_stage * 6:
                    current_step_in_sequence = 6
                    print(f"DEBUG: Opening gripper (step {frame_idx})")

            elif current_step_in_sequence == 6: # Open gripper (release FPC)
                robot.open_gripper()
                if frame_idx >= steps_per_stage * 7:
                    current_step_in_sequence = 7
                    print(f"DEBUG: Retreating (step {frame_idx})")

            elif current_step_in_sequence == 7: # Retreat (move robot away)
                target_pos = [mb_connector_pos_world[0], mb_connector_pos_world[1], mb_connector_pos_world[2] + RETREAT_HEIGHT]
                robot.move_to_cartesian_pose(target_pos, grasp_orientation_quat)
                # Let it finish the episode here (no further action needed)


            # --- Data Collection ---
            # Always capture data at the end of each simulation step if needed,
            # or continue with the sampling rate.
            if frame_idx % 5 == 0: # Capture data more frequently during action for better granularity
                # Pass current_step_in_sequence to ground_truth_data for assembly_stage
                capture_data(frame_idx, episode_dir, robot, motherboardId, fpcId)

        print(f"Episode {episode_idx + 1} completed. Data saved to {episode_dir}")

    print("PyBullet disconnected. Dataset generation finished.")
    p.disconnect()
    
if __name__ == "__main__":
    generate_dataset()