🤖 Embodied Intelligence Benchmarking in Industrial Manufacturing



🔑 What Is Embodied Intelligence?

Embodied Intelligence refers to the idea that intelligence is not just in the brain 🧠 but is also influenced by the body 🦾 and the environment 🌍 it operates in.
In the context of Artificial Intelligence, this means designing systems with a physical presence that can interact with the world, learn, and adapt through sensory and motor experiences.

🔍 What Is This Project About?

This project focuses on Embodied Intelligence Benchmarking in industrial manufacturing environments, using a framework called KubeEdge-Ianvs.

🛠️ What Is KubeEdge-Ianvs?

KubeEdge-Ianvs is a benchmarking platform for evaluating AI algorithms in real-world robotic tasks — especially in industrial settings 🏭.
It is built on top of KubeEdge (a Kubernetes-based edge computing framework), and helps test how well different AI models perform under real-world constraints.


💡 Advanced Task: Create Your Own Dataset


Objective: Propose a novel industrial use case and design a custom dataset around it — could be real or simulated.

🧩 Chosen Industrial Scenario

🔧 Precision Assembly of Thin and Soft Components

➡️ Specifically:

Precision Assembly of Flexible Printed Circuit (FPC) Cables onto Smartphone Motherboards 📱

✅ Why This Scenario Is a Strong Choice:
Criteria	           ✅ Reason It Fits
Relevance             	High-precision real-world task (e.g., FPC to motherboard)
Challenge          	Involves soft materials, sub-mm precision, and dynamic control
Benchmarkability              	Easy to measure: accuracy, control, perception, etc.
Customizability	             Perfect for query → response format in Ianvs
Compatibility	              Suitable for “inference-then-mining” or router-based evaluation
Uniqueness	           Not commonly found in public datasets


🏷️ Dataset Name
precision_soft_assembly

🧪 Data Generation Strategy
Simulation using PyBullet
Robotic arm: UR5 or Panda

Equipped with:

Simple gripper

Force/Torque sensor at wrist

Overhead RGB-D camera (front view + optional side view)

📦 Dataset Contents (Per Frame)
🎥 Visual Data
rgb_image_front.png: RGB image (front camera)
depth_image_front.png: Depth image (front camera)
(Optional): rgb_image_side.png, depth_image_side.png

🤖 Robot State Data
joint_positions.json: Robot joint angles
end_effector_pose.json: Cartesian position + orientation (quaternion)
force_torque_sensor_readings.json: Fx, Fy, Fz, Tx, Ty, Tz

🧭 Ground Truth Data
fpc_pose_world.json: 6-DOF pose of FPC
motherboard_pose_world.json: 6-DOF pose of motherboard
fpc_connector_pose_relative_motherboard.json: Relative target alignment
assembly_stage.txt: Stage label (e.g., pre-pick, aligned_fine, inserted, failed_collision, etc.)

🗂 Scene Metadata
scene_id.txt: Unique ID for each attempt
initial_fpc_pose_variant.txt: Pose variant ID
lighting_condition.txt: Lighting setup (e.g., normal, low_light, glare)

🧪 Dataset Variations for Robustness
🔄 Initial FPC Poses: Randomize start positions
📐 Motherboard Poses: Slight pose variations
💡 Lighting Conditions: Simulate normal, glare, and low_light
🌬️ Deformability Settings: Adjust stiffness/damping to simulate material variability
❌ Failure Modes: Include misalignment/collision data for training recovery behaviors

📊 Dataset Size
Aim: Hundreds to thousands of unique sequences

Each sequence: 50–200 frames

Total: Tens to hundreds of thousands of data points (frames)

🗃️ Data Format
Type	Format
Images	PNG
Numerical Data	JSON
Metadata	TXT / JSON

