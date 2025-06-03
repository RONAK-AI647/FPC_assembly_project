import pybullet as p
import pybullet_data
import time
import os

print("--- Starting simple_test.py ---")

# 1. Connect to PyBullet in DIRECT (headless) mode
print("Connecting to PyBullet in DIRECT (headless) mode...")
try:
    physicsClient = p.connect(p.DIRECT)
    print(f"Successfully connected to PyBullet. Client ID: {physicsClient}")
except p.error as e:
    print(f"ERROR: Failed to connect to PyBullet: {e}")
    exit()

# 2. Set the additional search path to pybullet_data
print("Setting additional search path for pybullet_data...")
try:
    pybullet_data_path = pybullet_data.getDataPath()
    p.setAdditionalSearchPath(pybullet_data_path)
    print(f"PyBullet data path added: {pybullet_data_path}")
except Exception as e:
    print(f"ERROR: Could not get PyBullet data path or set search path: {e}")
    p.disconnect()
    exit()

# 3. Load a very simple URDF (e.g., plane.urdf)
# We will NOT use the absolute path hack here, to see if setAdditionalSearchPath works for *this* simple case.
plane_urdf_name = "plane.urdf"
print(f"Attempting to load {plane_urdf_name} using search path...")
try:
    planeId = p.loadURDF(plane_urdf_name)
    print(f"Successfully loaded {plane_urdf_name}. Object ID: {planeId}")
except p.error as e:
    print(f"ERROR: Failed to load {plane_urdf_name}: {e}")
    print(f"PyBullet was looking for the file in its search paths, including: {pybullet_data_path}")
    p.disconnect()
    exit()

# 4. Set gravity and time step (basic simulation setup)
p.setGravity(0, 0, -9.81)
p.setTimeStep(1./240.)

# 5. Run a few simulation steps
print("Running a few simulation steps...")
for i in range(240): # Simulate 1 second
    p.stepSimulation()
    # time.sleep(1./240.) # No need to sleep in headless mode

print("Simulation steps completed.")

# 6. Disconnect from PyBullet
print("Disconnecting from PyBullet...")
p.disconnect()
print("PyBullet disconnected.")

print("--- simple_test.py finished successfully (if no errors above) ---")