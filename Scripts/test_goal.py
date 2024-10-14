import airsim
import numpy as np
from style import Format as f

name = f"{f.BOLD}Drone:{f.END} "

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print(f"{name}Taking off...")
client.takeoffAsync().join()

goal = np.array([19.70, -12.00, -0.45])

print(f"{name}On my way...")
client.moveToPositionAsync(19.70, -12.00, -0.45, 1).join()
print(f"{name}Made it!")

print(f"{name}Landing...")
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
