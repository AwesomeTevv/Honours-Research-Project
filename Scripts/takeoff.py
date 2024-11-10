import airsim
from style import Format as f

name = f"{f.BOLD}Drone:{f.END} "

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
print(f"{name}Ready to go!")

print(f"{name}Taking off...")
client.takeoffAsync().join()

print(f"{f.YELLOW}[Press any key to land]{f.END}")
airsim.wait_key()

print(f"{name}Landing...")
client.landAsync().join()
print(f"{name}Landed.")
client.armDisarm(False)
client.enableApiControl(False)
print(f"{name}I'm done.")
