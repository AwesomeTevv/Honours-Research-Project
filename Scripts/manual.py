import airsim
import pygame
import time

# Initialize the connection to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off
client.takeoffAsync().join()

# Initialize pygame
pygame.init()

# Set up display (not necessary for functionality but required by pygame)
screen = pygame.display.set_mode((100, 100))

# Initial velocities
vx, vy, vz = 0, 0, 0
yaw_rate = 0

# Acceleration factor
accel = 0.1
max_speed = 5

# Main loop to listen for keyboard inputs
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    keys = pygame.key.get_pressed()

    # Update velocities based on key presses
    if keys[pygame.K_w]:
        vx = min(vx + accel, max_speed)
    elif keys[pygame.K_s]:
        vx = max(vx - accel, -max_speed)
    else:
        vx = 0  # Decelerate to 0 when key is released

    if keys[pygame.K_a]:
        vy = max(vy - accel, -max_speed)
    elif keys[pygame.K_d]:
        vy = min(vy + accel, max_speed)
    else:
        vy = 0  # Decelerate to 0 when key is released

    if keys[pygame.K_UP]:
        vz = max(vz - accel, -max_speed)
    elif keys[pygame.K_DOWN]:
        vz = min(vz + accel, max_speed)
    else:
        vz = 0  # Decelerate to 0 when key is released

    if keys[pygame.K_q]:
        yaw_rate = max(yaw_rate - accel, -max_speed)
    elif keys[pygame.K_e]:
        yaw_rate = min(yaw_rate + accel, max_speed)
    else:
        yaw_rate = 0  # Stop rotating when key is released

    # Send velocity and yaw commands to the drone
    client.moveByVelocityAsync(vx, vy, vz, 0.1, yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate))

    # Get and print the drone's position
    position = client.getMultirotorState().kinematics_estimated.position
    print(f"Position: x={position.x_val:.2f}, y={position.y_val:.2f}, z={position.z_val:.2f}, Yaw rate: {yaw_rate:.2f}")
    
    # Limit the update rate to 10 Hz
    time.sleep(0.1)

# Disarm and release control before exiting
client.armDisarm(False)
client.enableApiControl(False)
pygame.quit()
