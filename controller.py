import airsim
import pygame
import time

# Initialize the PS5 controller
pygame.init()
pygame.joystick.init()

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Setup joystick (PS5 controller)
joystick = pygame.joystick.Joystick(0)
joystick.init()

def get_joystick_values():
    # Get axis values for pitch, roll, throttle, yaw
    pitch = joystick.get_axis(1)   # Left stick vertical
    roll = joystick.get_axis(0)    # Left stick horizontal
    throttle = -joystick.get_axis(4)  # Right stick vertical (inverted to match drone)
    yaw = joystick.get_axis(3)     # Right stick horizontal

    return pitch, roll, throttle, yaw

def move_drone(pitch, roll, throttle, yaw):
    # AirSim expects velocity values between -1 and 1
    vx = pitch  # forward/backward (pitch)
    vy = roll   # left/right (roll)
    vz = throttle  # up/down (throttle)
    yaw_rate = yaw  # rotate (yaw)

    # Send move command
    client.moveByVelocityAsync(vx, vy, vz, duration=0.1, yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate))

try:
    print("Press Ctrl+C to stop")

    while True:
        pygame.event.pump()  # Process joystick events

        # Get the current values from the controller
        pitch, roll, throttle, yaw = get_joystick_values()

        # Move the drone according to the controller input
        move_drone(pitch, roll, throttle, yaw)

        # Adjust the sleep time to match control rate
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Disarming and resetting drone.")
    client.armDisarm(False)
    client.reset()
    client.enableApiControl(False)
    pygame.quit()
