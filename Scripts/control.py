import airsim
import pygame
import time
import threading
import sys

from style import Format

pygame.init()
screen = pygame.display.set_mode((100, 100))
pygame.display.set_caption("Drone Control")

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()

move_speed = 5.0  # Speed in m/s
yaw_rate = 30  # Degrees per second
vertical_speed = 2.0  # Speed in m/s

keys = {
    "W": "Forward",
    "S": "Backward",
    "A": "Left",
    "D": "Right",
    "Q": "Rotate left",
    "E": "Rotate right",
    "Space": "Up",
    "L Shift": "Down",
}

print(f"Drone control ready.\n{Format.UNDERLINE}Use the following keys:{Format.END}")
for key, move in keys.items():
    print(f"{Format.BOLD}{key.rjust(7)}{Format.END} — {move}")
print(f"\nClose the Pygame window to {Format.RED}quit{Format.END}")


def print_position():
    while True:
        try:
            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            sys.stdout.write(
                f"\rDrone Position: x={Format.RED}{pos.x_val:.2f}{Format.END}, y={Format.GREEN}{pos.y_val:.2f}{Format.END}, z={Format.BLUE}{pos.z_val:.2f}{Format.END}"
            )
            sys.stdout.flush()
        except Exception as e:
            # sys.stdout.write(f"\rError getting drone position: {str(e)}")
            # sys.stdout.flush()
            pass
        time.sleep(0.1)


position_thread = threading.Thread(target=print_position, daemon=True)
position_thread.start()


def move_drone(vx, vy, vz, yaw):
    try:
        client.moveByVelocityBodyFrameAsync(
            vx,
            vy,
            vz,
            0.1,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(True, yaw),
        )
    except Exception as e:
        # print(f"Error moving drone: {str(e)}")
        pass


try:
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        vx, vy, vz, yaw = 0, 0, 0, 0

        if keys[pygame.K_w]:
            vx = move_speed
        if keys[pygame.K_s]:
            vx = -move_speed
        if keys[pygame.K_a]:
            vy = -move_speed
        if keys[pygame.K_d]:
            vy = move_speed
        if keys[pygame.K_SPACE]:
            vz = -vertical_speed
        if keys[pygame.K_LSHIFT]:
            vz = vertical_speed
        if keys[pygame.K_q]:
            yaw = -yaw_rate
        if keys[pygame.K_e]:
            yaw = yaw_rate

        move_drone(vx, vy, vz, yaw)

        clock.tick(30)  # Limit to 30 FPS

except KeyboardInterrupt:
    print("Program interrupted by user")

finally:
    print("Landing drone...")
    try:
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        print("Drone landed and disarmed.")
    except Exception as e:
        # print(f"Error during landing: {str(e)}")
        pass
    pygame.quit()
