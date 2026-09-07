# -*- coding: utf-8 -*-
"""
arm_control.py
==============
Robot arm control — imported by Aruco_detection_V2_USB.py

BEFORE USING:
  1. Set COM_PORT to your Arduino port (e.g. 'COM4' or 'COM3')
  2. Measure arm link lengths → update in solverNNA.py
  3. Calibrate gripper open/close angles below
  4. Set your camera position in camera_compensation()
  5. Set your drop position in pick_up()

USAGE from main file:
    import arm_control
    arm_control.home()
    arm_control.pick_up(x_mm, y_mm)
"""

import serial
import time
import solverNNA

# ── SETTINGS — update these ───────────────────────────────────────
COM_PORT   = 'COM4'    # ← change to your Arduino COM port
BAUD_RATE  = 115200
MOVE_SPEED = 20        # ms per degree (lower = faster, min 5)

# ── JOINT DEFINITIONS [default, min, max, index] ──────────────────
base      = [90,   0, 180, 0]
shoulder  = [90,  15, 165, 1]
elbow     = [90,   0, 180, 2]
wrist     = [90,   0, 180, 3]
wristRot  = [90,   0, 180, 4]
gripper   = [73,  20,  90, 5]   # 20=open  90=closed — calibrate!

# ── Serial connection ─────────────────────────────────────────────
try:
    arm = serial.Serial(COM_PORT, BAUD_RATE, timeout=5)
    print(f"[ARM] Connected on {COM_PORT}")
    time.sleep(2)
    # Home arm on startup
    arm.write(b'H90,90,90,90,90,73,20\n')
    time.sleep(3)
    print("[ARM] Arm homed and ready.")
except serial.SerialException as e:
    print(f"[ARM] ERROR: Could not connect to {COM_PORT}")
    print(f"      {e}")
    print(f"      Check COM port and try again.")
    arm = None


# ── Core write function ───────────────────────────────────────────

def write_arduino(angles, speed=MOVE_SPEED):
    """
    Send angle command to Arduino.
    angles = [base, shoulder, elbow, wrist, wristRot, gripper]
    """
    if arm is None:
        print("[ARM] Not connected.")
        return

    a = list(angles)
    a[0] = 180 - a[0]    # invert base
    a[3] = 180 - a[3]    # invert wrist

    cmd = "P" + ",".join(str(int(x)) for x in a) + f",{speed}\n"
    arm.write(cmd.encode())
    # Wait for OK response
    try:
        resp = arm.readline().decode().strip()
    except Exception:
        pass


def _save_angles(angles):
    """Save current angles to prev_teta.txt for backlash compensation."""
    try:
        with open("prev_teta.txt", "w") as f:
            for a in angles:
                f.write(str(a) + ";")
    except Exception:
        pass


# ── High-level arm functions ──────────────────────────────────────

def home(speed=20):
    """Move arm to home position."""
    angles = [base[0], shoulder[0], elbow[0],
              wrist[0], wristRot[0], gripper[0]]
    write_arduino(angles, speed)
    _save_angles(angles)
    print("[ARM] Homed.")


def open_gripper():
    """Open gripper."""
    prev = solverNNA.get_previous_teta()
    angles = [prev[0], prev[1], prev[2],
              prev[3], prev[4], gripper[2]]   # gripper[2] = open angle
    write_arduino(angles)
    _save_angles(angles)


def close_gripper():
    """Close gripper."""
    prev = solverNNA.get_previous_teta()
    angles = [prev[0], prev[1], prev[2],
              prev[3], prev[4], gripper[1]]   # gripper[1] = closed angle
    write_arduino(angles)
    _save_angles(angles)


def write_position(theta_base=90, theta_shoulder=90, theta_elbow=90,
                   theta_wrist=90, theta_wristRot=90, grip="closed"):
    """Move to specific joint angles."""
    theta_gripper = gripper[1] if grip == "closed" else gripper[2]

    # Apply backlash compensation to base
    theta_base_comp = solverNNA.backlash_compensation_base(theta_base)

    angles = [theta_base_comp, theta_shoulder, theta_elbow,
              theta_wrist, theta_wristRot, theta_gripper]
    write_arduino(angles)

    # Save without compensation for next backlash calc
    _save_angles([theta_base, theta_shoulder, theta_elbow,
                  theta_wrist, theta_wristRot, theta_gripper])


def go_to_coordinate(x, y, z, grip="closed"):
    """
    Move arm tip to cartesian coordinate (x,y,z) in mm.
    Origin = base centre of arm.
    """
    theta_list = solverNNA.move_to_position_cart(x, y, z)
    write_position(
        theta_list[0], theta_list[1],
        theta_list[2], theta_list[3],
        grip=grip)


# ── Pick and place ────────────────────────────────────────────────

def pick_up(x, y):
    """
    Full pick-and-place sequence.
    x, y = object position in mm from arm base origin.

    TUNE THESE VALUES:
      drop_pos      → where to drop the object
      pick_up_height → how high object sits off table
    """
    drop_pos       = [310, 95]   # ← tune: where to drop object (mm)
    pick_up_height = 10          # ← tune: object height off table (mm)
    delay          = 1.2         # seconds between steps

    print(f"[ARM] Picking up at ({x},{y})")

    home()
    time.sleep(delay)

    # Move above object
    go_to_coordinate(x, y, 100, "closed")
    time.sleep(delay)

    # Open gripper
    open_gripper()
    time.sleep(delay)

    # Lower to object
    go_to_coordinate(x, y, pick_up_height, "open")
    time.sleep(delay)

    # Grip
    close_gripper()
    time.sleep(delay)

    # Lift
    go_to_coordinate(x, y, 200, "closed")
    time.sleep(delay)

    # Move to drop position
    go_to_coordinate(drop_pos[0], drop_pos[1], 200, "closed")
    time.sleep(delay)

    # Lower to drop height
    go_to_coordinate(drop_pos[0], drop_pos[1], 80, "closed")
    time.sleep(delay)

    # Release
    open_gripper()
    time.sleep(delay)

    home()
    print("[ARM] Pick complete.")


# ── Camera perspective compensation ──────────────────────────────

def camera_compensation(x_coordinate, y_coordinate):
    """
    Correct pixel-to-mm coordinate for camera height offset.

    TUNE THESE VALUES to match your actual setup:
      h_object        → height of objects off table (mm)
      camera_position → [x, y, z] of camera from arm base (mm)
    """
    h_object        = 30                  # ← tune: object height (mm)
    camera_position = [480, 150, 880]     # ← tune: camera position (mm)

    offset = 300
    x_coordinate = (offset - x_coordinate) + (camera_position[0] - offset)

    x_comp = x_coordinate - (h_object / (camera_position[2] / x_coordinate))

    if y_coordinate < camera_position[1]:
        y_comp = y_coordinate - (h_object / (camera_position[2] / y_coordinate))
    else:
        y_comp = y_coordinate + (h_object / (camera_position[2] / y_coordinate))

    x_comp = offset - (x_comp - (camera_position[0] - offset))

    return int(x_comp), int(y_comp)
