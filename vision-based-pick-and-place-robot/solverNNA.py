# -*- coding: utf-8 -*-
"""
solverNNA.py
============
Inverse kinematics solver for kitsguru 6DOF arm.

IMPORTANT — measure your actual arm and update these link lengths:
  l0 = height from ground to shoulder pivot  (mm)
  l1 = shoulder to elbow                     (mm)
  l2 = elbow to wrist                        (mm)
  l3 = wrist to gripper tip                  (mm)

Current values are estimates — tune after testing.
"""

from math import sqrt, acos, asin, atan, degrees, radians, sin
import numpy as np

# ── LINK LENGTHS — UPDATE THESE to match your actual arm ──────────
l0 = 95.0    # ground → shoulder pivot  (mm)
l1 = 110.0   # shoulder → elbow         (mm)
l2 = 110.0   # elbow → wrist            (mm)
l3 = 155.0   # wrist → gripper tip      (mm)


def move_to_position_cart(x, y, z):
    """
    Convert cartesian (x,y,z) mm coordinate to
    [base, shoulder, elbow, wrist] joint angles in degrees.

    Origin = base centre of arm on table.
    x = forward/backward (mm)
    y = left/right       (mm)
    z = height           (mm)
    """
    r_compensation = 1.02   # 2% reach compensation
    z = z + 15              # backlash compensation

    r_hor = sqrt(x**2 + y**2)
    r     = sqrt(r_hor**2 + (z - l0)**2) * r_compensation

    # Base rotation angle
    if y == 0:
        theta_base = 180 if x <= 0 else 0
    else:
        theta_base = 90 - degrees(atan(x / y))

    # Shoulder / elbow / wrist
    try:
        alpha1        = acos((r - l2) / (l1 + l3))
        theta_shoulder = degrees(alpha1)

        alpha3        = asin((sin(alpha1) * l3 - sin(alpha1) * l1) / l2)
        theta_elbow   = (90 - degrees(alpha1)) + degrees(alpha3)
        theta_wrist   = (90 - degrees(alpha1)) - degrees(alpha3)

        if theta_wrist <= 0:
            alpha1         = acos((r - l2) / (l1 + l3))
            theta_shoulder = degrees(alpha1 + asin((l3 - l1) / r))
            theta_elbow    = 90 - degrees(alpha1)
            theta_wrist    = 90 - degrees(alpha1)

        # Height adjustment
        if z != l0:
            theta_shoulder += degrees(atan((z - l0) / r))

        # Mount compensation
        theta_elbow += 5
        theta_wrist += 5

    except (ValueError, ZeroDivisionError):
        # Out of reach — return safe home angles
        print(f"[SOLVER] Position ({x},{y},{z}) out of reach — returning home.")
        return [90, 90, 90, 90]

    return [
        round(theta_base),
        round(theta_shoulder),
        round(theta_elbow),
        round(theta_wrist)
    ]


def get_previous_teta():
    """Read last used angles from prev_teta.txt"""
    try:
        with open("prev_teta.txt", "r") as f:
            content = f.read()
        parts = content.split(";")
        parts = [p for p in parts if p.strip()]
        return [int(p) for p in parts[:6]]
    except Exception:
        return [90, 90, 90, 90, 90, 73]   # safe default


def backlash_compensation_base(theta_base):
    """Apply backlash compensation to base angle."""
    theta_base      = round(theta_base)
    theta_base_comp = theta_base

    compensation_CW  = 8
    compensation_CCW = np.linspace(0, 14, 135)

    try:
        prev_angles      = get_previous_teta()
        theta_base_prev  = prev_angles[0]
        delta            = theta_base - theta_base_prev

        if delta > 1:
            if theta_base > 45:
                idx             = int(round(theta_base - 46))
                idx             = min(idx, len(compensation_CCW) - 1)
                theta_base_comp = round(theta_base + compensation_CCW[idx])
        if delta < -1:
            theta_base_comp = theta_base - compensation_CW

    except Exception:
        pass

    return theta_base_comp
