# Arduino firmware

Add the Arduino sketch used to control the robotic arm in this directory, for example:

```text
firmware/
└── arduino_mg995_arm/
    └── arduino_mg995_arm.ino
```

The current Python controller expects serial commands using a format similar to:

- `H90,90,90,90,90,73,20` for homing
- `P<base>,<shoulder>,<elbow>,<wrist>,<wrist_rotation>,<gripper>,<speed>` for positioning

The exact Arduino implementation must match the physical arm, servo pins, limits, and serial protocol.
