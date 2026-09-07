Vision-Based Pick-and-Place Robotic Arm
A final-year B.Tech project that combines computer vision, voice commands, inverse kinematics, and Arduino-based servo control to detect an object and move a robotic arm to pick and place it.

Main features
Detects the working area using four ArUco markers
Corrects the camera view using perspective transformation
Detects common objects using YOLO-World or YOLOv8
Accepts an object name through voice input
Converts image coordinates into physical coordinates
Calculates robot joint angles using inverse kinematics
Sends movement commands to an Arduino through serial communication
Performs an automatic pick-and-place sequence
System flow
A USB camera captures the workspace.
ArUco markers with IDs 1, 2, 3, and 4 define its boundaries.
YOLO identifies objects inside the workspace.
The user selects an object using a voice command.
The program converts the object's pixel position to millimetres.
The inverse-kinematics solver calculates the joint angles.
The Arduino moves the robotic arm to pick and place the object.
Project structure
File	Purpose
Aruco_detection_V2_USB.py	Main program for camera input, detection, voice commands, and robot-arm operation
ArucoDetection_definitions.py	ArUco-marker and perspective-transformation helper functions
arm_control.py	Serial communication and pick-and-place motion sequence
solverNNA.py	Inverse-kinematics and backlash-compensation calculations
requirements.txt	Required Python packages
firmware/README.md	Instructions for adding the Arduino firmware
Hardware used
Arduino Uno
5/6-DOF robotic arm with servo motors
USB camera
External regulated servo power supply
Four printed ArUco markers
Computer running Python
Never power all arm servos directly from the Arduino 5 V pin. Use a suitable external power supply and connect its ground to the Arduino ground.

Installation
Python 3.10 or 3.11 is recommended.

git clone https://github.com/Bajilep/AUTOMATIC-PICK-AND-PLACE-ROBOT.git
cd AUTOMATIC-PICK-AND-PLACE-ROBOT
python -m venv .venv
Activate the virtual environment on Windows:

.venv\Scripts\activate
Install the dependencies:

pip install -r requirements.txt
Ultralytics downloads the required pretrained YOLO model automatically during the first run. Therefore, model weight files are not stored in this repository.

Configuration
Before running the program:

Upload the matching Arduino firmware to the Arduino Uno.
Set COM_PORT in arm_control.py, for example COM4 on Windows.
Check the arm link lengths l0, l1, l2, and l3 in solverNNA.py.
Set the measured workspace width and height in Aruco_detection_V2_USB.py.
Calibrate the gripper angles, drop position, camera position, and movement limits.
Place ArUco markers 1–4 at the four workspace corners.
Set CAMERA_INDEX to 0 or 1, depending on the connected camera.
Run
python Aruco_detection_V2_USB.py
Controls
Click the blue voice button and say the object name.
Press C to clear the selected object.
Press + or - to adjust detection confidence.
Press T to change the detection-object list.
Press Q to stop the program and return the arm home.
Safety
Test with low movement speed and a clear workspace.
Keep an emergency power-disconnect switch nearby.
Verify servo limits before running automatic movement.
Confirm every coordinate is reachable before lowering the gripper.
Keep hands away from the arm while it is powered.
Current limitation
The Arduino firmware referenced by the Python program was not present in the original project archive. Add the matching .ino file under firmware/ before reproducing the complete hardware setup.

Author
Bajil Mohammed E.P.
