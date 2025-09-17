# Kinova Gen3 Vision-Guided Gelatin Cutting System

A robotics research project implementing vision-guided manipulation for precision cutting operations on gelatin materials using a Kinova Gen3 7-DOF robotic arm with built-in camera system.

## System Overview

This project demonstrates:
- Real-time computer vision for line detection and path planning
- Compliant impedance control for safe interaction with soft materials
- Hand-eye calibration for accurate vision-robot coordination
- Integration of built-in RGB+Depth cameras with robotic control

## Hardware Configuration

### Robot Platform
- **Model**: Kinova Gen3 7-DOF robotic arm
- **Specifications**: 902mm reach, 4kg payload capacity
- **End-effector**: Robotiq 2F-140 gripper with custom cutting tool attachment
- **Control**: 1 kHz closed-loop control with integrated torque sensors
- **Interface**: Ethernet connection to robot base

### Built-in Vision System
- **RGB Camera**: Omnivision OV5640 sensor (built-in to robot wrist)
  - Available resolutions: 1920×1080, 1280×720, 640×480 pixels
  - Frame rates: 30/15 fps configurable
- **Depth Sensor**: Intel RealSense D410 stereo depth module (built-in to robot wrist)
  - Provides synchronized RGBD data with RGB camera
  - Optimized for close-range manipulation tasks

### Network Setup
- **Robot IP Address**: 192.168.1.10 (configurable)
- **Connection Type**: Direct Ethernet to robot base controller
- **Network Requirements**: Robot and workstation must be on same subnet

## Software Environment

### Operating System Requirements
- Ubuntu 20.04 LTS
- ROS Noetic Ninjemys distribution

### Core ROS Packages Used
- `gen3_robotiq_2f_140_move_it_config` - MoveIt motion planning configuration
- `kinova_vision` - Built-in camera interface and drivers
- `line_follower` - Custom cutting application package
- `cv_calibration` - Hand-eye calibration utilities

### Python Dependencies
```
numpy>=1.19.0
opencv-python>=4.5.0
scipy>=1.7.0
matplotlib>=3.3.0
PyYAML>=5.4.0
```

## Installation Instructions

### 1. Clone Repository
```bash
git clone https://github.com/akhiljoshi7060/kinova-gelatin-cutting-system.git
cd kinova-gelatin-cutting-system
```

### 2. Install System Dependencies
```bash
# Install ROS packages
sudo apt-get update
sudo apt-get install ros-noetic-moveit
sudo apt-get install ros-noetic-cv-bridge
sudo apt-get install ros-noetic-image-transport
sudo apt-get install ros-noetic-tf2-ros

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Copy Files to Catkin Workspace
```bash
# Navigate to your catkin workspace
cd ~/catkin_workspace/src

# Ensure line_follower package exists (create if needed)
mkdir -p line_follower/scripts
mkdir -p line_follower/launch
mkdir -p cv_calibration/src

# Copy source files to correct locations
cp ~/kinova-gelatin-cutting-system/src/vision/Line_detection.py ~/catkin_workspace/src/line_follower/scripts/
cp ~/kinova-gelatin-cutting-system/src/control/gelatin_cutter.py ~/catkin_workspace/src/line_follower/scripts/
cp ~/kinova-gelatin-cutting-system/src/control/gen3_impedance_controller_v4.py ~/catkin_workspace/src/line_follower/scripts/
cp ~/kinova-gelatin-cutting-system/src/control/line_follower_impedance.py ~/catkin_workspace/src/line_follower/scripts/
cp ~/kinova-gelatin-cutting-system/src/planning/move_to_position.py ~/catkin_workspace/src/line_follower/scripts/
cp ~/kinova-gelatin-cutting-system/src/calibration/rviz_calibration.py ~/catkin_workspace/src/cv_calibration/src/

# Copy launch files
cp ~/kinova-gelatin-cutting-system/launch/*.launch ~/catkin_workspace/src/line_follower/launch/

# Make scripts executable
chmod +x ~/catkin_workspace/src/line_follower/scripts/*.py
chmod +x ~/catkin_workspace/src/cv_calibration/src/*.py
```

### 4. Build Workspace
```bash
cd ~/catkin_workspace
catkin_make
source devel/setup.bash

# Add to bashrc for automatic sourcing
echo "source ~/catkin_workspace/devel/setup.bash" >> ~/.bashrc
```

## System Operation

### Pre-Operation Checklist
1. Verify robot is connected to power and network
2. Confirm robot IP address is reachable: `ping 192.168.1.10`
3. Ensure workspace is clear of obstacles
4. Check that cutting tool is properly attached

### Launch Sequence (3 Terminals Required)

#### Terminal 1: Robot System with MoveIt
```bash
# Source ROS environment
source ~/catkin_workspace/devel/setup.bash

# Launch robot with motion planning (adjust IP if different)
roslaunch gen3_robotiq_2f_140_move_it_config real_robot_moveit.launch ip_address:=192.168.1.10
```

#### Terminal 2: Built-in Vision System
```bash
# Source ROS environment
source ~/catkin_workspace/devel/setup.bash

# Start built-in camera drivers
roslaunch kinova_vision kinova_vision.launch
```

#### Terminal 3: Cutting Application
```bash
# Source ROS environment
source ~/catkin_workspace/devel/setup.bash

# Execute main gelatin cutting program
rosrun line_follower gelatin_cutter.py
```

### Alternative Launch Commands
```bash
# For impedance control mode only
rosrun line_follower line_follower_impedance.py

# For motion planning utilities
rosrun line_follower move_to_position.py

# For vision processing only
rosrun line_follower Line_detection.py

# For hand-eye calibration
rosrun cv_calibration rviz_calibration.py
```

## Project File Structure

```
kinova-gelatin-cutting-system/
├── src/
│   ├── vision/
│   │   └── Line_detection.py                    # Computer vision algorithms
│   ├── control/
│   │   ├── gelatin_cutter.py                    # Main cutting application (41KB)
│   │   ├── gen3_impedance_controller_v4.py      # Custom impedance controller
│   │   └── line_follower_impedance.py           # Vision-force integration
│   ├── planning/
│   │   └── move_to_position.py                  # Motion planning utilities
│   └── calibration/
│       └── rviz_calibration.py                  # Hand-eye calibration system
├── config/
│   └── robot/                                   # Robot configuration files
│       ├── joint_limits.yaml
│       ├── kinematics.yaml
│       ├── cartesian_limits.yaml
│       └── [other MoveIt configs]
├── launch/                                      # ROS launch files
│   ├── hand_eye_calibration.launch
│   ├── latest_camera_pose.launch
│   └── line_follower_impedance.launch
├── examples/
│   └── images/                                  # Calibration sample images
│       ├── left-0000.png through left-0044.png # 45 calibration images
└── docs/                                        # Documentation
```

## Core System Components

### Vision Processing Module (`Line_detection.py`)
- Real-time line detection using OpenCV algorithms
- Processes built-in camera feed at 1280×720 resolution @ 30 fps
- Converts image pixel coordinates to robot coordinate frame
- Generates waypoint trajectories for cutting paths
- Handles lighting variations and noise filtering

### Main Cutting Application (`gelatin_cutter.py`)
- Primary application file (41,387 bytes)
- Integrates vision processing with robot control
- Implements safety monitoring and emergency stop procedures
- Manages cutting sequence execution and progress tracking
- Provides real-time force feedback monitoring

### Impedance Control System (`gen3_impedance_controller_v4.py`)
- Custom compliant control algorithms for soft material interaction
- Maintains cutting forces below 5N threshold
- Real-time torque sensor feedback integration
- Adaptive stiffness control based on material properties
- Safety-critical force limiting mechanisms

### Vision-Force Integration (`line_follower_impedance.py`)
- Combines visual servoing with force control
- Real-time path correction based on visual feedback
- Adaptive cutting parameters based on material response
- Continuous monitoring of cutting quality

### Motion Planning Interface (`move_to_position.py`)
- MoveIt motion planning integration
- Collision-free trajectory generation
- Joint space and Cartesian space planning modes
- Workspace safety boundary enforcement
- Emergency stop and recovery procedures

### Calibration System (`rviz_calibration.py`)
- Hand-eye calibration implementation
- Camera intrinsic and extrinsic parameter estimation
- Calibration accuracy validation tools
- Automated calibration data collection

## Performance Specifications

### Demonstrated Capabilities
- **Cutting Precision**: ±0.5mm accuracy achieved
- **Force Control**: Maintains cutting forces <5N consistently
- **Vision Processing**: Real-time operation at 30 FPS
- **Complex Patterns**: Successfully cuts geometric shapes on gelatin
- **Calibration**: 45-point hand-eye calibration for accuracy

### System Response Times
- Vision processing latency: <33ms per frame
- Force control loop: 1 kHz update rate
- Motion planning: Variable based on complexity
- Emergency stop response: <100ms

## Troubleshooting Guide

### Common Issues and Solutions

#### Robot Connection Problems
```bash
# Test robot connectivity
ping 192.168.1.10

# Check ROS master configuration
echo $ROS_MASTER_URI

# Verify robot joint states
rostopic echo /joint_states
```

#### Vision System Issues
```bash
# Check camera topics availability
rostopic list | grep camera

# Test image stream
rostopic echo /camera/color/image_raw

# Verify camera calibration
rostopic echo /camera/color/camera_info
```

#### MoveIt Planning Issues
```bash
# Check MoveIt status
rostopic echo /move_group/status

# Verify planning scene
rostopic echo /planning_scene

# Test robot model loading
rosrun tf tf_echo base_link tool_frame
```

### Error Recovery Procedures
1. Emergency stop: Press robot emergency stop button
2. Software stop: Ctrl+C in all terminal windows
3. Robot reset: Power cycle robot and restart launch sequence
4. Calibration check: Re-run hand-eye calibration if accuracy issues

## Safety Protocols

### Pre-Operation Safety Checks
- Verify emergency stop functionality
- Check workspace boundaries are configured
- Ensure cutting tool is securely attached
- Confirm force sensor calibration is current

### During Operation Monitoring
- Continuous force feedback monitoring
- Visual inspection of cutting progress
- Emergency stop accessibility maintained
- Workspace clear of personnel

### Post-Operation Procedures
- Return robot to safe home position
- Power down in correct sequence
- Secure cutting tools safely
- Document any operational issues

## Development and Customization

### Adding New Cutting Patterns
1. Modify vision processing algorithms in `Line_detection.py`
2. Update path planning logic as needed
3. Test with simulation before hardware deployment
4. Validate cutting quality and safety

### Calibration Maintenance
- Recommend re-calibration every 30 days of operation
- After any camera or robot maintenance
- If cutting accuracy degrades below specifications
- When changing end-effector tools

## Contact Information

- **Author**: Akhil Joshi
- **GitHub**: [@akhiljoshi7060](https://github.com/akhiljoshi7060)
- **Email**: akhiljoshi436@gmail.com

## License

This project is licensed under the MIT License - see the LICENSE file for complete details.

## Acknowledgments

- Kinova Robotics for Gen3 platform and built-in vision system
- Intel RealSense team for depth sensing technology
- MoveIt community for motion planning framework
- OpenCV contributors for computer vision libraries

## Citation

If you use this work in research, please cite:
```bibtex
@misc{joshi2025kinova_gelatin_cutting,
  title={Vision-Guided Gelatin Cutting with Kinova Gen3 7-DOF Robotic Arm},
  author={Joshi, Akhil},
  year={2025},
  howpublished={\url{https://github.com/akhiljoshi7060/kinova-gelatin-cutting-system}}
}
