# Kinova Gen3 Vision-Guided Gelatin Cutting System

A robotics research project that combines computer vision, impedance control, and motion planning to perform precision cutting operations on gelatin materials using a Kinova Gen3 7-DOF robotic arm with embedded vision module.

## Overview

This system demonstrates vision-guided manipulation using:
- Real-time line detection and path planning
- Compliant impedance control for soft material interaction
- Hand-eye calibration for accurate vision-robot coordination
- Integration of embedded RGB+Depth cameras with robotic control

## Hardware Setup

### Robot Configuration
- **Model**: Kinova Gen3 7-DOF robotic arm
- **Reach**: 902mm
- **Payload**: 4kg
- **End-effector**: Robotiq 2F-140 gripper with custom cutting tool
- **Control**: 1 kHz closed-loop with integrated torque sensors

### Vision System
- **RGB Camera**: Omnivision OV5640 (embedded in wrist)
  - Resolutions: 1920×1080, 1280×720, 640×480
  - Frame rates: 30/15 fps
- **Depth Sensor**: Intel RealSense D410 (embedded in wrist)
  - Stereo depth sensing
  - RGBD data capture synchronized with RGB

### Network Configuration
- **Robot IP**: 192.168.1.10 (configurable)
- **Connection**: Ethernet to robot base
- **Requirements**: Same network subnet for robot and workstation

## Software Dependencies

### Operating System
- Ubuntu 20.04 LTS
- ROS Noetic Ninjemys

### ROS Packages
- `gen3_robotiq_2f_140_move_it_config` - MoveIt configuration
- `kinova_vision` - Vision system interface
- `line_follower` - Custom cutting application package
- `cv_calibration` - Hand-eye calibration utilities

### Python Dependencies
- OpenCV 4.x
- NumPy
- SciPy
- Matplotlib
- ROS Python libraries (rospy, cv_bridge, etc.)

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/akhiljoshi7060/kinova-gelatin-cutting-system.git
cd kinova-gelatin-cutting-system
```

### 2. Copy to Catkin Workspace
```bash
# Copy files to your catkin workspace
cp -r src/* ~/catkin_workspace/src/
cp -r config/* ~/catkin_workspace/src/your_package/config/
cp -r launch/* ~/catkin_workspace/src/your_package/launch/
```

### 3. Build Workspace
```bash
cd ~/catkin_workspace
catkin_make
source devel/setup.bash
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

## System Operation

### Startup Sequence

#### Terminal 1: Launch Robot System
```bash
# Start robot with MoveIt planning (adjust IP address as needed)
roslaunch gen3_robotiq_2f_140_move_it_config real_robot_moveit.launch ip_address:=192.168.1.10
```

#### Terminal 2: Start Vision System
```bash
# Launch embedded camera system
roslaunch kinova_vision kinova_vision.launch
```

#### Terminal 3: Run Cutting Application
```bash
# Execute main cutting program
rosrun line_follower gelatine_cutter.py
```

### Alternative Launch Commands
```bash
# For impedance control mode
rosrun line_follower line_follower_impedance.py

# For motion planning only
rosrun line_follower move_to_position.py

# For vision processing only
rosrun line_follower Line_detection.py
```

## Project Structure

```
kinova-gelatin-cutting-system/
├── src/
│   ├── vision/
│   │   └── Line_detection.py           # Computer vision algorithms
│   ├── control/
│   │   ├── gelatine_cutter.py          # Main cutting application
│   │   ├── gen3_impedance_controller_v4.py  # Impedance control
│   │   └── line_follower_impedance.py  # Vision-force integration
│   ├── planning/
│   │   └── move_to_position.py         # Motion planning utilities
│   └── calibration/
│       └── rviz_calibration.py         # Hand-eye calibration
├── config/
│   └── robot/                          # MoveIt and robot configs
├── launch/                             # ROS launch files
├── examples/
│   └── images/                         # Calibration images (left-*.png)
├── docs/                               # Documentation
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Core Components

### Vision Processing (`src/vision/`)
- **Line_detection.py**: Real-time line detection using OpenCV
  - Processes embedded camera feed (1280×720 @ 30 fps)
  - Converts image coordinates to robot coordinate frame
  - Generates cutting waypoints for path following

### Control System (`src/control/`)
- **gelatine_cutter.py**: Main application (41KB)
  - Integrates vision, planning, and control
  - Safety monitoring and emergency stops
  - Force feedback for compliant cutting
  
- **gen3_impedance_controller_v4.py**: Custom impedance controller
  - Compliant force control for soft materials
  - Maintains cutting forces below 5N
  - Real-time force/torque monitoring

- **line_follower_impedance.py**: Vision-control integration
  - Combines visual servoing with force control
  - Adaptive path correction based on visual feedback

### Planning System (`src/planning/`)
- **move_to_position.py**: Motion planning interface
  - MoveIt integration for collision-free paths
  - Joint space and Cartesian space planning
  - Safety boundary enforcement

### Calibration System (`src/calibration/`)
- **rviz_calibration.py**: Hand-eye calibration utility
  - Camera intrinsic and extrinsic calibration
  - Hand-eye transformation estimation
  - Calibration validation tools

## Configuration

### Robot Parameters
Located in `config/robot/`:
- `joint_limits.yaml` - Joint velocity and acceleration limits
- `kinematics.yaml` - Kinematic solver configuration
- `cartesian_limits.yaml` - Cartesian velocity limits
- `sensors_3d.yaml` - 3D sensor configuration

### Network Configuration
Update robot IP address in launch files:
```bash
# Edit the IP address in launch commands
ip_address:=192.168.1.10  # Change to your robot's IP
```

### Vision Parameters
Camera settings are configured through the `kinova_vision` package.

## Operation Workflow

### 1. System Preparation
- Connect robot to power and network
- Verify robot is in home position
- Check camera feeds are active

### 2. Calibration (if needed)
- Run hand-eye calibration procedure
- Validate calibration accuracy
- Save calibration parameters

### 3. Cutting Operation
- Place gelatin material in workspace
- Start vision processing
- Execute cutting sequence
- Monitor force feedback

### 4. Safety Features
- Emergency stop capability
- Force monitoring (torque sensors)
- Collision detection (MoveIt)
- Workspace boundary limits

## Results and Performance

### Demonstrated Capabilities
- Cutting accuracy: ±0.5mm precision
- Force control: Maintains <5N cutting forces
- Vision processing: Real-time at 30 FPS
- Complex pattern cutting on gelatin materials

### Sample Data
- Calibration images: `examples/images/left-*.png`
- 45 calibration images captured during hand-eye calibration
- Multiple cutting pattern examples

## Troubleshooting

### Common Issues
- **Robot connection failed**: Check IP address and network connectivity
- **Camera not detected**: Verify kinova_vision launch successful
- **Poor line detection**: Adjust lighting conditions and vision parameters
- **Force control issues**: Check torque sensor calibration

### Debug Commands
```bash
# Check robot connection
rostopic echo /joint_states

# Verify camera feeds
rostopic echo /camera/color/image_raw

# Monitor force feedback
rostopic echo /force_torque_sensor

# Check MoveIt planning
rostopic echo /move_group/status
```

## Development

### Adding New Features
1. Create feature branch from main
2. Implement changes in appropriate module
3. Test with robot hardware
4. Update documentation
5. Submit pull request

### Code Style
- Follow PEP 8 for Python code
- Use descriptive variable names
- Include docstrings for all functions
- Add safety checks for robot operations

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## Citation

If you use this work in research, please cite:

```bibtex
@misc{joshi2025kinova_cutting,
  title={Vision-Guided Gelatin Cutting with Kinova Gen3 Robotic Arm},
  author={Akhil Joshi},
  year={2025},
  howpublished={\url{https://github.com/akhiljoshi7060/kinova-gelatin-cutting-system}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Akhil Joshi
- **GitHub**: [@akhiljoshi7060](https://github.com/akhiljoshi7060)
- **Email**: akhiljoshi436@gmail.com

## Acknowledgments

- Kinova Robotics for Gen3 platform and embedded vision module
- Intel RealSense team for depth sensing technology
- MoveIt community for motion planning framework
- OpenCV contributors for computer vision tools
