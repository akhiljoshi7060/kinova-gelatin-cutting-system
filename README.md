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

### 3. Build Workspace
```bash
# Navigate to project directory
cd kinova-gelatin-cutting-system

# Build the project
catkin_make
source devel/setup.bash

# Add to bashrc for automatic sourcing
echo "source $(pwd)/devel/setup.bash" >> ~/.bashrc
```

## Project File Structure

```
kinova-gelatin-cutting-system/
├── Calibration_images_and_data/               # Camera calibration datasets and parameters
├── cartesian_impedance_controller/            # Custom impedance control implementation
├── config/                                    # Robot and system configuration files
├── depends/                                   # Project dependencies and external libraries
├── devel/                                     # Development build files (catkin workspace)
├── Frames/                                    # Reference frame definitions and transformations
├── launch/                                    # ROS launch files for system startup
├── Line_detection_images_analysis/           # Computer vision analysis and results
├── results/                                   # Experimental data and performance metrics
├── src/                                       # Source code (Python scripts and ROS nodes)
└── README.md                                 # Project documentation (this file)
```

## Core System Components

### Computer Vision System (`Line_detection_images_analysis/`)
- **Line Detection Algorithm**: Advanced OpenCV-based line detection and tracking
- **Real-time Processing**: Processes built-in camera feed at 1280×720 @ 30 fps
- **Image Analysis**: Comprehensive analysis tools and results
- **Coordinate Transformation**: Converts image pixels to robot coordinate frame
- **Path Planning**: Generates optimal cutting trajectories from detected lines

### Cartesian Impedance Controller (`cartesian_impedance_controller/`)
- **Based on Research**: Implementation inspired by Mayr & Salt-Ducaju (2024) C++ Cartesian Impedance Controller
- **Custom Implementation**: Specialized control algorithms for compliant manipulation
- **Compliant Control**: Maintains safe interaction forces with soft materials
- **Real-time Feedback**: 1 kHz control loop with integrated torque sensors
- **Adaptive Stiffness**: Dynamic stiffness adjustment based on material properties
- **Safety Mechanisms**: Force limiting and emergency stop integration

### Calibration System (`Calibration_images_and_data/`)
- **Hand-eye Calibration**: Comprehensive calibration datasets and parameters
- **Camera Intrinsics**: Pre-computed calibration values for built-in cameras
- **Calibration Images**: Complete dataset for system calibration
- **Validation Tools**: Calibration accuracy verification utilities

### Configuration Management (`config/`)
- **Robot Parameters**: Joint limits, kinematics, and workspace definitions
- **System Settings**: Camera parameters and vision processing configurations
- **Safety Parameters**: Force limits and emergency stop configurations
- **Calibration Data**: Stored calibration matrices and transformations

### Launch System (`launch/`)
- **Modular Launch Files**: Organized startup scripts for different system modes
- **System Integration**: Complete system launch configurations
- **Component Control**: Individual launch files for subsystem testing
- **Parameter Management**: Centralized configuration through launch parameters

### Frame Management (`Frames/`)
- **Coordinate Systems**: Robot base, tool, and camera frame definitions
- **Transformations**: Spatial relationships between coordinate frames
- **Calibration Results**: Hand-eye calibration transformation matrices
- **Visualization**: Frame relationship diagrams and validation tools

### Source Code (`src/`)
- **Core Algorithms**: Main implementation files for vision and control
- **ROS Nodes**: Custom ROS nodes for system integration
- **Utility Scripts**: Helper functions and debugging tools
- **Integration Layer**: Interfaces between vision, control, and planning systems

### Dependencies (`depends/`)
- **External Libraries**: Required third-party packages and libraries
- **ROS Packages**: Custom ROS package dependencies
- **API Interfaces**: Kinova Kortex API and related components
- **Development Tools**: Build and debugging utilities

## System Operation

### Launch Sequence (3 Terminals Required)

**Terminal 1: Robot System with MoveIt**
```bash
roslaunch gen3_robotiq_2f_140_move_it_config real_robot_moveit.launch ip_address:=192.168.1.10
```

**Terminal 2: Built-in Vision System**
```bash
roslaunch kinova_vision kinova_vision.launch
```

**Terminal 3: Main Cutting Application**
```bash
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

### Calibration Procedures
```bash
# Hand-eye calibration
rosrun cv_calibration rviz_calibration.py

# Verify camera topics
rostopic list | grep camera

# Check frame relationships
rosrun tf tf_echo base_link camera_link
```

## Performance Specifications

### Demonstrated Capabilities
- **Cutting Precision**: ±0.5mm accuracy achieved
- **Force Control**: Maintains cutting forces <5N consistently
- **Vision Processing**: Real-time operation at 30 FPS
- **Complex Patterns**: Successfully cuts geometric shapes on gelatin
- **Calibration**: Multi-point hand-eye calibration for sub-millimeter accuracy

### System Response Times
- Vision processing latency: <33ms per frame
- Force control loop: 1 kHz update rate
- Motion planning: Variable based on complexity
- Emergency stop response: <100ms

### Data Management
- **Results Storage**: Experimental data and performance metrics in `results/`
- **Calibration Archive**: Complete calibration datasets in `Calibration_images_and_data/`
- **Analysis Pipeline**: Computer vision analysis tools in `Line_detection_images_analysis/`
- **Configuration Backup**: System settings preserved in `config/`

## Troubleshooting Guide

### Build Issues
```bash
# Clean build (if using catkin structure)
rm -rf devel/
catkin_make clean
catkin_make

# Check dependencies
ls depends/
```

### Calibration Problems
```bash
# Check calibration data
ls -la Calibration_images_and_data/

# Verify frame definitions
ls -la Frames/

# Review calibration parameters
cat config/camera_calibration.yaml  # if exists
```

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
# Check camera topics
rostopic list | grep camera

# Test image stream
rostopic echo /camera/color/image_raw

# Verify camera calibration
rostopic echo /camera/color/camera_info
```

## Development and Customization

### Adding New Features
1. Implement new algorithms in `src/`
2. Add configuration files to `config/`
3. Create launch files in `launch/`
4. Test with existing calibration data
5. Document results in `results/`

### Calibration Maintenance
- Use existing calibration data in `Calibration_images_and_data/`
- Archive new calibrations: `tar -cf new_calibration.tar Calibration_images_and_data/`
- Validate with frame visualization tools
- Update launch files with new parameters

### Data Analysis
- Process experimental data in `results/`
- Use image analysis tools in `Line_detection_images_analysis/`
- Generate reports with video documentation from `Videos/`
- Archive important datasets for future reference

## Safety Protocols

### Pre-Operation Safety Checks
- Verify emergency stop functionality
- Check workspace boundaries are configured
- Ensure cutting tool is securely attached
- Confirm force sensor calibration is current
- Validate frame relationships using `frames.pdf`

### During Operation Monitoring
- Continuous force feedback monitoring through impedance controller
- Visual inspection of cutting progress
- Emergency stop accessibility maintained
- Real-time analysis of vision processing results

### Post-Operation Procedures
- Return robot to safe home position
- Save experimental data to `results/`
- Power down in correct sequence
- Backup calibration data if modified

## Contact Information

- **Author**: Akhil Joshi
- **GitHub**: [@akhiljoshi7060](https://github.com/akhiljoshi7060)
- **Email**: akhiljoshi436@gmail.com

## License

This project is licensed under the MIT License - see the LICENSE file for complete details.

## Acknowledgments

- **Kinova Robotics** for Gen3 platform, built-in vision system, and ROS integration
  - Repository: [https://github.com/kinovarobotics](https://github.com/kinovarobotics)
  - Kinova Gen3 ROS packages and SDK
- **Cartesian Impedance Controller** implementation based on:
  - Mayr, M. & Salt-Ducaju, J.M. (2024). A C++ Implementation of a Cartesian Impedance Controller for Robotic Manipulators. *Journal of Open Source Software*, 9(93), 5194.
- Intel RealSense team for depth sensing technology
- MoveIt community for motion planning framework
- OpenCV contributors for computer vision libraries
- ROS community for robotics middleware and tools

## Citation

If you use this work in research, please cite:

### This Work
```bibtex
@misc{joshi2025kinova_gelatin_cutting,
  title={Vision-Guided Gelatin Cutting with Kinova Gen3 7-DOF Robotic Arm},
  author={Joshi, Akhil},
  year={2025},
  howpublished={\url{https://github.com/akhiljoshi7060/kinova-gelatin-cutting-system}}
}
```

### Dependencies and References
```bibtex
@article{mayr2024cartesian,
  doi = {10.21105/joss.05194},
  url = {https://doi.org/10.21105/joss.05194},
  year = {2024},
  publisher = {The Open Journal},
  volume = {9},
  number = {93},
  pages = {5194},
  author = {Matthias Mayr and Julian M. Salt-Ducaju},
  title = {A C++ Implementation of a Cartesian Impedance Controller for Robotic Manipulators},
  journal = {Journal of Open Source Software}
}

@misc{kinova_robotics_ros,
  title = {Kinova Robotics ROS Packages},
  author = {Kinova Robotics},
  howpublished = {\url{https://github.com/kinovarobotics}},
  note = {Accessed: 2025}
}
```
