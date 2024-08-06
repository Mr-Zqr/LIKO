# LIKO
A LiDAR-Inertial-Kinematic Odometry (LIKO) for biped robot state estimation. 

Paper: [LIKO: LiDAR, Inertial, and Kinematic Odometry for Bipedal Robots](https://arxiv.org/abs/2404.18047)

The code implementation is based on [FAST_LIO](https://github.com/hku-mars/FAST_LIO).

## LIKO dataset
Rosbag can be downloaded from [google drive](https://drive.google.com/drive/folders/1tK65gU_lPM_HGoSTMXppqyoMq6ejWbsH?usp=drive_link).

Topic structure: 
```
/bhr_b3/foot_state    : sensor_msgs/JointState         # calculated robot kinematic
/bhr_b3/imu           : sensor_msgs/Imu                # imu measurement
/bhr_b3/l_foot_force  : geometry_msgs/WrenchStamped    # left foot F/T sensor
/bhr_b3/r_foot_force  : geometry_msgs/WrenchStamped    # right foot F/T sensor
/velodyne_points      : sensor_msgs/Pointcloud2        # LiDAR point cloud
/vicon/bhr_b3         : geometry_msgs/TransformStamped # VICON groundtruth
```

## System Overview
![system-overview](https://github.com/Mr-Zqr/LIKO/assets/62141613/60168789-2b43-41d6-88af-e759f3476c0f)

## 1. Prerequisites
### 1.1 **Ubuntu** and **ROS**
**Ubuntu >= 16.04**

For **Ubuntu 18.04 or higher**, the **default** PCL and Eigen is enough for FAST-LIO to work normally.

ROS    >= Melodic. [ROS Installation](http://wiki.ros.org/ROS/Installation)

### 1.2. **PCL && Eigen**
PCL    >= 1.8,   Follow [PCL Installation](http://www.pointclouds.org/downloads/linux.html).

Eigen  >= 3.3.4, Follow [Eigen Installation](http://eigen.tuxfamily.org/index.php?title=Main_Page).

### 1.3. **livox_ros_driver**
Follow [livox_ros_driver Installation](https://github.com/Livox-SDK/livox_ros_driver).

## 2. Build
Clone the repository and catkin_make:

```
    cd ~/$A_ROS_DIR$/src
    git clone https://github.com/Mr-Zqr/LIKO.git
    cd LIKO
    git submodule update --init
    cd ../..
    catkin_make
    source devel/setup.bash
```
- Remember to source the livox_ros_driver before build (follow 1.3 **livox_ros_driver**)
- If you want to use a custom build of PCL, add the following line to ~/.bashrc
```export PCL_ROOT={CUSTOM_PCL_PATH}```
## 3. Directly Run
Note: 

The LIKO takes calculated robot kinematic (foot position and velocity w.r.t. IMU frame) instead of raw joint encoder measurements as kinematic input. This is because these values are often calculated in the control end and there is no need for recalculation in such short time. 

The LIKO expects kinematic measurements as a `sensor_msgs::JointState` ROS topic with values in the following sequence: left foot position and velocity in x, y, and z axis; right foot position and velocity in x, y and z axis. 

---
To run the algorithm: 
```
    cd ~/$LIKO_ROS_DIR$
    source devel/setup.bash
    roslaunch liko mapping_$your_lidar_type$.launch
```
