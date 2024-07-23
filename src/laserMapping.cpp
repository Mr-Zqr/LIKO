// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <pinocchio/fwd.hpp>
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/WrenchStamped.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"

#include <ikd-Tree/ikd_Tree.h>
#include <pinocchio/parsers/urdf.hpp>
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/model.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include <ros/callback_queue.h>

#include "serow/serow_method.hpp"

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)
#define M_PI		3.14159265358979323846	/* pi */

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = true, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic, encoder_topic, l_foot_force_topic, r_foot_force_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited, can_clean_imu = false, new_imu_meas = false;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
deque<sensor_msgs::JointState::ConstPtr> encoder_buffer;
deque<geometry_msgs::WrenchStamped::ConstPtr> l_foot_force_buffer;
deque<geometry_msgs::WrenchStamped::ConstPtr> r_foot_force_buffer;

bool contact_changed = false;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 15, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;
sensor_msgs::Imu bitbotSE;
nav_msgs::Odometry VelocityMeas;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

// pinocchio variabiles.
pinocchio::Model model_;
pinocchio::Data data_;
Eigen::VectorXd q_;
std::string urdf_file;
Matrix3d R_base_imu;
Vector3d t_base_imu;
Vector3d t_imu_foot;

int Lfoot_id;
int Rfoot_id;
int IMU_id;

VectorXd joint_angle(10), joint_vel(10);
Vector3d v_meas;

static bool lidar_update=false;
static Eigen::Vector3d velocity_residual(0,0,0);
static Eigen::Vector3d position_residual(0,0,0);

int num_predicted_imu_meas = 0;
double kinematic_cov, kinematic_update_cov;


static V3D foot_position_kinematic, body_position_kinematic;
Matrix3d R_base_rfoot, R_base_lfoot, R_base_foot;
Vector3d t_base_rfoot, t_base_lfoot;
Matrix<double, 3, 3> R_se;
Matrix<double,3,10> J_fix_velocity;
Matrix<double, 6, 16> J_rfoot, J_lfoot;


using namespace std; //DEBUG

enum ContactPoint
{
    NONE = 0,
    LeftFoot,
    RightFoot,
    BothFeet
};

static ContactPoint next_contact;
static ContactPoint prev_contact;
static int contact_change_num = 0;

SerowMethod Serow_shimitt(false);
std::string support_leg_prev;
static ContactPoint contact_point_serow_shimitt = ContactPoint::NONE;
Eigen::Vector3d LLegGRF, RLegGRF, LLegGRT, RLegGRT;

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (Lidar_R_wrt_IMU*p_body + Lidar_T_wrt_IMU) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (Lidar_R_wrt_IMU*p_body + Lidar_T_wrt_IMU) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (Lidar_R_wrt_IMU*p_body + Lidar_T_wrt_IMU) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (Lidar_R_wrt_IMU*p_body + Lidar_T_wrt_IMU) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(Lidar_R_wrt_IMU*p_body_lidar + Lidar_T_wrt_IMU);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    // pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]); // fabs() 绝对值
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();

}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();
    
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void encoder_cbk(const sensor_msgs::JointState::ConstPtr &msg)
{
    // mtx_buffer.lock();
    encoder_buffer.push_back(msg);
    // mtx_buffer.unlock();
    // sig_buffer.notify_all();
}

void l_foot_force_cbk(const geometry_msgs::WrenchStamped::ConstPtr &msg)
{
    // mtx_buffer.lock();
    l_foot_force_buffer.push_back(msg);
    // mtx_buffer.unlock();
    // sig_buffer.notify_all();
}

void r_foot_force_cbk(const geometry_msgs::WrenchStamped::ConstPtr &msg)
{
    // mtx_buffer.lock();
    r_foot_force_buffer.push_back(msg);
    // mtx_buffer.unlock();
    // sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    /*** push imu data, and pop from imu buffer ***/
    if (can_clean_imu)
    {
        meas.imu.clear();
        can_clean_imu = false;
    }

    if (!imu_buffer.empty())
    {
        double imu_time = imu_buffer.front()->header.stamp.toSec();
        
        while ((!imu_buffer.empty()))
        {
            imu_time = imu_buffer.front()->header.stamp.toSec();
            meas.imu.push_back(imu_buffer.front());
            new_imu_meas = true;
            imu_buffer.pop_front();
        }
    }
    else
    {
        new_imu_meas = false;
    }

    if (lidar_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }
    
    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    can_clean_imu = true;
    return true;
}

bool new_force_and_encoder(MeasureGroup &meas)
{
    bool has_new_data = false;
    if (!l_foot_force_buffer.empty() && !r_foot_force_buffer.empty() && !encoder_buffer.empty()) 
    {
        has_new_data = true;
        // ROS_INFO("l_foot_force_buffer.size(): %d", int(l_foot_force_buffer.size()));
        // ROS_INFO("r_foot_force_buffer.size(): %d", int(r_foot_force_buffer.size()));
        meas.l_f_force = l_foot_force_buffer.back();
        meas.r_f_force = r_foot_force_buffer.back();
        meas.joint_encoder = encoder_buffer.back();
        LLegGRF << meas.l_f_force->wrench.force.x, meas.l_f_force->wrench.force.y, meas.l_f_force->wrench.force.z;
        RLegGRF << meas.r_f_force->wrench.force.x, meas.r_f_force->wrench.force.y, meas.r_f_force->wrench.force.z;
        LLegGRT << meas.l_f_force->wrench.torque.x, meas.l_f_force->wrench.torque.y, meas.l_f_force->wrench.torque.z;
        RLegGRT << meas.r_f_force->wrench.torque.x, meas.r_f_force->wrench.torque.y, meas.r_f_force->wrench.torque.z;
        l_foot_force_buffer.clear();
        r_foot_force_buffer.clear();
        encoder_buffer.clear();
    }
    return has_new_data;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
}

template<typename T>
void set_twiststamp(T & out)
{
    out.twist.linear.x = state_point.vel(0);
    out.twist.linear.y = state_point.vel(1);
    out.twist.linear.z = state_point.vel(2);    
    out.twist.angular.x = Measures.imu.back()->angular_velocity.x;
    out.twist.angular.y = Measures.imu.back()->angular_velocity.y;
    out.twist.angular.z = Measures.imu.back()->angular_velocity.z;
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    set_twiststamp(odomAftMapped.twist);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );
}

void publish_bitbot_se(const ros::Publisher & pubBitbotSE)
{
    bitbotSE.header.frame_id = "torso";
    bitbotSE.header.stamp = ros::Time::now();

    bitbotSE.linear_acceleration.x = state_point.pos(0);
    bitbotSE.linear_acceleration.y = state_point.pos(1);
    bitbotSE.linear_acceleration.z = state_point.pos(2);
    bitbotSE.angular_velocity.x = state_point.vel(0);
    bitbotSE.angular_velocity.y = state_point.vel(1);
    bitbotSE.angular_velocity.z = state_point.vel(2);

    // 使用eigen将q转为rpy之后付给bitbotSE
    V3D rpy = Log(state_point.rot.toRotationMatrix());
    bitbotSE.orientation.x = rpy(0);
    bitbotSE.orientation.y = rpy(1);
    bitbotSE.orientation.z = rpy(2);
    
    pubBitbotSE.publish(bitbotSE);
}
void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rviz will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

void publish_velocity_measure(const ros::Publisher pubVelocityMeas, const Vector3d &v_meas, const Vector3d &p_meas)
{
    VelocityMeas.header.stamp = ros::Time().fromSec(Measures.imu.back()->header.stamp.toSec());
    VelocityMeas.header.frame_id = "debug_data";
    VelocityMeas.pose.pose.position.x = p_meas(0);
    VelocityMeas.pose.pose.position.y = p_meas(1);
    VelocityMeas.pose.pose.position.z = p_meas(2);
    VelocityMeas.twist.twist.linear.x = v_meas(0);
    VelocityMeas.twist.twist.linear.y = v_meas(1);
    VelocityMeas.twist.twist.linear.z = v_meas(2);
    pubVelocityMeas.publish(VelocityMeas);
}

void UpdateKinematicState(MeasureGroup &meas, esekfom::esekf<state_ikfom, 15, input_ikfom> &kf, ContactPoint &contact)
{
    // 将meas.joint_encoder中的角度和角速度都存到eigen vector中
    J_rfoot.setZero();
    J_lfoot.setZero();
    R_se = state_point.rot.toRotationMatrix();

    // 这个别扭的操作是在为自己脑残的操作买单，在控制段输出joint_state信息的时候把左右顺序搞反了，导致和urdf中的左右关节顺序不一致，现在只能这样。
    joint_angle(0) = meas.joint_encoder->position[4];
    joint_vel(0)   = meas.joint_encoder->velocity[4];
    joint_angle(1) = meas.joint_encoder->position[5];
    joint_vel(1)   = meas.joint_encoder->velocity[5];
    joint_angle(2) = meas.joint_encoder->position[6];
    joint_vel(2)   = meas.joint_encoder->velocity[6];
    joint_angle(3) = meas.joint_encoder->position[7];
    joint_vel(3)   = meas.joint_encoder->velocity[7];
    joint_angle(4) = 0.0;
    joint_vel(4)   = 0.0;
    joint_angle(5) = meas.joint_encoder->position[0];
    joint_vel(5)   = meas.joint_encoder->velocity[0];
    joint_angle(6) = meas.joint_encoder->position[1];
    joint_vel(6)   = meas.joint_encoder->velocity[1];
    joint_angle(7) = meas.joint_encoder->position[2];
    joint_vel(7)   = meas.joint_encoder->velocity[2];
    joint_angle(8) = meas.joint_encoder->position[3];
    joint_vel(8)   = meas.joint_encoder->velocity[3];
    joint_angle(9) = 0.0;
    joint_vel(9)   = 0.0;

    joint_vel = 0.1047*joint_vel;

    constexpr int float_q = 7;
    q_.tail(17-float_q) = joint_angle;
    pinocchio::forwardKinematics(model_, data_, q_);
    pinocchio::framesForwardKinematics(model_, data_, q_);

    R_base_rfoot = data_.oMf[Rfoot_id].rotation();
    t_base_rfoot = data_.oMf[Rfoot_id].translation();
    R_base_lfoot = data_.oMf[Lfoot_id].rotation();
    t_base_lfoot = data_.oMf[Lfoot_id].translation();
    
    pinocchio::computeJointJacobians(model_, data_, q_);
    pinocchio::getFrameJacobian(model_, data_, Rfoot_id, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_rfoot);
    pinocchio::getFrameJacobian(model_, data_, Lfoot_id, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_lfoot);
}
void UpdateContactPositionAftSwitch(const ContactPoint & contact)
{
    if(contact == ContactPoint::NONE)
    {
        return;
    }
    if(contact == ContactPoint::LeftFoot)
    {
        state_point.pc = state_point.rot.toRotationMatrix()*(t_base_lfoot-t_base_imu) + state_point.pos;
        kf.change_x(state_point);
    }
    else if(contact == ContactPoint::RightFoot)
    {
        state_point.pc = state_point.rot.toRotationMatrix()*(t_base_rfoot-t_base_imu) + state_point.pos;
        kf.change_x(state_point);
    }
}

void UpdateBodyPositionKine(const ContactPoint &contact)
{
    if(contact == ContactPoint::NONE)
    {
        return;
    }
    if(contact == ContactPoint::LeftFoot)
    {
        body_position_kinematic = state_point.pc - state_point.rot.toRotationMatrix()*(t_base_lfoot-t_base_imu);
        R_base_foot = R_base_lfoot;
    }
    else if(contact == ContactPoint::RightFoot)
    {
        body_position_kinematic = state_point.pc - state_point.rot.toRotationMatrix()*(t_base_rfoot-t_base_imu);
        R_base_foot = R_base_rfoot;
    }
}

Vector3d KinematicVelocityEstimation(const MeasureGroup &meas, const ContactPoint &contact)
{
    if(contact == ContactPoint::NONE)
    {
        return Vector3d::Zero();
    }

    if(contact == ContactPoint::LeftFoot)
    {
        t_imu_foot = t_base_lfoot-t_base_imu;
        J_fix_velocity = J_lfoot.block<3,10>(0,6);
    }
    else if(contact == ContactPoint::RightFoot)
    {
        t_imu_foot = t_base_rfoot-t_base_imu;
        J_fix_velocity = J_rfoot.block<3,10>(0,6);
    }
    Vector3d v_base_meas;
    Vector3d w_base;    
    int imu_size = meas.imu.size();
    if (imu_size > 1)
    {
        w_base(0) = 0.5 * (meas.imu[imu_size-1]->angular_velocity.x+meas.imu[imu_size-2]->angular_velocity.x)-state_point.ba(0);
        w_base(1) = 0.5 * (meas.imu[imu_size-1]->angular_velocity.y+meas.imu[imu_size-2]->angular_velocity.y)-state_point.ba(1);
        w_base(2) = 0.5 * (meas.imu[imu_size-1]->angular_velocity.z+meas.imu[imu_size-2]->angular_velocity.z)-state_point.ba(2);

    }
    else
    {
        w_base <<  meas.imu.back()->angular_velocity.x, meas.imu.back()->angular_velocity.y, meas.imu.back()->angular_velocity.z;
    }
    
    v_base_meas = w_base.cross(t_imu_foot)+J_fix_velocity*joint_vel;
    // v_base_meas = w_base.cross(t_imu_foot);
    // v_base_meas = J_fix_velocity*joint_vel;
    v_base_meas = -R_se*v_base_meas;
    return v_base_meas;
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    if (lidar_update)
    {
        double match_start = omp_get_wtime();
        laserCloudOri->clear(); 
        corr_normvect->clear(); 
        total_residual = 0.0; 

        /** closest surface search and residual computation **/
        #ifdef MP_EN
            omp_set_num_threads(MP_PROC_NUM);
            #pragma omp parallel for
        #endif
        for (int i = 0; i < feats_down_size; i++)
        {
            PointType &point_body  = feats_down_body->points[i]; 
            PointType &point_world = feats_down_world->points[i]; 

            /* transform to world frame */
            V3D p_body(point_body.x, point_body.y, point_body.z);
            V3D p_global(s.rot * (Lidar_R_wrt_IMU*p_body + Lidar_T_wrt_IMU) + s.pos);
            point_world.x = p_global(0);
            point_world.y = p_global(1);
            point_world.z = p_global(2);
            point_world.intensity = point_body.intensity;

            vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

            auto &points_near = Nearest_Points[i];

            if (ekfom_data.converge) // 相当于过滤一下？不收敛就说明位置估计不准，就不用匹配了
            {
                /** Find the closest surfaces in the map **/
                ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
                point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
            }

            if (!point_selected_surf[i]) continue;

            VF(4) pabcd;
            point_selected_surf[i] = false;
            if (esti_plane(pabcd, points_near, 0.1f))
            {
                float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
                float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

                if (s > 0.9)
                {
                    point_selected_surf[i] = true;
                    normvec->points[i].x = pabcd(0);
                    normvec->points[i].y = pabcd(1);
                    normvec->points[i].z = pabcd(2);
                    normvec->points[i].intensity = pd2;
                    res_last[i] = abs(pd2);
                }
            }
        }
        
        effct_feat_num = 0;

        for (int i = 0; i < feats_down_size; i++)
        {
            if (point_selected_surf[i])
            {
                laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
                corr_normvect->points[effct_feat_num] = normvec->points[i];
                total_residual += res_last[i];
                effct_feat_num ++;
            }
        }

        if (effct_feat_num < 1)
        {
            ekfom_data.valid = false;
            ROS_WARN("No Effective Points! \n");
            return;
        }

        res_mean_last = total_residual / effct_feat_num;
        match_time  += omp_get_wtime() - match_start;
        double solve_start_  = omp_get_wtime();
        
        /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
        ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 18); //23
        ekfom_data.h.resize(effct_feat_num);

        for (int i = 0; i < effct_feat_num; i++)
        {
            const PointType &laser_p  = laserCloudOri->points[i];
            V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
            M3D point_be_crossmat;
            point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
            V3D point_this = Lidar_R_wrt_IMU * point_this_be + Lidar_T_wrt_IMU;
            M3D point_crossmat;
            point_crossmat<<SKEW_SYM_MATRX(point_this);

            /*** get the normal vector of closest surface/corner ***/
            const PointType &norm_p = corr_normvect->points[i];
            V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

            /*** calculate the Measuremnt Jacobian matrix H ***/
            V3D C(s.rot.conjugate() *norm_vec);
            V3D A(point_crossmat * C);
            // if (extrinsic_est_en)
            // {
            //     V3D B(point_be_crossmat * Lidar_R_wrt_IMU.conjugate() * C); //s.rot.conjugate()*norm_vec);
            //     ekfom_data.h_x.block<1, 15>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
            // }
            // else
            {
                ekfom_data.h_x.block<1, 18>(i,0) << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

            }

            /*** Measuremnt: distance to the closest surface/corner ***/
            ekfom_data.h(i) = -norm_p.intensity;
        }
        solve_time += omp_get_wtime() - solve_start_;

    }
    else
    {
        ekfom_data.h_x = MatrixXd::Zero(6, 18); //23
        ekfom_data.h.resize(6);
        M3D forward_kine_cross;
        forward_kine_cross << SKEW_SYM_MATRX(t_imu_foot);
        M3D H_omega_bias(s.rot.toRotationMatrix()*forward_kine_cross);
        M3D v_meas_cross;
        v_meas_cross << SKEW_SYM_MATRX(v_meas);
        M3D A(s.rot.toRotationMatrix()*v_meas_cross);
        ekfom_data.h_x.block<3,3>(0,0) = A;
        ekfom_data.h_x.block<3,3>(0,3) = Eigen::Matrix3d::Zero();
        ekfom_data.h_x.block<3,3>(0,6) = Eigen::Matrix3d::Identity();
        ekfom_data.h_x.block<3,3>(0,15) = H_omega_bias;
        ekfom_data.h_x.block<3,3>(0,15) = Eigen::Matrix3d::Zero();
        ekfom_data.h.block<3,1>(0,0) = velocity_residual;

        ekfom_data.h_x.block<3,3>(3,0) = H_omega_bias;
        ekfom_data.h_x.block<3,3>(3,3) = -Eigen::Matrix3d::Identity();
        ekfom_data.h_x.block<3,3>(3,6) = Eigen::Matrix3d::Zero();
        ekfom_data.h_x.block<3,3>(3,9) = Eigen::Matrix3d::Zero();
        ekfom_data.h_x.block<3,3>(3,15) = Eigen::Matrix3d::Zero();
        ekfom_data.h_x.block<3,3>(3,15) = Eigen::Matrix3d::Identity();
        ekfom_data.h.block<3,1>(3,0) = position_residual;
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);
    nh.param<string>("map_file_path",map_file_path,"");
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<string>("common/encoder_topic", encoder_topic,"/bitbot/joint_state");
    nh.param<string>("common/l_foot_force", l_foot_force_topic,"/bitbot/l_foot_force");
    nh.param<string>("common/r_foot_force", r_foot_force_topic,"/bitbot/r_foot_force");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",filter_size_map_min,0.5);
    nh.param<double>("cube_side_length",cube_len,200);
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<double>("mapping/fov_degree",fov_deg,180);
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 1);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    nh.param<double>("common/kinematic_cov",kinematic_cov,0.1);
    nh.param<double>("common/kinematic_update_cov", kinematic_update_cov, 1);
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    cout << "kineamtic_cov: " << kinematic_cov << endl;

    // init pinocchio
    urdf_file = "/home/zqr/catkin_ws/src/bitbot_kinematics_se/src/kuafu.urdf";
    pinocchio::urdf::buildModel(urdf_file, model_);
    data_ = pinocchio::Data(model_);
    q_ = pinocchio::neutral(model_);
    pinocchio::forwardKinematics(model_, data_, q_);
    pinocchio::framesForwardKinematics(model_, data_, q_);
    // print ros info that indicates pinocchio init down.
    cout << "Pinocchio init down, model: " << model_.name << "nq: " << model_.nq << endl;
    // PinocchioInit();
    joint_angle.setZero();
    joint_vel.setZero();
    Lfoot_id = model_.getFrameId("Lfoot");
    Rfoot_id = model_.getFrameId("Rfoot");
    IMU_id = model_.getFrameId("IMUlink");
    R_base_imu = data_.oMf[IMU_id].rotation();
    t_base_imu = data_.oMf[IMU_id].translation();
    cout << "R_base_imu: " << R_base_imu << endl;
    cout << "t_base_imu: " << t_base_imu.transpose() << endl;
    
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";
    ContactPoint contact = ContactPoint::NONE;

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    p_imu->set_contact_cov(V3D(kinematic_cov, kinematic_cov, kinematic_cov));
    v_meas.setZero();

    // 初始化GRF，GRT
    LLegGRF.setZero();
    LLegGRT.setZero();
    RLegGRF.setZero();
    RLegGRT.setZero();

    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, epsi);

    // init contact position state
    state_ikfom state_init_pc = kf.get_x();
    t_base_lfoot = data_.oMf[Lfoot_id].translation();
    state_init_pc.pc = state_init_pc.rot.toRotationMatrix()*(t_base_lfoot - t_base_imu) + state_init_pc.pos;
    kf.change_x(state_init_pc);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Subscriber sub_encoder = nh.subscribe(encoder_topic, 200000, encoder_cbk);
    ros::Subscriber sub_lfootf = nh.subscribe(l_foot_force_topic, 200000, l_foot_force_cbk);
    ros::Subscriber sub_rfootf = nh.subscribe(r_foot_force_topic, 200000, r_foot_force_cbk);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/Odometry", 200000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 100000);
    ros::Publisher pubBitbotSE = nh.advertise<sensor_msgs::Imu>("/bitbot_se", 20);

    // ros::Publisher pubVeocilityMeas = nh.advertise<nav_msgs::Odometry>("/velocity_meas", 20);
//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit) break;
        ros::spinOnce();
        if(sync_packages(Measures)) 
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }
            // p_imu->PredictImuState(Measures, kf, num_predicted_imu_meas);
            // num_predicted_imu_meas = Measures.imu.size();
            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort, R_base_foot); // 这里边就做完预测了。
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * Lidar_T_wrt_IMU;
            // std::cout << "foo/n " << std::endl;
            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            
            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/ //??
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(Lidar_R_wrt_IMU);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<Lidar_T_wrt_IMU.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            lidar_update = true;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time, NUM_MAX_ITERATIONS);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * Lidar_T_wrt_IMU;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);
            publish_bitbot_se(pubBitbotSE);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(Lidar_R_wrt_IMU);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<Lidar_T_wrt_IMU.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
            Measures.imu.clear();
        }
        else if(new_imu_meas)
        {
            if(num_predicted_imu_meas > Measures.imu.size())
            {
                num_predicted_imu_meas = 0;
            }
            p_imu->PredictImuState(Measures, kf, num_predicted_imu_meas, R_base_foot);
            num_predicted_imu_meas = Measures.imu.size();
            state_point = kf.get_x();
            // publish_bitbot_se(pubBitbotSE);
        }

        if(new_force_and_encoder(Measures) && !Measures.imu.empty())
        {
            // if(Measures.l_f_force->wrench.force.z>150 && Measures.r_f_force->wrench.force.z>150)
            // {
            //     contact = ContactPoint::BothFeet;
            // }
            // else
            // {
            //     if(Measures.l_f_force->wrench.force.z > 250)
            //     {
            //         contact = ContactPoint::LeftFoot;
            //     }
            //     else if(Measures.r_f_force->wrench.force.z > 250)
            //     {
            //         contact = ContactPoint::RightFoot;
            //     }
            //     else
            //     {
            //         contact = ContactPoint::NONE;
            //     }
            // }
            
            // // shimiit trigger
            // if(contact != prev_contact)
            // {
            //   if(contact_change_num == 0)
            //   {
            //     next_contact = contact;
            //     contact_change_num++;
            //   }
            //   else
            //   {
            //     if(contact == next_contact)
            //     {
            //       contact_change_num++;
            //     }
            //     else
            //     {
            //       contact_change_num=0;
            //     }
            //   }
            //   if(contact_change_num > 2)
            //   {
            //     prev_contact = contact;
            //     UpdateKinematicState(Measures, kf, prev_contact);
            //     UpdateContactPositionAftSwitch(prev_contact);
            //   }
            // }    
            // else
            // {
            //     UpdateKinematicState(Measures, kf, prev_contact);
            // }
            Serow_shimitt.LLeg_FT(LLegGRF, LLegGRT); 
            Serow_shimitt.RLeg_FT(RLegGRF, RLegGRT);
            std::string support_leg = Serow_shimitt.computeKinTFs();
            if(support_leg == "LLeg")
            {
                contact_point_serow_shimitt = ContactPoint::LeftFoot;
            }
            else if(support_leg == "RLeg")
            {
                contact_point_serow_shimitt = ContactPoint::RightFoot;
            }
            else if(support_leg == "NONE")
            {
                contact_point_serow_shimitt = ContactPoint::NONE;
            }
            if(support_leg != support_leg_prev)
            {
                support_leg_prev = support_leg;
                UpdateContactPositionAftSwitch(contact_point_serow_shimitt);                
                // UpdateKinematicState(Measures, kf, contact_point_serow_shimitt);
            }
            UpdateKinematicState(Measures, kf, contact_point_serow_shimitt);

            if(p_imu->imu_inited)
            {
                v_meas = KinematicVelocityEstimation(Measures, prev_contact);
                
                velocity_residual = v_meas - state_point.vel;

                UpdateBodyPositionKine(prev_contact);

                position_residual = body_position_kinematic - state_point.pos;
                // if(fabs(velocity_residual.x()) > 1 || fabs(velocity_residual.y()) > 1 || fabs(velocity_residual.z()) > 1)
                // {
                //     // ROS_WARN("velocity_residual: %f, %f, %f", velocity_residual.x(), velocity_residual.y(), velocity_residual.z());
                //     publish_bitbot_se(pubBitbotSE);
                //     // ROS_INFO("jump");
                //     continue;
                // }
                double solve_H_time = 0;
                // publish_velocity_measure(pubVeocilityMeas, v_meas, state_point.pc);
                if(v_meas!=Eigen::Vector3d::Zero())
                {
                    lidar_update = false;
                    kf.update_iterated_dyn_share_modified(kinematic_update_cov, solve_H_time, 1);
                    publish_bitbot_se(pubBitbotSE);
                    
                    ROS_INFO("published");
                }   
            }
            
        }

        status = ros::ok();
        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
