//
// Created by hyj on 18-1-19.
//

#ifndef IMUSIMWITHPOINTLINE_UTILITIES_H
#define IMUSIMWITHPOINTLINE_UTILITIES_H

#include "imu.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <vector>
#include <fstream>

// save 3d points to file
//Eigen::aligned_allocator:Eigen管理内存和C++11中的方法是不一样的，所以需要单独强调元素的内存分配和管理。
void save_points(std::string filename, std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > points);

// save 3d points and it's obs in image
void save_features(std::string filename,
                   std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > points,
                   std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > features);

// save line obs
void save_lines(std::string filename,
                std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > features);


void LoadPose(std::string filename, std::vector<MotionData>& pose);

// save imu body data
void save_Pose(std::string filename, std::vector<MotionData> pose);

// save pose as TUM style
void save_Pose_asTUM(std::string filename, std::vector<MotionData> pose);

#endif //IMUSIMWITHPOINTLINE_UTILITIES_H


