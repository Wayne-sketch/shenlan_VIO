#include "iostream"
#include "sophus/so3.hpp"
#include "sophus/se3.hpp"
int main(int argc, char** argv)
{	
	//w为旋转变量，因为是根据旋转矩阵R计算出的瞬间旋转向量，所以也是角速度，同样从矩阵相乘的角度思考是表示三维旋转的李代数，所以是R3空间的向量，题目中给定(0.01,0.02,0.03)
	Eigen::Vector3d w(0.01, 0.02, 0.03);
	//首先利用四元数计算更新后的旋转矩阵
	//创建一个单位四元数，单位四元数才能表示旋转
	Eigen::Quaterniond q = Eigen::Quaterniond::UnitRandom();
	//先输出变化更新前的旋转矩阵
	Eigen::Matrix3d R = q.toRotationMatrix();
	std::cout << "更新前的旋转矩阵R:" << std::endl << R << std::endl;
	//计算更新量对应的四元数，用q2代表
	Eigen::Quaterniond q2;
	//实部为1	
	q2.w() = 1;
	//虚部为w/2
	q2.vec() = 0.5 * w;
	//单位化处理，这样才能代表对应的旋转
	q2.normalized();
	//用四元数乘法计算更新后的旋转矩阵，记为R1，并输出表示
	//四元数乘法运算符重载
	//这里四元数相乘的顺序要和后面用旋转矩阵相乘的顺序一致，都为左乘或右乘
	Eigen::Matrix3d R1 = (q * q2).toRotationMatrix();
	std::cout << "利用四元数计算更新后的旋转矩阵R1:" << std::endl << R1 << std::endl;
	
	//用旋转矩阵计算更新后的旋转矩阵，结果记为R2，并输出表示
	//利用Sophus库的指数映射计算更新量对应的李群
	Eigen::Matrix3d exp_w = Sophus::SO3d::exp(w).matrix();
	Eigen::Matrix3d R2 = R * exp_w;
	std::cout << "利用旋转矩阵计算更新后的旋转矩阵R2:" << std::endl << R2 << std::endl;
	//输出R1与R2逆的结果，再次验证R1和R2几乎一致
	std::cout << "R1与R2逆的乘积:" << std::endl << R1*R2.transpose() << std::endl;
	return 0;
}
