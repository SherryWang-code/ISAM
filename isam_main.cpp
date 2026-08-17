#include <iostream> 
#include <fstream>
#include <vector>      
#include <string>    
#include <cmath>     
#include <stdio.h> 
#include<Windows.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
//#include "isam.h"
#include "interface.h"

//ISAMfast ISAM_FAST;
// 多项式拟合函数
Eigen::VectorXd polyfit(const std::vector<float>& x, const std::vector<float>& y, int order) {
	Eigen::MatrixXd A(x.size(), order + 1);
	Eigen::VectorXd b(y.size());

	// 填充矩阵A和向量b
	for (size_t i = 0; i < x.size(); ++i) {
		for (int j = 0; j <= order; ++j) {
			A(i, j) = std::pow(x[i], j);
		}
		b(i) = y[i];
	}

	// 使用SVD进行最小二乘拟合
	return A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
}

// 多项式求值函数
std::vector<float> polyval(const Eigen::VectorXd& coefficients, const std::vector<float>& x) {
	std::vector<float> result(x.size());

	for (size_t i = 0; i < x.size(); ++i) {
		float value = 0.0;
		for (int j = 0; j < coefficients.size(); ++j) {
			value += coefficients(j) * std::pow(x[i], j);
		}
		result[i] = value;
	}

	return result;
}
void ReadTxtToFile(const std::string& file_name, std::vector<float>& input) {
	std::ifstream inFile(file_name);

	if (inFile.is_open()) {
		float value;
		while (inFile >> value) {
			input.push_back(value);
		}
		inFile.close();
	}
}
int main() {
		// 	
		int N_z = 1024;
		int N_x = 500;
		int N_y = 500;
		int Z_slice_index = 346;
		float Delta_x = 1 * 1e-3;
		float Delta_y = Delta_x;
		float refractive_index = 1;
		int focus_index = 70;
		bool use_focus_shift = TRUE;
		//std::string filePath = "D:\data\OCT-17\20251027\20251027_142846142_3mm-defocus-500_origin1.raw";
		std::string filePath = "D:/data/OCT-11/20250804/20250805_144600833_defocus-330-1mm-500_origin1.raw";

		// k process
		std::string pixel2wavelengthPath = "OCT_OSA_calibration.txt";
		std::vector<float> OCT_OSA_calibration;
		ReadTxtToFile(pixel2wavelengthPath, OCT_OSA_calibration);
		size_t half_size = OCT_OSA_calibration.size() / 2;
		std::vector<float> wavelength(half_size);
		std::vector<float> pixel_number(half_size);
		for (size_t i = 0; i < half_size; ++i) {
			wavelength[i] = OCT_OSA_calibration[i] * 1e-9;
			pixel_number[i] = OCT_OSA_calibration[i + half_size];  // 
		}
		std::vector<float> k(wavelength.size());
		for (size_t i = 0; i < wavelength.size(); ++i) {
			k[i] = 2.0 * PI / wavelength[i];
		}
		int order = 1;
		Eigen::VectorXd p = polyfit(pixel_number, k, order);
		std::vector<float> new_pixels(2 * N_z);
		for (int i = 0; i < 2 * N_z; ++i) {
			new_pixels[i] = i;
		}
		Eigen::VectorXd coefficients(2);
		coefficients(0) = p(order - 1);
		coefficients(1) = p(order);
		std::vector<float> k_result = polyval(coefficients, new_pixels);
		ISAMFastCalculation(filePath, k_result, Delta_x, Delta_y, refractive_index,
			focus_index, Z_slice_index, use_focus_shift, N_z, N_x, N_y);
		//ISAMregular(filePath, k_result, Delta_x, Delta_y, refractive_index,
		//	focus_index, N_z, N_x, N_y);
		
	}
	// 
//int main() {
//	// 	
//	int N_z = 1024;
//	int N_x = 500;
//	int N_y = 500;
//	int X_slice_index = 250;
//	int Y_slice_index = 250;
//	int Z_slice_index = 180;
//	float dz = 5 * 1e-6;
//	isamParams params;
//	params.Delta_x = 3 * 1e-3;
//	params.Delta_y = params.Delta_x;
//	params.Delta_z = Z_slice_index * dz;
//	params.refractive_index = 1;
//	params.focus_index = 100;
//	params.zShiftFlag = TRUE;
//	// Signal read
//	std::string filePath = "D:/data/highRes/20250619/20250619_104524541_500-defocus-D3mm-f18_origin1.raw";
//	int zRangeMin = 30;
//	int zRangeMax = 1024;
//	std::vector<std::complex<float>> S_x_y_k_complex;
//	ISAM_ERROR error = ISAM_FAST.getCompData(filePath, 2 * N_z, N_x, N_y, zRangeMin, zRangeMax, S_x_y_k_complex);
//	// k process
//	std::string pixel2wavelengthPath = "OCT_OSA_calibration.txt";
//	std::vector<float> OCT_OSA_calibration;
//	ReadTxtToFile(pixel2wavelengthPath, OCT_OSA_calibration);
//	size_t half_size = OCT_OSA_calibration.size() / 2;
//	std::vector<float> wavelength(half_size);
//	std::vector<float> pixel_number(half_size);
//	for (size_t i = 0; i < half_size; ++i) {
//		wavelength[i] = OCT_OSA_calibration[i] * 1e-9;
//		pixel_number[i] = OCT_OSA_calibration[i + half_size];  // 
//	}
//	std::vector<float> k(wavelength.size());
//	for (size_t i = 0; i < wavelength.size(); ++i) {
//		k[i] = 2.0 * PI / wavelength[i];
//	}
//	int order = 1;
//	Eigen::VectorXd p = polyfit(pixel_number, k, order);
//	std::vector<float> new_pixels(2 * N_z);
//	for (int i = 0; i < 2 * N_z; ++i) {
//		new_pixels[i] = i;
//	}
//	Eigen::VectorXd coefficients(2);
//	coefficients(0) = p(order - 1);
//	coefficients(1) = p(order);
//	std::vector<float> k_result = polyval(coefficients, new_pixels);
//	std::vector<float> k_new(k_result.size());
//	float sum = 0;
//	for (size_t i = 0; i < k_result.size(); ++i) {
//		k_new[i] = k_result[i] * params.refractive_index;
//		sum += k_new[i];
//	}
//
//	float k0 = params.refractive_index* sum / k_new.size();
//
//	// Q calculation
//	std::vector<float> Q(N_x * N_y);
//	ISAM_FAST.QCalculation(Q, N_x, N_y, params);
//	//std::string filenameQ = "Q.txt";
//	//SaveVectorToFile(filenameQ, Q, N_x * N_y);
//
//	// ISAM calculation
//	std::vector<Complex> eta_3D_x_y_z(2 * N_z * N_x * N_y);
//	ISAM_FAST.ISAMcalculationfast(eta_3D_x_y_z, S_x_y_k_complex, params.focus_index, k0, Q, 2 * N_z, N_x, N_y, params.Delta_z, params.zShiftFlag);
//
//}


