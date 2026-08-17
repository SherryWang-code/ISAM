#include <iostream> 
#include <fstream>
#include <vector>      
#include <string>    
#include <cmath>     
#include <stdio.h> 
#include<Windows.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "interface.h"


int ISAMFastCalculation(std::string filePath, std::vector<float>& k, float Delta_x, float Delta_y,float refractive_index,
		float focus_index, int Z_slice_index, bool use_focus_shift, int N_z, int N_x, int N_y) {
	if (N_z <= 0 || N_x <= 0 || N_y <= 0|| Z_slice_index<0|| refractive_index<=0) {
		OutputDebugStringA("[ERROR] ISAMFastCalculation: input parameters is not valid.\n");
		return 1;
	}	
	if (k.empty() || filePath.empty()) {
		OutputDebugStringA("[ERROR] ISAMFastCalculation: input is empty.\n");
		return 1;
	}
	// Parameters
	ISAMfast ISAM_FAST;
	float dz = 5 * 1e-6;
	isamParams params;
	params.Delta_x = Delta_x;
	params.Delta_y = Delta_y;
	params.Delta_z = (Z_slice_index - focus_index)* dz;
	params.refractive_index = refractive_index;
	params.focus_index = focus_index;
	params.zShiftFlag = use_focus_shift;
	
	// Signal read
	int zRangeMin = 30;
	int zRangeMax = 1024;
	std::vector<std::complex<float>> S_x_y_k_complex;
	ISAM_ERROR error_getCompData = ISAM_FAST.getCompData(filePath, 2 * N_z, N_x, N_y, zRangeMin, zRangeMax, S_x_y_k_complex);
	if (error_getCompData != ISAM_ERROR::SUCCESS) {  // 检查getCompData返回的错误码
		OutputDebugStringA("[ERROR] ISAMFastCalculation: getCompData failed.\n");
		return 1;
	}
	//k calculation
	float sum = 0;
	for (size_t i = 0; i < k.size(); ++i) {
		sum += k[i];
	}
	float k0 = params.refractive_index * sum / k.size();
	// Q calculation
	std::vector<float> Q(N_x * N_y);
	ISAM_FAST.QCalculation(Q, N_x, N_y, params);
	// ISAM calculation
	std::vector<Complex> eta_3D_x_y_z(2 * N_z * N_x * N_y);
	ISAM_ERROR error_ISAMcalculationfast = ISAM_FAST.ISAMcalculationfast(eta_3D_x_y_z, S_x_y_k_complex, 
													params.focus_index, k0, Q, 2 * N_z, N_x, N_y, params.Delta_z, params.zShiftFlag);
	if (error_ISAMcalculationfast != ISAM_ERROR::SUCCESS) {  // 检查ISAMcalculationfast返回的错误码
		OutputDebugStringA("[ERROR] ISAMFastCalculation: ISAMcalculationfast failed.\n");
		return 1;
	}
	return 0;
}

int ISAMregular(std::string filePath, std::vector<float>& k, float Delta_x, float Delta_y, float refractive_index,
	float focus_index, int N_z, int N_x, int N_y) {
	if (N_z <= 0 || N_x <= 0 || N_y <= 0 || refractive_index <= 0) {
		OutputDebugStringA("[ERROR] ISAMFastCalculation: input parameters is not valid.\n");
		return 1;
	}
	if (k.empty() || filePath.empty()) {
		OutputDebugStringA("[ERROR] ISAMFastCalculation: input is empty.\n");
		return 1;
	}
	// Parameters
	ISAMfast ISAM_FAST;
	float dz = 5 * 1e-6;
	isamParams params;
	params.Delta_x = Delta_x;
	params.Delta_y = Delta_y;
	params.refractive_index = refractive_index;

	// Signal read
	int zRangeMin = 30;
	int zRangeMax = 1024;
	std::vector<std::complex<float>> S_x_y_k_complex;
	ISAM_ERROR error_getCompData = ISAM_FAST.getCompData(filePath, 2 * N_z, N_x, N_y, zRangeMin, zRangeMax, S_x_y_k_complex);
	if (error_getCompData != ISAM_ERROR::SUCCESS) {  // 检查getCompData返回的错误码
		OutputDebugStringA("[ERROR] ISAMFastCalculation: getCompData failed.\n");
		return 1;
	}
	//k calculation
	float sum = 0;
	for (size_t i = 0; i < k.size(); ++i) {
		sum += k[i];
	}
	float k0 = params.refractive_index * sum / k.size();
	// Q calculation
	std::vector<float> Q(N_x * N_y);
	ISAM_FAST.QCalculation(Q, N_x, N_y, params);
	// ISAM calculation
	std::vector<Complex> eta_3D_x_y_z(2 * N_z * N_x * N_y);
	ISAM_ERROR ISAMregular = ISAM_FAST.ISAM3D(eta_3D_x_y_z, S_x_y_k_complex,
		focus_index, k, Q, 2 * N_z, N_x, N_y);

	if (ISAMregular != ISAM_ERROR::SUCCESS) {  // 检查ISAMcalculationfast返回的错误码
		OutputDebugStringA("[ERROR] ISAMFastCalculation: ISAMcalculationfast failed.\n");
		return 1;
	}
	return 0;
}