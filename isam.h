#pragma once
#include <iostream> 
#include <fstream>
#include <vector>      
#include <string>    
#include <cmath>  
#include<Windows.h>
#include <stdio.h> 
constexpr float PI = 3.14159265359f;
using Complex = std::complex<float>;


// Optical parameters determined by the system
//struct opticalParams {
//    float working_distance;
//    float focal_length;
//    float z0 = 0 * 1e-3;
//    float refraction_index;
//    float NA = sin(atan(working_distance / (2 * focal_length)));
//    float alpha = PI / NA;
//
//};
enum struct ISAM_ERROR {
    SUCCESS = 0,
    PARAM_ERROR = 1,
    OPEN_ERROR = 2,
    SIZE_ERROR = 3,
    UNKNOWN_ERROR = 4
};
// Key parameters for ISAM
struct isamParams {
    float Delta_x;
    float Delta_y;
    float Delta_z;
    float refractive_index = 1;
    int focus_index = 0;
    float Q_factor = 1;
    bool zShiftFlag;
};

class ISAMfast{
public:
    ISAM_ERROR getCompData(const std::string& path, int N_Ascan_raw, int N_Bscan, int N_Cscan, int zRangeMin, int zRangeMax, std::vector<std::complex<float>>& compData);

    //std::vector<float> QCalculation(int N_x, int N_y, isamParams& isam_params);
    ISAM_ERROR QCalculation(std::vector<float>& Q, int N_x, int N_y, isamParams& isam_params);

    //void ReadTxtToFile(const std::string& file_name, std::vector<float>& input);

    //void ISAM3D(float* S_3D_x_y_k_complex, int N_x, int N_y, int N_z, std::vector<float>& Spectrum, std::vector<float>& k,
    //    opticalParams& opt_params, isamParams& isam_params);

    ISAM_ERROR ISAMcalculationfast(std::vector<Complex>& eta_3D_x_y_z, std::vector<Complex>& S_x_y_k_complex,
        int focus_index, float k0, std::vector<float>& Q, int N_Ascan_raw, int N_Bscan, int N_Cscan, float Delta_z, bool zShiftFlag);

    ISAM_ERROR ISAM3D(std::vector<Complex>& eta_3D_x_y_z, std::vector<Complex>& S_x_y_k_complex,
        int focus_index, std::vector<float> k, std::vector<float>& Q, int N_Ascan_raw, int N_Bscan, int N_Cscan);

private:
    void meshgrid(const std::vector<float>& x, const std::vector<float>& y,
        std::vector<std::vector<float>>& X, std::vector<std::vector<float>>& Y);

    void fftshift(const cv::Mat& input);
    void SaveVectorToBinaryFile(const std::string& filename, const std::vector<float>& data);

    void SaveVectorToFile(const std::string& file_name, const std::vector<float> z_correct, int point_cloud_size);
};