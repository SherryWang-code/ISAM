#include <iostream> 
#include <fstream>
#include <vector>      
#include <string>    
#include <cmath>
#include <complex>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "isam.h"
using namespace std;
using Complex = std::complex<float>;

// 线性插值函数（单个点插值）
// x: 原始x坐标数组
// y: 原始y值数组（复数）
// xi: 待插值的x坐标
Complex linear_interpolate(std::vector<float>& x, std::vector<Complex>& y, float xi) {
	if (x.empty() || y.empty() || x.size() != y.size()) {
		throw std::invalid_argument("x and y must be non-empty and of the same size");
	}

	// 找到xi所在的区间 [x[i], x[i+1]]
	auto it = std::lower_bound(x.begin(), x.end(), xi);

	// 处理边界情况
	if (it == x.begin()) {
		return y[0]; // xi <= x[0]，返回第一个点
	}
	if (it == x.end()) {
		return y.back(); // xi >= x[-1]，返回最后一个点
	}

	// 计算插值索引
	int i = std::distance(x.begin(), it) - 1;
	float x0 = x[i];
	float x1 = x[i + 1];
	const Complex& y0 = y[i];
	const Complex& y1 = y[i + 1];

	// 避免除零（理论上x是单调的，x1 != x0）
	if (std::abs(x1 - x0) < 1e-9f) {
		return y0;
	}

	// 线性插值公式：y = y0 + (xi - x0) * (y1 - y0) / (x1 - x0)
	float t = (xi - x0) / (x1 - x0);
	return Complex(
		y0.real() + t * (y1.real() - y0.real()),
		y0.imag() + t * (y1.imag() - y0.imag())
	);
}

// 
void ISAMfast::meshgrid(const std::vector<float>& x, const std::vector<float>& y,
	std::vector<std::vector<float>>& X, std::vector<std::vector<float>>& Y) {
	int nx = x.size();
	int ny = y.size();
	X.resize(ny, std::vector<float>(nx));
	Y.resize(ny, std::vector<float>(nx));

	for (int i = 0; i < ny; ++i) {
		for (int j = 0; j < nx; ++j) {
			X[i][j] = x[j];  // X的每一行相同，为x
			Y[i][j] = y[i];  // Y的每一列相同，为y
		}
	}
}

void ISAMfast::fftshift(const cv::Mat& input) {
	// 交换象限，使FFT中心移到图像中心
	if (input.cols % 2 != 0 || input.rows % 2 != 0) {
		OutputDebugStringA("[Error]fftshift: input size is not even number!");
		return;
	}
	else{
		int cx = input.cols / 2;
		int cy = input.rows / 2;

		cv::Mat q0(input, cv::Rect(0, 0, cx, cy));   // 左上区域
		cv::Mat q1(input, cv::Rect(cx, 0, cx, cy));  // 右上区域
		cv::Mat q2(input, cv::Rect(0, cy, cx, cy));  // 左下区域
		cv::Mat q3(input, cv::Rect(cx, cy, cx, cy)); // 右下区域

		cv::Mat tmp;                           // 创建临时交换矩阵
		q0.copyTo(tmp);                        // 交换左上和右下
		q3.copyTo(q0);
		tmp.copyTo(q3);

		q1.copyTo(tmp);                        // 交换右上和左下
		q2.copyTo(q1);
		tmp.copyTo(q2);
	}	
}

ISAM_ERROR ISAMfast::getCompData(const std::string& path, int N_Ascan_raw, int N_Bscan, int N_Cscan, int zRangeMin, int zRangeMax, std::vector<Complex>& compData) {
	// 基本参数检查

	if (path.empty() || N_Ascan_raw <= 0 || N_Bscan <= 0 || N_Cscan <= 0) {
		std::string msg = "[ERROR]getCompData: invalid parameters";
		OutputDebugStringA((msg + "\n").c_str());
		return ISAM_ERROR::PARAM_ERROR;
	}

	ifstream f(path, ios::binary);
	if (!f) {
		std::string msg = "[ERROR]getCompData: cannot open: " + path;
		OutputDebugStringA((msg + "\n").c_str());
		return ISAM_ERROR::OPEN_ERROR;
	}
	size_t totalElements = static_cast<size_t>(N_Ascan_raw) * N_Bscan * N_Cscan;
	// 读取为 uint8 类型
	vector<uint8_t> mirror(totalElements);
	f.read(reinterpret_cast<char*>(mirror.data()), totalElements);
	if (!f) {
		std::string msg = "[ERROR]getCompData: fail to read:" + path;
		OutputDebugStringA((msg + "\n").c_str());
		return ISAM_ERROR::OPEN_ERROR;
	}
	f.close();
	try {
		cv::Mat data3D = cv::Mat::zeros(N_Bscan * N_Cscan, N_Ascan_raw, CV_32F);    // 2048* (500*500)
		for (int i = 0; i < N_Bscan * N_Cscan; ++i) {
			for (int j = 0; j < N_Ascan_raw; ++j) {
				size_t index = j + i * N_Ascan_raw;
				data3D.at<float>(i, j) = static_cast<float>(mirror[index]);
			}
		}
		mirror.clear();


		cv::Mat data3Dfft;
		cv::dft(data3D, data3Dfft, cv::DFT_COMPLEX_OUTPUT | cv::DFT_ROWS);

		for (size_t xy = 0; xy < N_Bscan * N_Cscan; xy++)
		{
			for (int z = 0; z < (zRangeMin - 1); ++z)
			{
				data3Dfft.at<cv::Vec2f>(xy, z) = cv::Vec2f(0, 0);
			}
			for (int z = (zRangeMax - 1); z < N_Ascan_raw; ++z)
			{
				data3Dfft.at<cv::Vec2f>(xy, z) = cv::Vec2f(0, 0);
			}
		}

		cv::Mat data3DComplexSignal;
		cv::dft(data3Dfft * 2.0f, data3DComplexSignal, cv::DFT_COMPLEX_OUTPUT | cv::DFT_ROWS);


		compData.resize(totalElements);
		for (int c = 0; c < N_Cscan; ++c) {
			for (int b = 0; b < N_Bscan; ++b) {
				for (int a = 0; a < N_Ascan_raw; ++a) {
					//int matIdx = a + (b + c * N_Bscan) * N_Ascan_raw;
					int idx = a + b * N_Ascan_raw + c * N_Ascan_raw * N_Bscan;
					compData[idx] = Complex(
						data3DComplexSignal.at<cv::Vec2f>(b + c * N_Bscan, a)[0],
						data3DComplexSignal.at<cv::Vec2f>(b + c * N_Bscan, a)[1]
					);
				}
			}
		}


		OutputDebugStringA("[Info]: getCompData success\n");
		return ISAM_ERROR::SUCCESS;

	}
	catch (...) {
		OutputDebugStringA("[ERROR]: getCompData fail\n");
	}
}

// Calculate space frequency
ISAM_ERROR ISAMfast::QCalculation(std::vector<float>&Q, int N_x, int N_y, isamParams& isam_params) {
	if (N_x <= 0 || N_y <= 0 || N_x == NULL || N_y == NULL) {
		OutputDebugStringA("[ERROR]: QCalculation: invalid input parameters\n");
		return ISAM_ERROR::PARAM_ERROR;
	}
	float delta_x = isam_params.Delta_x / N_x;
	float delta_y = isam_params.Delta_y / N_y;
	float step_x = 1.0f / isam_params.Delta_x;
	float step_y = 1.0f / isam_params.Delta_y;
	float start_x = -1.0f / (2.0f * delta_x);
	float end_x = 1.0f / (2.0f * delta_x) - step_x;
	float start_y = -1.0f / (2.0f * delta_y);
	float end_y = 1.0f / (2.0f * delta_y) - step_y;
	std::vector<float> q_x(N_x);
	std::vector<float> q_y(N_y);
	for (int i = 0; i < N_x; ++i) {
		q_x[i] = start_x + i * (end_x - start_x) / (N_x - 1);
		q_x[i] *= 2.0f * PI;
	}
	for (int i = 0; i < N_y; ++i) {
		q_y[i] = start_y + i * (end_y - start_y) / (N_y - 1);
		q_y[i] *= 2.0f * PI;
	}
	//std::vector<float> Q(N_x * N_y);

	for (int j = 0; j < N_y; ++j) {
		for (int i = 0; i < N_x; ++i) {
			float Q_x = q_x[i];
			float Q_y = q_y[j];
			float Q0 = std::sqrt(Q_x * Q_x + Q_y * Q_y);
			Q[j * N_x + i] = Q0 * isam_params.Q_factor; // 500*500 按行排列
		}
	}

	return ISAM_ERROR::SUCCESS;
}


ISAM_ERROR ISAMfast::ISAM3D(std::vector<Complex>& eta_3D_x_y_z, std::vector<Complex>& S_x_y_k_complex,
	int focus_index, std::vector<float> k, std::vector<float>& Q, int N_Ascan_raw, int N_Bscan, int N_Cscan) {
	
	int totalElements = static_cast<size_t>(N_Ascan_raw) * N_Bscan * N_Cscan;
	// 定义聚焦变换
	std::vector<Complex> focus_trans(N_Ascan_raw);
	std::vector<float> indexZ(N_Ascan_raw);
	for (int i = 0; i < N_Ascan_raw; ++i) {
		indexZ[i] = static_cast<float>(i) / (N_Ascan_raw);
	}
	for (int i = 0; i < N_Ascan_raw; ++i) {
		focus_trans[i] = std::exp(Complex(0, 2 * PI * indexZ[i] * focus_index));
	}
	//  应用聚焦变换
	for (int idx = 0; idx < totalElements; idx++) {
		S_x_y_k_complex[idx] *= focus_trans[idx % N_Ascan_raw];
	}
	OutputDebugString(L"[Info] ISAMcalculationfast 应用聚焦变换完成\n");

	// 对整个3D数据进行二维fft
	std::vector<Complex> S_3D_fft3_Q_k(totalElements);
	for (int a = 0; a < N_Ascan_raw; a++) {
		cv::Mat S_2D_x_y_k = cv::Mat::zeros(N_Bscan, N_Cscan, CV_32FC2);
		for (int c = 0; c < N_Cscan; c++) {
			for (int b = 0; b < N_Bscan; b++) {
				int idx = a + b * N_Ascan_raw + c * N_Ascan_raw * N_Bscan;
				S_2D_x_y_k.at<cv::Vec2f>(b, c)[0] = S_x_y_k_complex[idx].real();
				S_2D_x_y_k.at<cv::Vec2f>(b, c)[1] = S_x_y_k_complex[idx].imag();
			}
		}
		cv::Mat S_2D_Q_k;
		cv::dft(S_2D_x_y_k, S_2D_Q_k);
		fftshift(S_2D_Q_k);

		for (int c = 0; c < N_Cscan; c++) {
			for (int b = 0; b < N_Bscan; b++) {
				int idx = a + b * N_Ascan_raw + c * N_Ascan_raw * N_Bscan;  // 三维线性索引
				cv::Vec2f complex_val = S_2D_Q_k.at<cv::Vec2f>(b, c);
				S_3D_fft3_Q_k[idx] = Complex(complex_val[0], complex_val[1]);
			}
		}
	}


	// 对S_3D_fft3_Q_k 变量进行k域插值
	std::vector<Complex>eta_3D_Q_beta(totalElements);
	std::vector<float> x_vec(N_Ascan_raw);
	for (int a = 0; a < N_Ascan_raw; ++a) {
		x_vec[a] = 2 * k[a];
	}
	for (int b = 0; b < N_Bscan; b++) {
		for (int c = 0; c < N_Cscan; c++) {
			int Aline_index = c + b * N_Cscan;
			float q_val = Q[Aline_index];
			std::vector<float>beta_ISAM(N_Ascan_raw);
			for (int a = 0; a < N_Ascan_raw; a++){
				beta_ISAM[a] = -std::sqrt(std::pow(2 * k[a], 2) - std::pow(q_val, 2));
			}
			//eta_3D_fft3_Q_beta(Aline_index) = 
			std::vector<Complex> y_vec(N_Ascan_raw);
			for (int a = 0; a < N_Ascan_raw; ++a) {
				int s_idx = a + b * N_Ascan_raw + c * N_Ascan_raw * N_Bscan;
				y_vec[a] = S_3D_fft3_Q_k[s_idx];
			}


			for (int a = 0; a < N_Ascan_raw; ++a) {
				float xi = -beta_ISAM[a]; // 对应MATLAB中的-beta_ISAM
				Complex interpolated = linear_interpolate(x_vec, y_vec, xi);

				// 计算存储索引（扁平化3D数组）
				int eta_idx = a + b * N_Ascan_raw + c * N_Ascan_raw * N_Bscan;
				eta_3D_Q_beta[eta_idx] = interpolated;
			}

		}
	}

	// 对整个3D数据进行二维ifft
	std::vector<Complex> eta_3D_x_y_beta(totalElements);
	for (int a = 0; a < N_Ascan_raw; a++) {
		cv::Mat S_2D_x_y_k = cv::Mat::zeros(N_Bscan, N_Cscan, CV_32FC2);
		for (int c = 0; c < N_Cscan; c++) {
			for (int b = 0; b < N_Bscan; b++) {
				int idx = a + b * N_Ascan_raw + c * N_Ascan_raw * N_Bscan;
				S_2D_x_y_k.at<cv::Vec2f>(b, c)[0] = eta_3D_Q_beta[idx].real();
				S_2D_x_y_k.at<cv::Vec2f>(b, c)[1] = eta_3D_Q_beta[idx].imag();
			}
		}
		cv::Mat S_2D_Q_k;
		cv::idft(S_2D_x_y_k, S_2D_Q_k);
		for (int c = 0; c < N_Cscan; c++) {
			for (int b = 0; b < N_Bscan; b++) {
				int idx = a + b * N_Ascan_raw + c * N_Ascan_raw * N_Bscan;  // 三维线性索引
				cv::Vec2f complex_val = S_2D_Q_k.at<cv::Vec2f>(b, c);
				eta_3D_x_y_beta[idx] = Complex(complex_val[0], complex_val[1]);
			}
		}
	}

	// 应用聚焦变换共轭
	for (int idx = 0; idx < totalElements; idx++) {
		eta_3D_x_y_beta[idx] *= std::conj(focus_trans[idx % N_Ascan_raw]);
	}
	// 沿Aline方向进行ifft
	eta_3D_x_y_z.resize(totalElements);
	for (int c = 0; c < N_Cscan; ++c) {
		for (int b = 0; b < N_Bscan; ++b) {
			// 提取当前Bscan和Cscan位置的Ascan数据
			std::vector<Complex> currentAscan(N_Ascan_raw);
			for (int a = 0; a < N_Ascan_raw; ++a) {
				int idx = a + b * N_Ascan_raw + c * N_Ascan_raw * N_Bscan;
				currentAscan[a] = eta_3D_x_y_beta[idx];
			}
			// 转换为OpenCV矩阵进行IFFT
			cv::Mat ascanMat = cv::Mat::zeros(N_Ascan_raw, 1, CV_32FC2);
			for (int a = 0; a < N_Ascan_raw; ++a) {
				ascanMat.at<cv::Vec2f>(a, 0)[0] = currentAscan[a].real();
				ascanMat.at<cv::Vec2f>(a, 0)[1] = currentAscan[a].imag();
			}
			// 执行IFFT
			cv::Mat ifftResult;
			cv::idft(ascanMat, ifftResult, cv::DFT_SCALE);

			// 存储结果
			for (int a = 0; a < N_Ascan_raw; ++a) {
				int idx = a + b * N_Ascan_raw + c * N_Ascan_raw * N_Bscan;
				cv::Vec2f val = ifftResult.at<cv::Vec2f>(a, 0);
				eta_3D_x_y_z[idx] = Complex(val[0], val[1]);
			}
		}
	}
	std::vector<float>eta_3D_x_y_z_real(totalElements);
	std::vector<float>eta_3D_x_y_z_imag(totalElements);
	for (int i = 0; i < totalElements; i++) {
		eta_3D_x_y_z_real[i] = eta_3D_x_y_z[i].real();
		eta_3D_x_y_z_imag[i] = eta_3D_x_y_z[i].imag();
	}
	SaveVectorToBinaryFile("eta_3D_x_y_z_real.bin", eta_3D_x_y_z_real);
	SaveVectorToBinaryFile("eta_3D_x_y_z_imag.bin", eta_3D_x_y_z_imag);


}


ISAM_ERROR ISAMfast::ISAMcalculationfast(std::vector<Complex>& eta_3D_x_y_z, std::vector<Complex>& S_x_y_k_complex,
	int focus_index, float k0, std::vector<float>& Q, int N_Ascan_raw, int N_Bscan, int N_Cscan, float Delta_z, bool zShiftFlag) {
	OutputDebugString(L"[Info] ISAMcalculationfast: start. \n");
	if (S_x_y_k_complex.empty()) {
		std::string msg = "[ERROR] ISAMcalculationfast: S_x_y_k_complex is empty.";
		OutputDebugStringA((msg + "\n").c_str());
		return ISAM_ERROR::PARAM_ERROR;
	}
	if (N_Ascan_raw <= 0 || N_Bscan <= 0 || N_Cscan <= 0) {
		OutputDebugStringA("[ERROR] ISAMcalculationfast: input parameters is not valid.\n");
		return ISAM_ERROR::PARAM_ERROR;
	}
	int totalElements = static_cast<size_t>(N_Ascan_raw) * N_Bscan * N_Cscan;
	if (S_x_y_k_complex.size() != totalElements) {
		std::string msg = "[ERROR] ISAMcalculationfast: S_x_y_k_complex size is not valid.";
		OutputDebugStringA((msg + "\n").c_str());
		return ISAM_ERROR::SIZE_ERROR;
	}
	try {
		std::vector<Complex> focus_trans(N_Ascan_raw);
		if (zShiftFlag == 1) {
			//  定义聚焦变换	
			std::vector<float> indexZ(N_Ascan_raw);
			for (int i = 0; i < N_Ascan_raw; ++i) {
				indexZ[i] = static_cast<float>(i) / (N_Ascan_raw);
			}
			for (int i = 0; i < N_Ascan_raw; ++i) {
				focus_trans[i] = std::exp(Complex(0, -2 * PI * indexZ[i] * focus_index));
			}
			//  应用聚焦变换
			for (int idx = 0; idx < totalElements; idx++) {
				S_x_y_k_complex[idx] *= focus_trans[idx % N_Ascan_raw];
			}
			OutputDebugString(L"[Info] ISAMcalculationfast 应用聚焦变换完成\n");

		}
		else {
			OutputDebugString(L"[Info] ISAMcalculationfast: 不应用聚焦变换\n");
		}

		// 定义振幅和相位因子
		std::vector<float> amplitude_factor(N_Bscan * N_Cscan);
		std::vector<Complex> phase_factor(N_Bscan * N_Cscan);
		for (int y = 0; y < N_Cscan; y++) {
			for (int x = 0; x < N_Bscan; x++) {
				// 计算振幅和相位因子
				int idx = x + y * N_Bscan;
				float q_val = Q[idx];
				float sqrt_val = std::sqrt(1.0f - std::pow(q_val / (2.0f * k0), 2.0f));
				amplitude_factor[idx] = 1.0f / std::sqrt(1.0f - std::pow(q_val / (2.0f * k0), 2.0f));
				phase_factor[idx] = std::exp(Complex(0, 2.0f * k0 * (1.0f - sqrt_val) * Delta_z));
			}
		}
		OutputDebugString(L"[Info] ISAMcalculationfast 计算振幅和相位因子完成\n");

		//  为结果分配空间
		std::vector<Complex> eta_3D_x_y_beta(totalElements);
		//  对每个Aline方向的enface进行遍历
		for (int a = 0; a < N_Ascan_raw; a++) {

			cv::Mat S_2D_x_y_k = cv::Mat::zeros(N_Bscan, N_Cscan, CV_32FC2);

			for (int c = 0; c < N_Cscan; c++) {
				for (int b = 0; b < N_Bscan; b++) {
					int idx = a + b * N_Ascan_raw + c * N_Ascan_raw * N_Bscan;
					S_2D_x_y_k.at<cv::Vec2f>(b, c)[0] = S_x_y_k_complex[idx].real();
					S_2D_x_y_k.at<cv::Vec2f>(b, c)[1] = S_x_y_k_complex[idx].imag();
				}
			}

			cv::Mat S_2D_Q_k;
			cv::dft(S_2D_x_y_k, S_2D_Q_k);
			fftshift(S_2D_Q_k);

			// 将因子应用到FFTshift结果（逐像素相乘）
			cv::Mat S_2D_Q_beta = S_2D_Q_k.clone();
			for (int c = 0; c < N_Cscan; c++) {
				for (int b = 0; b < N_Bscan; b++) {
					int idx = b + c * N_Bscan;
					cv::Vec2f& complex_val = S_2D_Q_beta.at<cv::Vec2f>(b, c);// 
					Complex factor = amplitude_factor[idx] * phase_factor[idx];
					// 复数相乘：(a+jb)*(c+jd) = (ac-bd) + j(ad+bc)
					float real_part = complex_val[0] * factor.real() - complex_val[1] * factor.imag();
					float imag_part = complex_val[0] * factor.imag() + complex_val[1] * factor.real();
					complex_val[0] = real_part;
					complex_val[1] = imag_part;
				}
			}

			cv::Mat eta_2D_x_y_beta;
			cv::idft(S_2D_Q_beta, eta_2D_x_y_beta);
			//std::vector<float> testreal1(eta_2D_x_y_beta.total());
			//std::vector<float> testimag1(eta_2D_x_y_beta.total());
			//for (int c = 0; c < N_Cscan; c++) {
			//	for (int b = 0; b < N_Bscan; b++) {
			//		int i = b + c * N_Bscan;
			//		cv::Vec2f& complex_val = eta_2D_x_y_beta.at<cv::Vec2f>(b, c);
			//		testreal1[i] = complex_val[0];
			//		testimag1[i] = complex_val[1];
			//	}
			//}
			//SaveVectorToBinaryFile("eta_2D_x_y_beta_testre.bin", testreal1);
			//SaveVectorToBinaryFile("eta_2D_x_y_beta_testim.bin", testimag1);
			// 将结果写回输出向量（x: Bscan，y: Cscan，z: Ascan）
			for (int c = 0; c < N_Cscan; c++) {
				for (int b = 0; b < N_Bscan; b++) {
					int idx = a + b * N_Ascan_raw + c * N_Ascan_raw * N_Bscan;  // 三维线性索引
					cv::Vec2f complex_val = eta_2D_x_y_beta.at<cv::Vec2f>(b, c);
					eta_3D_x_y_beta[idx] = Complex(complex_val[0], complex_val[1]);
				}
			}
		}
		OutputDebugString(L"[Info] ISAMcalculationfast eta_3D_Q_z计算结束\n");


		if (zShiftFlag == 1) {
			//  应用聚焦变换
			for (int idx = 0; idx < totalElements; idx++) {
				eta_3D_x_y_beta[idx] *= std::conj(focus_trans[idx % N_Ascan_raw]);
			}   OutputDebugString(L"[Info] ISAMcalculationfast 应用共轭聚焦变换完成\n");
		}
		else {
			OutputDebugString(L"[Info] ISAMcalculationfast: 不应用共轭聚焦变换\n");
		}


		// 执行最后一个IFFT操作（沿Ascan方向）
		eta_3D_x_y_z.resize(totalElements);
		for (int c = 0; c < N_Cscan; ++c) {
			for (int b = 0; b < N_Bscan; ++b) {
				// 提取当前Bscan和Cscan位置的Ascan数据
				std::vector<Complex> currentAscan(N_Ascan_raw);
				for (int a = 0; a < N_Ascan_raw; ++a) {
					int idx = a + b * N_Ascan_raw + c * N_Ascan_raw * N_Bscan;
					currentAscan[a] = eta_3D_x_y_beta[idx];
				}

				// 转换为OpenCV矩阵进行IFFT
				cv::Mat ascanMat = cv::Mat::zeros(N_Ascan_raw, 1, CV_32FC2);
				for (int a = 0; a < N_Ascan_raw; ++a) {
					ascanMat.at<cv::Vec2f>(a, 0)[0] = currentAscan[a].real();
					ascanMat.at<cv::Vec2f>(a, 0)[1] = currentAscan[a].imag();
				}

				// 执行IFFT
				cv::Mat ifftResult;
				cv::idft(ascanMat, ifftResult, cv::DFT_SCALE);

				// 存储结果
				for (int a = 0; a < N_Ascan_raw; ++a) {
					int idx = a + b * N_Ascan_raw + c * N_Ascan_raw * N_Bscan;
					cv::Vec2f val = ifftResult.at<cv::Vec2f>(a, 0);
					eta_3D_x_y_z[idx] = Complex(val[0], val[1]);
				}
			}
		}

		std::vector<float>eta_3D_Q_z_real(totalElements);
		std::vector<float>eta_3D_Q_z_imag(totalElements);
		for (int i = 0; i < totalElements; i++) {
			eta_3D_Q_z_real[i] = eta_3D_x_y_z[i].real();
			eta_3D_Q_z_imag[i] = eta_3D_x_y_z[i].imag();
		}
		SaveVectorToBinaryFile("eta_3D_x_y_z_real.bin", eta_3D_Q_z_real);
		SaveVectorToBinaryFile("eta_3D_x_y_z_imag.bin", eta_3D_Q_z_imag);
	}
	catch (...) {
		// 统一捕获所有异常，简化处理
		OutputDebugStringA("[FATAL] 函数执行过程中发生异常\n");
		return ISAM_ERROR::UNKNOWN_ERROR;
	}

	return ISAM_ERROR::SUCCESS;
	OutputDebugString(L"[Info] ISAMcalculationfast 处理完成\n");
}


void ISAMfast::SaveVectorToFile(const std::string& file_name, vector<float> z_correct, int point_cloud_size) {
	std::ofstream outFile(file_name);

	if (outFile.is_open()) {
		for (const auto& val : z_correct) {
			outFile << val << std::endl;
		}
		outFile.close();
		std::cout << "数据已成功保存到文件: " << file_name << std::endl;
	}
	else {
		std::cerr << "无法打开文件用于写入: " << file_name << std::endl;
	}
}

void ISAMfast::SaveVectorToBinaryFile(const std::string& filename, const std::vector<float>& data) {
	std::ofstream file(filename, std::ios::binary);
	if (!file) throw std::runtime_error("无法打开文件: " + filename);

	file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
	file.close();
	OutputDebugStringA("Vector to bin file saved.\n");

}
