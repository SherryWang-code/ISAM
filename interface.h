#pragma once
#include "isam.h";

#ifndef ISAM_DLL_EXPORTS
#define ISAM_DLL_API __declspec(dllexport)
#else
#define ISAM_DLL_API __declspec(dllimport)
#endif
/// <summary>
/// 执行 ISAM（干涉合成孔径成像）快速计算，处理三维成像数据并生成结果
/// </summary>
/// <param name="filePath"><输入原始数据文件路径/param>
/// <param name="k"><波数向量，从pixel_wavenumber计算得到/param>
/// <param name="Delta_x"><x 方向的空间采样间隔，单位：米/param>
/// <param name="Delta_y"><y 方向的空间采样间隔，单位：米/param>
/// <param name="refractive_index"><介质的折射率/param>
/// <param name="focus_index"><聚焦索引，指定聚焦变换的参考位置/param>
/// <param name="Z_slice_index"><z 方向切片索引,用于提取某一层/param>
/// <param name="use_focus_shift"><是否启用聚焦偏移功能，true 表示执行聚焦变换步骤，false 表示跳过/param>
/// <param name="N_z"><z 方向（深度方向）的采样点数，Aline个数/param>
/// <param name="N_x"><x 方向的采样点数/param>
/// <param name="N_y"><y 方向的采样点数/param>
/// <returns></returns>
ISAM_DLL_API int ISAMFastCalculation(std::string filePath, std::vector<float>& k, float Delta_x, float Delta_y, float refractive_index,
	float focus_index, int Z_slice_index, bool use_focus_shift, int N_z, int N_x, int N_y);

ISAM_DLL_API int ISAMregular(std::string filePath, std::vector<float>& k, float Delta_x, float Delta_y, float refractive_index,
	float focus_index, int N_z, int N_x, int N_y);