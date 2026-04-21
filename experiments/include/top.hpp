#include <stdlib.h>
//#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ap_fixed.h>
#include <ap_int.h>
//#include <CL/cl2.hpp>

#pragma once

/*
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
*/

// #define NO_SYNTH


typedef ap_fixed<16, 4, AP_RND, AP_SAT> FIX_W;
typedef ap_ufixed<32, 4> FIX_bn;
typedef ap_fixed<16, 11, AP_RND, AP_SAT> FIX_F;
typedef ap_uint<10> uint9;
typedef ap_uint<16> uint16;
typedef ap_uint<512> uint512;


#ifdef NO_SYNTH
	FIX_F *image = (FIX_F*)malloc(224*224*3*sizeof(FIX_F));
	FIX_W *sparse_weight_1x1_all = (FIX_W*)malloc(32*33198*sizeof(FIX_W));
	FIX_W *weight_3x3_all = (FIX_W*)malloc(226*32*3*3*sizeof(FIX_W));
	FIX_W *bias_all = (FIX_W*)malloc(1000*sizeof(FIX_W));
	uint9 *block_c_all = (uint9*)malloc(265584*sizeof(uint9));
	// static uint16 block_col_r_all[2506];
	FIX_W *bn_all = (FIX_W*)malloc(32*2132*sizeof(FIX_W));
	FIX_W *conv_last_1x1_weight_1 = (FIX_W*)malloc(160*1280*sizeof(FIX_W));
	FIX_W *conv_last_1x1_weight_2 = (FIX_W*)malloc(160*1280*sizeof(FIX_W));
	FIX_W *conv_last_1x1_weight_3 = (FIX_W*)malloc(160*1280*sizeof(FIX_W));
	FIX_W *conv_last_1x1_weight_4 = (FIX_W*)malloc(160*1280*sizeof(FIX_W));
	FIX_W *conv_last_1x1_weight_5 = (FIX_W*)malloc(160*1280*sizeof(FIX_W));
	FIX_W *conv_last_1x1_weight_6 = (FIX_W*)malloc(160*1280*sizeof(FIX_W));
	FIX_W *conv_last_1x1_weight_7 = (FIX_W*)malloc(40*1280*sizeof(FIX_W));

	uint512 *image_raw = (uint512*)malloc(9408*sizeof(uint512));
	uint512 *sparse_1x1_weight_raw = (uint512*)malloc(33198*sizeof(uint512));
	uint512 *weight_3x3_raw = (uint512*)malloc(226*3*3*sizeof(uint512));
	uint512 *weight_3x3_17_raw = (uint512*)malloc(960*3*3*sizeof(uint512));
	uint512 *conv_last_1x1_bias_raw = (uint512*)malloc(32*sizeof(uint512));
	uint512 *block_c_raw = (uint512*)malloc(8300*sizeof(uint512));// 32*8300=265600;
	uint512 *block_c_16_raw = (uint512*)malloc(1200*sizeof(uint512));
	// static uint512 block_col_r_all_512bit[79]; // 32*79=2528;
	uint512 *bn_raw = (uint512*)malloc(2132*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_11 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_21 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_31 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_41 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_51 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_61 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_12 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_22 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_32 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_42 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_52 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_62 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_13 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_23 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_33 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_43 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_53 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_63 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_14 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_24 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_34 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_44 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_54 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_64 = (uint512*)malloc(1600*sizeof(uint512));
	uint512 *conv_last_1x1_weight_raw_7 = (uint512*)malloc(1600*sizeof(uint512));

	uint512 *feature_map_raw = (uint512*)malloc(5000*sizeof(uint512));
	uint512 *network_output_raw = (uint512*)malloc(32*sizeof(uint512));
	FIX_F *fm_16_1_raw = (FIX_F*)malloc(96*112*112*sizeof(FIX_F));
	FIX_F *feature_map_2_raw = (FIX_F*)malloc(96*112*112*sizeof(FIX_F));
	FIX_F *hls_output = (FIX_F*)malloc(1000*sizeof(FIX_F));

	float *golden_data = (float*)malloc(1000*sizeof(float));

	uint16 *pw1_last_1x1_block_col_r_raw = (uint16*)malloc((16/4 + 1)*sizeof(uint16));
	uint16 *pw2_first_1x1_block_col_r_raw = (uint16*)malloc((96/4 + 1)*sizeof(uint16));
	uint16 *pw2_last_1x1_block_col_r_raw = (uint16*)malloc((24/4 + 1)*sizeof(uint16));
	uint16 *pw3_first_1x1_block_col_r_raw = (uint16*)malloc((144/4 + 1)*sizeof(uint16));
	uint16 *pw3_last_1x1_block_col_r_raw = (uint16*)malloc((24/4 + 1)*sizeof(uint16));
	uint16 *pw4_first_1x1_block_col_r_raw = (uint16*)malloc((144/4 +1)*sizeof(uint16));
	uint16 *pw4_last_1x1_block_col_r_raw = (uint16*)malloc((32/4 + 1)*sizeof(uint16));
	uint16 *pw5_first_1x1_block_col_r_raw = (uint16*)malloc((192/4 + 1)*sizeof(uint16));
	uint16 *pw5_last_1x1_block_col_r_raw = (uint16*)malloc((32/4 + 1)*sizeof(uint16));
	uint16 *pw6_first_1x1_block_col_r_raw = (uint16*)malloc((192/4 + 1)*sizeof(uint16));
	uint16 *pw6_last_1x1_block_col_r_raw = (uint16*)malloc((192/4 + 1)*sizeof(FIX_W));
	uint16 *pw7_first_1x1_block_col_r_raw = (uint16*)malloc((192/4 + 1)*sizeof(uint16));
	uint16 *pw7_last_1x1_block_col_r_raw = (uint16*)malloc((64/4 + 1)*sizeof(uint16));
	uint16 *pw8_first_1x1_block_col_r_raw = (uint16*)malloc((384/4 + 1)*sizeof(uint16));
	uint16 *pw8_last_1x1_block_col_r_raw = (uint16*)malloc((64/4 + 1)*sizeof(uint16));
	uint16 *pw9_first_1x1_block_col_r_raw = (uint16*)malloc((384/4 + 1)*sizeof(uint16));
	uint16 *pw9_last_1x1_block_col_r_raw = (uint16*)malloc((64/4 + 1)*sizeof(uint16));
	uint16 *pw10_first_1x1_block_col_r_raw = (uint16*)malloc((384/4 + 1)*sizeof(uint16));
	uint16 *pw10_last_1x1_block_col_r_raw = (uint16*)malloc((64/4 + 1)*sizeof(uint16));
	uint16 *pw11_first_1x1_block_col_r_raw = (uint16*)malloc((384/4 + 1)*sizeof(uint16));
	uint16 *pw11_last_1x1_block_col_r_raw = (uint16*)malloc((96/4 + 1)*sizeof(uint16));
	uint16 *pw12_first_1x1_block_col_r_raw = (uint16*)malloc((576/4 + 1)*sizeof(uint16));
	uint16 *pw12_last_1x1_block_col_r_raw = (uint16*)malloc((96/4 + 1)*sizeof(uint16));
	uint16 *pw13_first_1x1_block_col_r_raw = (uint16*)malloc((576/4 + 1)*sizeof(uint16));
	uint16 *pw13_last_1x1_block_col_r_raw = (uint16*)malloc((96/4 + 1)*sizeof(uint16));
	uint16 *pw14_first_1x1_block_col_r_raw = (uint16*)malloc((576/4 + 1)*sizeof(uint16));
	uint16 *pw14_last_1x1_block_col_r_raw = (uint16*)malloc((160/4 + 1)*sizeof(uint16));
	uint16 *pw15_first_1x1_block_col_r_raw = (uint16*)malloc((960/4 + 1)*sizeof(uint16));
	uint16 *pw15_last_1x1_block_col_r_raw = (uint16*)malloc((160/4 + 1)*sizeof(uint16));
	uint16 *pw16_first_1x1_block_col_r_raw = (uint16*)malloc((960/4 + 1)*sizeof(uint16));
	uint16 *pw16_last_1x1_block_col_r_raw = (uint16*)malloc((160/4 + 1)*sizeof(uint16));
	uint16 *pw17_first_1x1_block_col_r_raw = (uint16*)malloc((960/4 + 1)*sizeof(uint16));
	uint16 *pw17_last_1x1_block_col_r_raw = (uint16*)malloc((320/4 + 1)*sizeof(uint16));
	uint16 *conv_1x1_block_col_r_raw = (uint16*)malloc((1280/4 + 1)*sizeof(uint16));

#else
	FIX_F image[224*224*3];
	FIX_W sparse_weight_1x1_all[32*33198];
	FIX_W weight_3x3_all[226*32*3*3];
	FIX_W bias_all[1000];
	uint9 block_c_all[265584];
	// static uint16 block_col_r_all[2506];
	FIX_W bn_all[32*2132];
	FIX_W conv_last_1x1_weight_1[160*1280];
	FIX_W conv_last_1x1_weight_2[160*1280];
	FIX_W conv_last_1x1_weight_3[160*1280];
	FIX_W conv_last_1x1_weight_4[160*1280];
	FIX_W conv_last_1x1_weight_5[160*1280];
	FIX_W conv_last_1x1_weight_6[160*1280];
	FIX_W conv_last_1x1_weight_7[40*1280];

	uint512 image_raw[9408];
	uint512 sparse_1x1_weight_raw[33198];
	uint512 weight_3x3_raw[226*3*3];
	uint512 weight_3x3_17_raw[960*3*3];
	uint512 conv_last_1x1_bias_raw[32];
	uint512 block_c_raw[8300];// 32*8300=265600;
	uint512 block_c_16_raw[1200];
	// static uint512 block_col_r_all_512bit[79]; // 32*79=2528;
	uint512 bn_raw[2132];
	uint512 conv_last_1x1_weight_raw_11[1600];
	uint512 conv_last_1x1_weight_raw_21[1600];
	uint512 conv_last_1x1_weight_raw_31[1600];
	uint512 conv_last_1x1_weight_raw_41[1600];
	uint512 conv_last_1x1_weight_raw_51[1600];
	uint512 conv_last_1x1_weight_raw_61[1600];
	uint512 conv_last_1x1_weight_raw_12[1600];
	uint512 conv_last_1x1_weight_raw_22[1600];
	uint512 conv_last_1x1_weight_raw_32[1600];
	uint512 conv_last_1x1_weight_raw_42[1600];
	uint512 conv_last_1x1_weight_raw_52[1600];
	uint512 conv_last_1x1_weight_raw_62[1600];
	uint512 conv_last_1x1_weight_raw_13[1600];
	uint512 conv_last_1x1_weight_raw_23[1600];
	uint512 conv_last_1x1_weight_raw_33[1600];
	uint512 conv_last_1x1_weight_raw_43[1600];
	uint512 conv_last_1x1_weight_raw_53[1600];
	uint512 conv_last_1x1_weight_raw_63[1600];
	uint512 conv_last_1x1_weight_raw_14[1600];
	uint512 conv_last_1x1_weight_raw_24[1600];
	uint512 conv_last_1x1_weight_raw_34[1600];
	uint512 conv_last_1x1_weight_raw_44[1600];
	uint512 conv_last_1x1_weight_raw_54[1600];
	uint512 conv_last_1x1_weight_raw_64[1600];
	uint512 conv_last_1x1_weight_raw_7[1600];

	uint512 feature_map_raw[5000];
	uint512 network_output_raw[32];
	FIX_F hls_output[1000];
	FIX_F fm_16_1_raw[96*112*112];
	FIX_F feature_map_2_raw[96*112*112];

	float golden_data[1000];

	uint16 pw1_last_1x1_block_col_r_raw[(16/4 + 1)];
	uint16 pw2_first_1x1_block_col_r_raw[(96/4 + 1)];
	uint16 pw2_last_1x1_block_col_r_raw[(24/4 + 1)];
	uint16 pw3_first_1x1_block_col_r_raw[(144/4 + 1)];
	uint16 pw3_last_1x1_block_col_r_raw[(24/4 + 1)];
	uint16 pw4_first_1x1_block_col_r_raw[(144/4 +1)];
	uint16 pw4_last_1x1_block_col_r_raw[(32/4 + 1)];
	uint16 pw5_first_1x1_block_col_r_raw[(192/4 + 1)];
	uint16 pw5_last_1x1_block_col_r_raw[(32/4 + 1)];
	uint16 pw6_first_1x1_block_col_r_raw[(192/4 + 1)];
	uint16 pw6_last_1x1_block_col_r_raw[(192/4 + 1)];
	uint16 pw7_first_1x1_block_col_r_raw[(192/4 + 1)];
	uint16 pw7_last_1x1_block_col_r_raw[(64/4 + 1)];
	uint16 pw8_first_1x1_block_col_r_raw[(384/4 + 1)];
	uint16 pw8_last_1x1_block_col_r_raw[(64/4 + 1)];
	uint16 pw9_first_1x1_block_col_r_raw[(384/4 + 1)];
	uint16 pw9_last_1x1_block_col_r_raw[(64/4 + 1)];
	uint16 pw10_first_1x1_block_col_r_raw[(384/4 + 1)];
	uint16 pw10_last_1x1_block_col_r_raw[(64/4 + 1)];
	uint16 pw11_first_1x1_block_col_r_raw[(384/4 + 1)];
	uint16 pw11_last_1x1_block_col_r_raw[(96/4 + 1)];
	uint16 pw12_first_1x1_block_col_r_raw[(576/4 + 1)];
	uint16 pw12_last_1x1_block_col_r_raw[(96/4 + 1)];
	uint16 pw13_first_1x1_block_col_r_raw[(576/4 + 1)];
	uint16 pw13_last_1x1_block_col_r_raw[(96/4 + 1)];
	uint16 pw14_first_1x1_block_col_r_raw[(576/4 + 1)];
	uint16 pw14_last_1x1_block_col_r_raw[(160/4 + 1)];
	uint16 pw15_first_1x1_block_col_r_raw[(960/4 + 1)];
	uint16 pw15_last_1x1_block_col_r_raw[(160/4 + 1)];
	uint16 pw16_first_1x1_block_col_r_raw[(960/4 + 1)];
	uint16 pw16_last_1x1_block_col_r_raw[(160/4 + 1)];
	uint16 pw17_first_1x1_block_col_r_raw[(960/4 + 1)];
	uint16 pw17_last_1x1_block_col_r_raw[(320/4 + 1)];
	uint16 conv_1x1_block_col_r_raw[(1280/4 + 1)];
#endif

//Customized buffer allocation for 4K boundary alignment
template <typename T>
struct aligned_allocator
{
  using value_type = T;
  T* allocate(std::size_t num)
  {
    void* ptr = nullptr;
    if (posix_memalign(&ptr,4096,num*sizeof(T)))
      throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t num)
  {
    free(p);
  }
};
