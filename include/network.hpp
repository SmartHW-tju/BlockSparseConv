#include <cstddef>
#include <stdio.h>
#include <hls_math.h>
#include <ap_fixed.h>
#include <ap_int.h>
//#include "hls_stream.h"
#include <iostream>
#include <fstream>
//#include <cmath>

#if !defined(__SYNTHESIS__) && !defined(NO_SYNTH)
#define NO_SYNTH
#endif

// Shared types and kernel entry points for the block-sparse CNN accelerator.
// The design uses 512-bit AXI transfers for off-chip tensors and fixed-point
// arithmetic inside the compute kernels.

// h0903
typedef ap_fixed<16, 4, AP_RND, AP_SAT> FIX_W;
//typedef FIX_F FIX_W;
typedef ap_ufixed<32, 4> FIX_bn;
//typedef float FIX_bn;
// h0903
typedef ap_fixed<16, 11, AP_RND, AP_SAT> FIX_F;
//typedef float16 FIX_F;
typedef ap_uint<10> uint9;
typedef ap_uint<16> uint16;
typedef ap_uint<512> uint512;

typedef ap_fixed<16, 4, AP_RND, AP_SAT> FIX_M;


void pw_1x1_sparse(FIX_F *input, FIX_W *weight, uint9 *block_c, uint16 *block_col_r,
                    int ic, int ih, int oc, int oh, FIX_F *output);

void pw_1x1_sparse_new(FIX_F *input, FIX_W *weight, uint9 *block_c, uint16 *block_col_r, int ic, int ih, int iw, int oc, int oh, int ow, FIX_F *output);

void pw_1x1(FIX_F *input, FIX_W *weight, FIX_W *bias, int ic, int number, int start, FIX_F *output);

// void pw_3x3_first(uint8 input[3][224][224], FIX_F weight[960][3][3], FIX_F bias[1280], FIX_F output[96][112][112]);

void pw_3x3(FIX_F *input, FIX_W *weight, FIX_F *output);

void pw_3x3_new(FIX_F *input, FIX_W *weight, FIX_F *output);

void pw_3x3_new_2(FIX_F *input, FIX_W *weight, FIX_F *output);

void dw_3x3_s1(FIX_F *input, FIX_W *weight, int ic, int ih, int oc, int oh, FIX_F *output);

void dw_3x3_s2(FIX_F *input, FIX_W *weight, int ic, int ih, int oc, int oh, FIX_F *output);

void batchnorm2d(FIX_F *input, FIX_W *parameter, int ic, int ih, int iw, FIX_F *output);

void relu6(FIX_F *input, int ic, int ih, FIX_F *output);

void globalaveragepooling(FIX_F *input, int ic, int ih, FIX_F *output);

void bottleneck_1(FIX_F *buffer_x, FIX_F *buffer_y, int ic, int ih, int oc, int oh, int n1, int n2, int n3, int n4,
				uint512 *sparse_1x1_weight, uint512 *index1, uint16 *index12, uint16 *index22, uint512 *weight_3x3, uint512 *bn);

void bottleneck_2(FIX_F *buffer_x, FIX_F *buffer_y, int ic, int ih, int oc, int oh, int n1, int n2, int n3, int n4,
				uint512 *sparse_1x1_weight, uint512 *index1, uint16 *index12, uint16 *index22, uint512 *weight_3x3, uint512 *bn);

extern "C" {

void network(uint512 *image_raw,
	uint512 *sparse_1x1_weight_raw,
	uint512 *block_c_raw,
	uint512 *block_c_16_raw,
	uint512 *bn_raw,
	uint512 *weight_3x3_raw,
	uint512 *weight_3x3_17_raw,
	uint512 *conv_last_1x1_weight_raw_11,
	uint512 *conv_last_1x1_weight_raw_12,
	uint512 *conv_last_1x1_weight_raw_13,
	uint512 *conv_last_1x1_weight_raw_14,
	uint512 *conv_last_1x1_weight_raw_21,
	uint512 *conv_last_1x1_weight_raw_22,
	uint512 *conv_last_1x1_weight_raw_23,
	uint512 *conv_last_1x1_weight_raw_24,
	uint512 *conv_last_1x1_weight_raw_31,
	uint512 *conv_last_1x1_weight_raw_32,
	uint512 *conv_last_1x1_weight_raw_33,
	uint512 *conv_last_1x1_weight_raw_34,
	uint512 *conv_last_1x1_weight_raw_41,
	uint512 *conv_last_1x1_weight_raw_42,
	uint512 *conv_last_1x1_weight_raw_43,
	uint512 *conv_last_1x1_weight_raw_44,
	uint512 *conv_last_1x1_weight_raw_51,
	uint512 *conv_last_1x1_weight_raw_52,
	uint512 *conv_last_1x1_weight_raw_53,
	uint512 *conv_last_1x1_weight_raw_54,
	uint512 *conv_last_1x1_weight_raw_61,
	uint512 *conv_last_1x1_weight_raw_62,
	uint512 *conv_last_1x1_weight_raw_63,
	uint512 *conv_last_1x1_weight_raw_64,
	uint512 *conv_last_1x1_weight_raw_7,
	uint512 *conv_last_1x1_bias_raw,
	uint16 *pw1_last_1x1_block_col_r_raw, 
	uint16 *pw2_first_1x1_block_col_r_raw,
	uint16 *pw2_last_1x1_block_col_r_raw,
	uint16 *pw3_first_1x1_block_col_r_raw,
	uint16 *pw3_last_1x1_block_col_r_raw,
	uint16 *pw4_first_1x1_block_col_r_raw,
	uint16 *pw4_last_1x1_block_col_r_raw,
	uint16 *pw5_first_1x1_block_col_r_raw,
	uint16 *pw5_last_1x1_block_col_r_raw,
	uint16 *pw6_first_1x1_block_col_r_raw,
	uint16 *pw6_last_1x1_block_col_r_raw,
	uint16 *pw7_first_1x1_block_col_r_raw,
	uint16 *pw7_last_1x1_block_col_r_raw,
	uint16 *pw8_first_1x1_block_col_r_raw,
	uint16 *pw8_last_1x1_block_col_r_raw,
	uint16 *pw9_first_1x1_block_col_r_raw,
	uint16 *pw9_last_1x1_block_col_r_raw,
	uint16 *pw10_first_1x1_block_col_r_raw,
	uint16 *pw10_last_1x1_block_col_r_raw,
	uint16 *pw11_first_1x1_block_col_r_raw,
	uint16 *pw11_last_1x1_block_col_r_raw,
	uint16 *pw12_first_1x1_block_col_r_raw,
	uint16 *pw12_last_1x1_block_col_r_raw,
	uint16 *pw13_first_1x1_block_col_r_raw,
	uint16 *pw13_last_1x1_block_col_r_raw,
	uint16 *pw14_first_1x1_block_col_r_raw,
	uint16 *pw14_last_1x1_block_col_r_raw,
	uint16 *pw15_first_1x1_block_col_r_raw,
	uint16 *pw15_last_1x1_block_col_r_raw,
	uint16 *pw16_first_1x1_block_col_r_raw,
	uint16 *pw16_last_1x1_block_col_r_raw,
	uint16 *pw17_first_1x1_block_col_r_raw,
	uint16 *pw17_last_1x1_block_col_r_raw,
	uint16 *conv_1x1_block_col_r_raw,
	uint512 *feature_map_raw, 
	uint512 *network_output_raw,
	FIX_F *fm_16_1_raw,
	FIX_F *feature_map_2_raw);

}
