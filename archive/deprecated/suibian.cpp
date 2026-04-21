void network(image,
1x1_sparse_weight,
block_c,
bn,
3x3_weight,
conv_last_1x1_weight,
conv_last_1x1_bias,
pw1_last_1x1_block_col_r, 
pw2_first_1x1_block_col_r,
pw2_last_1x1_block_col_r,
pw3_first_1x1_block_col_r,
pw3_last_1x1_block_col_r,
pw4_first_1x1_block_col_r,
pw4_last_1x1_block_col_r,
pw5_first_1x1_block_col_r,
pw5_last_1x1_block_col_r,
pw6_first_1x1_block_col_r,
pw6_last_1x1_block_col_r,
pw7_first_1x1_block_col_r,
pw7_last_1x1_block_col_r,
pw8_first_1x1_block_col_r,
pw8_last_1x1_block_col_r,
pw9_first_1x1_block_col_r,
pw9_last_1x1_block_col_r,
pw10_first_1x1_block_col_r,
pw10_last_1x1_block_col_r,
pw11_first_1x1_block_col_r,
pw11_last_1x1_block_col_r,
pw12_first_1x1_block_col_r,
pw12_last_1x1_block_col_r,
pw13_first_1x1_block_col_r,
pw13_last_1x1_block_col_r,
pw14_first_1x1_block_col_r,
pw14_last_1x1_block_col_r,
pw15_first_1x1_block_col_r,
pw15_last_1x1_block_col_r,
pw16_first_1x1_block_col_r,
pw16_last_1x1_block_col_r,
pw17_first_1x1_block_col_r,
pw17_last_1x1_block_col_r,
conv_1x1_block_col_r,
feature_map,
network_output
)


static uint16 pw1_last_1x1_block_col_r[16/4 + 1];
static uint16 pw2_first_1x1_block_col_r[96/4 + 1];
static uint16 pw2_last_1x1_block_col_r[24/4 + 1];
static uint16 pw3_first_1x1_block_col_r[144/4 + 1];
static uint16 pw3_last_1x1_block_col_r[24/4 + 1];
static uint16 pw4_first_1x1_block_col_r[144/4 + 1];
static uint16 pw4_last_1x1_block_col_r[32/4 + 1];
static uint16 pw5_first_1x1_block_col_r[192/4 + 1];
static uint16 pw5_last_1x1_block_col_r[32/4 + 1];
static uint16 pw6_first_1x1_block_col_r[192/4 + 1];
static uint16 pw6_last_1x1_block_col_r[192/4 + 1];
static uint16 pw7_first_1x1_block_col_r[192/4 + 1];
static uint16 pw7_last_1x1_block_col_r[64/4 + 1];
static uint16 pw8_first_1x1_block_col_r[384/4 + 1]; //97
static uint16 pw8_last_1x1_block_col_r[64/4 + 1];//17
static uint16 pw9_first_1x1_block_col_r[384/4 + 1];
static uint16 pw9_last_1x1_block_col_r[64/4 + 1];
static uint16 pw10_first_1x1_block_col_r[384/4 + 1];
static uint16 pw10_last_1x1_block_col_r[64/4 + 1];
static uint16 pw11_first_1x1_block_col_r[384/4 + 1];//97
static uint16 pw11_last_1x1_block_col_r[96/4 + 1];//25
static uint16 pw12_first_1x1_block_col_r[576/4 + 1];//145
static uint16 pw12_last_1x1_block_col_r[96/4 + 1];//25
static uint16 pw13_first_1x1_block_col_r[576/4 + 1];
static uint16 pw13_last_1x1_block_col_r[96/4 + 1];
static uint16 pw14_first_1x1_block_col_r[576/4 + 1];//145
static uint16 pw14_last_1x1_block_col_r[160/4 + 1];//41
static uint16 pw15_first_1x1_block_col_r[960/4 + 1];//241
static uint16 pw15_last_1x1_block_col_r[160/4 + 1];//41
static uint16 pw16_first_1x1_block_col_r[960/4 + 1];
static uint16 pw16_last_1x1_block_col_r[160/4 + 1];
static uint16 pw17_first_1x1_block_col_r[960/4 + 1];//241
static uint16 pw17_last_1x1_block_col_r[320/4 + 1];//81
static uint16 conv_1x1_block_col_r[1280/4 + 1];


#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <ap_fixed.h>
#include <ap_int.h>
// #include "hls_stream.h"

#include "network.hpp"

#define NO_SYNTH

// #include "function.h"
// #include <ap_fixed.h>

/*
ic, ih, iw
oc, kc
pad
t = oc*kc/8
oh = ((ih + 2*pad - 1)/stride + 1)
ow = ((iw + 2*pad - 1)/stride + 1)
*/

// 瀵硅緭鍏mg2col灞曞紑骞惰繘琛屽嵎绉绠楋�?

// stride = 1; pad = 0;
void pw_1x1_sparse(FIX_F *input, FIX_W *weight, uint9 *block_c, uint16 *block_col_r, /*FIX_F bias[1280], */int ic, int ih, int oc, int oh, FIX_F *output)
{
    int iw = ih;
    int ow = oh;
    int ih_img2col = ow*oh;
    int iw_img2col = ic*1*1;

    #ifdef NO_SYNTH
	    // FIX_F input_img2col[ih_img2col][iw_img2col];
        FIX_F *input_img2col = (FIX_F*)malloc(56*56*144*sizeof(FIX_F));
    	// FIX_F *input_img2col;
	    // FIX_F matrix[ic][ih_img2col][1*1];
        FIX_F *matrix = (FIX_F*)malloc(144*56*56*1*1*sizeof(FIX_F));
	    // FIX_F block_multi[oc*ic/8][4];
        FIX_F *block_multi = (FIX_F*)malloc(320*1280/8*4*sizeof(FIX_F));
        //FIX_F output_img2col[ow*oh][oc];
        FIX_F *output_img2col = (FIX_F*)malloc(112*112*96*sizeof(FIX_F));
    #else
        //FIX_F input_img2col[56*56*144];
        FIX_F *input_img2col;
        FIX_F matrix[144*56*56*1*1];
        FIX_F block_multi[320*1280/8*4];
        FIX_F output_img2col[112*112*96];
    #endif

    // img2col;
    /*for(int a = 0; a < ic; a++){
        for(int i = 0; i < oh; i++){
            for(int j = 0; j < ow; j++){
                matrix[a*ih_img2col*1 + (i*ow + j)*1 + 0] = input[a*oh*ow + i*ow + j];
                // matrix[a][i*ow + j][1] = input_[a][i*stride][j*stride + 1];
                // matrix[a][i*ow + j][2] = input_[a][i*stride + 1][j*stride];
                // matrix[a][i*ow + j][3] = input_[a][i*stride + 1][j*stride + 1];
            }
        }
    }

    for(int i = 0; i < ih_img2col; i++){
         for(int j = 0; j < 1*1; j++){
        	for(int a = 0; a < ic; a++){
            #pragma HLS PIPELINE II=1
                input_img2col[i*iw_img2col + (j + a*1*1)] = matrix[a*ih_img2col*1 + i*1 + j];
            }
        }
    }*/

    /*for (int c = 0; c < ic; ++c) {
        FIX_F *input_c = input + c * ih *iw;
        for (int m = 0; m < 1; ++m) {
            FIX_F *input_m = input_c + m * iw;
            for (int n = 0; n < 1; ++n) {
                FIX_F *input_n = input_m + n;
                for (int h = 0; h < ih; ++h) {
                    FIX_F *input_h = input_n + h * iw;
                    for (int w = 0; w < iw; ++w) {
                        input_img2col[0] = input_h[w];
                        ++input_img2col;
                    }
                }
            }
        }
    }*/

    for (int c = 0; c < ic; c++){
        for (int h = 0; h < ih; h++){
            for (int w = 0; w < iw; w++){
                int im_row = h*iw + w;
                int im_col = c;
                int index = im_row*ic + im_col;
                input_img2col[index] = input[c*ih*iw + h*iw + w];
            }
        }
    }

    #ifdef NO_SYNTH
        free(matrix);
    #endif

    // calculate;
    for(int i = 0; i < ow*oh; i++){
        int j = 0;
        for(int m; j < (oc/4); j++){
            m = block_col_r[j];
            // img2col_label9:for(int n = 0; n < block_col_t[j]; n++){
            for(int n = 0; n < (block_col_r[j + 1] - block_col_r[j]); n++){
                //#pragma HLS loop_tripcount min=1 max=2
                block_multi[n*4 + 0] = input_img2col[i*iw_img2col + block_c[m + n]] * weight[(m + n*1)*4 + 0];
                block_multi[n*4 + 1] = input_img2col[i*iw_img2col + block_c[m + n]] * weight[(m + n*1)*4 + 1];
                block_multi[n*4 + 2] = input_img2col[i*iw_img2col + block_c[m + n]] * weight[(m + n*1)*4 + 2];
                block_multi[n*4 + 3] = input_img2col[i*iw_img2col + block_c[m + n]] * weight[(m + n*1)*4 + 3];
            }

            output_img2col[i*oc + (j*4 + 0)] = 0;
            output_img2col[i*oc + (j*4 + 1)] = 0;
            output_img2col[i*oc + (j*4 + 2)] = 0;
            output_img2col[i*oc + (j*4 + 3)] = 0;
        
            // img2col_label10:for(int n = 0; n < block_col_t[j]; n++){
            for(int n = 0; n < (block_col_r[j + 1] - block_col_r[j]); n++){
                /*output_img2col[i][j*4 + 0] += block_multi[n][0] + bias[j*4 + 0];
                output_img2col[i][j*4 + 1] += block_multi[n][1] + bias[j*4 + 1];
                output_img2col[i][j*4 + 2] += block_multi[n][2] + bias[j*4 + 2];
                output_img2col[i][j*4 + 3] += block_multi[n][3] + bias[j*4 + 3];*/
                output_img2col[i*oc + (j*4 + 0)] += block_multi[n*4 + 0];
                output_img2col[i*oc + (j*4 + 1)] += block_multi[n*4 + 1];
                output_img2col[i*oc + (j*4 + 2)] += block_multi[n*4 + 2];
                output_img2col[i*oc + (j*4 + 3)] += block_multi[n*4 + 3];
            }
        }
    }

    #ifdef NO_SYNTH
        free(input_img2col);
        free(block_multi);
    #endif

    // 瀵筰mg2col寰楀埌鐨勮緭鍑鸿繘琛屾仮澶嶄互渚夸綔涓轰笅涓�灞傜殑杈撳叆锛�?
    for(int n = 0; n < oc; n++){
        for(int i = 0; i < oh; i++){
            for(int j = 0; j < ow; j++){
                output[n*oh*ow + i*ow + j] = output_img2col[(i*3 + j)*oc + n];
            }
        }
    }

    #ifdef NO_SYNTH
        free(output_img2col);
    #endif
}

    size_t size_in_bytes_image_raw = N_w * sizeof(uint512);
    size_t size_in_bytes_1x1 = N_m * sizeof(uint512);
    size_t size_in_bytes_block_c = N_i * sizeof(uint512);
    size_t size_in_bytes_block_c_16 = N_i * sizeof(uint512);
    size_t size_in_bytes_bn = oc * sizeof(uint512);
    size_t size_in_bytes_3x3 = out_size * sizeof(uint512);
    size_t size_in_bytes_conv_last_1x1 = 1600 * sizeof(uint512);
    size_t size_in_bytes_bias = out_size * sizeof(uint512);
    size_t size_in_bytes_fm = * sizeof(uint512);
    size_t size_in_bytes_output = * sizeof(uint512);
    size_t size_in_bytes_fm_16_1 = * sizeof(FIX_F);
    size_t size_in_bytes_fm_2 = * sizeof(FIX_F);
    size_t size_in_bytes_pw1_last_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw2_first_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw2_last_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw3_first_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw3_last_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw4_first_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw4_last_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw5_first_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw5_last_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw6_first_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw6_last_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw7_first_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw7_last_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw8_first_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw8_last_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw9_first_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw9_last_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw10_first_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw10_last_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw11_first_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw11_last_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw12 _first_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw12_last_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw13_first_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw13_last_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw14_first_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw14_last_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw15_first_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw15_last_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw16_first_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw16_last_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw17_first_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_pw17_last_block_col_r = out_size * sizeof(uint16);
    size_t size_in_bytes_conv_1x1_block_col_r = out_size * sizeof(uint16);

    cl::Buffer buffer_image_raw(context,  CL_MEM_READ_ONLY, size_in_bytes_weights);
    cl::Buffer buffer_1x1(context, CL_MEM_READ_ONLY, size_in_bytes_mask);
    cl::Buffer buffer_block_c(context, CL_MEM_READ_ONLY, size_in_bytes_input);
    cl::Buffer buffer_block_c_16(context,  CL_MEM_READ_ONLY, size_in_bytes_bias);
    cl::Buffer buffer_bn(context,  CL_MEM_WRITE_ONLY, size_in_bytes_output);
    cl::Buffer buffer_3x3(context,  CL_MEM_WRITE_ONLY, size_in_bytes_output);
    cl::Buffer buffer_3x3_17(context,  CL_MEM_WRITE_ONLY, size_in_bytes_output);
    cl::Buffer buffer_last_1x1_11(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_12(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_13(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_14(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_21(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_22(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_23(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_24(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_31(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_32(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_33(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_34(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_41(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_42(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_43(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_44(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_51(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_52(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_53(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_54(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_61(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_62(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_63(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_64(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_last_1x1_7(context, CL_MEM_READ_ONLY, size_in_bytes_conv_last_1x1);
    cl::Buffer buffer_bias(context, CL_MEM_READ_ONLY, size_in_bytes_bias);
    cl::Buffer buffer_fm(context, CL_MEM_READ_ONLY, size_in_bytes_fm);
    cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY, size_in_bytes_output);
    cl::Buffer buffer_fm_16_1(context, CL_MEM_READ_ONLY, size_in_bytes_fm_16_1);
    cl::Buffer buffer_fm_2(context, CL_MEM_READ_ONLY, size_in_bytes_fm_2);
    cl::Buffer buffer_pw1_last_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw1_last_block_col_r);
    cl::Buffer buffer_pw2_first_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw2_first_block_col_r);
    cl::Buffer buffer_pw2_last_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw2_last_block_col_r);
    cl::Buffer buffer_pw3_first_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw3_first_block_col_r);
    cl::Buffer buffer_pw3_last_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw3_last_block_col_r);
    cl::Buffer buffer_pw4_first_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw4_first_block_col_r);
    cl::Buffer buffer_pw4_last_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw4_last_block_col_r);
    cl::Buffer buffer_pw5_first_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw5_first_block_col_r);
    cl::Buffer buffer_pw5_last_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw5_last_block_col_r);
    cl::Buffer buffer_pw6_first_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw6_first_block_col_r);
    cl::Buffer buffer_pw6_last_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw6_last_block_col_r);
    cl::Buffer buffer_pw7_first_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw7_first_block_col_r);
    cl::Buffer buffer_pw7_last_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw7_last_block_col_r);
    cl::Buffer buffer_pw8_first_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw8_first_block_col_r);
    cl::Buffer buffer_pw8_last_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw8_last_block_col_r);
    cl::Buffer buffer_pw9_first_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw9_first_block_col_r);
    cl::Buffer buffer_pw9_last_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw9_last_block_col_r);
    cl::Buffer buffer_pw10_first_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw10_first_block_col_r);
    cl::Buffer buffer_pw10_last_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw10_last_block_col_r);
    cl::Buffer buffer_pw11_first_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw11_first_block_col_r);
    cl::Buffer buffer_pw11_last_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw11_last_block_col_r);
    cl::Buffer buffer_pw12_first_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw12_first_block_col_r);
    cl::Buffer buffer_pw12_last_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw12_last_block_col_r);
    cl::Buffer buffer_pw13_first_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw13_first_block_col_r);
    cl::Buffer buffer_pw13_last_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw13_last_block_col_r);
    cl::Buffer buffer_pw14_first_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw14_first_block_col_r);
    cl::Buffer buffer_pw14_last_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw14_last_block_col_r);
    cl::Buffer buffer_pw15_first_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw15_first_block_col_r);
    cl::Buffer buffer_pw15_last_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw15_last_block_col_r);
    cl::Buffer buffer_pw16_first_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw16_first_block_col_r);
    cl::Buffer buffer_pw16_last_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw16_last_block_col_r);
    cl::Buffer buffer_pw17_first_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw17_first_block_col_r);
    cl::Buffer buffer_pw17_last_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_pw17_last_block_col_r);
    cl::Buffer buffer_conv_1x1_block_col_r(context, CL_MEM_READ_ONLY, size_in_bytes_conv_1x1_block_col_r);


    q.enqueueMigrateMemObjects({buffer_last_1x1_42);
    q.enqueueMigrateMemObjects({buffer_last_1x1_43);
    q.enqueueMigrateMemObjects({buffer_last_1x1_44);
    q.enqueueMigrateMemObjects({buffer_last_1x1_51);
    q.enqueueMigrateMemObjects({buffer_last_1x1_52);
    q.enqueueMigrateMemObjects({buffer_last_1x1_53);
    q.enqueueMigrateMemObjects({buffer_last_1x1_54);
    q.enqueueMigrateMemObjects({buffer_last_1x1_61);
    q.enqueueMigrateMemObjects({buffer_last_1x1_62);
    q.enqueueMigrateMemObjects({buffer_last_1x1_63);
    q.enqueueMigrateMemObjects({buffer_last_1x1_64);
    q.enqueueMigrateMemObjects({buffer_last_1x1_7);
    q.enqueueMigrateMemObjects({buffer_bias);
    q.enqueueMigrateMemObjects({buffer_fm);
    q.enqueueMigrateMemObjects({buffer_output);
    q.enqueueMigrateMemObjects({buffer_fm_16_1);
    q.enqueueMigrateMemObjects({buffer_fm_2);
    q.enqueueMigrateMemObjects({buffer_pw1_last_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw2_first_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw2_last_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw3_first_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw3_last_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw4_first_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw4_last_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw5_first_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw5_last_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw6_first_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw6_last_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw7_first_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw7_last_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw8_first_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw8_last_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw9_first_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw9_last_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw10_first_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw10_last_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw11_first_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw11_last_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw12_first_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw12_last_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw13_first_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw13_last_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw14_first_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw14_last_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw15_first_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw15_last_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw16_first_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw16_last_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw17_first_block_col_r);
    q.enqueueMigrateMemObjects({buffer_pw17_last_block_col_r);
    q.enqueueMigrateMemObjects({buffer_conv_1x1_block_col_r);