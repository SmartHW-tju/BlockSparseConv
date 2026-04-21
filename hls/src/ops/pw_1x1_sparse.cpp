#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <hls_math.h>
#include <iostream>
#include <ap_fixed.h>
#include <ap_int.h>
// #include "hls_stream.h"

#include "network.hpp"

#if !defined(__SYNTHESIS__) && !defined(NO_SYNTH)
#define NO_SYNTH
#endif

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

// Sparse pointwise convolution with a compressed 1x4 block format.
// For each output-channel group `j`, `block_col_r[j]..block_col_r[j+1]-1`
// enumerate the nonzero input-channel blocks stored in `block_c` and `weight`.
void pw_1x1_sparse(FIX_F *input, FIX_W *weight, uint9 *block_c, uint16 *block_col_r, int ic, int ih, int oc, int oh, FIX_F *output)
{

    int iw = ih;
    int ow = oh;
    int ih_img2col = ow*oh;
    int iw_img2col = ic*1*1;

    #ifdef NO_SYNTH
	    // FIX_F input_img2col[ih_img2col][iw_img2col];
        FIX_F *input_img2col = (FIX_F*)malloc(56*56*144*sizeof(FIX_F));
	    // FIX_F matrix[ic][ih_img2col][1*1];
        //FIX_F *matrix = (FIX_F*)malloc(144*56*56*1*1*sizeof(FIX_F));
	    // FIX_F block_multi[oc*ic/8][4];
        //FIX_F *block_multi = (FIX_F*)malloc(320*1280/8*4*sizeof(FIX_F));
        //FIX_F output_img2col[ow*oh][oc];
        //FIX_F *output_img2col = (FIX_F*)malloc(112*112*96*sizeof(FIX_F));
    #else
        FIX_F input_img2col[56*56*144];
        //FIX_F matrix[144*56*56*1*1];
       // FIX_F block_multi[320*1280/8*4];
        //FIX_F output_img2col[112*112*96];
    #endif

    for (int c = 0; c < ic; c++){
        for (int h = 0; h < ih; h++){
            for (int w = 0; w < iw; w++){
                int col_index = (h*iw + w)*ic + c;
                input_img2col[col_index] = input[c*ih*iw + h*iw + w];
                //std::cout << col_index  << " " << input_img2col[col_index] << " " << im_col + iw*(im_row + ih*ic) << " " << input[im_col + iw*(im_row + ih*ic)] << std::endl;
            }
        }
    }

    // calculate;
    for(int i = 0; i < ow*oh; i++){
        int j = 0;
        for(int m; j < (oc/4); j++){
            m = block_col_r[j];

            int q0 = (i + 1)/ow;
            int r0 = (i + 1)%ow - 1;

            output[(j*4 + 0)*oh*ow + q0*ow + r0] = 0;
            output[(j*4 + 1)*oh*ow + q0*ow + r0] = 0;
            output[(j*4 + 2)*oh*ow + q0*ow + r0] = 0;
            output[(j*4 + 3)*oh*ow + q0*ow + r0] = 0;
            // img2col_label9:for(int n = 0; n < block_col_t[j]; n++){
            for(int n = 0; n < (block_col_r[j + 1] - block_col_r[j]); n++){
                //#pragma HLS loop_tripcount min=1 max=2
                output[(j*4 + 0)*oh*ow + q0*ow + r0] +=  input_img2col[i*iw_img2col + block_c[m + n]] * weight[(m + n*1)*4 + 0];
                output[(j*4 + 1)*oh*ow + q0*ow + r0] +=  input_img2col[i*iw_img2col + block_c[m + n]] * weight[(m + n*1)*4 + 1];
                output[(j*4 + 2)*oh*ow + q0*ow + r0] +=  input_img2col[i*iw_img2col + block_c[m + n]] * weight[(m + n*1)*4 + 2];
                output[(j*4 + 3)*oh*ow + q0*ow + r0] +=  input_img2col[i*iw_img2col + block_c[m + n]] * weight[(m + n*1)*4 + 3];
            }
            // img2col_label10:for(int n = 0; n < block_col_t[j]; n++){
        }
    }

    #ifdef NO_SYNTH
        free(input_img2col);
        //free(block_multi);
    #endif
}

// Same sparse pointwise operator, but with explicit non-square spatial sizes.
void pw_1x1_sparse_new(FIX_F *input, FIX_W *weight, uint9 *block_c, uint16 *block_col_r, int ic, int ih, int iw, int oc, int oh, int ow, FIX_F *output)
{
    
    int ih_img2col = ow*oh;
    int iw_img2col = ic*1*1;

    #ifdef NO_SYNTH
	    // FIX_F input_img2col[ih_img2col][iw_img2col];
        FIX_F *input_img2col = (FIX_F*)malloc(56*56*144*sizeof(FIX_F));
	    // FIX_F matrix[ic][ih_img2col][1*1];
        // FIX_F *matrix = (FIX_F*)malloc(144*56*56*1*1*sizeof(FIX_F));
	    // FIX_F block_multi[oc*ic/8][4];
        // FIX_F *block_multi = (FIX_F*)malloc(320*1280/8*4*sizeof(FIX_F));
        //FIX_F output_img2col[ow*oh][oc];
        //FIX_F *output_img2col = (FIX_F*)malloc(112*112*96*sizeof(FIX_F));
    #else
        FIX_F input_img2col[56*56*144];
        //FIX_F matrix[144*56*56*1*1];
        //FIX_F block_multi[320*1280/8*4];
        //FIX_F output_img2col[112*112*96];
    #endif

    for (int c = 0; c < ic; c++){
        for (int h = 0; h < ih; h++){
            for (int w = 0; w < iw; w++){
                int col_index = (h*iw + w)*ic + c;
                input_img2col[col_index] = input[c*ih*iw + h*iw + w];
                //std::cout << col_index  << " " << input_img2col[col_index] << " " << im_col + iw*(im_row + ih*ic) << " " << input[im_col + iw*(im_row + ih*ic)] << std::endl;
            }
        }
    }

    //#ifdef NO_SYNTH
        //free(matrix);
    //#endif

    // calculate;
    for(int i = 0; i < ow*oh; i++){
        int j = 0;
        for(int m; j < (oc/4); j++){
            m = block_col_r[j];
            int q0 = (i + 1)/ow;
            int r0 = (i + 1)%ow - 1;

            output[(j*4 + 0)*oh*ow + q0*ow + r0] = 0;
            output[(j*4 + 1)*oh*ow + q0*ow + r0] = 0;
            output[(j*4 + 2)*oh*ow + q0*ow + r0] = 0;
            output[(j*4 + 3)*oh*ow + q0*ow + r0] = 0;
            // img2col_label9:for(int n = 0; n < block_col_t[j]; n++){
            for(int n = 0; n < (block_col_r[j + 1] - block_col_r[j]); n++){
//#pragma HLS dataflow
            	//#pragma HLS loop_tripcount min=1 max=2
                output[(j*4 + 0)*oh*ow + q0*ow + r0] +=  input_img2col[i*iw_img2col + block_c[m + n]] * weight[(m + n*1)*4 + 0];
                output[(j*4 + 1)*oh*ow + q0*ow + r0] +=  input_img2col[i*iw_img2col + block_c[m + n]] * weight[(m + n*1)*4 + 1];
                output[(j*4 + 2)*oh*ow + q0*ow + r0] +=  input_img2col[i*iw_img2col + block_c[m + n]] * weight[(m + n*1)*4 + 2];
                output[(j*4 + 3)*oh*ow + q0*ow + r0] +=  input_img2col[i*iw_img2col + block_c[m + n]] * weight[(m + n*1)*4 + 3];
                /*if((ic == 16)&&(ih == 28)){
                    std::cout << ((j*4 + 0)*oh*ow + q0*ow + r0) << " " << output[(j*4 + 0)*oh*ow + q0*ow + r0] << " " <<  input_img2col[i*iw_img2col + block_c[m + n]] << " " << weight[(m + n*1)*4 + 0] << std::endl;
                }*/
            }
        }
    }

    #ifdef NO_SYNTH
        free(input_img2col);
    #endif

}
