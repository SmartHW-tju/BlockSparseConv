#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <hls_math.h>
#include <iostream>
#include <ap_fixed.h>
#include <ap_int.h>

#include "network.hpp"

#if !defined(__SYNTHESIS__) && !defined(NO_SYNTH)
#define NO_SYNTH
#endif

#ifdef NO_SYNTH
	// C-simulation uses heap allocation to avoid very large stack frames.
	FIX_F *fm_buffer_1 = (FIX_F*)malloc(144*56*56*sizeof(FIX_F));
	FIX_F *fm_buffer_2 = (FIX_F*)malloc(144*56*56*sizeof(FIX_F));
	uint9 *block_c_buf = (uint9*)malloc(320*1280/8*sizeof(uint9));
	uint16 *block_col_r_buf = (uint16*)malloc((1280/4+1)*sizeof(uint16));
	FIX_W *weight_1x1_buffer = (FIX_W*)malloc(320*1280/8*4*sizeof(FIX_W));
	FIX_W *weight_3x3_buffer = (FIX_W*)malloc(960*3*3*sizeof(FIX_W));
	FIX_W *bn_buffer = (FIX_W*)malloc(4*1280*sizeof(FIX_W));
#else
	FIX_F fm_buffer_1[144*56*56];
	FIX_F fm_buffer_2[144*56*56];
	uint9 block_c_buf[320*1280/8];
	uint16 block_col_r_buf[1280/4+1];
	FIX_W weight_1x1_buffer[320*1280/8*4];
	FIX_W weight_3x3_buffer[960*3*3];
	FIX_W bn_buffer[4*1280];
#endif

// Load sparse 1x1 weights from DDR into a local buffer.
// The stored sparse primitive is a 1x4 output-channel block packed into
// contiguous 512-bit words.
void load_weight_1x1_from_axi(uint512 *weight_1x1_from_axi, int t, int n, FIX_W *buffer)
{
	int g = t*4/32;

	for(int i = 0; i < g; i++)
		for(int x = 0; x < 8; x++){
			for(int y = 0; y < 4; y++){
				buffer[(i*8 + x)*4 + y].range(15, 0) = weight_1x1_from_axi[n + i].range((x*4 + y)*16 + 15, (x*4 + y)*16);
		} 
	} 
}

// Load sparse 1x1 block metadata.
// `block_c` stores the flattened input-channel index of each nonzero 1x4 block.
void load_weight_1x1_block_c_from_axi(uint512 *index1_from_axi, int t, int n, uint9 *buffer)
{
	int g  = (t + 16)/32;

	for(int i = 0; i < g; i++){
		for(int x = 0; x < 32; x++){
			buffer[i*32 + x].range(9, 0) = index1_from_axi[n + i].range(x*16 + 9, x*16);
		}
	}
}

// `block_col_r` is a prefix pointer array of length `oc/4 + 1`.
// Entry j marks where the nonzero list for output-channel group j begins.
void load_weight_1x1_block_col_r_from_axi(uint16 *index2_from_axi, int t, uint16 *buffer)
{
	for(int i = 0; i < t; i++){
		buffer[i] = index2_from_axi[i];
	}
}

// Load dense depthwise 3x3 weights from DDR into the local working set.
void load_weight_3x3_from_axi(uint512 *weight_3x3_from_axi, int w_i_c, int n, FIX_W *buffer)
{
	int t = (w_i_c + 16)/32;
	
	for(int i = 0; i < t; i++){
    	for(int j = 0; j < 3; j++) {
    		for(int k = 0; k < 3; k++) {
                uint512 DATA = 0;
				DATA.range(511, 0) = weight_3x3_from_axi[(n + i)*3*3 + j*3 + k].range(511, 0);
				for(int l = 0; l < 32; l++){
					buffer[(i*32 + l)*3*3 + j*3 + k].range(15, 0) = DATA.range(l*16 + 15, l*16);
					//if((n == 197) && (i == 29)){
						//std::cout << l*3*3 + j*3 + k << " " << buffer[(i*32 + l)*3*3 + j*3 + k] << std::endl;
					//}
				}
			}
    	}
    }
}


// Load classifier bias terms. This path is only used by the final dense 1x1 layer.
void load_bias_from_axi(uint512 *bias_from_axi, FIX_W *buffer)
{

	for(int i = 0; i < 31; i++){
		for(int j = 0; j < 32; j++){
			buffer[i*32 + j].range(15, 0) = bias_from_axi[i].range(j*16 + 15, j*16);
		}
	}

	for(int i = 0; i < 8; i++){
		buffer[992 + i].range(15, 0) = bias_from_axi[31].range(i*16 + 15, i*16);
	}
}

// Load per-channel batch-norm parameters.
// The host packs four parameter planes per channel into one flat tensor.
void load_bn_from_axi(uint512 *bn_from_axi, int bn_i_c, int n, FIX_W *buffer)
{
	int g  = bn_i_c*4/32;

	for(int i = 0; i < g; i++){
		for(int j = 0; j < 32; j++){
			buffer[i*32 + j].range(15, 0) = bn_from_axi[i + n].range(j*16 + 15, j*16);
		}
	}
}


void load_image_from_axi(uint512 *image_from_axi, FIX_F *buffer)
{
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 224; j++){
			for(int k = 0; k < 224; k++){
				buffer[i*224*224 + j*224 + k] = image_from_axi[i*224*224 + j*224 + k];
			}
		}
	}
}


// Load a feature map packed as 32 fixed-point values per 512-bit word.
void load_fm_from_axi(uint512 *fm_from_axi, int fm_i_h, int fm_i_w, int fm_i_c, FIX_F *buffer)
{
	int n = fm_i_h*fm_i_w*fm_i_c/32;

	for(int m = 0; m < n; m++){
		for(int i = 0; i < 32; i++){
			buffer[m*32 + i].range(15, 0) = fm_from_axi[m].range(i*16 + 15, i*16);
		}
	}
}


// Spill an intermediate feature map back to DDR. The top kernel uses this path
// when residual branches exceed the local scratch capacity.
void write_fm_to_ddr(FIX_F *buffer, int fm_o_h, int fm_o_w, int fm_o_c, int start, uint512 *fm_to_axi)
{
	int n = fm_o_h*fm_o_w*fm_o_c/32;

	for(int i = 0; i < n; i++){
		for(int j = 0; j < 32; j++){
			fm_to_axi[start/32 + i].range(j*16 + 15, j*16) = buffer[i*32 + j].range(15, 0);
		}
	}
}


void load_last_weight_from_axi(uint512 *weight_last_from_axi, int number, FIX_W *buffer)
{
	for(int i = 0; i < number; i++){
		for(int j = 0; j < 32; j++){
			buffer[i*32 + j].range(15, 0) = weight_last_from_axi[i].range(j*16 + 15, j*16);
		}
	}
}


void write_output_to_ddr(FIX_F *buffer, int start, int number, uint512 *output_to_axi)
{
	int n = number/32;

	for(int i = 0; i < n; i++){
		for(int j = 0; j < 32; j++){
			output_to_axi[start/32 + i].range(j*16 + 15, j*16) = buffer[i*32 + j].range(15, 0);	
		}
	}
}


void load_conv_last_from_ddr(uint512 *conv_last_axi, int index, FIX_W *buffer)
{
    for(int i = 0; i < 1600; i++){
	    for(int j = 0; j < 32; j++){
			buffer[index + i*32 + j].range(15, 0) = conv_last_axi[i].range(j*16 + 15, j*16);
		}
	}
}


// Utility used by the first large stride-2 block. The feature map is partitioned
// into overlapping stripes so the manually scheduled stem can reuse a smaller buffer.
void load_ptn(FIX_F *from_axi, int ih, int iw, int ic, int start, FIX_F *buffer)
{
	for(int n = 0; n < ic; n++){
		for(int i = 0; i < ih; i++){
			for(int j = 0; j < iw; j++){
				buffer[n*ih*iw + i*iw + j] = from_axi[n*iw*iw + (i + start)*iw + j];
			}
		}
	}
}

void to_ptn(FIX_F *buffer, int ih, int iw, int ic, int start, FIX_F *to_axi)
{
	for(int n = 0; n < ic; n++){
		for(int i = 0; i < ih; i++){
			for(int j = 0; j < iw; j++){
				to_axi[n*iw*iw + (i + start)*iw + j] = buffer[n*ih*iw + i*iw + j];
			}
		}
	}
}

// Compute the main body of a stride-2 depthwise stage over a spatial stripe.
void ptn_calculate(FIX_F *buffer_1, FIX_W *weight, FIX_F *buffer_2)
{
    for(int n = 0; n < 96; n++){
        for(int i = 0; i < 14; i++){
            for(int j = 0; j < 55; j++){
                buffer_2[n*14*56 + i*56 + j] = buffer_1[n*29*112 + i*2*112 + j*2] * weight[n*3*3 + 0*3 + 0]
                                    + buffer_1[n*29*112 + i*2*112 + (j*2 + 1)] * weight[n*3*3 +0*3 + 1]
                                    + buffer_1[n*29*112 + i*2*112 + (j*2 + 2)] * weight[n*3*3 +0*3 + 2]
                                    + buffer_1[n*29*112 + (i*2 + 1)*112 + j*2] * weight[n*3*3 +1*3 + 0]
                                    + buffer_1[n*29*112 + (i*2 + 1)*112 + (j*2 + 1)] * weight[n*3*3 + 1*3 + 1]
                                    + buffer_1[n*29*112 + (i*2 + 1)*112 + (j*2 + 2)] * weight[n*3*3 + 1*3 + 2]
                                    + buffer_1[n*29*112 + (i*2 + 2)*112 + j*2] * weight[n*3*3 +2*3 + 0]
                                    + buffer_1[n*29*112 + (i*2 + 2)*112 + (j*2 + 1)] * weight[n*3*3 + 2*3 + 1]
                                    + buffer_1[n*29*112 + (i*2 + 2)*112 + (j*2 + 2)] * weight[n*3*3 + 2*3 + 2];
            }
        }
    }
}

// Handle the truncated right edge for the manually partitioned stride-2 depthwise stage.
void ptn_calculate_6(FIX_F *buffer_1, FIX_W *weight, FIX_F *buffer_2)
{
	for(int n = 0; n < 96; n++){
		for(int i = 0; i < 14; i++){
            buffer_2[n*14*56 + i*56 + 55] = buffer_1[n*29*112 + i*2*112 + 55*2] * weight[n*3*3 + 0*3 + 0]
                                + buffer_1[n*29*112 + i*2*112 + (55*2 + 1)] * weight[n*3*3 +0*3 + 1]
                                + buffer_1[n*29*112 + (i*2 + 1)*112 + 55*2] * weight[n*3*3 +1*3 + 0]
                                + buffer_1[n*29*112 + (i*2 + 1)*112 + (55*2 + 1)] * weight[n*3*3 + 1*3 + 1]
                                + buffer_1[n*29*112 + (i*2 + 2)*112 + 55*2] * weight[n*3*3 +2*3 + 0]
                                + buffer_1[n*29*112 + (i*2 + 2)*112 + (55*2 + 1)] * weight[n*3*3 + 2*3 + 1];
		}
	}
}

// Generic inverted residual block:
// sparse 1x1 expansion -> BN -> ReLU6 -> depthwise 3x3 -> BN -> ReLU6
// -> sparse 1x1 projection -> BN.
// Skip-path accumulation is handled by the top-level scheduler around this helper.
void bottleneck(FIX_F *buffer_x, FIX_F *buffer_y, int ic, int ih, int oc, int oh, int s_mode, int n1, int n2, int n3, int n4,
				uint512 *sparse_1x1_weight, uint512 *index1, uint16 *index12, uint16 *index22, uint512 *weight_3x3, uint512 *bn)
{
//#pragma HLS ALLOCATION function instances=pw_1x1_sparse limit=1
//#pragma HLS ALLOCATION function instances=batchnorm2d limit=1
//#pragma HLS ALLOCATION function instances=relu6 limit=1

	int b_t;

	FILE* fpp;

	b_t = ic*ic*6/8;

	// 1x1 Pointwise Convolution (sparse) + bn + relu;
	load_weight_1x1_from_axi(sparse_1x1_weight, b_t, n1, weight_1x1_buffer);

	/*if((ic == 24) && (oc == 32)){
		FILE* fpp;
		fpp = fopen("layer4_pw1_weight_d.txt","w");
		for(int i = 0; i < 1728; i++){
        	fprintf(fpp, "%f \n", weight_1x1_buffer[i].to_float());
        }
		fclose(fpp);
	}*/

	load_weight_1x1_block_c_from_axi(index1, b_t, n2, block_c_buf);

	/*if((ic == 24) && (oc == 32)){
		FILE* fpp;
		fpp = fopen("layer4_pw1_block_c_d.txt","w");
		for(int i = 0; i < 448; i++){
        	fprintf(fpp, "%f \n", block_c_buf[i].to_float());
        }
		fclose(fpp);
	}*/

	load_weight_1x1_block_col_r_from_axi(index12, ic*6/4+1, block_col_r_buf); // 1x1_oc = ic*6;

	/*if((ic == 24) && (oc == 32)){
		FILE* fpp;
		fpp = fopen("layer4_pw1_block_col_r_d.txt","w");
		for(int i = 0; i < 37; i++){
        	fprintf(fpp, "%f \n",block_col_r_buf[i].to_float());
        }
		fclose(fpp);
	}*/

	load_bn_from_axi(bn, ic*6, n3, bn_buffer);

	pw_1x1_sparse(buffer_x, weight_1x1_buffer, block_c_buf, block_col_r_buf,
                    ic, ih, ic*6, ih, buffer_y); // 1x1_oc = ic*6; 1x1_oh = ih;

	/*if((ic == 24) && (oc == 32)){
		FILE* fpp;
		fpp = fopen("layer4_pw1_1_d.txt","w");
		for(int i = 0; i < 144; i++){
			for(int j = 0; j < 28; j++){
				for(int k = 0; k < 56; k++){
                    fprintf(fpp, "%f \n", buffer_y[i*56*56 + j*56 + k].to_float());
				}
			}
		}
		fclose(fpp);
	}*/

	batchnorm2d(buffer_y, bn_buffer, ic*6, ih, ih, buffer_x); // 1x1_bn_ic = 1x1_oc = ic*6;

	/*if((ic == 24) && (oc == 32)){
		FILE* fpp;
		fpp = fopen("layer4_pw1_1_bn_d.txt","w");
		for(int i = 0; i < 144; i++){
			for(int j = 0; j < 28; j++){
				for(int k = 0; k < 56; k++){
                    fprintf(fpp, "%f \n", buffer_x[i*56*56 + j*56 + k].to_float());
				}
			}
		}
		fclose(fpp);
	}*/

	relu6(buffer_x, ic*6, ih, buffer_y); // 1x1_relu6_ic = 1x1_bn_oc = 1x1_bn_ic = 1x1_oc = ic*6;

	/*if((ic == 24) && (oc == 32)){
		FILE* fpp;
		fpp = fopen("layer4_pw1_1_relu_d.txt","w");
		for(int i = 0; i < 144; i++){
			for(int j = 0; j < 28; j++){
				for(int k = 0; k < 56; k++){
                    fprintf(fpp, "%f \n", buffer_y[i*56*56 + j*56 + k].to_float());
				}
			}
		}
		fclose(fpp);
	}*/

	/*if((ic == 24) && (oc == 32)){
		FILE* fpp;
		fpp = fopen("layer4_pw1_2_relu_d.txt","w");
		for(int i = 0; i < 144; i++){
			for(int j = 28; j < 56; j++){
				for(int k = 0; k < 56; k++){
                    fprintf(fpp, "%f \n", buffer_y[i*56*56 + j*56 + k].to_float());
				}
			}
		}
		fclose(fpp);
	}*/

	/*if((ic == 160) && (oc == 320)){
		FILE* fpp;
		fpp = fopen("layer17_pw1_relu.txt","w");
		for(int i = 0; i < 7*7*960; i++){
			fprintf(fpp, "%f \n", buffer_y[i].to_float());
		}
		fclose(fpp);
	}*/


	// 3x3 Depthwise Convolution + bn + relu;
	load_weight_3x3_from_axi(weight_3x3, ic*6, n4, weight_3x3_buffer);

	/*if((ic == 24) && (oc == 32)){
		FILE* fpp;
		fpp = fopen("layer4_dw_weight_hls_d.txt","w");
		for(int i = 0; i < 1296; i++){
			fprintf(fpp, "%f \n", weight_3x3_buffer[i].to_float());
		}
		fclose(fpp);
	}

	if((ic == 160) && (oc == 320)){
		FILE* fpp;
		fpp = fopen("layer17_dw_weight_d.txt","w");
		for(int i = 0; i < 3*3*960; i++){
			fprintf(fpp, "%f \n", weight_3x3_buffer[i].to_float());
		}
		fclose(fpp);
	}*/

	//load_bn_from_axi(bn, ic*6, n3 + ic*6/32, bn_buffer);
	load_bn_from_axi(bn, ic*6, n3 + ic*6*4/32, bn_buffer);
	
	if (s_mode == 1)
		dw_3x3_s1(buffer_y, weight_3x3_buffer, ic*6, ih, ic*6, oh, buffer_x); // 3x3_oc = ic*6;
	else
		dw_3x3_s2(buffer_y, weight_3x3_buffer, ic*6, ih, ic*6, oh, buffer_x); // 3x3_oc = ic*6;
	
	batchnorm2d(buffer_x, bn_buffer, ic*6, oh, oh, buffer_y);
	relu6(buffer_y, ic*6, oh, buffer_x);

	/*if((ic == 24) && (oc == 32)){
		FILE* fpp;
		fpp = fopen("layer4_dw_1_relu_d.txt","w");
		for(int i = 0; i < 144; i++){
			for(int j = 0; j < 14; j++){
				for(int k = 0; k < 28; k++){
                    fprintf(fpp, "%f \n", buffer_x[i*28*28 + j*28 + k].to_float());
				}
			}
		}
		fclose(fpp);
	}*/

	/*if((ic == 24) && (oc == 32)){
		FILE* fpp;
		fpp = fopen("layer4_dw_2_relu_d.txt","w");
		for(int i = 0; i < 144; i++){
			for(int j = 14; j < 28; j++){
				for(int k = 0; k < 28; k++){
                    fprintf(fpp, "%f \n", buffer_x[i*28*28 + j*28 + k].to_float());
				}
			}
		}
		fclose(fpp);
	}

	if((ic == 160) && (oc == 320)){
		FILE* fpp;
		fpp = fopen("layer17_dw_relu.txt","w");
		for(int i = 0; i < 7*7*960; i++){
			fprintf(fpp, "%f \n", buffer_y[i].to_float());
		}
		fclose(fpp);
	}*/


	b_t = ic*6*oc/8;

	int temp = (ic*ic*6/8 + 16)/32;

	// 1x1 Pointwise Convolution (sparse) + bn;
	load_weight_1x1_from_axi(sparse_1x1_weight, b_t, n1 + ic*ic*6/8*4/32, weight_1x1_buffer);
	load_weight_1x1_block_c_from_axi(index1, b_t, n2 + temp, block_c_buf);
	load_weight_1x1_block_col_r_from_axi(index22, oc/4+1, block_col_r_buf);
	// load_bn_from_axi(bn, oc, n3 + ic*6*2/32, bn_buffer);
	load_bn_from_axi(bn, oc, n3 + ic*6*4*2/32, bn_buffer);

	pw_1x1_sparse(buffer_x, weight_1x1_buffer, block_c_buf, block_col_r_buf,
                    ic*6, oh, oc, oh, buffer_y);
	batchnorm2d(buffer_y, bn_buffer, oc, oh, oh, buffer_x);
}



void bottleneck_1(FIX_F *buffer_x, FIX_F *buffer_y, int ic, int ih, int oc, int oh, int n1, int n2, int n3, int n4,
				uint512 *sparse_1x1_weight, uint512 *index1, uint16 *index12, uint16 *index22, uint512 *weight_3x3, uint512 *bn)
{
	int b_t;

	FILE* fpp;

	b_t = ic*ic*6/8;

	// 1x1 Pointwise Convolution (sparse) + bn + relu;
	load_weight_1x1_from_axi(sparse_1x1_weight, b_t, n1, weight_1x1_buffer);

	load_weight_1x1_block_c_from_axi(index1, b_t, n2, block_c_buf);

	load_weight_1x1_block_col_r_from_axi(index12, ic*6/4+1, block_col_r_buf); // 1x1_oc = ic*6;

	load_bn_from_axi(bn, ic*6, n3, bn_buffer);

	pw_1x1_sparse(buffer_x, weight_1x1_buffer, block_c_buf, block_col_r_buf,
                    ic, ih, ic*6, ih, buffer_y); // 1x1_oc = ic*6; 1x1_oh = ih;

	batchnorm2d(buffer_y, bn_buffer, ic*6, ih, ih, buffer_x); // 1x1_bn_ic = 1x1_oc = ic*6;

	relu6(buffer_x, ic*6, ih, buffer_y); // 1x1_relu6_ic = 1x1_bn_oc = 1x1_bn_ic = 1x1_oc = ic*6;

	//load_bn_from_axi(bn, ic*6, n3 + ic*6/32, bn_buffer);
	load_bn_from_axi(bn, ic*6, n3 + ic*6*4/32, bn_buffer);
	
	dw_3x3_s1(buffer_y, weight_3x3_buffer, ic*6, ih, ic*6, oh, buffer_x); // 3x3_oc = ic*6;

	batchnorm2d(buffer_x, bn_buffer, ic*6, oh, oh, buffer_y);
	relu6(buffer_y, ic*6, oh, buffer_x);

	b_t = ic*6*oc/8;

	int temp = (ic*ic*6/8 + 16)/32;

	// 1x1 Pointwise Convolution (sparse) + bn;
	load_weight_1x1_from_axi(sparse_1x1_weight, b_t, n1 + ic*ic*6/8*4/32, weight_1x1_buffer);
	load_weight_1x1_block_c_from_axi(index1, b_t, n2 + temp, block_c_buf);
	load_weight_1x1_block_col_r_from_axi(index22, oc/4+1, block_col_r_buf);
	// load_bn_from_axi(bn, oc, n3 + ic*6*2/32, bn_buffer);
	load_bn_from_axi(bn, oc, n3 + ic*6*4*2/32, bn_buffer);

	pw_1x1_sparse(buffer_x, weight_1x1_buffer, block_c_buf, block_col_r_buf,
                    ic*6, oh, oc, oh, buffer_y);
	batchnorm2d(buffer_y, bn_buffer, oc, oh, oh, buffer_x);
}

void bottleneck_2(FIX_F *buffer_x, FIX_F *buffer_y, int ic, int ih, int oc, int oh, int n1, int n2, int n3, int n4,
				uint512 *sparse_1x1_weight, uint512 *index1, uint16 *index12, uint16 *index22, uint512 *weight_3x3, uint512 *bn)
{
	int b_t;

	FILE* fpp;

	b_t = ic*ic*6/8;

	// 1x1 Pointwise Convolution (sparse) + bn + relu;
	load_weight_1x1_from_axi(sparse_1x1_weight, b_t, n1, weight_1x1_buffer);

	load_weight_1x1_block_c_from_axi(index1, b_t, n2, block_c_buf);

	load_weight_1x1_block_col_r_from_axi(index12, ic*6/4+1, block_col_r_buf); // 1x1_oc = ic*6;

	load_bn_from_axi(bn, ic*6, n3, bn_buffer);

	pw_1x1_sparse(buffer_x, weight_1x1_buffer, block_c_buf, block_col_r_buf,
                    ic, ih, ic*6, ih, buffer_y); // 1x1_oc = ic*6; 1x1_oh = ih;

	batchnorm2d(buffer_y, bn_buffer, ic*6, ih, ih, buffer_x); // 1x1_bn_ic = 1x1_oc = ic*6;

	relu6(buffer_x, ic*6, ih, buffer_y); // 1x1_relu6_ic = 1x1_bn_oc = 1x1_bn_ic = 1x1_oc = ic*6;

	//load_bn_from_axi(bn, ic*6, n3 + ic*6/32, bn_buffer);
	load_bn_from_axi(bn, ic*6, n3 + ic*6*4/32, bn_buffer);
	
	dw_3x3_s2(buffer_y, weight_3x3_buffer, ic*6, ih, ic*6, oh, buffer_x); // 3x3_oc = ic*6;

	batchnorm2d(buffer_x, bn_buffer, ic*6, oh, oh, buffer_y);
	relu6(buffer_y, ic*6, oh, buffer_x);

	b_t = ic*6*oc/8;

	int temp = (ic*ic*6/8 + 16)/32;

	// 1x1 Pointwise Convolution (sparse) + bn;
	load_weight_1x1_from_axi(sparse_1x1_weight, b_t, n1 + ic*ic*6/8*4/32, weight_1x1_buffer);
	load_weight_1x1_block_c_from_axi(index1, b_t, n2 + temp, block_c_buf);
	load_weight_1x1_block_col_r_from_axi(index22, oc/4+1, block_col_r_buf);
	// load_bn_from_axi(bn, oc, n3 + ic*6*2/32, bn_buffer);
	load_bn_from_axi(bn, oc, n3 + ic*6*4*2/32, bn_buffer);

	pw_1x1_sparse(buffer_x, weight_1x1_buffer, block_c_buf, block_col_r_buf,
                    ic*6, oh, oc, oh, buffer_y);
	batchnorm2d(buffer_y, bn_buffer, oc, oh, oh, buffer_x);
}




extern "C" {

// Top-level kernel. Large tensors live in external DDR through independent AXI
// interfaces, while the kernel reuses a small set of local scratch buffers and
// explicit DDR spill buffers to fit the full MobileNetV2-style pipeline.
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
	FIX_F *feature_map_2_raw)
{
//#pragma HLS STREAM variable=fm_buffer_1 depth=512 dim=1
//#pragma HLS STREAM variable=fm_buffer_2 depth=512 dim=1
//#pragma HLS ALLOCATION instances=bottleneck limit=1 function
//#pragma HLS ALLOCATION function instances=bottleneck limit=1


	int ic;
	int ih;
	int iw;
	int pad;
	int stride;
	int oc;
	int oh;
	int ow;
	int b_t;
	int bias_t;
	int col_t;

	
	//////////////////// image + conv_3x3 ////////////////////
	// Separate AXI master ports reduce contention between weights, metadata,
	// scratch feature maps, and final outputs.
	#pragma HLS INTERFACE m_axi depth=9408 port=image_raw offset=slave bundle=input

	#pragma HLS INTERFACE m_axi depth=33198 port=sparse_1x1_weight_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=8300 port=block_c_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1200 port=block_c_16_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=2132 port=bn_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=2034 port=weight_3x3_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=112*112*96 port=fm_16_1_raw offset=slave bundle=inout
	#pragma HLS INTERFACE m_axi depth=112*112*96 port=feature_map_2_raw offset=slave bundle=inout

	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_11 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_12 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_13 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_14 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_21 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_22 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_23 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_24 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_31 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_32 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_33 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_34 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_41 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_42 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_43 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_44 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_51 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_52 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_53 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_54 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_61 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_62 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_63 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1600 port=conv_last_1x1_weight_raw_64 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=1700 port=conv_last_1x1_weight_raw_7 offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=32 port=conv_last_1x1_bias_raw offset=slave bundle=input

	#pragma HLS INTERFACE m_axi depth=5 port=pw1_last_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=25 port=pw2_first_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=7 port=pw2_last_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=37 port=pw3_first_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=7 port=pw3_last_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=37 port=pw4_first_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=9 port=pw4_last_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=48 port=pw5_first_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=9 port=pw5_last_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=49 port=pw6_first_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=49 port=pw6_last_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=49 port=pw7_first_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=17 port=pw7_last_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=97 port=pw8_first_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=17 port=pw8_last_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=97 port=pw9_first_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=17 port=pw9_last_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=97 port=pw10_first_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=17 port=pw10_last_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=97 port=pw11_first_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=25 port=pw11_last_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=155 port=pw12_first_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=25 port=pw12_last_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=145 port=pw13_first_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=25 port=pw13_last_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=145 port=pw14_first_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=41 port=pw14_last_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=241 port=pw15_first_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=41 port=pw15_last_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=241 port=pw16_first_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=41 port=pw16_last_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=241 port=pw17_first_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=81 port=pw17_last_1x1_block_col_r_raw offset=slave bundle=input
	#pragma HLS INTERFACE m_axi depth=321 port=conv_1x1_block_col_r_raw offset=slave bundle=input
	
	#pragma HLS INTERFACE m_axi depth=40000 port=feature_map_raw offset=slave bundle=inout
	#pragma HLS INTERFACE m_axi depth=40000 port=network_output_raw offset=slave bundle=output
	
    // #pragma HLS dataflow

	
	/*FIX_W test11;
    FIX_W test22;
    test11.range(15, 0) = conv_last_1x1_weight_raw_6[6144].range(15, 0);
    test22.range(15, 0) = conv_last_1x1_weight_raw_6[6144].range(31, 16);
    std::cout << test11 << std::endl;
    std::cout << test22 << std::endl;*/
	
	load_fm_from_axi(image_raw, 224, 224, 3, fm_buffer_2);

	printf("layer 0 start\n");

	ic = 3;
	ih = 224;
	iw = 224;
	pad = 1;
	stride = 2;
	oc = 32;
	oh = 112;
	ow = 112;

	load_weight_3x3_from_axi(weight_3x3_raw, 96, 0, weight_3x3_buffer);
	load_bn_from_axi(bn_raw, 32, 0, bn_buffer);



	// pw_3x3(image_buffer, weight_3x3_buffer, bias_buffer, fm_buffer_1);
    pw_3x3_new(fm_buffer_2, weight_3x3_buffer, fm_buffer_1);

    batchnorm2d(fm_buffer_1, bn_buffer, 32, 56, 112, fm_buffer_2);

	relu6(fm_buffer_2, 16, 112, fm_buffer_1); //112*56*32 -> 112*112*16;

	for(int n = 0; n < 32; n++){
        for(int i = 0; i < 56; i++){
		    for(int j = 0; j < 112; j++){
		    	fm_16_1_raw[n*112*112 + i*112 + j] = fm_buffer_1[n*56*112 + i*112 + j];
		    }
		}
	}

	/*fp = fopen("layer0_1_hls_d.txt","w");
	for(int i = 0; i < 112*112*16; i++){
		fprintf(fp, "%f \n", fm_buffer_1[i].to_float());
	}

	fclose(fp);*/

	load_fm_from_axi(image_raw, 224, 224, 3, fm_buffer_2);

    pw_3x3_new_2(fm_buffer_2, weight_3x3_buffer, fm_buffer_1);

    batchnorm2d(fm_buffer_1, bn_buffer, 32, 56, 112, fm_buffer_2); 

	relu6(fm_buffer_2, 16, 112, fm_buffer_1); //112*56*32 -> 112*112*16;

	for(int m = 0; m < 32; m++){
        for(int i = 0; i < 56; i++){
		    for(int j = 0; j < 112; j++){
				fm_16_1_raw[m*112*112 + (i + 56)*112 + j] = fm_buffer_1[m*56*112 + i*112 + j];
			}
		}
	}

	printf("3*3 Conv Done\n");
	std::cout << "here0" << std::endl;

	/*fp = fopen("layer0_2_hls_d.txt","w");
	for(int i = 0; i < 112*112*16; i++){
		fprintf(fp, "%f \n", fm_buffer_1[i].to_float());
	}

	fclose(fp);*/


	//////////////////// pw1 + dw1 + pw1 ////////////////////

	printf("layer 1 start\n");

	ic = 32;
	ih = 112;
	iw = 112;
	oc = 16;

	// 3x3 Depthwise Convolution + bn + relu;
	load_weight_3x3_from_axi(weight_3x3_raw, 32, 3, weight_3x3_buffer);
	load_bn_from_axi(bn_raw, 32, 4, bn_buffer);

    load_weight_1x1_from_axi(sparse_1x1_weight_raw, 64, 0, weight_1x1_buffer);                                                                                        
	load_weight_1x1_block_c_from_axi(block_c_raw, 64, 0, block_c_buf);
	load_weight_1x1_block_col_r_from_axi(pw1_last_1x1_block_col_r_raw, 5, block_col_r_buf);
	
	for(int n = 0; n < 32; n++){
        for(int m = 0; m < 112; m++){
		    for(int i = 0; i < 112; i++){
			    fm_buffer_1[n*112*112 + m*112 + i] = fm_16_1_raw[n*112*112 + m*112 + i];
            }
		}
	}

    dw_3x3_s1(fm_buffer_1, weight_3x3_buffer, ic, ih, ic, ih, fm_buffer_2);
    //std::cout << fm_buffer_2[111] << std::endl;
    batchnorm2d(fm_buffer_2, bn_buffer, 32, 112, 112, fm_buffer_1); 
    //std::cout << fm_buffer_1[111] << std::endl;
	relu6(fm_buffer_1, 32, 112, fm_buffer_2); //112*56*32 -> 112*112*16;
    //std::cout << fm_buffer_2[111] << std::endl;
	load_bn_from_axi(bn_raw, 16, 8, bn_buffer);

    pw_1x1_sparse_new(fm_buffer_2, weight_1x1_buffer, block_c_buf, block_col_r_buf, 32, 112, 112, 16, 112, 112, fm_buffer_1);
    //std::cout << fm_buffer_1[111] << std::endl;
    batchnorm2d(fm_buffer_1, bn_buffer, 16, 112, 112, fm_buffer_2);
    //std::cout << fm_buffer_2[111] << std::endl;
	/*fp = fopen("layer1_hls_d.txt","w");
	for(int i = 0; i < 112*112*16; i++){
		fprintf(fp, "%f \n", fm_buffer_2[i].to_float());
	}

	fclose(fp); */


	for(int m = 0; m < 16; m++){
        for(int i = 0; i < 112; i++){
		    for(int j = 0; j < 112; j++){
				fm_16_1_raw[m*112*112 + i*112 + j] = fm_buffer_2[m*112*112 + i*112 + j];
			}
		}
	}

	std::cout << "here1" << std::endl;


	//////////////////// pw2 + dw2 + pw2 ////////////////////

	printf("layer 2 start\n");

    load_weight_1x1_from_axi(sparse_1x1_weight_raw, 192, 8, weight_1x1_buffer);                                                                                        
	load_weight_1x1_block_c_from_axi(block_c_raw, 192, 2, block_c_buf);
	load_weight_1x1_block_col_r_from_axi(pw2_first_1x1_block_col_r_raw, 25, block_col_r_buf);
	load_bn_from_axi(bn_raw, 96, 10, bn_buffer);
	
	// 1
	/*for(int n = 0; n < 16; n++){
		for(int i = 0; i < 28; i++){
			for(int j = 0; j < 112; j++){
				fm_buffer_1[n*28*112 + i*112 + j] = fm_16_1_raw[n*112*112 + i*112 + j];
			}
		}
	}*/

	load_ptn(fm_16_1_raw, 28, 112, 16, 0, fm_buffer_1);

	pw_1x1_sparse_new(fm_buffer_1, weight_1x1_buffer, block_c_buf, block_col_r_buf, 16, 28, 112, 96, 28, 112, fm_buffer_2);

    batchnorm2d(fm_buffer_2, bn_buffer, 96, 28, 112, fm_buffer_1);

	relu6(fm_buffer_1, 24, 112, fm_buffer_2); //112*28*96 -> 112*112*24;

	/*for(int n = 0; n < 96; n++){
		for(int i = 0; i < 28; i++){
			for(int j = 0; j < 112; j++){
				feature_map_2_raw[n*112*112 + i*112 + j] = fm_buffer_2[n*28*112 + i*112 + j];
			}
		}
	}*/

	/*fp = fopen("layer2_pw1_1_relu_hls_109_d.txt","w");
	for(int n = 0; n < 96; n++){
		fprintf(fp, "%f \n", fm_buffer_2[n*28*112 + 0*112 + 109].to_float());
	}

	fclose(fp);

	fp = fopen("layer2_pw1_1_relu_hls_110_d.txt","w");
	for(int n = 0; n < 96; n++){
		fprintf(fp, "%f \n", fm_buffer_2[n*28*112 + 0*112 + 110].to_float());
	}

	fclose(fp);

	fp = fopen("layer2_pw1_1_relu_hls_111_d.txt","w");
	for(int n = 0; n < 96; n++){
		fprintf(fp, "%f \n", fm_buffer_2[n*28*112 + 0*112 + 111].to_float());
	}

	fclose(fp);

	fp = fopen("layer2_pw1_1_relu_hls_221_d.txt","w");
	for(int n = 0; n < 96; n++){
		fprintf(fp, "%f \n", fm_buffer_2[n*28*112 + 1*112 + 109].to_float());
	}

	fclose(fp);

	fp = fopen("layer2_pw1_1_relu_hls_222_d.txt","w");
	for(int n = 0; n < 96; n++){
		fprintf(fp, "%f \n", fm_buffer_2[n*28*112 + 1*112 + 110].to_float());
	}

	fclose(fp);

	fp = fopen("layer2_pw1_1_relu_hls_223_d.txt","w");
	for(int n = 0; n < 96; n++){
		fprintf(fp, "%f \n", fm_buffer_2[n*28*112 + 1*112 + 111].to_float());
	}

	fclose(fp);*/

	to_ptn(fm_buffer_2, 28, 112, 96, 0, feature_map_2_raw);

	//2
	/*for(int n = 0; n < 16; n++){
		for(int i = 0; i < 28; i++){
			for(int j = 0; j < 112; j++){
				fm_buffer_1[n*28*112 + i*112 + j] = fm_16_1_raw[n*112*112 + (i + 28)*112 + j];
			}
		}
	}*/

	load_ptn(fm_16_1_raw, 28, 112, 16, 28, fm_buffer_1);

	pw_1x1_sparse_new(fm_buffer_1, weight_1x1_buffer, block_c_buf, block_col_r_buf, 16, 28, 112, 96, 28, 112, fm_buffer_2);
    batchnorm2d(fm_buffer_2, bn_buffer, 96, 28, 112, fm_buffer_1);
	relu6(fm_buffer_1, 24, 112, fm_buffer_2); //112*28*96 -> 112*112*24;

	/*for(int n = 0; n < 96; n++){
		for(int i = 0; i < 28; i++){
			for(int j = 0; j < 112; j++){
				feature_map_2_raw[n*112*112 + (i + 28)*112 + j] = fm_buffer_2[n*28*112 + i*112 + j];
				//FIX_F test;
				//test.range(15, 0) = network_output_raw[n*392 + i + 98].range(j*16 + 15, j*16);
				//std::cout << (n*98*32 + i*32 + j) << " " << fm_buffer_2[n*98*32 + i*32 + j] << " " << n*392 + i + 98 << " " << test << std::endl;
			}
		}
	}*/

	to_ptn(fm_buffer_2, 28, 112, 96, 28, feature_map_2_raw);


	//FIX_F test;
	//test.range(15, 0) = network_output_raw[98].range(15, 0);
	//std::cout << test << std::endl;

	//3
	/*for(int n = 0; n < 16; n++){
		for(int i = 0; i < 28; i++){
			for(int j = 0; j < 112; j++){
				fm_buffer_1[n*28*112 + i*112 + j] = fm_16_1_raw[n*112*112 + (i + 56)*112 + j];
			}
		}
	}*/

	load_ptn(fm_16_1_raw, 28, 112, 16, 56, fm_buffer_1);

	pw_1x1_sparse_new(fm_buffer_1, weight_1x1_buffer, block_c_buf, block_col_r_buf, 16, 28, 112, 96, 28, 112, fm_buffer_2);
    batchnorm2d(fm_buffer_2, bn_buffer, 96, 28, 112, fm_buffer_1);
	relu6(fm_buffer_1, 24, 112, fm_buffer_2); //112*28*96 -> 112*112*24;

	/*for(int n = 0; n < 96; n++){
		for(int i = 0; i < 28; i++){
			for(int j = 0; j < 112; j++){
				feature_map_2_raw[n*112*112 + (i + 56)*112 + j] = fm_buffer_2[n*28*112 + i*112 + j];
			}
		}
	}*/

	to_ptn(fm_buffer_2, 28, 112, 96, 56, feature_map_2_raw);

	//4
	/*for(int n = 0; n < 16; n++){
		for(int i = 0; i < 28; i++){
			for(int j = 0; j < 112; j++){
				fm_buffer_1[n*28*112 + i*112 + j] = fm_16_1_raw[n*112*112 + (i + 84)*112 + j];
			}
		}
	}*/

	load_ptn(fm_16_1_raw, 28, 112, 16, 84, fm_buffer_1);

	pw_1x1_sparse_new(fm_buffer_1, weight_1x1_buffer, block_c_buf, block_col_r_buf, 16, 28, 112, 96, 28, 112, fm_buffer_2);
    batchnorm2d(fm_buffer_2, bn_buffer, 96, 28, 112, fm_buffer_1);
	relu6(fm_buffer_1, 24, 112, fm_buffer_2); //112*28*96 -> 112*112*24;

	/*for(int n = 0; n < 96; n++){
		for(int i = 0; i < 28; i++){
			for(int j = 0; j < 112; j++){
				feature_map_2_raw[n*112*112 + (i + 84)*112 + j] = fm_buffer_2[n*28*112 + i*112 + j];
			}
		}
	}*/

	to_ptn(fm_buffer_2, 28, 112, 96, 84, feature_map_2_raw);

	load_weight_3x3_from_axi(weight_3x3_raw, 96, 4, weight_3x3_buffer);
	load_bn_from_axi(bn_raw, 96, 22, bn_buffer);


	//1

	/*for(int n = 0; n < 96; n++){
		for(int i = 0; i < 29; i++){
			for(int j = 0; j < 112; j++){
				fm_buffer_1[n*29*112 + i*112 + j] = feature_map_2_raw[n*112*112 + i*112 + j];
			}
		}
	}*/

	load_ptn(feature_map_2_raw, 29, 112, 96, 0, fm_buffer_1);

    /*for(int n = 0; n < 96; n++){
        for(int i = 0; i < 14; i++){
            for(int j = 0; j < 55; j++){
                fm_buffer_2[n*14*56 + i*56 + j] = fm_buffer_1[n*29*112 + i*2*112 + j*2] * weight_3x3_buffer[n*3*3 + 0*3 + 0]
                                    + fm_buffer_1[n*29*112 + i*2*112 + (j*2 + 1)] * weight_3x3_buffer[n*3*3 +0*3 + 1]
                                    + fm_buffer_1[n*29*112 + i*2*112 + (j*2 + 2)] * weight_3x3_buffer[n*3*3 +0*3 + 2]
                                    + fm_buffer_1[n*29*112 + (i*2 + 1)*112 + j*2] * weight_3x3_buffer[n*3*3 +1*3 + 0]
                                    + fm_buffer_1[n*29*112 + (i*2 + 1)*112 + (j*2 + 1)] * weight_3x3_buffer[n*3*3 + 1*3 + 1]
                                    + fm_buffer_1[n*29*112 + (i*2 + 1)*112 + (j*2 + 2)] * weight_3x3_buffer[n*3*3 + 1*3 + 2]
                                    + fm_buffer_1[n*29*112 + (i*2 + 2)*112 + j*2] * weight_3x3_buffer[n*3*3 +2*3 + 0]
                                    + fm_buffer_1[n*29*112 + (i*2 + 2)*112 + (j*2 + 1)] * weight_3x3_buffer[n*3*3 + 2*3 + 1]
                                    + fm_buffer_1[n*29*112 + (i*2 + 2)*112 + (j*2 + 2)] * weight_3x3_buffer[n*3*3 + 2*3 + 2];
            }
        }
    }*/

	ptn_calculate(fm_buffer_1, weight_3x3_buffer, fm_buffer_2);
	
	/*for(int n = 0; n < 96; n++){
		for(int i = 0; i < 14; i++){
            fm_buffer_2[n*14*56 + i*56 + 55] = fm_buffer_1[n*29*112 + i*2*112 + 55*2] * weight_3x3_buffer[n*3*3 + 0*3 + 0]
                                + fm_buffer_1[n*29*112 + i*2*112 + (55*2 + 1)] * weight_3x3_buffer[n*3*3 +0*3 + 1]
                                + fm_buffer_1[n*29*112 + (i*2 + 1)*112 + 55*2] * weight_3x3_buffer[n*3*3 +1*3 + 0]
                                + fm_buffer_1[n*29*112 + (i*2 + 1)*112 + (55*2 + 1)] * weight_3x3_buffer[n*3*3 + 1*3 + 1]
                                + fm_buffer_1[n*29*112 + (i*2 + 2)*112 + 55*2] * weight_3x3_buffer[n*3*3 +2*3 + 0]
                                + fm_buffer_1[n*29*112 + (i*2 + 2)*112 + (55*2 + 1)] * weight_3x3_buffer[n*3*3 + 2*3 + 1];
		}
	}*/

	ptn_calculate_6(fm_buffer_1, weight_3x3_buffer, fm_buffer_2);

	batchnorm2d(fm_buffer_2, bn_buffer, 96, 14, 56, fm_buffer_1); 

	relu6(fm_buffer_1, 24, 56, fm_buffer_2); //14*56*96 -> 56*56*24;


	/*for(int n = 0; n < 96; n++){
		for(int i = 0; i < 14; i++){
			for(int j = 0; j < 56; j++){
				fm_16_1_raw[n*56*56 + i*56 + j] = fm_buffer_2[n*14*56 + i*56 + j];
			}
		}
	}*/

	to_ptn(fm_buffer_2, 14, 56, 96, 0, fm_16_1_raw);

	/*fp = fopen("layer2_dw_1_relu_hls_55_d.txt","w");
	for(int n = 0; n < 96; n++){
		fprintf(fp, "%f \n", fm_buffer_2[n*14*56 + 55].to_float());
	}

	fclose(fp);*/

	//2

	/*for(int n = 0; n < 96; n++){
		for(int i = 0; i < 29; i++){
			for(int j = 0; j < 112; j++){
				fm_buffer_1[n*29*112 + i*112 + j] = feature_map_2_raw[n*112*112 + (i + 28)*112 + j];
				//FIX_F test;
				//test.range(15, 0) = network_output_raw[n*392 + i + 98].range(j*16 + 15, j*16);
				//std::cout << (n*30*112 + i*32 + j) << " " << fm_buffer_1[n*30*112 + i*32 + j] << " " << (n*392 + i + 98) << " " << test << std::endl;
			}
		}
	}*/

	load_ptn(feature_map_2_raw, 29, 112, 96, 28, fm_buffer_1);

    /*for(int n = 0; n < 96; n++){
        for(int i = 0; i < 14; i++){
            for(int j = 0; j < 55; j++){
                fm_buffer_2[n*14*56 + i*56 + j] = fm_buffer_1[n*29*112 + i*2*112 + j*2] * weight_3x3_buffer[n*3*3 + 0*3 + 0]
                                    + fm_buffer_1[n*29*112 + i*2*112 + (j*2 + 1)] * weight_3x3_buffer[n*3*3 +0*3 + 1]
                                    + fm_buffer_1[n*29*112 + i*2*112 + (j*2 + 2)] * weight_3x3_buffer[n*3*3 +0*3 + 2]
                                    + fm_buffer_1[n*29*112 + (i*2 + 1)*112 + j*2] * weight_3x3_buffer[n*3*3 +1*3 + 0]
                                    + fm_buffer_1[n*29*112 + (i*2 + 1)*112 + (j*2 + 1)] * weight_3x3_buffer[n*3*3 + 1*3 + 1]
                                    + fm_buffer_1[n*29*112 + (i*2 + 1)*112 + (j*2 + 2)] * weight_3x3_buffer[n*3*3 + 1*3 + 2]
                                    + fm_buffer_1[n*29*112 + (i*2 + 2)*112 + j*2] * weight_3x3_buffer[n*3*3 +2*3 + 0]
                                    + fm_buffer_1[n*29*112 + (i*2 + 2)*112 + (j*2 + 1)] * weight_3x3_buffer[n*3*3 + 2*3 + 1]
                                    + fm_buffer_1[n*29*112 + (i*2 + 2)*112 + (j*2 + 2)] * weight_3x3_buffer[n*3*3 + 2*3 + 2];
            }
        }
    }*/

    ptn_calculate(fm_buffer_1, weight_3x3_buffer, fm_buffer_2); 
	
	/*for(int n = 0; n < 96; n++){
		for(int i = 0; i < 14; i++){
            fm_buffer_2[n*14*56 + i*56 + 55] = fm_buffer_1[n*29*112 + i*2*112 + 55*2] * weight_3x3_buffer[n*3*3 + 0*3 + 0]
                                + fm_buffer_1[n*29*112 + i*2*112 + (55*2 + 1)] * weight_3x3_buffer[n*3*3 +0*3 + 1]
                                + fm_buffer_1[n*29*112 + (i*2 + 1)*112 + 55*2] * weight_3x3_buffer[n*3*3 +1*3 + 0]
                                + fm_buffer_1[n*29*112 + (i*2 + 1)*112 + (55*2 + 1)] * weight_3x3_buffer[n*3*3 + 1*3 + 1]
                                + fm_buffer_1[n*29*112 + (i*2 + 2)*112 + 55*2] * weight_3x3_buffer[n*3*3 +2*3 + 0]
                                + fm_buffer_1[n*29*112 + (i*2 + 2)*112 + (55*2 + 1)] * weight_3x3_buffer[n*3*3 + 2*3 + 1];
		}
	}*/

	ptn_calculate_6(fm_buffer_1, weight_3x3_buffer, fm_buffer_2);

	batchnorm2d(fm_buffer_2, bn_buffer, 96, 14, 56, fm_buffer_1); 

	relu6(fm_buffer_1, 24, 56, fm_buffer_2); //14*56*96 -> 56*56*24;
	
	/*for(int n = 0; n < 96; n++){
		for(int i = 0; i < 14; i++){
			for(int j = 0; j < 56; j++){
				fm_16_1_raw[n*56*56 + (i + 14)*56 + j] = fm_buffer_2[n*14*56 + i*56 + j];
			}
		}
	}*/

	to_ptn(fm_buffer_2, 14, 56, 96, 14, fm_16_1_raw);


	//3
	/*for(int n = 0; n < 96; n++){
		for(int i = 0; i < 29; i++){
			for(int j = 0; j < 112; j++){
				fm_buffer_1[n*29*112 + i*112 + j] = feature_map_2_raw[n*112*112 + (i + 56)*112 + j];
			}
		}
	}*/

	load_ptn(feature_map_2_raw, 29, 112, 96, 56, fm_buffer_1);

    /*for(int n = 0; n < 96; n++){
        for(int i = 0; i < 14; i++){
            for(int j = 0; j < 55; j++){
                fm_buffer_2[n*14*56 + i*56 + j] = fm_buffer_1[n*29*112 + i*2*112 + j*2] * weight_3x3_buffer[n*3*3 + 0*3 + 0]
                                    + fm_buffer_1[n*29*112 + i*2*112 + (j*2 + 1)] * weight_3x3_buffer[n*3*3 +0*3 + 1]
                                    + fm_buffer_1[n*29*112 + i*2*112 + (j*2 + 2)] * weight_3x3_buffer[n*3*3 +0*3 + 2]
                                    + fm_buffer_1[n*29*112 + (i*2 + 1)*112 + j*2] * weight_3x3_buffer[n*3*3 +1*3 + 0]
                                    + fm_buffer_1[n*29*112 + (i*2 + 1)*112 + (j*2 + 1)] * weight_3x3_buffer[n*3*3 + 1*3 + 1]
                                    + fm_buffer_1[n*29*112 + (i*2 + 1)*112 + (j*2 + 2)] * weight_3x3_buffer[n*3*3 + 1*3 + 2]
                                    + fm_buffer_1[n*29*112 + (i*2 + 2)*112 + j*2] * weight_3x3_buffer[n*3*3 +2*3 + 0]
                                    + fm_buffer_1[n*29*112 + (i*2 + 2)*112 + (j*2 + 1)] * weight_3x3_buffer[n*3*3 + 2*3 + 1]
                                    + fm_buffer_1[n*29*112 + (i*2 + 2)*112 + (j*2 + 2)] * weight_3x3_buffer[n*3*3 + 2*3 + 2];
            }
        }
    }*/

	ptn_calculate(fm_buffer_1, weight_3x3_buffer, fm_buffer_2);
	
	/*for(int n = 0; n < 96; n++){
		for(int i = 0; i < 14; i++){
            fm_buffer_2[n*14*56 + i*56 + 55] = fm_buffer_1[n*29*112 + i*2*112 + 55*2] * weight_3x3_buffer[n*3*3 + 0*3 + 0]
                                + fm_buffer_1[n*29*112 + i*2*112 + (55*2 + 1)] * weight_3x3_buffer[n*3*3 +0*3 + 1]
                                + fm_buffer_1[n*29*112 + (i*2 + 1)*112 + 55*2] * weight_3x3_buffer[n*3*3 +1*3 + 0]
                                + fm_buffer_1[n*29*112 + (i*2 + 1)*112 + (55*2 + 1)] * weight_3x3_buffer[n*3*3 + 1*3 + 1]
                                + fm_buffer_1[n*29*112 + (i*2 + 2)*112 + 55*2] * weight_3x3_buffer[n*3*3 +2*3 + 0]
                                + fm_buffer_1[n*29*112 + (i*2 + 2)*112 + (55*2 + 1)] * weight_3x3_buffer[n*3*3 + 2*3 + 1];
		}
	}*/

	ptn_calculate_6(fm_buffer_1, weight_3x3_buffer, fm_buffer_2);

	batchnorm2d(fm_buffer_2, bn_buffer, 96, 14, 56, fm_buffer_1); 

	relu6(fm_buffer_1, 24, 56, fm_buffer_2); //14*56*96 -> 56*56*24;

	/*for(int n = 0; n < 96; n++){
		for(int i = 0; i < 14; i++){
			for(int j = 0; j < 56; j++){
				fm_16_1_raw[n*56*56 + (i + 28)*56 + j] = fm_buffer_2[n*14*56 + i*56 + j];
			}
		}
	}*/

	to_ptn(fm_buffer_2, 14, 56, 96, 28, fm_16_1_raw);


	//4
	/*for(int n = 0; n < 96; n++){
		for(int i = 0; i < 28; i++){
			for(int j = 0; j < 112; j++){
				fm_buffer_1[n*28*112 + i*112 + j] = feature_map_2_raw[n*112*112 + (i + 84)*112 + j];
			}
		}
	}*/

	load_ptn(feature_map_2_raw, 28, 112, 96, 84, fm_buffer_1);

    for(int n = 0; n < 96; n++){
        for(int i = 0; i < 13; i++){
            for(int j = 0; j < 55; j++){
                fm_buffer_2[n*14*56 + i*56 + j] = fm_buffer_1[n*28*112 + i*2*112 + j*2] * weight_3x3_buffer[n*3*3 + 0*3 + 0]
                                    + fm_buffer_1[n*28*112 + i*2*112 + (j*2 + 1)] * weight_3x3_buffer[n*3*3 +0*3 + 1]
                                    + fm_buffer_1[n*28*112 + i*2*112 + (j*2 + 2)] * weight_3x3_buffer[n*3*3 +0*3 + 2]
                                    + fm_buffer_1[n*28*112 + (i*2 + 1)*112 + j*2] * weight_3x3_buffer[n*3*3 +1*3 + 0]
                                    + fm_buffer_1[n*28*112 + (i*2 + 1)*112 + (j*2 + 1)] * weight_3x3_buffer[n*3*3 + 1*3 + 1]
                                    + fm_buffer_1[n*28*112 + (i*2 + 1)*112 + (j*2 + 2)] * weight_3x3_buffer[n*3*3 + 1*3 + 2]
                                    + fm_buffer_1[n*28*112 + (i*2 + 2)*112 + j*2] * weight_3x3_buffer[n*3*3 +2*3 + 0]
                                    + fm_buffer_1[n*28*112 + (i*2 + 2)*112 + (j*2 + 1)] * weight_3x3_buffer[n*3*3 + 2*3 + 1]
                                    + fm_buffer_1[n*28*112 + (i*2 + 2)*112 + (j*2 + 2)] * weight_3x3_buffer[n*3*3 + 2*3 + 2];
            }
        }
    }
	
	for(int n = 0; n < 96; n++){
		for(int i = 0; i < 13; i++){
            fm_buffer_2[n*14*56 + i*56 + 55] = fm_buffer_1[n*28*112 + i*2*112 + 55*2] * weight_3x3_buffer[n*3*3 + 0*3 + 0]
                                + fm_buffer_1[n*28*112 + i*2*112 + (55*2 + 1)] * weight_3x3_buffer[n*3*3 +0*3 + 1]
                                + fm_buffer_1[n*28*112 + (i*2 + 1)*112 + 55*2] * weight_3x3_buffer[n*3*3 +1*3 + 0]
                                + fm_buffer_1[n*28*112 + (i*2 + 1)*112 + (55*2 + 1)] * weight_3x3_buffer[n*3*3 + 1*3 + 1]
                                + fm_buffer_1[n*28*112 + (i*2 + 2)*112 + 55*2] * weight_3x3_buffer[n*3*3 +2*3 + 0]
                                + fm_buffer_1[n*28*112 + (i*2 + 2)*112 + (55*2 + 1)] * weight_3x3_buffer[n*3*3 + 2*3 + 1];
		}
	}

	for(int n = 0; n < 96; n++){
		for(int j = 0; j < 55; j++){
            fm_buffer_2[n*14*56 + 13*56 + j] = fm_buffer_1[n*28*112 + 13*2*112 + j*2] * weight_3x3_buffer[n*3*3 + 0*3 + 0]
                                + fm_buffer_1[n*28*112 + 13*2*112 + (j*2 + 1)] * weight_3x3_buffer[n*3*3 +0*3 + 1]
                                + fm_buffer_1[n*28*112 + 13*2*112 + (j*2 + 2)] * weight_3x3_buffer[n*3*3 +0*3 + 2]
                                + fm_buffer_1[n*28*112 + (13*2 + 1)*112 + j*2] * weight_3x3_buffer[n*3*3 +1*3 + 0]
                                + fm_buffer_1[n*28*112 + (13*2 + 1)*112 + (j*2 + 1)] * weight_3x3_buffer[n*3*3 + 1*3 + 1]
                                + fm_buffer_1[n*28*112 + (13*2 + 1)*112 + (j*2 + 2)] * weight_3x3_buffer[n*3*3 + 1*3 + 2];
		}
	}

	for(int n = 0; n < 96; n++){
		fm_buffer_2[n*14*56 + 13*56 + 55] = fm_buffer_1[n*28*112 + 13*2*112 + 55*2] * weight_3x3_buffer[n*3*3 + 0*3 + 0]
                                + fm_buffer_1[n*28*112 + 13*2*112 + (55*2 + 1)] * weight_3x3_buffer[n*3*3 +0*3 + 1]
                                + fm_buffer_1[n*28*112 + (13*2 + 1)*112 + 55*2] * weight_3x3_buffer[n*3*3 +1*3 + 0]
                                + fm_buffer_1[n*28*112 + (13*2 + 1)*112 + (55*2 + 1)] * weight_3x3_buffer[n*3*3 + 1*3 + 1];
	}

	batchnorm2d(fm_buffer_2, bn_buffer, 96, 14, 56, fm_buffer_1); 

	relu6(fm_buffer_1, 24, 56, fm_buffer_2); //14*56*96 -> 56*56*24;

	/*for(int n = 0; n < 96; n++){
		for(int i = 0; i < 14; i++){
			for(int j = 0; j < 56; j++){
				fm_16_1_raw[n*56*56 + (i + 42)*56 + j] = fm_buffer_2[n*14*56 + i*56 + j];
			}
		}
	}*/

	to_ptn(fm_buffer_2, 14, 56, 96, 42, fm_16_1_raw);

    load_weight_1x1_from_axi(sparse_1x1_weight_raw, 288, 32, weight_1x1_buffer);                                                                                        
	load_weight_1x1_block_c_from_axi(block_c_raw, 288, 8, block_c_buf);
	load_weight_1x1_block_col_r_from_axi(pw2_last_1x1_block_col_r_raw, 7, block_col_r_buf);
	load_bn_from_axi(bn_raw, 24, 34, bn_buffer);

	//load_fm_from_axi(feature_map_raw, 56, 56, 96, fm_buffer_1);//gai__________________

	/*for(int n = 0; n < 96; n++){
		for(int i = 0; i < 56; i++){
			for(int j = 0; j < 56; j++){
				fm_buffer_1[n*56*56 + i*56 + j] = fm_16_1_raw[n*56*56 + i*56 + j];
			}
		}
	}*/

	load_ptn(fm_16_1_raw, 56, 56, 96, 0, fm_buffer_2);

	/*fp = fopen("layer2_dw_1_relu_hls_55_fix_d.txt","w");
	for(int n = 0; n < 96; n++){
		fprintf(fp, "%f \n", fm_buffer_2[n*56*56 + 55].to_float());
	}

	fclose(fp);*/

	pw_1x1_sparse(fm_buffer_2, weight_1x1_buffer, block_c_buf, block_col_r_buf,
                    96, 56, 24, 56, fm_buffer_1);

	/*fp = fopen("layer2_pw2_hls_55_d.txt","w");
	for(int n = 0; n < 96; n++){
		fprintf(fp, "%f \n", fm_buffer_1[n*56*56 + 55].to_float());
	}

	fclose(fp);*/

	batchnorm2d(fm_buffer_1, bn_buffer, 24, 56, 56, fm_buffer_2); 

	/*fp = fopen("layer2_pw2_bn_hls_55_d.txt","w");
	for(int n = 0; n < 96; n++){
		fprintf(fp, "%f \n", fm_buffer_2[n*56*56 + 55].to_float());
	}

	fclose(fp);*/

	/*for(int n = 0; n < 24; n++){
		for(int i = 0; i < 56; i++){
			for(int j = 0; j < 56; j++){
				feature_map_2_raw[n*56*56 + i*56 + j] = fm_buffer_1[n*56*56 + i*56 + j];
			}
		}
	}*/

	to_ptn(fm_buffer_2, 56, 56, 24, 0, feature_map_2_raw);

	for(int i = 0; i < 2352; i++){
		for(int j = 0; j < 32; j++){
			feature_map_raw[i].range(j*16 + 15, j*16) = fm_buffer_2[i*32 + j].range(15, 0);
		}
	}

	/*fp = fopen("layer2_hls_d.txt","w");
	for(int i = 0; i < 56*56*24; i++){
		fprintf(fp, "%f \n", fm_buffer_2[i].to_float());
	}

	fclose(fp);*/

	std::cout << "here2" << std::endl;


	//////////////////// pw3 + dw3 + pw3 ////////////////////

	ic = 24;
	ih = 56;
	iw = 56;
	oc = 24;
	oh = 56;
	ow = 56;
	
	printf("Bottleneck_3 Start\n");

	bottleneck_1(fm_buffer_2, fm_buffer_1, ic, ih, oc, oh, 68, 17, 37, 7,
				sparse_1x1_weight_raw, block_c_raw, pw3_first_1x1_block_col_r_raw, pw3_last_1x1_block_col_r_raw, weight_3x3_raw, bn_raw);
	
	printf("Bottleneck_3 Done\n");
	std::cout << "here3" << std::endl;

	/*fp = fopen("layer3_hls_d.txt","w");
	for(int i = 0; i < 56*56*24; i++){
		fprintf(fp, "%f \n", fm_buffer_2[i].to_float());
	}

	fclose(fp);*/


	/*fp = fopen("feature_map_2_raw_23pre.txt","w");
	for(int i = 0; i < 56*56*24; i++){
		fprintf(fp, "%f \n", feature_map_2_raw[i].to_float());
	}

	fclose(fp);*/	


	//////////////////// 2 + 3 ////////////////////
	
	printf("2 From DDR Start\n");
	// load_fm_from_axi(feature_map_raw, 56, 56, 24, fm_buffer_1);

	for(int i = 0; i < 2352; i++){
		for(int j = 0; j < 32; j++){
			fm_buffer_1[i*32 + j].range(15, 0) = feature_map_raw[i].range(j*16 + 15, j*16);
		}
	}

	/*fp = fopen("layer2_to_hls.txt","w");
	for(int i = 0; i < 56*56*24; i++){
		fprintf(fp, "%f \n", fm_buffer_1[i].to_float());
	}

	fclose(fp);*/


	printf("2 + 3 Start\n");
	for(int i = 0; i < 24; i++){
		for(int j = 0; j < 56; j++){
			for(int k = 0; k < 56; k++){
				fm_buffer_2[i*56*56 + j*56 + k] += fm_buffer_1[i*56*56 + j*56 + k];
            }
		}
	}
	printf("2 + 3 Done\n");

	for(int n = 0 ; n < 24; n++){
		for(int i = 0; i < 56; i++){
			for(int j = 0; j < 56; j++){
				fm_16_1_raw[n*56*56 + i*56 + j] = fm_buffer_2[n*56*56 + i*56 + j];
			}
		}
	}

	/*fp = fopen("layer23_hls.txt","w");
	for(int i = 0; i < 56*56*24; i++){
		fprintf(fp, "%f \n", fm_buffer_2[i].to_float());
	}

	fclose(fp);*/

	//////////////////// pw4 + dw4 + pw4 ////////////////////

	ic = 24;
	ih = 56;
	iw = 56;
	oc = 32;
	oh = 28;
	ow = 28;

	printf("Bottleneck_4 Start\n");

	bottleneck_2(fm_buffer_2, fm_buffer_1, ic, ih, oc, oh, 176, 45, 76, 12,
				sparse_1x1_weight_raw, block_c_raw, pw4_first_1x1_block_col_r_raw, pw4_last_1x1_block_col_r_raw, weight_3x3_raw, bn_raw);

	printf("Bottleneck_4 Done\n");

	write_fm_to_ddr(fm_buffer_2, 28, 28, 32, 0, feature_map_raw);
	printf("4 To DDR Done\n");
	std::cout << "here4" << std::endl;

	/*fp = fopen("layer4_hls_d.txt","w");
	for(int i = 0; i < 28*28*32; i++){
		fprintf(fp, "%f \n", fm_buffer_2[i].to_float());
	}

	fclose(fp);*/



	//////////////////// pw5 + dw5 + pw5 ////////////////////
	ic = 32;
	ih = 28;
	iw = 28;
	oc = 32;
	oh = 28;
	ow = 28;

	printf("Bottleneck_5 Start\n");
	bottleneck_1(fm_buffer_2, fm_buffer_1, ic, ih, oc, oh, 302, 77, 116, 17,
				sparse_1x1_weight_raw, block_c_raw, pw5_first_1x1_block_col_r_raw, pw5_last_1x1_block_col_r_raw, weight_3x3_raw, bn_raw);
	printf("Bottleneck_5 Done\n");

	write_fm_to_ddr(fm_buffer_2, oh, ow, oc, 28*28*32, feature_map_raw);
	printf("5 To DDR Done\n");
	std::cout << "here5" << std::endl;

	/*fp = fopen("layer5_hls_d.txt","w");
	for(int i = 0; i < 28*28*32; i++){
		fprintf(fp, "%f \n", fm_buffer_2[i].to_float());
	}

	fclose(fp);*/




	//////////////////// pw6 + dw6 + pw6 ////////////////////
	ic = 32;
	ih = 28;
	iw = 28;
	oc = 32;
	oh = 28;
	ow = 28;

	printf("Bottleneck_6 Start\n");
	bottleneck_1(fm_buffer_2, fm_buffer_1, ic, ih, oc, oh, 494, 125, 168, 23,
				sparse_1x1_weight_raw, block_c_raw, pw6_first_1x1_block_col_r_raw, pw6_last_1x1_block_col_r_raw, weight_3x3_raw, bn_raw);
	printf("Bottleneck_6 Done\n");
	std::cout << "here6" << std::endl;

	/*fp = fopen("layer6_hls_d.txt","w");
	for(int i = 0; i < 28*28*32; i++){
		fprintf(fp, "%f \n", fm_buffer_2[i].to_float());
	}

	fclose(fp);*/



	//////////////////// 4 + 5 + 6 ////////////////////

	printf("4 5 From DDR Starting\n");
	load_fm_from_axi(feature_map_raw, 28, 28, 32*2, fm_buffer_1);

	printf("4 + 5 + 6 Start\n");
	for(int i = 0; i < oc; i++){
		for(int j = 0; j < oh; j++){
			for(int k = 0; k < ow; k++){
				fm_buffer_2[i*oh*ow + j*ow + k] = fm_buffer_2[i*oh*ow + j*ow + k] + fm_buffer_1[i*oh*ow + j*ow + k] + fm_buffer_1[(i + 32)*oh*ow + j*ow + k];
			}
		}
	}
	printf("4 + 5 + 6 Done\n");



	//////////////////// pw7 + dw7 + pw7 ////////////////////
	ic = 32;
	ih = 28;
	iw = 28;
	oc = 64;
	oh = 14;
	ow = 14;

	printf("Bottleneck_7 Start\n");
	bottleneck_2(fm_buffer_2, fm_buffer_1, ic, ih, oc, oh, 686, 173, 220, 29,
				sparse_1x1_weight_raw, block_c_raw, pw7_first_1x1_block_col_r_raw, pw7_last_1x1_block_col_r_raw, weight_3x3_raw, bn_raw);
	printf("Bottleneck_7 Done\n");

	write_fm_to_ddr(fm_buffer_2, oh, ow, oc, 0, feature_map_raw);
	printf("7 To DDR Done\n");
	std::cout << "here7" << std::endl;

	/*fp = fopen("layer7_hls_d.txt","w");
	for(int i = 0; i < 14*14*64; i++){
		fprintf(fp, "%f \n", fm_buffer_2[i].to_float());
	}

	fclose(fp);*/


	//////////////////// pw8 + dw8 + pw8 ////////////////////
	ic = 64;
	ih = 14;
	iw = 14;
	oc = 64;
	oh = 14;
	ow = 14;

	printf("Bottleneck_8 Start\n");
	bottleneck_1(fm_buffer_2, fm_buffer_1, ic, ih, oc, oh, 974, 245, 276, 35,
				sparse_1x1_weight_raw, block_c_raw, pw8_first_1x1_block_col_r_raw, pw8_last_1x1_block_col_r_raw, weight_3x3_raw, bn_raw);
	printf("Bottleneck_8 Done\n");

	write_fm_to_ddr(fm_buffer_2, oh, ow, oc, oc*oh*ow, feature_map_raw);
	printf("8 To DDR Done\n");
	std::cout << "here8" << std::endl;

	/*fp = fopen("layer8_hls_d.txt","w");
	for(int i = 0; i < 14*14*64; i++){
		fprintf(fp, "%f \n", fm_buffer_2[i].to_float());
	}

	fclose(fp);*/



	//////////////////// pw9 + dw9 + pw9 ////////////////////
	ic = 64;
	ih = 14;
	iw = 14;
	oc = 64;
	oh = 14;
	ow = 14;

	printf("Bottleneck_9 Start\n");
	bottleneck_1(fm_buffer_2, fm_buffer_1, ic, ih, oc, oh, 1742, 437, 380, 47,
				sparse_1x1_weight_raw, block_c_raw, pw9_first_1x1_block_col_r_raw, pw9_last_1x1_block_col_r_raw, weight_3x3_raw, bn_raw);
	printf("Bottleneck_9 Done\n");

	write_fm_to_ddr(fm_buffer_2, oh, ow, oc, oc*oh*ow*2, feature_map_raw);
	printf("9 To DDR Done\n");
	std::cout << "here9" << std::endl;

	/*fp = fopen("layer9_hls_d.txt","w");
	for(int i = 0; i < 14*14*64; i++){
		fprintf(fp, "%f \n", fm_buffer_2[i].to_float());
	}

	fclose(fp);*/



	//////////////////// pw10 + dw10 + pw10 ////////////////////
	ic = 64;
	ih = 14;
	iw = 14;
	oc = 64;
	oh = 14;
	ow = 14;

	printf("Bottleneck_10 Start\n");
	bottleneck_1(fm_buffer_2, fm_buffer_1, ic, ih, oc, oh, 2510, 629, 484, 59,
				sparse_1x1_weight_raw, block_c_raw, pw10_first_1x1_block_col_r_raw, pw10_last_1x1_block_col_r_raw, weight_3x3_raw, bn_raw);
	printf("Bottleneck_10 Done\n");
	std::cout << "here10" << std::endl;

	/*fp = fopen("layer10_hls_d.txt","w");
	for(int i = 0; i < 14*14*64; i++){
		fprintf(fp, "%f \n", fm_buffer_2[i].to_float());
	}

	fclose(fp);*/



	//////////////////// 7 + 8 + 9 + 10 ////////////////////

	printf("7 8 9 From DDR Starting\n");
	load_fm_from_axi(feature_map_raw, 14, 14, 64*3, fm_buffer_1);
	printf("7 + 8 + 9 + 10 Start\n");
	for(int i = 0; i < oc; i++){
		for(int j = 0; j < oh; j++){
			for(int k = 0; k < ow; k++){
				fm_buffer_2[i*oh*ow + j*ow + k] = fm_buffer_2[i*oh*ow + j*ow + k] + fm_buffer_1[i*oh*ow + j*ow + k] + fm_buffer_1[(i + 64)*oh*ow + j*ow + k] + fm_buffer_1[(i + 64*2)*oh*ow + j*ow + k];
			}
		}
	}
	printf("7 + 8 + 9 + 10 Done\n");



	//////////////////// pw11 + dw11 + pw11 ////////////////////
	ic = 64;
	ih = 14;
	iw = 14;
	oc = 96;
	oh = 14;
	ow = 14;

	printf("Bottleneck_11 Start\n");
	bottleneck_1(fm_buffer_2, fm_buffer_1, ic, ih, oc, oh, 3278, 821, 588, 71,
				sparse_1x1_weight_raw, block_c_raw, pw11_first_1x1_block_col_r_raw, pw11_last_1x1_block_col_r_raw, weight_3x3_raw, bn_raw);
	printf("Bottleneck_11 Done\n");

	write_fm_to_ddr(fm_buffer_2, oh, ow, oc, 0, feature_map_raw);
	printf("11 To DDR Done\n");
	std::cout << "here11" << std::endl;

	/*fp = fopen("layer11_hls_d.txt","w");
	for(int i = 0; i < 14*14*96; i++){
		fprintf(fp, "%f \n", fm_buffer_2[i].to_float());
	}

	fclose(fp);*/



	//////////////////// pw12 + dw12 + pw12 ////////////////////
	ic = 96;
	ih = 14;
	iw = 14;
	oc = 96;
	oh = 14;
	ow = 14;

	printf("Bottleneck_12 Start\n");
	bottleneck_1(fm_buffer_2, fm_buffer_1, ic, ih, oc, oh, 4238, 1061, 696, 83,
				sparse_1x1_weight_raw, block_c_raw, pw12_first_1x1_block_col_r_raw, pw12_last_1x1_block_col_r_raw, weight_3x3_raw, bn_raw);
	printf("Bottleneck_12 Done\n");

	write_fm_to_ddr(fm_buffer_2, oh, ow, oc, 14*14*96, feature_map_raw);
	printf("12 To DDR Done\n");
	std::cout << "here12" << std::endl;

	/*fp = fopen("layer12_hls_d.txt","w");
	for(int i = 0; i < 14*14*64; i++){
		fprintf(fp, "%f \n", fm_buffer_2[i].to_float());
	}

	fclose(fp);*/



	//////////////////// pw13 + dw13 + pw13 ////////////////////
	ic = 96;
	ih = 14;
	iw = 14;
	oc = 96;
	oh = 14;
	ow = 14;

	printf("Bottleneck_13 Start\n");
	bottleneck_1(fm_buffer_2, fm_buffer_1, ic, ih, oc, oh, 5966, 1493, 852, 101,
				sparse_1x1_weight_raw, block_c_raw, pw13_first_1x1_block_col_r_raw, pw13_last_1x1_block_col_r_raw, weight_3x3_raw, bn_raw);
	printf("Bottleneck_13 Done\n");
	std::cout << "here13" << std::endl;

	/*fp = fopen("layer13_hls_d.txt","w");
	for(int i = 0; i < 14*14*96; i++){
		fprintf(fp, "%f \n", fm_buffer_2[i].to_float());
	}

	fclose(fp);*/


	//////////////////// 11 + 12 + 13 ////////////////////

	printf("11 12 From DDR Start\n");
	load_fm_from_axi(feature_map_raw, 14, 14, 96*2, fm_buffer_1);
	printf("11 + 12 + 13 Start\n");
	for(int i = 0; i < oc; i++){
		for(int j = 0; j < oh; j++){
			for(int k = 0; k < ow; k++){
				fm_buffer_2[i*oh*ow + j*ow + k] = fm_buffer_2[i*oh*ow + j*ow + k] + fm_buffer_1[i*oh*ow + j*ow + k] + fm_buffer_1[(i + 96)*oh*ow + j*ow + k];
			}
		}
	}
	printf("11 + 12 + 13 Done\n");



	//////////////////// pw14 + dw14 + pw14 ////////////////////
	ic = 96;
	ih = 14;
	iw = 14;
	oc = 160;
	oh = 7;
	ow = 7;

	printf("Bottleneck_14 Start\n");
	bottleneck_2(fm_buffer_2, fm_buffer_1, ic, ih, oc, oh, 7694, 1925, 1008, 119,
				sparse_1x1_weight_raw, block_c_raw, pw14_first_1x1_block_col_r_raw, pw14_last_1x1_block_col_r_raw, weight_3x3_raw, bn_raw);
	printf("Bottleneck_14 Done\n");

	write_fm_to_ddr(fm_buffer_2, oh, ow, oc, 0, feature_map_raw);
	printf("14 To DDR Done\n");

	/*fp = fopen("layer14_hls_d.txt","w");
	for(int i = 0; i < 7*7*160; i++){
		fprintf(fp, "%f \n", fm_buffer_2[i].to_float());
	}

	fclose(fp);*/



	//////////////////// pw15 + dw15 + pw15 ////////////////////
	ic = 160;
	ih = 7;
	iw = 7;
	oc = 160;
	oh = 7;
	ow = 7;

	printf("Bottleneck_15 Start\n");
	bottleneck_1(fm_buffer_2, fm_buffer_1, ic, ih, oc, oh, 9998, 2501, 1172, 137,
				sparse_1x1_weight_raw, block_c_raw, pw15_first_1x1_block_col_r_raw, pw15_last_1x1_block_col_r_raw, weight_3x3_raw, bn_raw);
	printf("Bottleneck_15 Done\n");

	write_fm_to_ddr(fm_buffer_2, oh, ow, oc, 160*160*7, feature_map_raw);
	printf("15 To DDR Done\n");


	for(int i = 0; i < 320*1280/8; i++){
		block_c_buf[i] = 0;
	}

	/*fp = fopen("layer15_hls_d.txt","w");
	for(int i = 0; i < 7*7*160; i++){
		fprintf(fp, "%f \n", fm_buffer_2[i].to_float());
	}

	fclose(fp);*/

	//////////////////// pw16 + dw16 + pw16 ////////////////////
	ic = 160;
	ih = 7;
	iw = 7;
	oc = 160;
	oh = 7;
	ow = 7;

	printf("Bottleneck_16 Start\n");
	bottleneck_1(fm_buffer_2, fm_buffer_1, ic, ih, oc, oh, 14798, 0, 1432, 167,
				sparse_1x1_weight_raw, block_c_16_raw, pw16_first_1x1_block_col_r_raw, pw16_last_1x1_block_col_r_raw, weight_3x3_raw, bn_raw);
	printf("Bottleneck_16 Done\n");

	/*fp = fopen("layer16_hls_d.txt","w");
	for(int i = 0; i < 7*7*160; i++){
		fprintf(fp, "%f \n", fm_buffer_2[i].to_float());
	}

	fclose(fp);*/



	//////////////////// 14 + 15 + 16 ////////////////////

	printf("14 15 From DDR Starting\n");
	load_fm_from_axi(feature_map_raw, 7, 7, 160*2, fm_buffer_1);
	printf("14 + 15 + 16 Start\n");
	for(int i = 0; i < oc; i++){
		for(int j = 0; j < oh; j++){
			for(int k = 0; k < ow; k++){
				fm_buffer_2[i*oh*ow + j*ow + k] = fm_buffer_2[i*oh*ow + j*ow + k] + fm_buffer_1[i*oh*ow + j*ow + k] + fm_buffer_1[(i + 160)*oh*ow + j*ow + k];
			}
		}
	}
	printf("14 + 15 + 16 Done\n");



	//////////////////// pw17 + dw17 + pw17 ////////////////////
	ic = 160;
	ih = 7;
	iw = 7;
	oc = 320;
	oh = 7;
	ow = 7;

	printf("Bottleneck_17 Start\n");
	bottleneck_1(fm_buffer_2, fm_buffer_1, ic, ih, oc, oh, 19598, 4901, 1692, 0,
				sparse_1x1_weight_raw, block_c_raw, pw17_first_1x1_block_col_r_raw, pw17_last_1x1_block_col_r_raw, weight_3x3_17_raw, bn_raw);
	printf("Bottleneck_17 Done\n");

	/*fp = fopen("layer17_hls_d.txt","w");
	for(int i = 0; i < 7*7*320; i++){
		fprintf(fp, "%f \n", fm_buffer_2[i].to_float());
	}

	fclose(fp);*/



	//////////////////// conv2d (sparse) 1x1 ////////////////////
	ic = 320;
	ih = 7;
	iw = 7;
	oc = 1280;
	oh = 7;
	ow = 7;

	b_t = 320*1280/8;

	printf("Conv2d 1*1 Loading Start\n");
	load_weight_1x1_from_axi(sparse_1x1_weight_raw, b_t, 26798, weight_1x1_buffer);
	load_weight_1x1_block_c_from_axi(block_c_raw, b_t, 6701, block_c_buf);
	load_weight_1x1_block_col_r_from_axi(conv_1x1_block_col_r_raw, oc/4+1, block_col_r_buf); 
	load_bn_from_axi(bn_raw, oc, 1972, bn_buffer);
	printf("Conv2d 1*1 Loading Done\n");

	printf("Conv2d 1*1 Start\n");
	pw_1x1_sparse(fm_buffer_2, weight_1x1_buffer, block_c_buf, block_col_r_buf,
                    ic, ih, oc, oh, fm_buffer_1); 
	batchnorm2d(fm_buffer_1, bn_buffer, oc, oh, oh, fm_buffer_2); 
	relu6(fm_buffer_2, oc, oh, fm_buffer_1); 
	printf("Conv2d 1*1 Done\n");

	/*fp = fopen("layer18_hls_d.txt","w");
	for(int i = 0; i < 7*7*1280; i++){
		fprintf(fp, "%f \n", fm_buffer_2[i].to_float());
	}

	fclose(fp);*/



	//////////////////// avgpool ////////////////////
	ic = 1280;
	ih = 7;
	iw = 7;

	printf("Global Average Pooling Start\n");
	globalaveragepooling(fm_buffer_1, ic, ih, fm_buffer_2);
	printf("Global Average Pooling Done\n");


	//////////////////// conv2d 1x1 ////////////////////
	ic = 1280;
	oc = 1000;
	bias_t = 1000;

	int start;
	start = 0;

	printf("The Last Conv2d 1*1 Start\n");

	//load_last_weight_from_axi(conv_last_1x1_weight_raw_1, 6400, weight_1x1_buffer);
	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_11[i].range(j*16 + 15, j*16);
		}
	}*/
	load_conv_last_from_ddr(conv_last_1x1_weight_raw_11, 0, weight_1x1_buffer);

	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[51200 + i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_12[i].range(j*16 + 15, j*16);
		}
	}*/
    load_conv_last_from_ddr(conv_last_1x1_weight_raw_12, 51200, weight_1x1_buffer);

	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[102400 + i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_13[i].range(j*16 + 15, j*16);
		}
	}*/
    load_conv_last_from_ddr(conv_last_1x1_weight_raw_13, 102400, weight_1x1_buffer);

	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[153600 + i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_14[i].range(j*16 + 15, j*16);
		}
	}*/
    load_conv_last_from_ddr(conv_last_1x1_weight_raw_14, 153600, weight_1x1_buffer);

	//free(conv_last_1x1_weight_raw_1);
	load_bias_from_axi(conv_last_1x1_bias_raw, weight_3x3_buffer);

	pw_1x1(fm_buffer_2, weight_1x1_buffer, weight_3x3_buffer, ic, 160, 0, fm_buffer_1);

	//free(weight_1x1_buffer);

	write_output_to_ddr(fm_buffer_1, 0, 160, network_output_raw);

	start = 6400;

	//load_last_weight_from_axi(conv_last_1x1_weight_raw_2, 6400, weight_1x1_buffer);
	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_21[i].range(j*16 + 15, j*16);
		}
	}*/
    load_conv_last_from_ddr(conv_last_1x1_weight_raw_21, 0, weight_1x1_buffer);

	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[51200 + i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_22[i].range(j*16 + 15, j*16);
		}
	}*/
    load_conv_last_from_ddr(conv_last_1x1_weight_raw_22, 51200, weight_1x1_buffer);

	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[102400 + i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_23[i].range(j*16 + 15, j*16);
		}
	}*/
    load_conv_last_from_ddr(conv_last_1x1_weight_raw_23, 102400, weight_1x1_buffer);

	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[153600 + i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_24[i].range(j*16 + 15, j*16);
		}
	}*/
    load_conv_last_from_ddr(conv_last_1x1_weight_raw_24, 153600, weight_1x1_buffer);


	//free(conv_last_1x1_weight_raw_2);
	pw_1x1(fm_buffer_2, weight_1x1_buffer, weight_3x3_buffer, ic, 160, 160, fm_buffer_1);


	write_output_to_ddr(fm_buffer_1, 160, 160, network_output_raw);

	start = 12800;

	//load_last_weight_from_axi(conv_last_1x1_weight_raw_3, 6400, weight_1x1_buffer);
	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_31[i].range(j*16 + 15, j*16);
		}
	}*/
    load_conv_last_from_ddr(conv_last_1x1_weight_raw_31, 0, weight_1x1_buffer);

	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[51200 + i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_32[i].range(j*16 + 15, j*16);
		}
	}*/
    load_conv_last_from_ddr(conv_last_1x1_weight_raw_32, 51200, weight_1x1_buffer);

	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[102400 + i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_33[i].range(j*16 + 15, j*16);
		}
	}*/
    load_conv_last_from_ddr(conv_last_1x1_weight_raw_33, 102400, weight_1x1_buffer);

	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[153600 + i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_34[i].range(j*16 + 15, j*16);
		}
	}*/
    load_conv_last_from_ddr(conv_last_1x1_weight_raw_34, 153600, weight_1x1_buffer);

	//free(conv_last_1x1_weight_raw_3);
	pw_1x1(fm_buffer_2, weight_1x1_buffer, weight_3x3_buffer, ic, 160, 320, fm_buffer_1);

	//free(weight_1x1_buffer);
	write_output_to_ddr(fm_buffer_1, 320, 160, network_output_raw);

	start = 19200;

	//load_last_weight_from_axi(conv_last_1x1_weight_raw_4, 6400, weight_1x1_buffer);
	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_41[i].range(j*16 + 15, j*16);
		}
	}*/
    load_conv_last_from_ddr(conv_last_1x1_weight_raw_41, 0, weight_1x1_buffer);

	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[51200 + i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_42[i].range(j*16 + 15, j*16);
		}
	}*/
    load_conv_last_from_ddr(conv_last_1x1_weight_raw_42, 51200, weight_1x1_buffer);

	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[102400 + i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_43[i].range(j*16 + 15, j*16);
		}
	}*/
    load_conv_last_from_ddr(conv_last_1x1_weight_raw_43, 102400, weight_1x1_buffer);

	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[153600 + i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_44[i].range(j*16 + 15, j*16);
		}
	}*/
    load_conv_last_from_ddr(conv_last_1x1_weight_raw_44, 153600, weight_1x1_buffer);

	//free(conv_last_1x1_weight_raw_4);
	pw_1x1(fm_buffer_2, weight_1x1_buffer, weight_3x3_buffer, ic, 160, 480, fm_buffer_1);


	//free(weight_1x1_buffer);
	write_output_to_ddr(fm_buffer_1, 480, 160, network_output_raw);

	start = 25600;

	//load_last_weight_from_axi(conv_last_1x1_weight_raw_5, 6400, weight_1x1_buffer);
	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_51[i].range(j*16 + 15, j*16);
		}
	}*/
    load_conv_last_from_ddr(conv_last_1x1_weight_raw_51, 0, weight_1x1_buffer);

	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[51200 + i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_52[i].range(j*16 + 15, j*16);
		}
	}*/
    load_conv_last_from_ddr(conv_last_1x1_weight_raw_52, 51200, weight_1x1_buffer);

	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[102400 + i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_53[i].range(j*16 + 15, j*16);
		}
	}*/
    load_conv_last_from_ddr(conv_last_1x1_weight_raw_53, 102400, weight_1x1_buffer);

	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[153600 + i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_54[i].range(j*16 + 15, j*16);
		}
	}*/
	load_conv_last_from_ddr(conv_last_1x1_weight_raw_54, 153600, weight_1x1_buffer);


	//free(conv_last_1x1_weight_raw_5);
	pw_1x1(fm_buffer_2, weight_1x1_buffer, weight_3x3_buffer, ic, 160, 640, fm_buffer_1);

	//free(weight_1x1_buffer);
	write_output_to_ddr(fm_buffer_1, 640, 160, network_output_raw);

	start = 32000;

	//load_last_weight_from_axi(conv_last_1x1_weight_raw_6, 6400, weight_1x1_buffer);
	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_61[i].range(j*16 + 15, j*16);
		}
	}*/
	load_conv_last_from_ddr(conv_last_1x1_weight_raw_61, 0, weight_1x1_buffer);

	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[51200 + i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_62[i].range(j*16 + 15, j*16);
		}
	}*/
	load_conv_last_from_ddr(conv_last_1x1_weight_raw_62, 51200, weight_1x1_buffer);

	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[102400 + i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_63[i].range(j*16 + 15, j*16);
		}
	}*/
    load_conv_last_from_ddr(conv_last_1x1_weight_raw_63, 102400, weight_1x1_buffer);

	/*for(int i = 0; i < 1600; i++){
		for(int j = 0; j < 32; j++){
			weight_1x1_buffer[153600 + i*32 + j].range(15, 0) = conv_last_1x1_weight_raw_64[i].range(j*16 + 15, j*16);
		}
	}*/
	load_conv_last_from_ddr(conv_last_1x1_weight_raw_64, 153600, weight_1x1_buffer);


    //free(conv_last_1x1_weight_raw_6);
	pw_1x1(fm_buffer_2, weight_1x1_buffer, weight_3x3_buffer, ic, 160, 800, fm_buffer_1);



	write_output_to_ddr(fm_buffer_1, 800, 160, network_output_raw);

	start = 38400;

	load_last_weight_from_axi(conv_last_1x1_weight_raw_7, 1600, weight_1x1_buffer);
	//free(conv_last_1x1_weight_raw_7);
	pw_1x1(fm_buffer_2, weight_1x1_buffer, weight_3x3_buffer, ic, 40, 960, fm_buffer_1);


	for(int i = 0; i < 1; i++){
		for(int j = 0; j < 32; j++){
			network_output_raw[960/32 + i].range(j*16 + 15, j*16) = fm_buffer_1[i*32 + j].range(15, 0);
			// std::cout << i*16 + j << " " << fm_buffer_1[i*16 + j] << std::endl;
		}
	}

	for(int i = 0; i < 8; i++){
		network_output_raw[31].range(i*16 + 15, i*16) = fm_buffer_1[32 + i].range(15, 0);
	}

	printf("The Last Conv2d 1*1 Done\n");
}

}
