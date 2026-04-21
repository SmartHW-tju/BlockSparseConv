#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <hls_math.h>
#include <iostream>
#include <ap_fixed.h>
#include <ap_int.h>
//#include "hls_stream.h"

#include "network.hpp"

#if !defined(__SYNTHESIS__) && !defined(NO_SYNTH)
#define NO_SYNTH
#endif


/*
define k 3
ih, iw, ic
oc, kc
pad
oh = ((ih + 2*pad - 3)/stride + 1) // floor
ow = ((iw + 2*pad - 3)/stride + 1) // floor
ih_pad = (ih + pad*2)
iw_pad = (iw + pad*2)
*/

// Dense depthwise 3x3 convolution, stride 1.
// Boundary cases are expanded explicitly instead of building a padded tensor,
// which avoids a large temporary buffer in the synthesized design.
void dw_3x3_s1(FIX_F *input, FIX_W *weight,
                int ic, int ih, int oc, int oh, FIX_F *output)
{
#pragma HLS ALLOCATION instances=Multiplier limit=189 core
//#pragma HLS ALLOCATION instances=add limit=36 operation
    //set_directive_resource -core RAM_1P dw_3x3_s1 output
#pragma HLS ALLOCATION instances=icmp limit=15 operation
#pragma HLS bind_storage variable=output type=RAM_1P impl=bram
#pragma HLS BIND_OP variable=output op=add impl=dsp


    int iw = ih;
    int ow = oh;
    int ih_pad = ih + 2;
    int iw_pad = iw + 2;
    
    // FIX_F input_pad[ic][ih_pad][iw_pad];
    /*#ifdef NO_SYNTH
        FIX_F *input_pad = (FIX_F*)malloc(144*58*58*sizeof(FIX_F));
    #else
        FIX_F input_pad[144*58*58];
    #endif
    
    // padding
    for(int m = 0; m < ic; m++){
        for(int i = 0; i < ih_pad; i++){
            for(int j = 0; j < iw_pad; j++){
                input_pad[m*ih_pad*iw_pad + i*iw_pad + j] = 0;
            }
        }
    }
    for(int m = 0; m < ic; m++){
        for(int i = 1; i < (ih_pad - 1); i++){
            for(int j = 1; j < (iw_pad - 1); j++){
            	input_pad[m*ih_pad*iw_pad + i*iw_pad + j] = input[m*(ih_pad - 1)*(iw_pad - 1) + (i - 1)*(iw_pad - 1) + j - 1];
            }
        }
    }
    
    // conv(3*3);
    for(int n = 0; n < oc; n++){
        for(int i = 0; i < oh; i++){
            for(int j = 0; j < ow; j++){
                output[n*oh*ow + i*ow + j] = input_pad[n*ih_pad*iw_pad + i*iw_pad + j] * weight[n*3*3 + 0*3 + 0]
                                    + input_pad[n*ih_pad*iw_pad + i*iw_pad + (j + 1)] * weight[n*3*3 +0*3 + 1]
                                    + input_pad[n*ih_pad*iw_pad + i*iw_pad + (j + 2)] * weight[n*3*3 +0*3 + 2]
                                    + input_pad[n*ih_pad*iw_pad + (i + 1)*iw_pad + j] * weight[n*3*3 +1*3 + 0]
                                    + input_pad[n*ih_pad*iw_pad + (i + 1)*iw_pad + (j + 1)] * weight[n*3*3 + 1*3 + 1]
                                    + input_pad[n*ih_pad*iw_pad + (i + 1)*iw_pad + (j + 2)] * weight[n*3*3 + 1*3 + 2]
                                    + input_pad[n*ih_pad*iw_pad + (i + 2)*iw_pad + j] * weight[n*3*3 +2*3 + 0]
                                    + input_pad[n*ih_pad*iw_pad + (i + 2)*iw_pad + (j + 1)] * weight[n*3*3 + 2*3 + 1]
                                    + input_pad[n*ih_pad*iw_pad + (i + 2)*iw_pad + (j + 2)] * weight[n*3*3 + 2*3 + 2];
            }
        }
        // output[n][i][j] = output[n][i][j] + bias[n];
    }*/

    dw_s1_for_1:
    for(int n = 0; n < oc; n++){
//#pragma HLS ALLOCATION instances=mul limit=8 operation
//#pragma HLS ALLOCATION instances=add limit=8 operation
        for(int i = 1; i < (oh-1); i++){
            for(int j = 1; j < (ow-1); j++){
                output[n*oh*ow + i*ow + j] = input[n*ih*iw + (i-1)*iw + (j-1)] * weight[n*9] //input[n*ih*iw + (i-1)*iw + (j-1)] * weight[n*3*3 + 0*3 + 0]
                                    + input[n*ih*iw + (i-1)*iw + (j)] * weight[n*9+1]  //input[n*ih*iw + (i-1)*iw + (j-1 + 1)] * weight[n*3*3 +0*3 + 1]
                                    + input[n*ih*iw + (i-1)*iw + (j+1)] * weight[n*9+2]  //input[n*ih*iw + (i-1)*iw + (j-1 + 2)] * weight[n*3*3 +0*3 + 2]
                                    + input[n*ih*iw + (i)*iw + j-1] * weight[n*9+3]  //input[n*ih*iw + (i-1 + 1)*iw + j-1] * weight[n*3*3 +1*3 + 0]
                                    + input[n*ih*iw + (i)*iw + (j)] * weight[n*9+4]  //input[n*ih*iw + (i-1 + 1)*iw + (j-1 + 1)] * weight[n*3*3 + 1*3 + 1]
                                    + input[n*ih*iw + (i)*iw + (j+1)] * weight[n*9+5]  //input[n*ih*iw + (i-1 + 1)*iw + (j-1 + 2)] * weight[n*3*3 + 1*3 + 2]
                                    + input[n*ih*iw + (i+1)*iw + j-1] * weight[n*9+6]  //input[n*ih*iw + (i-1 + 2)*iw + j-1] * weight[n*3*3 +2*3 + 0]
                                    + input[n*ih*iw + (i+1)*iw + (j)] * weight[n*9+7]  //input[n*ih*iw + (i-1 + 2)*iw + (j-1 + 1)] * weight[n*3*3 + 2*3 + 1]
                                    + input[n*ih*iw + (i+1)*iw + (j+1)] * weight[n*9+8];  //input[n*ih*iw + (i-1 + 2)*iw + (j-1 + 2)] * weight[n*3*3 + 2*3 + 2]
            }
        }
        // output[n][i][j] = output[n][i][j] + bias[n];
    }

    dw_s1_for_2:
    for(int n = 0; n < oc; n++){
//#pragma HLS ALLOCATION instances=add limit=5 operation
        for(int j = 1; j < (ow-1); j++){
            output[n*oh*ow + 0*ow + j] = input[n*ih*iw + j-1] * weight[n*3*3 +3]
                                    + input[n*ih*iw + j] * weight[n*3*3 +4]
                                    + input[n*ih*iw + j+1] * weight[n*3*3 +5]
                                    + input[n*ih*iw + iw + j-1] * weight[n*3*3 +6]
                                    + input[n*ih*iw + iw + j] * weight[n*3*3 +7]
                                    + input[n*ih*iw + iw + j+1] * weight[n*3*3 +8];  //input[n*ih*iw + (0-1 + 1)*iw + j-1] * weight[n*3*3 +1*3 + 0]+ input[n*ih*iw + (0-1 + 1)*iw + (j-1 + 1)] * weight[n*3*3 + 1*3 + 1]+ input[n*ih*iw + (0-1 + 1)*iw + (j-1 + 2)] * weight[n*3*3 + 1*3 + 2]+ input[n*ih*iw + (0-1 + 2)*iw + j-1] * weight[n*3*3 +2*3 + 0]+ input[n*ih*iw + (0-1 + 2)*iw + (j-1 + 1)] * weight[n*3*3 + 2*3 + 1]+ input[n*ih*iw + (0-1 + 2)*iw + (j-1 + 2)] * weight[n*3*3 + 2*3 + 2];
        }
    }

    dw_s1_for_3:
    for(int n = 0; n < oc; n++){
//#pragma HLS ALLOCATION instances=add limit=5 operation
        for(int j = 1; j < (ow-1); j++){
            output[n*oh*ow + (oh-1)*ow + j] = input[n*ih*iw + (oh-2)*iw + (j-1)] * weight[n*9 +0]
                                    + input[n*ih*iw + (oh-2)*iw + (j)] * weight[n*9 +1]
                                    + input[n*ih*iw + (oh-2)*iw + (j+1)] * weight[n*9 +2]
                                    + input[n*ih*iw + (oh-1)*iw + j-1] * weight[n*9 +3]
                                    + input[n*ih*iw + (oh-1)*iw + (j)] * weight[n*9 +4]
                                    + input[n*ih*iw + (oh-1)*iw + (j+1)] * weight[n*9 +5];
        }
    }

    /*output[n*oh*ow + (oh-1)*ow + j] = input[n*ih*iw + (oh-1-1)*iw + (j-1)] * weight[n*3*3 + 0*3 + 0]
                                    + input[n*ih*iw + (oh-1-1)*iw + (j-1 + 1)] * weight[n*3*3 +0*3 + 1]
                                    + input[n*ih*iw + (oh-1-1)*iw + (j-1 + 2)] * weight[n*3*3 +0*3 + 2]
                                    + input[n*ih*iw + (oh-1-1 + 1)*iw + j-1] * weight[n*3*3 +1*3 + 0]
                                    + input[n*ih*iw + (oh-1-1 + 1)*iw + (j-1 + 1)] * weight[n*3*3 + 1*3 + 1]
                                    + input[n*ih*iw + (oh-1-1 + 1)*iw + (j-1 + 2)] * weight[n*3*3 + 1*3 + 2];*/

    dw_s1_for_4:
    for(int n = 0; n < oc; n++){
//#pragma HLS ALLOCATION instances=add limit=5 operation
        for(int i = 1; i < (oh-1); i++){
            output[n*oh*ow + i*ow + 0] = input[n*ih*iw + (i-1)*iw] * weight[n*9 +1]
                                    + input[n*ih*iw + (i-1)*iw + 1] * weight[n*9 +2]
                                    + input[n*ih*iw + i*iw] * weight[n*9 +4]
                                    + input[n*ih*iw + i*iw + 1] * weight[n*9 +5]
                                    + input[n*ih*iw + (i+1)*iw] * weight[n*9 +7]
                                    + input[n*ih*iw + (i+1)*iw + 1] * weight[n*9 +8];
        }
    }

    /*output[n*oh*ow + i*ow + 0] = input[n*ih*iw + (i-1)*iw + (0-1 + 1)] * weight[n*3*3 +0*3 + 1]
                                    + input[n*ih*iw + (i-1)*iw + (0-1 + 2)] * weight[n*3*3 +0*3 + 2]
                                    + input[n*ih*iw + (i-1 + 1)*iw + (0-1 + 1)] * weight[n*3*3 + 1*3 + 1]
                                    + input[n*ih*iw + (i-1 + 1)*iw + (0-1 + 2)] * weight[n*3*3 + 1*3 + 2]
                                    + input[n*ih*iw + (i-1 + 2)*iw + (0-1 + 1)] * weight[n*3*3 + 2*3 + 1]
                                    + input[n*ih*iw + (i-1 + 2)*iw + (0-1 + 2)] * weight[n*3*3 + 2*3 + 2];*/

    dw_s1_for_5:
    for(int n = 0; n < oc; n++){
//#pragma HLS ALLOCATION instances=add limit=5 operation
        for(int i = 1; i < (oh-1); i++){
            output[n*oh*ow + i*ow + ow-1] = input[n*ih*iw + (i-1)*iw + ow-2] * weight[n*9]
                                    + input[n*ih*iw + (i-1)*iw + (ow-1)] * weight[n*9 +1]
                                    + input[n*ih*iw + (i)*iw + ow-2] * weight[n*9 +3]
                                    + input[n*ih*iw + (i)*iw + (ow-1)] * weight[n*9 +4]
                                    + input[n*ih*iw + (i+1)*iw + ow-2] * weight[n*9 +6]
                                    + input[n*ih*iw + (i+1)*iw + (ow-1)] * weight[n*9 +7];
        }
    }

    /*            output[n*oh*ow + i*ow + ow-1] = input[n*ih*iw + (i-1)*iw + ow-1-1] * weight[n*3*3 + 0*3 + 0]
                                    + input[n*ih*iw + (i-1)*iw + (ow-1-1 + 1)] * weight[n*3*3 +0*3 + 1]
                                    + input[n*ih*iw + (i-1 + 1)*iw + ow-1-1] * weight[n*3*3 +1*3 + 0]
                                    + input[n*ih*iw + (i-1 + 1)*iw + (ow-1-1 + 1)] * weight[n*3*3 + 1*3 + 1]
                                    + input[n*ih*iw + (i-1 + 2)*iw + ow-1-1] * weight[n*3*3 +2*3 + 0]
                                    + input[n*ih*iw + (i-1 + 2)*iw + (ow-1-1 + 1)] * weight[n*3*3 + 2*3 + 1];*/

    dw_s1_for_6:
    for(int n = 0; n < oc; n++){
//#pragma HLS ALLOCATION instances=add limit=3 operation
        output[n*oh*ow + 0*ow + 0] = input[n*ih*iw] * weight[n*9 + 4]
                                    + input[n*ih*iw +1] * weight[n*9 + 5]
                                    + input[n*ih*iw + iw ] * weight[n*9 + 7]
                                    + input[n*ih*iw + iw + 1] * weight[n*9 + 8];
    }

    /*output[n*oh*ow + 0*ow + 0] = input[n*ih*iw + 0*iw + 0] * weight[n*3*3 + 1*3 + 1]
                                    + input[n*ih*iw + 0*iw + 1] * weight[n*3*3 + 1*3 + 2]
                                    + input[n*ih*iw + 1*iw + 0] * weight[n*3*3 + 2*3 + 1]
                                    + input[n*ih*iw + 1*iw + 1] * weight[n*3*3 + 2*3 + 2];*/
    
    dw_s1_for_7:
    for(int n = 0; n < oc; n++){
//#pragma HLS ALLOCATION instances=add limit=3 operation
        output[n*oh*ow + 0*ow + ow-1] = input[n*ih*iw + iw-2] * weight[n*9 +3]
                                    + input[n*ih*iw + iw-1] * weight[n*9 + 4]
                                    + input[n*ih*iw + 2*iw -2] * weight[n*9 +6]
                                    + input[n*ih*iw + 2*iw -1] * weight[n*9 +7];
    }

    /*output[n*oh*ow + 0*ow + ow-1] = input[n*ih*iw + 0*iw + iw-2] * weight[n*3*3 +1*3 + 0]
                                    + input[n*ih*iw + 0*iw + iw-1] * weight[n*3*3 + 1*3 + 1]
                                    + input[n*ih*iw + 1*iw + iw-2] * weight[n*3*3 +2*3 + 0]
                                    + input[n*ih*iw + 1*iw + iw-1] * weight[n*3*3 + 2*3 + 1];*/

    dw_s1_for_8:
    for(int n = 0; n < oc; n++){
//#pragma HLS ALLOCATION instances=add limit=3 operation
        output[n*oh*ow + (oh-1)*ow + 0] = input[n*ih*iw + (ih-2)*iw + 0] * weight[n*3*3 +0*3 + 1]
                                    + input[n*ih*iw + (ih-2)*iw + 1] * weight[n*3*3 +0*3 + 2]
                                    + input[n*ih*iw + (ih-1)*iw + 0] * weight[n*3*3 + 1*3 + 1]
                                    + input[n*ih*iw + (ih-1)*iw + 1] * weight[n*3*3 + 1*3 + 2];

    }

    /*        output[n*oh*ow + (oh-1)*ow + 0] = input[n*ih*iw + (ih-2)*iw + 0] * weight[n*3*3 +0*3 + 1]
                                    + input[n*ih*iw + (ih-2)*iw + 1] * weight[n*3*3 +0*3 + 2]
                                    + input[n*ih*iw + (ih-1)*iw + 0] * weight[n*3*3 + 1*3 + 1]
                                    + input[n*ih*iw + (ih-1)*iw + 1] * weight[n*3*3 + 1*3 + 2];*/

    dw_s1_for_9:
    for(int n = 0; n < oc; n++){
//#pragma HLS ALLOCATION instances=add limit=3 operation
        output[n*oh*ow + (oh-1)*ow + ow-1] = input[n*ih*iw + (ih-2)*iw + iw-2] * weight[n*9]
                                    + input[n*ih*iw + (ih-2)*iw + iw-1] * weight[n*9 +1]
                                    + input[n*ih*iw + (ih-1)*iw + iw-2] * weight[n*9 +3]
                                    + input[n*ih*iw + (ih-1)*iw + iw-1] * weight[n*9 +4];
                                    
    }

    /*        output[n*oh*ow + (oh-1)*ow + ow-1] = input[n*ih*iw + (ih-2)*iw + iw-2] * weight[n*3*3 + 0*3 + 0]
                                    + input[n*ih*iw + (ih-2)*iw + iw-1] * weight[n*3*3 +0*3 + 1]
                                    + input[n*ih*iw + (ih-1)*iw + iw-2] * weight[n*3*3 +1*3 + 0]
                                    + input[n*ih*iw + (ih-1)*iw + iw-1] * weight[n*3*3 + 1*3 + 1];*/

    /*#ifdef NO_SYNTH
        free(input_pad);
    #endif*/
}


// Dense depthwise 3x3 convolution, stride 2.
// This path uses an explicit padded buffer because the stride-2 access pattern
// is harder to express with the boundary-specialized form used above.
void dw_3x3_s2(FIX_F *input, FIX_W *weight,
                int ic, int ih, int oc, int oh, FIX_F *output)
{
//#pragma HLS ALLOCATION instances=icmp limit=13 operation
#pragma HLS BIND_OP variable=output op=add impl=dsp

    int iw = ih;
    int ow = oh;
    int ih_pad = ih + 1;
    int iw_pad = iw + 1;
    
    // FIX_F input_pad[ic][ih_pad][iw_pad];
    #ifdef NO_SYNTH
        FIX_F *input_pad = (FIX_F*)malloc(96*113*113*sizeof(FIX_F));
    #else
        FIX_F input_pad[96*113*113];
    #endif
    
    // padding
    /*for(int m = 0; m < ic; m++){
        for(int i = 0; i < ih_pad; i++){
            for(int j = 0; j < iw_pad; j++){
                input_pad[m*ih_pad*iw_pad + i*iw_pad + j] = 0;
            }
        }
    }
    for(int m = 0; m < ic; m++){
        for(int i = 0; i < (ih_pad - 1); i++){
            for(int j = 0; j < (iw_pad - 1); j++){
            	input_pad[m*ih_pad*iw_pad + i*iw_pad + j] = input[m*(ih_pad - 1)*(iw_pad - 1) + i*(iw_pad - 1) + j];
            }
        }
    }
    
    // conv(3*3);
    for(int n = 0; n < oc; n++){
        for(int i = 0; i < oh; i++){
            for(int j = 0; j < ow; j++){
                output[n*oh*ow + i*ow + j] = input_pad[n*ih_pad*iw_pad + i*2*iw_pad + j*2] * weight[n*3*3 + 0*3 + 0]
                                    + input_pad[n*ih_pad*iw_pad + i*2*iw_pad + (j*2 + 1)] * weight[n*3*3 +0*3 + 1]
                                    + input_pad[n*ih_pad*iw_pad + i*2*iw_pad + (j*2 + 2)] * weight[n*3*3 +0*3 + 2]
                                    + input_pad[n*ih_pad*iw_pad + (i*2 + 1)*iw_pad + j*2] * weight[n*3*3 +1*3 + 0]
                                    + input_pad[n*ih_pad*iw_pad + (i*2 + 1)*iw_pad + (j*2 + 1)] * weight[n*3*3 + 1*3 + 1]
                                    + input_pad[n*ih_pad*iw_pad + (i*2 + 1)*iw_pad + (j*2 + 2)] * weight[n*3*3 + 1*3 + 2]
                                    + input_pad[n*ih_pad*iw_pad + (i*2 + 2)*iw_pad + j*2] * weight[n*3*3 +2*3 + 0]
                                    + input_pad[n*ih_pad*iw_pad + (i*2 + 2)*iw_pad + (j*2 + 1)] * weight[n*3*3 + 2*3 + 1]
                                    + input_pad[n*ih_pad*iw_pad + (i*2 + 2)*iw_pad + (j*2 + 2)] * weight[n*3*3 + 2*3 + 2];
            }
        }
        // output[n][i][j] = output[n][i][j] + bias[n];
    }*/

    for(int n = 0; n < oc; n++){
//#pragma HLS ALLOCATION instances=add limit=8 operation
        for(int i = 0; i < oh-1; i++){
            for(int j = 0; j < ow-1; j++){
                output[n*oh*ow + i*ow + j] = input[n*ih*iw + i*2*iw + j*2] * weight[n*3*3 + 0*3 + 0]
                                    + input[n*ih*iw + i*2*iw + (j*2 + 1)] * weight[n*3*3 +0*3 + 1]               
                                    + input[n*ih*iw + i*2*iw + (j*2 + 2)] * weight[n*3*3 +0*3 + 2]            
                                    + input[n*ih*iw + (i*2 + 1)*iw + j*2] * weight[n*3*3 +1*3 + 0]            
                                    + input[n*ih*iw + (i*2 + 1)*iw + (j*2 + 1)] * weight[n*3*3 + 1*3 + 1]          
                                    + input[n*ih*iw + (i*2 + 1)*iw + (j*2 + 2)] * weight[n*3*3 + 1*3 + 2]    
                                    + input[n*ih*iw + (i*2 + 2)*iw + j*2] * weight[n*3*3 +2*3 + 0]    
                                    + input[n*ih*iw + (i*2 + 2)*iw + (j*2 + 1)] * weight[n*3*3 + 2*3 + 1]            
                                    + input[n*ih*iw + (i*2 + 2)*iw + (j*2 + 2)] * weight[n*3*3 + 2*3 + 2];
            }
        }
        // output[n][i][j] = output[n][i][j] + bias[n];
    }

    for(int n = 0; n < oc; n++){
//#pragma HLS ALLOCATION instances=add limit=5 operation
        for(int i = 0; i < oh-1; i++){
            output[n*oh*ow + i*ow + ow-1] = input[n*ih*iw + i*2*iw + (ow-1)*2] * weight[n*3*3 + 0*3 + 0]
                                    + input[n*ih*iw + i*2*iw + ((ow-1)*2 + 1)] * weight[n*3*3 +0*3 + 1]   
                                    + input[n*ih*iw + (i*2 + 1)*iw + (ow-1)*2] * weight[n*3*3 +1*3 + 0]    
                                    + input[n*ih*iw + (i*2 + 1)*iw + ((ow-1)*2 + 1)] * weight[n*3*3 + 1*3 + 1]    
                                    + input[n*ih*iw + (i*2 + 2)*iw + (ow-1)*2] * weight[n*3*3 +2*3 + 0]
                                    + input[n*ih*iw + (i*2 + 2)*iw + ((ow-1)*2 + 1)] * weight[n*3*3 + 2*3 + 1];
        }
    }

    for(int n = 0; n < oc; n++){
//#pragma HLS ALLOCATION instances=add limit=5 operation
        for(int j = 0; j < ow-1; j++){
            output[n*oh*ow + (oh-1)*ow + j] = input[n*ih*iw + (oh-1)*2*iw + j*2] * weight[n*3*3 + 0*3 + 0]
                                    + input[n*ih*iw + (oh-1)*2*iw + (j*2 + 1)] * weight[n*3*3 +0*3 + 1]           
                                    + input[n*ih*iw + (oh-1)*2*iw + (j*2 + 2)] * weight[n*3*3 +0*3 + 2]
                                    + input[n*ih*iw + ((oh-1)*2 + 1)*iw + j*2] * weight[n*3*3 +1*3 + 0]
                                    + input[n*ih*iw + ((oh-1)*2 + 1)*iw + (j*2 + 1)] * weight[n*3*3 + 1*3 + 1]
                                    + input[n*ih*iw + ((oh-1)*2 + 1)*iw + (j*2 + 2)] * weight[n*3*3 + 1*3 + 2];
        }
    }

    for(int n = 0; n < oc; n++){
//#pragma HLS ALLOCATION instances=add limit=3 operation
        output[n*oh*ow + (oh-1)*ow + ow-1] = input[n*ih*iw + (ih-2)*iw + iw-2] * weight[n*3*3 + 0*3 + 0]
                                    + input[n*ih*iw + (ih-2)*iw + iw-1] * weight[n*3*3 +0*3 + 1]    
                                    + input[n*ih*iw + (ih-1)*iw + iw-2] * weight[n*3*3 +1*3 + 0]
                                    + input[n*ih*iw + (ih-1)*iw + iw-1] * weight[n*3*3 + 1*3 + 1];
    }

    #ifdef NO_SYNTH
        free(input_pad);
    #endif
}
