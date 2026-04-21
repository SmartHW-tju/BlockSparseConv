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

/*
define k 1
ih, iw, ic
oc, kc
pad
oh = ((ih + 2*pad - 1)/stride + 1)
ow = ((iw + 2*pad - 1)/stride + 1)
ih_pw_pad = (ih + pad*2)
iw_pw_pad = (iw + pad*2)
*/

// Dense pointwise convolution used by the final classifier after global average pooling.
void pw_1x1(FIX_F *input, FIX_W *weight, FIX_W *bias, int ic, int number, int start, FIX_F *output) // mumber = 160 or 40;
{
    #ifdef NO_SYNTH
        // FIX_F output_1[number][ic][1][1];
	    FIX_F *output_1 = (FIX_F*)malloc(160*1280*1*1*sizeof(FIX_F));
    #else
        FIX_F output_1[160*1280*1*1];
    #endif

    // conv(1*1);
    for(int m = 0; m < number; m++){
        // for(int i = 0; i < 1; i++){
            // for(int j = 0; j < 1; j++){
                output[m*1*1 + 0 + 0] = 0;
                for(int n = 0; n < ic; n++){
                    output_1[m*1280*1*1 + n*1*1 + 0 + 0] = input[n*1*1 + 0 + 0] * weight[m*1280 + n];
                    output[m*1*1 + 0 + 0] += output_1[m*1280*1*1 + n*1*1 + 0 + 0];
                    /*if((m == 153) && (start == 800)){
                        std::cout << "n: " << n << " input: " << input[n*1*1 + 0 + 0] << " weight: " << weight[m*1280 + n] << " output_1: " << output_1[m*1280*1*1 + n*1*1 + 0 + 0] << " output: " << output[m*1*1 + 0 + 0] << std::endl;
                    }*/
                }
            // } 
        // }
        output[m*1*1 + 0 + 0] += bias[(m + start)*1*1 + 0 + 0];
        /*if(start == 800){
            std::cout << output[m*1*1 + 0 + 0] << " " << bias[(m + start)*1*1 + 0 + 0] << std::endl;
        }*/
    }

    #ifdef NO_SYNTH
        free(output_1);
    #endif
}
