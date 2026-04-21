#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <hls_math.h>
#include <iostream>
#include <algorithm>
#include <ap_fixed.h>
#include <ap_int.h>
//#include <hls_dsp.h>
//#include "hls_stream.h"

#include "network.hpp"


// Basic post-processing layers shared by all backbone stages.
// Batch-norm parameters are pre-packed by channel as:
// [scale | bias | mean | folded_denominator].

// parameter[0][]: weight
// parameter[1][]: bias
// parameter[2][]: running_mean
// parameter[3][]: running_var

void batchnorm2d(FIX_F *input, FIX_W *parameter, int ic, int ih, int iw, FIX_F *output)
{
    //#pragma HLS ALLOCATION instances=sdiv limit=500 operation
    
    //int iw = ih;

    for(int k = 0; k < ic; k++){
        //FIX_bn temp = FIX_bn(float(parameter[3*ic + k]) + 1/256);
        //std::cout << k << " " << temp.to_float() <<std::endl;
        for(int i = 0; i < ih; i++){
            for(int j = 0; j < iw; j++){ 
                // output[k][i][j] = (input[k][i][j] - parameter[2][k]) * parameter[0][k]/FIX_F(sqrt(float(parameter[3][k] + FIX_F(0.00001)))) + parameter[1][k];
                //output[k*ih*iw + i*iw + j] = FIX_F((input[k*ih*iw + i*iw + j].to_float() - parameter[2*ic + k].to_float()) * parameter[0*ic + k].to_float()/sqrt(float(parameter[3*ic + k]) + 0.00001)) + parameter[1*ic + k];
                //output[k*ih*iw + i*iw + j] = FIX_F(float((input[k*ih*iw + i*iw + j] - parameter[2*ic + k]) * parameter[0*ic + k])/float(hls::sqrt(temp)) + parameter[1*ic + k].to_float());
                //output[k*ih*iw + i*iw + j] = FIX_F(float((input[k*ih*iw + i*iw + j] - parameter[2*ic + k]) * parameter[0*ic + k])/float(parameter[3*ic + k]) + parameter[1*ic + k].to_float());
                // The denominator term is precomputed offline, so the kernel does
                // not need to evaluate a square root during inference.
                output[k*ih*iw + i*iw + j] = (input[k*ih*iw + i*iw + j] - parameter[2*ic + k]) * parameter[0*ic + k]/parameter[3*ic + k] + parameter[1*ic + k];
                //if((ic == 32) && (ih == 112))
                    //{std::cout << "input:" << input[k*ih*iw + i*iw + j] << " 2:" << parameter[2*ic + k] << " 0:" << parameter[0*ic + k] << " 3:" << parameter[3*ic + k] << " 1:" << parameter[1*ic + k] << " output:" << output[k*ih*iw + i*iw + j] << std::endl;}
            }
        }
    }
}

void relu6(FIX_F *input, int ic, int ih, FIX_F *output)
{
    int iw = ih;
    
    for(int k = 0; k < ic; k++){
        for(int i = 0; i < ih; i++){
            for(int j = 0; j < iw; j++){
                // output[k][i][j] = min(max(input[k][i][j], 0), 6);
                if (input[k*ih*iw + i*iw + j] > 0.0000000000000000000000001)
                    output[k*ih*iw + i*iw + j] = std::min(input[k*ih*iw + i*iw + j], FIX_F(6));
                else
                    output[k*ih*iw + i*iw + j] = 0;
            }
        }
    }
}

void globalaveragepooling(FIX_F *input, int ic, int ih, FIX_F *output)
{   
    int iw = ih;
    int oc = ic;
    FIX_F gap_sum[1][1][1280];

    for(int i = 0; i < ic; i++){
        gap_sum[0][0][i] = 0;
        for(int j = 0; j < ih; j++){
            for(int k = 0; k < iw; k++){
                gap_sum[0][0][i] += input[i*ih*iw + j*iw + k];
            }
        }
        output[0 + 0 + i] = gap_sum[0][0][i]/(ih*iw);
    }
}
