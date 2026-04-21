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

// Dense 3x3 stem convolution for the input image. The implementation is split
// into two spatial halves (`pw_3x3_new` and `pw_3x3_new_2`) so the top-level
// scheduler can keep the temporary buffers smaller.
void pw_3x3(FIX_F *input, FIX_W *weight, /*FIX_F bias[1280], */FIX_F *output)
{
    #ifdef NO_SYNTH
        FIX_F *input_pad = (FIX_F*)malloc(3*225*225*sizeof(FIX_F));
        FIX_F *output_1 = (FIX_F*)malloc(32*112*112*3*sizeof(FIX_F));
    #else
        FIX_F input_pad[3*225*225];
        FIX_F output_1[32*112*112*3];
    #endif

    // padding
    /*for(int k = 0; k < 3; k++){
        for(int i = 0; i < 225; i++){
            for(int j = 0; j < 225; j++){
                input_pad[k*225*225 + i*225 + j] = 0;
            }
        }
    }

    for(int k = 0; k < 3; k++){
        for(int i = 0; i < 224; i++){
            for(int j = 0; j < 224; j++){
                input_pad[k*224*224 + i*224 + j] = input[k*224*224 + i*224 + j];
            }
        }
    }

    // conv(3*3);
    for(int m = 0; m < 32; m++){
        for(int i = 0; i < 112; i++){
            for(int j = 0; j < 112; j++){
                // float output[m][i][j] = 0;
                for(int n = 0; n < 3; n++){
                    output_1[m*112*112*3 + i*112*3 + j*3 + n] = input_pad[n*224*224 + i*2*224 + j*2] * weight[m*3*3*3 + n*3*3 + 0*3 + 0]
                                        + input_pad[n*224*224 + i*2*224 + j*2 + 1] * weight[m*3*3*3 + n*3*3 + 0*3 + 1]
                                        + input_pad[n*224*224 + i*2*224 + j*2 + 2] * weight[m*3*3*3 + n*3*3 + 0*3 + 2]
                                        + input_pad[n*224*224 + (i*2 + 1)*224 + j*2] * weight[m*3*3*3 + n*3*3 + 1*3 + 0]
                                        + input_pad[n*224*224 + (i*2 + 1)*224 + j*2 + 1] * weight[m*3*3*3 + n*3*3 + 1*3 + 1]
                                        + input_pad[n*224*224 + (i*2 + 1)*224 + j*2 + 2] * weight[m*3*3*3 + n*3*3 + 1*3 + 2]
                                        + input_pad[n*224*224 + (i*2 + 2)*224 + j*2] * weight[m*3*3*3 + n*3*3 + 2*3 + 0]
                                        + input_pad[n*224*224 + (i*2 + 2)*224 + j*2 + 1] * weight[m*3*3*3 + n*3*3 + 2*3 + 1]
                                        + input_pad[n*224*224 + (i*2 + 2)*224 + j*2 + 2] * weight[m*3*3*3 + n*3*3 + 2*3 + 2];
                    output[m*112*112 + i*112 + j] += output_1[m*112*112*3 + i*112*3 + j*3 + n];

                    //if (n == (3 - 1))
                        //output[m][i][j] = output[m][i][j] + bias[m];
                }
            }
        } 
    }*/

    // conv(3*3);
    for(int m = 0; m < 32; m++){
        for(int i = 0; i < 111; i++){
            for(int j = 0; j < 111; j++){
                // float output[m][i][j] = 0;
                for(int n = 0; n < 3; n++){
                    output_1[m*112*112*3 + i*112*3 + j*3 + n] = input[n*224*224 + i*2*224 + j*2] * weight[m*3*3*3 + n*3*3 + 0*3 + 0]
                                        + input[n*224*224 + i*2*224 + j*2 + 1] * weight[m*3*3*3 + n*3*3 + 0*3 + 1]
                                        + input[n*224*224 + i*2*224 + j*2 + 2] * weight[m*3*3*3 + n*3*3 + 0*3 + 2]
                                        + input[n*224*224 + (i*2 + 1)*224 + j*2] * weight[m*3*3*3 + n*3*3 + 1*3 + 0]
                                        + input[n*224*224 + (i*2 + 1)*224 + j*2 + 1] * weight[m*3*3*3 + n*3*3 + 1*3 + 1]
                                        + input[n*224*224 + (i*2 + 1)*224 + j*2 + 2] * weight[m*3*3*3 + n*3*3 + 1*3 + 2]
                                        + input[n*224*224 + (i*2 + 2)*224 + j*2] * weight[m*3*3*3 + n*3*3 + 2*3 + 0]
                                        + input[n*224*224 + (i*2 + 2)*224 + j*2 + 1] * weight[m*3*3*3 + n*3*3 + 2*3 + 1]
                                        + input[n*224*224 + (i*2 + 2)*224 + j*2 + 2] * weight[m*3*3*3 + n*3*3 + 2*3 + 2];
                    output[m*112*112 + i*112 + j] += output_1[m*112*112*3 + i*112*3 + j*3 + n];
                }
            }
        } 
    }

    for(int m = 0; m < 32; m++){
        for(int i = 0; i < 111; i++){
            for(int n = 0; n < 3; n++){
                output_1[m*112*112*3 + i*112*3 + 111*3 + n] = input[n*224*224 + i*2*224 + 111*2] * weight[m*3*3*3 + n*3*3 + 0*3 + 0]
                                        + input[n*224*224 + i*2*224 + 111*2 + 1] * weight[m*3*3*3 + n*3*3 + 0*3 + 1]
                                        + input[n*224*224 + (i*2 + 1)*224 + 111*2] * weight[m*3*3*3 + n*3*3 + 1*3 + 0]
                                        + input[n*224*224 + (i*2 + 1)*224 + 111*2 + 1] * weight[m*3*3*3 + n*3*3 + 1*3 + 1]
                                        + input[n*224*224 + (i*2 + 2)*224 + 111*2] * weight[m*3*3*3 + n*3*3 + 2*3 + 0]
                                        + input[n*224*224 + (i*2 + 2)*224 + 111*2 + 1] * weight[m*3*3*3 + n*3*3 + 2*3 + 1];
                /*if((m == 0)&&(i == 0)){
                    std::cout << m*112*112*3 + i*112*3 + 111*3 + n << " " << output_1[m*112*112*3 + i*112*3 + 111*3 + n] << std::endl;
                    std::cout << "input: " << n*224*224 + i*2*224 + 111*2 << " " << input[n*224*224 + i*2*224 + 111*2] << std::endl;
                    std::cout << "input: " << n*224*224 + i*2*224 + 111*2 + 1<< " " << input[n*224*224 + i*2*224 + 111*2+1] << std::endl;
                    std::cout << "input: " << n*224*224 + (i*2+1)*224 + 111*2 << " " << input[n*224*224 + (i*2+1)*224 + 111*2] << std::endl;
                    std::cout << "input: " << n*224*224 + (i*2+1)*224 + 111*2 + 1<< " " << input[n*224*224 + (i*2+1)*224 + 111*2+1] << std::endl;
                    std::cout << "input: " << n*224*224 + (i*2+2)*224 + 111*2 << " " << input[n*224*224 + (i*2+2)*224 + 111*2] << std::endl;
                    std::cout << "input: " << n*224*224 + (i*2+2)*224 + 111*2 + 1 << " " << input[n*224*224 + (i*2+2)*224 + 111*2+1] << std::endl;
                }*/
                output[m*112*112 + i*112 + 111] += output_1[m*112*112*3 + i*112*3 + 111*3 + n];
            }
        }
    }

    for(int m = 0; m < 32; m++){
        for(int j = 0; j < 111; j++){
            for(int n = 0; n < 3; n++){
                output_1[m*112*112*3 + 111*112*3 + j*3 + n] = input[n*224*224 + 111*2*224 + j*2] * weight[m*3*3*3 + n*3*3 + 0*3 + 0]
                                        + input[n*224*224 + 111*2*224 + j*2 + 1] * weight[m*3*3*3 + n*3*3 + 0*3 + 1]
                                        + input[n*224*224 + 111*2*224 + j*2 + 2] * weight[m*3*3*3 + n*3*3 + 0*3 + 2]
                                        + input[n*224*224 + (111*2 + 1)*224 + j*2] * weight[m*3*3*3 + n*3*3 + 1*3 + 0]
                                        + input[n*224*224 + (111*2 + 1)*224 + j*2 + 1] * weight[m*3*3*3 + n*3*3 + 1*3 + 1]
                                        + input[n*224*224 + (111*2 + 1)*224 + j*2 + 2] * weight[m*3*3*3 + n*3*3 + 1*3 + 2];
                output[m*112*112 + 111*112 + j] += output_1[m*112*112*3 + 111*112*3 + j*3 + n];
            }
        }
    }

    for(int m = 0; m < 32; m++){
        for(int n = 0; n < 3; n++){
            output_1[m*112*112*3 + 111*112*3 + 111*3 + n] = input[n*224*224 + 222*224 + 222] * weight[m*3*3*3 + n*3*3 + 0*3 + 0]
                                + input[n*224*224 + 222*224 + 223] * weight[m*3*3*3 + n*3*3 + 0*3 + 1]
                                + input[n*224*224 + 223*224 + 222] * weight[m*3*3*3 + n*3*3 + 1*3 + 0]
                                + input[n*224*224 + 223*224 + 223] * weight[m*3*3*3 + n*3*3 + 1*3 + 1];
            output[m*112*112 + 111*112 + 111] += output_1[m*112*112*3 + 111*112*3 + 111*3 + n];    
        }
    }

    #ifdef NO_SYNTH
        free(input_pad);
        free(output_1);
    #endif
}

void pw_3x3_new(FIX_F *input, FIX_W *weight, FIX_F *output)
{
    #ifdef NO_SYNTH
        FIX_F *output_1 = (FIX_F*)malloc(32*112*112*3/2*sizeof(FIX_F));
    #else
        //FIX_F input_pad[3*225*225];
        FIX_F output_1[32*112*112*3/2];
    #endif


    for(int m = 0; m < 32; m++){
        for(int i = 0; i < 56; i++){
            for(int j = 0; j < 111; j++){
                // float output[m][i][j] = 0;
                for(int n = 0; n < 3; n++){
                    output_1[m*56*112*3 + i*112*3 + j*3 + n] = input[n*224*224 + i*2*224 + j*2] * weight[m*3*3*3 + n*3*3 + 0*3 + 0]
                                        + input[n*224*224 + i*2*224 + j*2 + 1] * weight[m*3*3*3 + n*3*3 + 0*3 + 1]
                                        + input[n*224*224 + i*2*224 + j*2 + 2] * weight[m*3*3*3 + n*3*3 + 0*3 + 2]
                                        + input[n*224*224 + (i*2 + 1)*224 + j*2] * weight[m*3*3*3 + n*3*3 + 1*3 + 0]
                                        + input[n*224*224 + (i*2 + 1)*224 + j*2 + 1] * weight[m*3*3*3 + n*3*3 + 1*3 + 1]
                                        + input[n*224*224 + (i*2 + 1)*224 + j*2 + 2] * weight[m*3*3*3 + n*3*3 + 1*3 + 2]
                                        + input[n*224*224 + (i*2 + 2)*224 + j*2] * weight[m*3*3*3 + n*3*3 + 2*3 + 0]
                                        + input[n*224*224 + (i*2 + 2)*224 + j*2 + 1] * weight[m*3*3*3 + n*3*3 + 2*3 + 1]
                                        + input[n*224*224 + (i*2 + 2)*224 + j*2 + 2] * weight[m*3*3*3 + n*3*3 + 2*3 + 2];
                    output[m*56*112 + i*112 + j] += output_1[m*56*112*3 + i*112*3 + j*3 + n];
                }
            }
        } 
    }
    for(int m = 0; m < 32; m++){
        for(int i = 0; i < 56; i++){
                // float output[m][i][j] = 0;
            for(int n = 0; n < 3; n++){
                output_1[m*56*112*3 + i*112*3 + 111*3 + n] = input[n*224*224 + i*2*224 + 111*2] * weight[m*3*3*3 + n*3*3 + 0*3 + 0]
                                    + input[n*224*224 + i*2*224 + 111*2 + 1] * weight[m*3*3*3 + n*3*3 + 0*3 + 1]
                                    + input[n*224*224 + (i*2 + 1)*224 + 111*2] * weight[m*3*3*3 + n*3*3 + 1*3 + 0]
                                    + input[n*224*224 + (i*2 + 1)*224 + 111*2 + 1] * weight[m*3*3*3 + n*3*3 + 1*3 + 1]
                                    + input[n*224*224 + (i*2 + 2)*224 + 111*2] * weight[m*3*3*3 + n*3*3 + 2*3 + 0]
                                    + input[n*224*224 + (i*2 + 2)*224 + 111*2 + 1] * weight[m*3*3*3 + n*3*3 + 2*3 + 1];
                output[m*56*112 + i*112 + 111] += output_1[m*56*112*3 + i*112*3 + 111*3 + n];
            }
        } 
    }
} 

void pw_3x3_new_2(FIX_F *input, FIX_W *weight, FIX_F *output)
{
    #ifdef NO_SYNTH
        FIX_F *output_1 = (FIX_F*)malloc(32*112*112*3/2*sizeof(FIX_F));
    #else
        //FIX_F input_pad[3*225*225];
        FIX_F output_1[32*112*112*3/2];
    #endif

    for(int m = 0; m < 32; m++){
        for(int i = 56; i < 111; i++){
            for(int j = 0; j < 111; j++){
                // float output[m][i][j] = 0;
                output[m*56*112 + (i - 56)*112 + j] = 0;
                for(int n = 0; n < 3; n++){
                    output_1[m*56*112*3 + (i - 56)*112*3 + j*3 + n] = input[n*224*224 + i*2*224 + j*2] * weight[m*3*3*3 + n*3*3 + 0*3 + 0]
                                        + input[n*224*224 + i*2*224 + j*2 + 1] * weight[m*3*3*3 + n*3*3 + 0*3 + 1]
                                        + input[n*224*224 + i*2*224 + j*2 + 2] * weight[m*3*3*3 + n*3*3 + 0*3 + 2]
                                        + input[n*224*224 + (i*2 + 1)*224 + j*2] * weight[m*3*3*3 + n*3*3 + 1*3 + 0]
                                        + input[n*224*224 + (i*2 + 1)*224 + j*2 + 1] * weight[m*3*3*3 + n*3*3 + 1*3 + 1]
                                        + input[n*224*224 + (i*2 + 1)*224 + j*2 + 2] * weight[m*3*3*3 + n*3*3 + 1*3 + 2]
                                        + input[n*224*224 + (i*2 + 2)*224 + j*2] * weight[m*3*3*3 + n*3*3 + 2*3 + 0]
                                        + input[n*224*224 + (i*2 + 2)*224 + j*2 + 1] * weight[m*3*3*3 + n*3*3 + 2*3 + 1]
                                        + input[n*224*224 + (i*2 + 2)*224 + j*2 + 2] * weight[m*3*3*3 + n*3*3 + 2*3 + 2];
                    output[m*56*112 + (i - 56)*112 + j] += output_1[m*56*112*3 + (i - 56)*112*3 + j*3 + n];
                }
            }
        } 
    }

    for(int m = 0; m < 32; m++){
        for(int i = 56; i < 111; i++){
            // float output[m][i][j] = 0;
            output[m*56*112 + (i - 56)*112 + 111] = 0;
            for(int n = 0; n < 3; n++){
                    output_1[m*56*112*3 + (i - 56)*112*3 + 111*3 + n] = input[n*224*224 + i*2*224 + 111*2] * weight[m*3*3*3 + n*3*3 + 0*3 + 0]
                                        + input[n*224*224 + i*2*224 + 111*2 + 1] * weight[m*3*3*3 + n*3*3 + 0*3 + 1]
                                        + input[n*224*224 + (i*2 + 1)*224 + 111*2] * weight[m*3*3*3 + n*3*3 + 1*3 + 0]
                                        + input[n*224*224 + (i*2 + 1)*224 + 111*2 + 1] * weight[m*3*3*3 + n*3*3 + 1*3 + 1]
                                        + input[n*224*224 + (i*2 + 2)*224 + 111*2] * weight[m*3*3*3 + n*3*3 + 2*3 + 0]
                                        + input[n*224*224 + (i*2 + 2)*224 + 111*2 + 1] * weight[m*3*3*3 + n*3*3 + 2*3 + 1];
                    output[m*56*112 + (i - 56)*112 + 111] += output_1[m*56*112*3 + (i - 56)*112*3 + 111*3 + n];
            }
        }
    }

    for(int m = 0; m < 32; m++){
        for(int j = 0; j < 111; j++){
            // float output[m][i][j] = 0;
            output[m*56*112 + (111 - 56)*112 + j] = 0;
            for(int n = 0; n < 3; n++){
                output_1[m*56*112*3 + (111 - 56)*112*3 + j*3 + n] = input[n*224*224 + 111*2*224 + j*2] * weight[m*3*3*3 + n*3*3 + 0*3 + 0]
                                    + input[n*224*224 + 111*2*224 + j*2 + 1] * weight[m*3*3*3 + n*3*3 + 0*3 + 1]
                                    + input[n*224*224 + 111*2*224 + j*2 + 2] * weight[m*3*3*3 + n*3*3 + 0*3 + 2]
                                    + input[n*224*224 + (111*2 + 1)*224 + j*2] * weight[m*3*3*3 + n*3*3 + 1*3 + 0]
                                    + input[n*224*224 + (111*2 + 1)*224 + j*2 + 1] * weight[m*3*3*3 + n*3*3 + 1*3 + 1]
                                    + input[n*224*224 + (111*2 + 1)*224 + j*2 + 2] * weight[m*3*3*3 + n*3*3 + 1*3 + 2];
                output[m*56*112 + (111 - 56)*112 + j] += output_1[m*56*112*3 + (111 - 56)*112*3 + j*3 + n];
            }
        }
    } 

    for(int m = 0; m < 32; m++){
        // float output[m][i][j] = 0;
        output[m*56*112 + (111 - 56)*112 + 111] = 0;
        for(int n = 0; n < 3; n++){
            output_1[m*56*112*3 + (111 - 56)*112*3 + 111*3 + n] = input[n*224*224 + 111*2*224 + 111*2] * weight[m*3*3*3 + n*3*3 + 0*3 + 0]
                                + input[n*224*224 + 111*2*224 + 111*2 + 1] * weight[m*3*3*3 + n*3*3 + 0*3 + 1]
                                + input[n*224*224 + (111*2 + 1)*224 + 111*2] * weight[m*3*3*3 + n*3*3 + 1*3 + 0]
                                + input[n*224*224 + (111*2 + 1)*224 + 111*2 + 1] * weight[m*3*3*3 + n*3*3 + 1*3 + 1];
            output[m*56*112 + (111 - 56)*112 + 111] += output_1[m*56*112*3 + (111 - 56)*112*3 + 111*3 + n];

            //std::cout << "output_1" << " " << output_1[m*56*112*3 + (111 - 56)*112*3 + 111*3 + n] << " "  << "input" << input[n*224*224 + 111*2*224 + 111*2] << " " << " weight" << " " << weight[m*3*3*3 + n*3*3 + 0*3 + 0] << " " << "output " << output[m*56*112 + (111 - 56)*112 + 111] << std::endl;
        }
    }
}

