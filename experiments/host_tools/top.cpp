#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <fstream>
#include <iostream>
#include <stdlib.h>
//#include <cstdlib>
#include "top.hpp"
#include <CL/cl2.hpp>

using namespace std;

static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

void parameter_recorder()
{
    
    // load image from file;
    std::fstream fileimg("img1.txt");
    for(int i = 0; i < 3*224*224; i++){
        FIX_F n;
        fileimg >> n;
        image[i] = n;
    }
    fileimg.close();

    for(int i = 0; i < 4704; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = image[i*32 + j].range(15, 0);
    	}
    	image_raw[i].range(511, 0) = DATA.range(511, 0);
    }
    

    // load 1*1 sparse weight from file;
    std::fstream file11("1x1_sparse_weight.txt");
    for(int i = 0; i < 32*33198; i++){
        FIX_W n;
        file11 >> n;
        sparse_weight_1x1_all[i] = n;
    }
    file11.close();

    // fill 1*1 sparse weight into 512 bit-width bus;
    // 32*33198;
    for(int i = 0; i < 33198; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = sparse_weight_1x1_all[i*32 + j].range(15, 0);
    	}
    	sparse_1x1_weight_raw[i].range(511, 0) = DATA.range(511, 0);
    }
    std::cout << "1*1 512bit done" << std::endl;

    
    // load 3*3 weight from file;
    std::fstream file33("3x3_weight.txt");
    for(int i = 0; i < 226; i++){
        for(int j = 0; j < 32; j++){
            for(int k = 0; k < 3; k++){
                for(int l = 0; l < 3; l++){
                    FIX_W n;
                    file33 >> n;
                    weight_3x3_all[i*32*3*3 + j*3*3 + k*3 + l] = n;
                }
            }
        }
    }
    file33.close();

    FILE* fp1;
    fp1 = fopen("weight_3x3.txt","w");
	for(int i = 0; i < 3*3*3*32; i++){
		fprintf(fp1, "%3.14f \n", weight_3x3_all[i].to_float());
	}

	fclose(fp1);

    // fill 3*3 weight into 512 bit-width bus;
    // 32*2034;
    /*for(int i = 0; i < 2034; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = weight_3x3_all[i*32 + j].range(15, 0);
    	}
    	weight_3x3_all_512bit[i].range(511, 0) = DATA.range(511, 0);
    }*/

    for(int i = 0; i < 11; i++) {
    	for(int m = 0; m < 3; m++) {
    		for(int n = 0; n < 3; n++) {
                uint512 DATA = 0;
				for(int j = 0; j < 32; j++) {
					DATA.range(j*16 + 15, j*16) = weight_3x3_all[i*32*3*3 + j*3*3 + m*3 + n].range(15, 0);
				}
				weight_3x3_raw[i*3*3 + m*3 + n].range(511, 0) = DATA.range(511, 0);
    		}
    	}
    }


    for(int m = 0; m < 3; m++) {
    	for(int n = 0; n < 3; n++) {
            uint512 DATA = 0;
			for(int j = 0; j < 16; j++) {
				DATA.range(j*16 + 15, j*16) = weight_3x3_all[11*32*3*3 + j*3*3 + m*3 + n].range(15, 0);
			}
			weight_3x3_raw[11*3*3 + m*3 + n].range(511, 0) = DATA.range(511, 0);
    	}
    } 

    for(int i = 0; i < 4; i++) {
    	for(int m = 0; m < 3; m++) {
    		for(int n = 0; n < 3; n++) {
                uint512 DATA = 0;
				for(int j = 0; j < 32; j++) {
					DATA.range(j*16 + 15, j*16) = weight_3x3_all[3312 + i*32*3*3 + j*3*3 + m*3 + n].range(15, 0);
                }
				weight_3x3_raw[(i + 12)*3*3 + m*3 + n].range(511, 0) = DATA.range(511, 0);
    		}
    	}
    }

    for(int m = 0; m < 3; m++) {
    	for(int n = 0; n < 3; n++) {
            uint512 DATA = 0;
			for(int j = 0; j < 16; j++) {
				DATA.range(j*16 + 15, j*16) = weight_3x3_all[4464 + j*3*3 + m*3 + n].range(15, 0);
			}
			weight_3x3_raw[16*3*3 + m*3 + n].range(511, 0) = DATA.range(511, 0);
    	}
    }

    for(int i = 0; i < 210; i++) {
    	for(int m = 0; m < 3; m++) {
    		for(int n = 0; n < 3; n++) {
                uint512 DATA = 0;
				for(int j = 0; j < 32; j++) {
					DATA.range(j*16 + 15, j*16) = weight_3x3_all[4608 + i*32*3*3 + j*3*3 + m*3 + n].range(15, 0); 
                    if (i == 209){
                        std::cout << j*3*3 + m*3 + n << " " << weight_3x3_all[4608 + i*32*3*3 + j*3*3 + m*3 + n] << std::endl;
                    }
				}
				weight_3x3_raw[(i + 17)*3*3 + m*3 + n].range(511, 0) = DATA.range(511, 0);
    		}
    	}
    }

    for(int i = 0; i < 30; i++) {
    	for(int m = 0; m < 3; m++) {
    		for(int n = 0; n < 3; n++) {
                uint512 DATA = 0;
				for(int j = 0; j < 32; j++) {
					DATA.range(j*16 + 15, j*16) = weight_3x3_all[56448 + i*32*3*3 + j*3*3 + m*3 + n].range(15, 0); 
                    if (i == 209){
                        std::cout << j*3*3 + m*3 + n << " " << weight_3x3_all[56448 + i*32*3*3 + j*3*3 + m*3 + n] << std::endl;
                    }
				}
				weight_3x3_17_raw[(i*3*3 + m*3 + n)].range(511, 0) = DATA.range(511, 0);
    		}
    	}
    }

    std::cout << "3*3 512bit done" << std::endl;

    // load bias from file;
    std::fstream fileb("bias.txt");
    for(int i = 0; i < 1000; i++){
        FIX_W n;
        fileb >> n;
        bias_all[i] = n;
    }
    fileb.close();

    // fill bias into 512 bit-width bus;
    // 32*2034;
    for(int i = 0; i < 31; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = bias_all[i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_bias_raw[i].range(511, 0) = DATA.range(511, 0);
    }
    
    uint512 DATA1 = 0;
    for(int i = 0; i < 8; i++) {
    	DATA1.range(i*16 + 15, i*16) = bias_all[992 + i].range(15, 0);
    }
    	conv_last_1x1_bias_raw[31].range(511, 0) = DATA1.range(511, 0);

    std::cout << "bias 512bit done" << std::endl;


    // load block_c from file;
    std::fstream filei1("block_c.txt");
    for(int i = 0; i < 265584; i++){
        uint9 n;
        filei1 >> n;
        block_c_all[i] = n;
    }
    filei1.close();

    // fill block_c into 512 bit-width bus;
    // 32*8300;
    for(int i = 0; i < 30; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 9, j*16) = block_c_all[i*32 + j].range(9, 0);
    	}
    	block_c_raw[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 16; i++){
        block_c_raw[30].range(i*16 + 9, i*16) = block_c_all[960 + i].range(9, 0);
    }

    for(int i = 0; i < 13; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 9, j*16) = block_c_all[976 + i*32 + j].range(9, 0);
    	}
    	block_c_raw[31 + i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 16; i++){
        block_c_raw[44].range(i*16 + 9, i*16) = block_c_all[1392 + i].range(9, 0);
    }

    for(int i = 0; i < 13; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 9, j*16) = block_c_all[1408 + i*32 + j].range(9, 0);
    	}
    	block_c_raw[45 + i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 16; i++){
        block_c_raw[58].range(i*16 + 9, i*16) = block_c_all[1824 + i].range(9, 0);
    }

    for(int i = 0; i < 8242; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 9, j*16) = block_c_all[1840 + i*32 + j].range(9, 0);
    	}
    	block_c_raw[59 + i].range(511, 0) = DATA.range(511, 0);
    }


    for(int i = 0; i < 1200; i++){
        uint512 DATA = 0;
        for(int j = 0; j < 32; j++){
            DATA.range(j*16 + 9, j*16) = block_c_all[118384 + i*32 + j].range(9, 0);
        }
        block_c_16_raw[i].range(511, 0) = DATA.range(511, 0);
    }
    // load bn from file;
    std::fstream filebn("sqrt_bn.txt");
    for(int i = 0; i < 32*2132; i++){
        FIX_W n;
        filebn >> n;
        bn_all[i] = n;
    }
    filebn.close();

    for(int i = 0; i < 2132; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = bn_all[i*32 + j].range(15, 0);
    	}
    	bn_raw[i].range(511, 0) = DATA.range(511, 0);
    }


    // load the last conv weight from file;
    std::fstream filelast1("classifier_1.txt");
    for(int i = 0; i < 160*1280; i++){
        FIX_W n;
        filelast1 >> n;
        conv_last_1x1_weight_1[i] = n;
    }
    filelast1.close();

    std::fstream filelast2("classifier_2.txt");
    for(int i = 0; i < 160*1280; i++){
        FIX_W n;
        filelast2 >> n;
        conv_last_1x1_weight_2[i] = n;
    }
    filelast2.close();

    std::fstream filelast3("classifier_3.txt");
    for(int i = 0; i < 160*1280; i++){
        FIX_W n;
        filelast3 >> n;
        conv_last_1x1_weight_3[i] = n;
    }
    filelast3.close();

    std::fstream filelast4("classifier_4.txt");
    for(int i = 0; i < 160*1280; i++){
        FIX_W n;
        filelast4 >> n;
        conv_last_1x1_weight_4[i] = n;
    }
    filelast4.close();

    std::fstream filelast5("classifier_5.txt");
    for(int i = 0; i < 160*1280; i++){
        FIX_W n;
        filelast5 >> n;
        conv_last_1x1_weight_5[i] = n;
    }
    filelast5.close();

    std::fstream filelast6("classifier_6.txt");
    for(int i = 0; i < 160*1280; i++){
        FIX_W n;
        filelast6 >> n;
        conv_last_1x1_weight_6[i] = n;
    }
    filelast6.close();

    std::fstream filelast7("classifier_7.txt");
    for(int i = 0; i < 40*1280; i++){
        FIX_W n;
        filelast7 >> n;
        conv_last_1x1_weight_7[i] = n;
    }
    filelast7.close();

    // fill the last conv weight into 512 bit-width bus;
    // 32*40000;
    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_1[i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_11[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_1[51200 + i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_12[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_1[102400 + i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_13[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_1[153600 + i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_14[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_2[i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_21[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_2[51200 + i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_22[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_2[102400 + i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_23[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_2[153600 + i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_24[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_3[i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_31[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_3[51200 + i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_32[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_3[102400 + i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_33[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_3[153600 + i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_34[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_4[i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_41[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_4[51200 + i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_42[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_4[102400 + i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_43[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_4[153600 + i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_44[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_5[i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_51[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_5[51200 + i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_52[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_5[102400 + i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_53[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 3200; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_5[153600 + i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_54[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_6[i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_61[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_6[51200 + i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_62[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_6[102400 + i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_63[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_6[153600 + i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_64[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 1600; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_7[i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_raw_7[i].range(511, 0) = DATA.range(511, 0);
    }

    /*FIX_W test1;
    FIX_W test2;
    test1.range(15, 0) = conv_last_1x1_weight_6_512bit[6144].range(15, 0);
    test2.range(15, 0) = conv_last_1x1_weight_6_512bit[6144].range(31, 16);
    std::cout << test1 << std::endl;
    std::cout << test2 << std::endl;*/

    /*FIX_W test_array1;
    test_array1.range(15, 0) = conv_last_1x1_weight_all_512bit[6144].range(15, 0);
    std::cout << "tb: " << conv_last_1x1_weight_all_512bit[6144].range(15, 0) << std::endl;
    std::cout << "tb: " << test_array1.to_float() << std::endl;
    std::cout << "tb: " << conv_last_1x1_weight_all[196608].to_float() << std::endl;*/

}


void load_block_col_r(){
    std::fstream file0("features.1.conv.3.block.col.r.txt");
    std::fstream file1("features.2.conv.0.block.col.r.txt");
    std::fstream file2("features.2.conv.6.block.col.r.txt");
    std::fstream file3("features.3.conv.0.block.col.r.txt");
    std::fstream file4("features.3.conv.6.block.col.r.txt");
    std::fstream file5("features.4.conv.0.block.col.r.txt");
    std::fstream file6("features.4.conv.6.block.col.r.txt");
    std::fstream file7("features.5.conv.0.block.col.r.txt");
    std::fstream file8("features.5.conv.6.block.col.r.txt");
    std::fstream file9("features.6.conv.0.block.col.r.txt");
    std::fstream file10("features.6.conv.6.block.col.r.txt");
    std::fstream file11("features.7.conv.0.block.col.r.txt");
    std::fstream file12("features.7.conv.6.block.col.r.txt");
    std::fstream file13("features.8.conv.0.block.col.r.txt");
    std::fstream file14("features.8.conv.6.block.col.r.txt");
    std::fstream file15("features.9.conv.0.block.col.r.txt");
    std::fstream file16("features.9.conv.6.block.col.r.txt");
    std::fstream file17("features.10.conv.0.block.col.r.txt");
    std::fstream file18("features.10.conv.6.block.col.r.txt");
    std::fstream file19("features.11.conv.0.block.col.r.txt");
    std::fstream file20("features.11.conv.6.block.col.r.txt");
    std::fstream file21("features.12.conv.0.block.col.r.txt");
    std::fstream file22("features.12.conv.6.block.col.r.txt");
    std::fstream file23("features.13.conv.0.block.col.r.txt");
    std::fstream file24("features.13.conv.6.block.col.r.txt");
    std::fstream file25("features.14.conv.0.block.col.r.txt");
    std::fstream file26("features.14.conv.6.block.col.r.txt");
    std::fstream file27("features.15.conv.0.block.col.r.txt");
    std::fstream file28("features.15.conv.6.block.col.r.txt");
    std::fstream file29("features.16.conv.0.block.col.r.txt");
    std::fstream file30("features.16.conv.6.block.col.r.txt");
    std::fstream file31("features.17.conv.0.block.col.r.txt");
    std::fstream file32("features.17.conv.6.block.col.r.txt");
    std::fstream file33("features.18.0.block.col.r.txt");

    //1
    for(int i = 0; i < (16/4 + 1); i++){
        uint16 n;
        file0 >> n;
        pw1_last_1x1_block_col_r_raw[i] = n;
    }
    file0.close();
    //2
    for(int i = 0; i < (96/4 + 1); i++){
        uint16 n;
        file1 >> n;
        pw2_first_1x1_block_col_r_raw[i] = n;
    }
    file1.close();
    for(int i = 0; i < (24/4 + 1); i++){
        uint16 n;
        file2 >> n;
        pw2_last_1x1_block_col_r_raw[i] = n;
    }
    file2.close();
    //3
    for(int i = 0; i < (144/4 + 1); i++){
        uint16 n;
        file3 >> n;
        pw3_first_1x1_block_col_r_raw[i] = n;
    }
    file3.close();
    for(int i = 0; i < (24/4 + 1); i++){
        uint16 n;
        file4 >> n;
        pw3_last_1x1_block_col_r_raw[i] = n;
    }
    file4.close();
    //4
    for(int i = 0; i < (144/4 + 1); i++){
        uint16 n;
        file5 >> n;
        pw4_first_1x1_block_col_r_raw[i] = n;
    }
    file5.close();
    for(int i = 0; i < (32/4 + 1); i++){
        uint16 n;
        file6 >> n;
        pw4_last_1x1_block_col_r_raw[i] = n;
    }
    file6.close();
    //5
    for(int i = 0; i < (192/4 + 1); i++){
        uint16 n;
        file7 >> n;
        pw5_first_1x1_block_col_r_raw[i] = n;
    }
    file7.close();
    for(int i = 0; i < (32/4 + 1); i++){
        uint16 n;
        file8 >> n;
        pw5_last_1x1_block_col_r_raw[i] = n;
    }
    file8.close();
    //6
    for(int i = 0; i < (192/4 + 1); i++){
        uint16 n;
        file9 >> n;
        pw6_first_1x1_block_col_r_raw[i] = n;
    }
    file9.close();
    for(int i = 0; i < (32/4 + 1); i++){
        uint16 n;
        file10 >> n;
        pw6_last_1x1_block_col_r_raw[i] = n;
    }
    file10.close();
    //7
    for(int i = 0; i < (192/4 + 1); i++){
        uint16 n;
        file11 >> n;
        pw7_first_1x1_block_col_r_raw[i] = n;
    }
    file11.close();
    for(int i = 0; i < (64/4 + 1); i++){
        uint16 n;
        file12 >> n;
        pw7_last_1x1_block_col_r_raw[i] = n;
    }
    file12.close();
    //8
    for(int i = 0; i < (384/4 + 1); i++){
        uint16 n;
        file13 >> n;
        pw8_first_1x1_block_col_r_raw[i] = n;
    }
    file13.close();
    for(int i = 0; i < (64/4 + 1); i++){
        uint16 n;
        file14 >> n;
        pw8_last_1x1_block_col_r_raw[i] = n;
    }
    file14.close();
    //9
    for(int i = 0; i < (384/4 + 1); i++){
        uint16 n;
        file15 >> n;
        pw9_first_1x1_block_col_r_raw[i] = n;
    }
    file15.close();
    for(int i = 0; i < (64/4 + 1); i++){
        uint16 n;
        file16 >> n;
        pw9_last_1x1_block_col_r_raw[i] = n;
    }
    file16.close();
    //10
    for(int i = 0; i < (384/4 + 1); i++){
        uint16 n;
        file17 >> n;
        pw10_first_1x1_block_col_r_raw[i] = n;
    }
    file17.close();
    for(int i = 0; i < (64/4 + 1); i++){
        uint16 n;
        file18 >> n;
        pw10_last_1x1_block_col_r_raw[i] = n;
    }
    file18.close();
    //11
    for(int i = 0; i < (384/4 + 1); i++){
        uint16 n;
        file19 >> n;
        pw11_first_1x1_block_col_r_raw[i] = n;
    }
    file19.close();
    for(int i = 0; i < (96/4 + 1); i++){
        uint16 n;
        file20 >> n;
        pw11_last_1x1_block_col_r_raw[i] = n;
    }
    file20.close();
    //12
    for(int i = 0; i < (576/4 + 1); i++){
        uint16 n;
        file21 >> n;
        pw12_first_1x1_block_col_r_raw[i] = n;
    }
    file21.close();
    for(int i = 0; i < (96/4 + 1); i++){
        uint16 n;
        file22 >> n;
        pw12_last_1x1_block_col_r_raw[i] = n;
    }
    file22.close();
    //13
    for(int i = 0; i < (576/4 + 1); i++){
        uint16 n;
        file23 >> n;
        pw13_first_1x1_block_col_r_raw[i] = n;
    }
    file23.close();
    for(int i = 0; i < (96/4 + 1); i++){
        uint16 n;
        file24 >> n;
        pw13_last_1x1_block_col_r_raw[i] = n;
    }
    file24.close();
    //14
    for(int i = 0; i < (576/4 + 1); i++){
        uint16 n;
        file25 >> n;
        pw14_first_1x1_block_col_r_raw[i] = n;
    }
    file25.close();
    for(int i = 0; i < (160/4 + 1); i++){
        uint16 n;
        file26 >> n;
        pw14_last_1x1_block_col_r_raw[i] = n;
    }
    file26.close();
    //15
    for(int i = 0; i < (960/4 + 1); i++){
        uint16 n;
        file27 >> n;
        pw15_first_1x1_block_col_r_raw[i] = n;
    }
    file27.close();
    for(int i = 0; i < (160/4 + 1); i++){
        uint16 n;
        file28 >> n;
        pw15_last_1x1_block_col_r_raw[i] = n;
    }
    file28.close();
    //16
    for(int i = 0; i < (960/4 + 1); i++){
        uint16 n;
        file29 >> n;
        pw16_first_1x1_block_col_r_raw[i] = n;
    }
    file29.close();
    for(int i = 0; i < (160/4 + 1); i++){
        uint16 n;
        file30 >> n;
        pw16_last_1x1_block_col_r_raw[i] = n;
    }
    file30.close();
    //17
    for(int i = 0; i < (960/4 + 1); i++){
        uint16 n;
        file31 >> n;
        pw17_first_1x1_block_col_r_raw[i] = n;
    }
    file31.close();
    for(int i = 0; i < (320/4 + 1); i++){
        uint16 n;
        file32 >> n;
        pw17_last_1x1_block_col_r_raw[i] = n;
    }
    file32.close();
    //18
    for(int i = 0; i < (1280/4 + 1); i++){
        uint16 n;
        file33 >> n;
        conv_1x1_block_col_r_raw[i] = n;
    }
    file33.close();
}



int main(int argc, char* argv[])
{
    //int arr_in[N], arr_out[N];
    int retval = 0, i;
    float tmp1, tmp2, tmp3, tmp4;
    FILE *fp;

    std::cout << "Current directory: " << system("pwd") << " \n";

    //TARGET_DEVICE macro needs to be passed from gcc command line
    if(argc != 2) {
		std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
		return EXIT_FAILURE;
	}

    char* xclbinFilename = argv[1];
    
    // Compute the size of array in bytes
    //size_t size_in_bytes = DATA_SIZE * sizeof(int);
    size_t size_in_bytes_fm_1 = 144*56*56 * sizeof(FIX_F);
    size_t size_in_bytes_fm_2 = 144*56*56 * sizeof(FIX_F);
    size_t size_in_bytes_block_c = 320*1280/8 * sizeof(uint9);
    size_t size_in_bytes_block_col_r = (1280/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_weight_1x1 = 320*1280/8*4 * sizeof(FIX_W);
    size_t size_in_bytes_weight_3x3 = 960*3*3 * sizeof(FIX_W);
    size_t size_in_bytes_bn = 4*1280 * sizeof(FIX_W);
    std::cout << " size_in_bytes_fm_1 = '" << size_in_bytes_fm_1 << "'\n";
    std::cout << " size_in_bytes_fm_2 = '" << size_in_bytes_fm_2 << "'\n";
    std::cout << " size_in_bytes_block_c = '" << size_in_bytes_block_c << "'\n";
    std::cout << " size_in_bytes_block_col_r = '" << size_in_bytes_block_col_r << "'\n";
    std::cout << " size_in_bytes_weight_1x1 = '" << size_in_bytes_weight_1x1 << "'\n";
    std::cout << " size_in_bytes_weight_3x3 = '" << size_in_bytes_weight_3x3 << "'\n";
    std::cout << " size_in_bytes_bn = '" << size_in_bytes_bn << "'\n";
    std::cout << " sizeof(float) = '" << sizeof(float) << "'\n";

    
    std::vector<cl::Device> devices;
    cl::Device device;
    std::vector<cl::Platform> platforms;
    bool found_device = false;

    //traversing all Platforms To find Xilinx Platform and targeted
    //Device in Xilinx Platform
    cl::Platform::get(&platforms);
    for(size_t i = 0; (i < platforms.size() ) & (found_device == false) ;i++){
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if ( platformName == "Xilinx"){
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
	    if (devices.size()){
		    device = devices[0];
		    found_device = true;
		    break;
	    }
        }
    }
    if (found_device == false){
       std::cout << "Error: Unable to find Target Device " 
           << device.getInfo<CL_DEVICE_NAME>() << std::endl;
       return EXIT_FAILURE; 
    }

    // Creating Context and Command Queue for selected device
    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
    // cl::CommandQueue q(context, device);

    // Load xclbin 
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);
    
    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf, nb});
    devices.resize(1);
    cl::Program program(context, devices, bins);
    
    // This call will get the kernel object from program. A kernel is an 
    // OpenCL function that is executed on the FPGA. 
    cl::Kernel krnl_network(program,"network");

    // These commands will allocate memory on the Device. The cl::Buffer objects can
    // be used to reference the memory locations on the device. 
    cl::Buffer buffer_fm_1(context,  CL_MEM_READ_WRITE, size_in_bytes_fm_1);
    cl::Buffer buffer_fm_2(context, CL_MEM_READ_WRITE, size_in_bytes_fm_2);
    cl::Buffer buffer_block_c(context, CL_MEM_READ_WRITE, size_in_bytes_block_c);
    cl::Buffer buffer_block_col_r(context,  CL_MEM_READ_WRITE, size_in_bytes_block_col_r);
    cl::Buffer buffer_weight_1x1(context,  CL_MEM_READ_WRITE, size_in_bytes_weight_1x1);
    cl::Buffer buffer_weight_3x3(context,  CL_MEM_READ_WRITE, size_in_bytes_weight_3x3);
    cl::Buffer buffer_bn(context,  CL_MEM_READ_WRITE, size_in_bytes_bn);


    /*cl_mem_ext_ptr_t inExt, outExt;  // Declaring two extensions for both buffers
    inExt.flags  = 0|XCL_MEM_TOPOLOGY; // Specify Bank0 Memory for input memory
    outExt.flags = 1|XCL_MEM_TOPOLOGY; // Specify Bank1 Memory for output Memory
    inExt.obj = 0   ; outExt.obj = 0; // Setting Obj and Param to Zero
    inExt.param = 0 ; outExt.param = 0;

    int err;
    //Allocate Buffer in Bank0 of Global Memory for Input Image using Xilinx Extension
    cl_mem buffer_inImage = clCreateBuffer(world.context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
            image_size_bytes, &inExt, &err);
    if (err != CL_SUCCESS){
        std::cout << "Error: Failed to allocate device Memory" << std::endl;
        return EXIT_FAILURE;
    }

    //Allocate Buffer in Bank1 of Global Memory for Input Image using Xilinx Extension
    cl_mem feature_map_raw = clCreateBuffer(world.context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX,
            image_size_bytes, &outExt, NULL);
    if (err != CL_SUCCESS){
        std::cout << "Error: Failed to allocate device Memory" << std::endl;
        return EXIT_FAILURE;
    }

    cl_mem network_output_raw = clCreateBuffer(world.context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX,
            image_size_bytes, &outExt, NULL);
    if (err != CL_SUCCESS){
        std::cout << "Error: Failed to allocate device Memory" << std::endl;
        return EXIT_FAILURE;
    }

    cl_mem fm_16_1_raw = clCreateBuffer(world.context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX,
            image_size_bytes, &outExt, NULL);
    if (err != CL_SUCCESS){
        std::cout << "Error: Failed to allocate device Memory" << std::endl;
        return EXIT_FAILURE;
    }

    cl_mem feature_map_2_raw = clCreateBuffer(world.context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX,
            image_size_bytes, &outExt, NULL);
    if (err != CL_SUCCESS){
        std::cout << "Error: Failed to allocate device Memory" << std::endl;
        return EXIT_FAILURE;
    }*/



    //We then need to map our OpenCL buffers to get the pointers
    FIX_F *fm_buffer_1 = (FIX_F *) q.enqueueMapBuffer (buffer_fm_1 , CL_TRUE , CL_MAP_READ , 0, size_in_bytes_fm_1);
    FIX_F *fm_buffer_2 = (FIX_F *) q.enqueueMapBuffer (buffer_fm_2 , CL_TRUE , CL_MAP_READ , 0, size_in_bytes_fm_2);
    uint9 *block_c_buf = (uint9 *) q.enqueueMapBuffer (buffer_block_c , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_block_c);
    uint16 *block_col_r_buf = (uint16 *) q.enqueueMapBuffer (buffer_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_block_col_r);
    FIX_W *weight_1x1_buffer = (FIX_W *) q.enqueueMapBuffer (buffer_weight_1x1 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_weight_1x1);
    FIX_W *weight_3x3_buffer = (FIX_W *) q.enqueueMapBuffer (buffer_weight_3x3 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_weight_3x3);
    FIX_W *bn_buffer = (FIX_W *) q.enqueueMapBuffer (buffer_bn , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_bn);


    //set the kernel Arguments
    int narg=0;
    krnl_network.setArg(narg++, buffer_fm_1);
    krnl_network.setArg(narg++, buffer_fm_2);
    krnl_network.setArg(narg++, buffer_block_c);
    krnl_network.setArg(narg++, buffer_block_col_r);
    krnl_network.setArg(narg++, buffer_weight_1x1);
    krnl_network.setArg(narg++, buffer_weight_3x3);
    krnl_network.setArg(narg++, buffer_bn);

    // Migrate data to kernel space
    q.enqueueMigrateMemObjects({buffer_fm_1},0/* 0 means from host*/);
    q.enqueueMigrateMemObjects({buffer_fm_2},0/* 0 means from host*/);
    q.enqueueMigrateMemObjects({buffer_block_c},0/* 0 means from host*/);
    q.enqueueMigrateMemObjects({buffer_block_col_r},0/* 0 means from host*/);
    q.enqueueMigrateMemObjects({buffer_weight_1x1},0/* 0 means from host*/);
    q.enqueueMigrateMemObjects({buffer_weight_3x3},0/* 0 means from host*/);
    q.enqueueMigrateMemObjects({buffer_bn},0/* 0 means from host*/);
    //q.enqueueWriteBuffer(buffer_in,CL_TRUE,0,size_in_bytes,arr_in);

    //Launch the Kernel
    q.enqueueTask(krnl_network);

    // Migrate FPGA data back to host
    q.enqueueMigrateMemObjects({network_output_raw},CL_MIGRATE_MEM_OBJECT_HOST);
    q.flush();
    q.finish();


    load_block_col_r();
    parameter_recorder();

    /*// Generate Output file
    fp = fopen("output.dat","w");
    if (fp == NULL) {
    	printf("Could not create output.dat. \n");
        retval = 1;
    } else {
        std::cout << "Writing '" << fp << "'\n";
		for (i = 0; i < out_size; i++) {
			fprintf(fp, "%f \n", output[i]);
	        std::cout << "Output array index " << i << " has value " << output[i] << " \n";
		}
		fclose(fp);
    }*/

    std::fstream file_g("golden_data1.txt");
    for(int i = 0; i < 1000; i++){
        float n;
        file_g >> n;
        golden_data[i] = n;
    }
    file_g.close();

    for(int i = 0; i < 31; i++){
    	for(int j = 0; j < 32; j++){
    		hls_output[i*32 + j].range(15, 0) = network_output_raw[i].range(j*16 + 15, j*16);
    	}
    }
    for(int i = 0; i < 8; i++){
    	hls_output[992 + i].range(15, 0) = network_output_raw[31].range(i*16 + 15, i*16);
    }

    /*// Compare the results file with the golden results
    retval = system("diff --brief -w output.dat golden.dat");
    //retval = system("diff --brief -w /home/randyh/hls_lab_host_code/out.dat /home/randyh/hls_lab_host_code/in.dat");
    if (retval != 0) {
	 printf("Test failed  !!!\n");
	 retval=1;
    } else {
	 printf("Test passed !\n");
    }
    //return retval;*/

/*
    //Compare arr_in[i] to arr_out[i] and print results
    for (i=0; i<N; i++){
    	if (arr_in[i] == arr_out[i]) {
    		printf("Match at line %d: %d = %d \n", i, arr_in[i], arr_out[i]);
    	} else {
    		printf("No match at line %d: %d != %d \n", i, arr_in[i], arr_out[i]);
    		retval=1;
    	}
    }//End FOR
*/

    // compare;
    int error = 0;

    for(int i = 0; i < 1000; i++){
        if((float(hls_output[i]) - golden_data[i]) > 0.000001 || (golden_data[i] - float(hls_output[i])) > 0.000001){
            printf("Error!: [%d] %f - > %f\n", i, float(hls_output[i]), golden_data[i]);
            error = 1;
        }
    }

    printf("Comparison Finished!\n");
    /*if(error==1)
    	printf("FAILED!\n");
    else
    	printf("SUCCESS!\n");*/

    float max1 = 0;
    float max2 = 0;
    int index_max_hls;
    int index_max_golden;

    for(int i = 0; i < 1000; i++){
    	if (hls_output[i].to_float() > max1)
    		{
    		max1 = hls_output[i].to_float();
    		index_max_hls = i;
    		}
    }

    std::cout << "hls_output: " << index_max_hls << " " << max1 << std::endl;

    for(int i = 0; i < 1000; i++){
    	if (golden_data[i] > max2)
    		{
    		max2 = golden_data[i];
    		index_max_golden = i;
    		}
    }

    std::cout << "golden_data: " << index_max_golden << " " << max2 << std::endl;

    if(index_max_hls == index_max_golden)
        printf("SUCCESS!\n");
    else
        printf("FAILED!\n");

    return 0;

    // Free memory and garbage collect
    q.enqueueUnmapMemObject(buffer_fm_1, fm_buffer_1);
    q.enqueueUnmapMemObject(buffer_fm_2, fm_buffer_2);
    q.enqueueUnmapMemObject(buffer_block_c, block_c_buf);
    q.enqueueUnmapMemObject(buffer_block_col_r, block_col_r_buf);
    q.enqueueUnmapMemObject(buffer_weight_1x1, weight_1x1_buffer);
    q.enqueueUnmapMemObject(buffer_weight_3x3, weight_3x3_buffer);
    q.enqueueUnmapMemObject(buffer_bn, bn_buffer);
    q.finish();

    //std::cout << "TEST " << (retval ? "FAILED" : "PASSED") << std::endl;
    //return (retval ? EXIT_FAILURE :  EXIT_SUCCESS);

}