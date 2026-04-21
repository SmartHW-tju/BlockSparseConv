#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <ap_fixed.h>
#include <ap_int.h>

#include "network.hpp"

static FIX_W 1x1_sparse_weight_all[32*33198];
static FIX_W 3x3_weight_all[226][32][3][3];
static FIX_W bias_all[1000];
static uint9 block_c_all[265584];
static uint16 block_col_r_all[2506];
static FIX_W bn_all[32*2132];
static FIX_W conv_last_1x1_weight_all[1000*1280];

static uint512 1x1_sparse_weight_all_512bit[33198];
static uint512 3x3_weight_all_512bit[226][3][3];
static uint512 bias_all_512bit[32];
static uint512 block_c_all_512bit[8300];// 32*8300=265600;
static uint512 block_col_r_all_512bit[79]; // 32*79=2528;
static uint512 bn_all_512bit[2132];
static uint512 conv_last_1x1_weight_all_512bit[40000];

void parameter_reorder()
{
    // load 1*1 sparse weight from file;
    std::fstream file11("1x1_sparse_weight.txt");
    for(int i = 0; i < 32*33198; i++){
        FIX_W n;
        file11 >> n;
        1x1_sparse_weight_all[i] = n;
    }
    file11.close();

    // fill 1*1 sparse weight into 512 bit-width bus;
    // 32*33198;
    for(int i = 0; i < 33198; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = 1x1_sparse_weight_all[i*32 + j].range(15, 0);
    	}
    	1x1_sparse_weight_all_512bit[i].range(511, 0) = DATA.range(511, 0);
    }

    
    // load 3*3 weight from file;
    std::fstream file33("3x3_weight.txt");
    for(int i = 0; i < 226; i++){
        for(int j = 0; j < 32; j++){
            for(int k = 0; k < 3; k++){
                for(int l = 0; l < 3; l++){
                    FIX_W n;
                    file33 >> n;
                    3x3_weight_all[i][j][k][l] = n;
                }
            }
        }
    }
    file33.close();

    /*// fill 3*3 weight into 512 bit-width bus;
    // 32*2034;
    for(int i = 0; i < 2034; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = 3x3_weight_all[i*32 + j].range(15, 0);
    	}
    	3x3_weight_all_512bit[i].range(511, 0) = DATA.range(511, 0);
    }*/

    for(int i = 0; i < 226; i++) {
    	for(int m = 0; m < 3; m++) {
    		for(int n = 0; n < 3; n++) {
                uint512 DATA = 0;
				for(int j = 0; j < 32; j++) {
					DATA.range(j*16 + 15, j*16) = 3x3_weight_all[i][j][m][n].range(15, 0);
				}
				3x3_weight_all_512bit[i][m][n].range(511, 0) = DATA.range(511, 0);
    		}
    	}
    }


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
    	bias_all_512bit[i].range(511, 0) = DATA.range(511, 0);
    }
    
    uint512 DATA1 = 0;
    for(int i = 0; i < 8; i++) {
    	DATA1.range(i*16 + 15, i*16) = bias_all[992 + i].range(15, 0);
    }
    	bias_all_512bit[31].range(511, 0) = DATA1.range(511, 0);


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
    for(int i = 0; i < 58; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 8, j*16) = block_c_all[i*32 + j].range(8, 0);
    	}
    	block_c_all_512bit[i].range(511, 0) = DATA.range(511, 0);
    }

    for(int i = 0; i < 16; i++){
        block_c_all_512bit[58].range(j*16 + 8, 0) = block_c_all[1408 + j].range(8, 0);;
    }

    for(int i = 59; i < 8230; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 8, j*16) = block_c_all[i*32 + j].range(8, 0);
    	}
    	block_c_all_512bit[i].range(511, 0) = DATA.range(511, 0);
    }


    // load block_col_r from file;
    std::fstream filei2("block_col_r.txt");
    for(int i = 0; i < 2506; i++){
        uint16 n;
        filei2 >> n;
        block_c_all[i] = n;
    }
    filei1.close();

    // fill block_col_r into 512 bit-width bus;
    // 32*79;
    for(int i = 0; i < 78; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = block_c_all[i*32 + j].range(15, 0);
    	}
    	block_c_all_512bit[i].range(511, 0) = DATA.range(511, 0);
    }

    uint512 DATA3 = 0;
    for(int i = 0; i < 11; i++) {
    	DATA3.range(i*16 + 15, i*16) = bias_all[2495 + i].range(15, 0);
    }
    bias_all_512bit[78].range(511, 0) = DATA1.range(511, 0);


    // load bn from file;
    std::fstream filebn("bn.txt");
    for(int i = 0; i < 32*2132; i++){
        FIX_W n;
        filebn >> n;
        bn_all[i] = n;
    }
    filebn.close();

    // fill 1*1 sparse weight into 512 bit-width bus;
    // 32*2132;
    for(int i = 0; i < 2132; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = bn_all[i*32 + j].range(15, 0);
    	}
    	bn_all_512bit[i].range(511, 0) = DATA.range(511, 0);
    }


    // load the last conv weight from file;
    std::fstream filelast("classifier.txt");
    for(int i = 0; i < 32*2132; i++){
        FIX_W n;
        filelast >> n;
        conv_last_1x1_weight_all[i] = n;
    }
    filebn.close();

    // fill the last conv weight into 512 bit-width bus;
    // 32*40000;
    for(int i = 0; i < 40000; i++) {
    	uint512 DATA = 0;
    	for(int j = 0; j < 32; j ++) {
    		DATA.range(j*16 + 15, j*16) = conv_last_1x1_weight_all[i*32 + j].range(15, 0);
    	}
    	conv_last_1x1_weight_all_512bit[i].range(511, 0) = DATA.range(511, 0);
    }
}