#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>
#include "host.hpp"
#include <CL/cl2.hpp>
#include <ap_fixed.h>
#include <ap_int.h>
#include <cl_ext_xilinx.h>
//#include <CL/cl_ext.h>

using namespace std;

// OpenCL/XRT host application for the `network` kernel.
// It reads text-format parameter dumps, repacks them into the 512-bit AXI layout
// expected by the accelerator, launches the bitstream, and compares the result
// with a golden top-1 classification reference.

static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

static std::string resolve_data_path(const char* filename) {
    const char* env_dir = std::getenv("BSC_DATA_DIR");
    const std::string env_prefix = env_dir ? std::string(env_dir) : std::string();
    const std::string runtime_prefix = "data/runtime/";
    const std::string sample_prefix = "data/samples/";

    std::ifstream env_file;
    if (!env_prefix.empty()) {
        const std::string candidate = env_prefix + "/" + filename;
        env_file.open(candidate);
        if (env_file.good()) {
            return candidate;
        }
    }

    std::ifstream runtime_file(runtime_prefix + filename);
    if (runtime_file.good()) {
        return runtime_prefix + filename;
    }

    std::ifstream sample_file(sample_prefix + filename);
    if (sample_file.good()) {
        return sample_prefix + filename;
    }

    return std::string(filename);
}

static std::ifstream open_required_file(const char* filename) {
    const std::string resolved = resolve_data_path(filename);
    std::ifstream file(resolved);
    if (!file.is_open()) {
        std::cerr << "Failed to open required data file: " << resolved << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return file;
}



int main(int argc, char* argv[])
{
    //int arr_in[N], arr_out[N];
    int retval = 0, i;
    float tmp1, tmp2, tmp3, tmp4;
    FILE *fp;

    cl_mem_ext_ptr_t inExt, outExt, inoutExt;  // Declaring two extensions for both buffers
    inExt.flags  = XCL_MEM_DDR_BANK0; // Specify Bank0 Memory for input memory
    outExt.flags = XCL_MEM_DDR_BANK1; // Specify Bank1 Memory for output Memory
    inoutExt.flags  = XCL_MEM_DDR_BANK2;
    inExt.obj = 0   ; outExt.obj = 0; inoutExt.obj = 0;// Setting Obj and Param to Zero
    inExt.param = 0 ; outExt.param = 0; inoutExt.param = 0;

    std::cout << "Current directory: " << system("pwd") << " \n";

    //TARGET_DEVICE macro needs to be passed from gcc command line
    if(argc != 2) {
		std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
		return EXIT_FAILURE;
	}

    char* xclbinFilename = argv[1];
    
    // Compute the size of array in bytes
    //size_t size_in_bytes = DATA_SIZE * sizeof(int);
    size_t size_in_bytes_image_raw = 9408 * sizeof(uint512);
    size_t size_in_bytes_1x1 = 33198 * sizeof(uint512);
    size_t size_in_bytes_block_c = 8300 * sizeof(uint512);
    size_t size_in_bytes_block_c_16 = 1200 * sizeof(uint512);
    size_t size_in_bytes_bn = 2132 * sizeof(uint512);
    size_t size_in_bytes_3x3 = 226*3*3 * sizeof(uint512);
    size_t size_in_bytes_3x3_17 = 960*3*3 * sizeof(uint512);
    size_t size_in_bytes_conv_last_1x1 = 1600 * sizeof(uint512);
    size_t size_in_bytes_bias = 32 * sizeof(uint512);
    size_t size_in_bytes_fm = 5000 * sizeof(uint512);
    size_t size_in_bytes_output = 1000 * sizeof(uint512);
    size_t size_in_bytes_fm_16_1 = 96*112*112 * sizeof(FIX_F);
    size_t size_in_bytes_fm_2 = 96*112*112 * sizeof(FIX_F);
    size_t size_in_bytes_pw1_last_block_col_r = (16/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw2_first_block_col_r = (96/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw2_last_block_col_r = (24/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw3_first_block_col_r = (144/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw3_last_block_col_r = (24/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw4_first_block_col_r = (144/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw4_last_block_col_r = (32/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw5_first_block_col_r = (192/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw5_last_block_col_r = (32/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw6_first_block_col_r = (192/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw6_last_block_col_r = (192/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw7_first_block_col_r = (192/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw7_last_block_col_r = (64/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw8_first_block_col_r = (384/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw8_last_block_col_r = (64/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw9_first_block_col_r = (384/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw9_last_block_col_r = (64/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw10_first_block_col_r = (384/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw10_last_block_col_r = (64/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw11_first_block_col_r = (384/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw11_last_block_col_r = (96/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw12_first_block_col_r = (576/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw12_last_block_col_r = (96/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw13_first_block_col_r = (576/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw13_last_block_col_r = (96/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw14_first_block_col_r = (576/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw14_last_block_col_r = (160/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw15_first_block_col_r = (960/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw15_last_block_col_r = (160/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw16_first_block_col_r = (960/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw16_last_block_col_r = (160/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw17_first_block_col_r = (960/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_pw17_last_block_col_r = (320/4 + 1) * sizeof(uint16);
    size_t size_in_bytes_conv_1x1_block_col_r = (1280/4 + 1) * sizeof(uint16);
    /*std::cout << " size_in_bytes_weights = '" << size_in_bytes_weights << "'\n";
    std::cout << " size_in_bytes_mask = '" << size_in_bytes_mask << "'\n";
    std::cout << " size_in_bytes_input = '" << size_in_bytes_input << "'\n";
    std::cout << " size_in_bytes_bias = '" << size_in_bytes_bias << "'\n";
    std::cout << " size_in_bytes_output = '" << size_in_bytes_output << "'\n";
    std::cout << " sizeof(float) = '" << sizeof(float) << "'\n";*/

    
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
    bins.push_back({buf,nb});
    devices.resize(1);
    cl::Program program(context, devices, bins);
    
    // This call will get the kernel object from program. A kernel is an 
    // OpenCL function that is executed on the FPGA. 
    cl::Kernel krnl_network(program,"network");

    // These commands will allocate memory on the Device. The cl::Buffer objects can
    // be used to reference the memory locations on the device. 
    cl::Buffer buffer_image_raw(context,  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_image_raw, &inExt);
    cl::Buffer buffer_1x1(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_1x1, &inExt);
    cl::Buffer buffer_block_c(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_block_c, &inExt);
    cl::Buffer buffer_block_c_16(context,  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_block_c_16, &inExt);
    cl::Buffer buffer_bn(context,  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_bn, &inExt);
    cl::Buffer buffer_3x3(context,  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_3x3, &inExt);
    cl::Buffer buffer_3x3_17(context,  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_3x3_17, &inExt);
    cl::Buffer buffer_last_1x1_11(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_12(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_13(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_14(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_21(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_22(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_23(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_24(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_31(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_32(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_33(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_34(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_41(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_42(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_43(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_44(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_51(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_52(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_53(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_54(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_61(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_62(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_63(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_64(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_last_1x1_7(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_last_1x1, &inExt);
    cl::Buffer buffer_bias(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_bias, &inExt);
    cl::Buffer buffer_fm(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size_in_bytes_fm, &inoutExt);
    cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_output, &outExt);
    cl::Buffer buffer_fm_16_1(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size_in_bytes_fm_16_1, &inoutExt);
    cl::Buffer buffer_fm_2(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size_in_bytes_fm_2, &inoutExt);
    cl::Buffer buffer_pw1_last_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw1_last_block_col_r, &inExt);
    cl::Buffer buffer_pw2_first_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw2_first_block_col_r, &inExt);
    cl::Buffer buffer_pw2_last_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw2_last_block_col_r, &inExt);
    cl::Buffer buffer_pw3_first_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw3_first_block_col_r, &inExt);
    cl::Buffer buffer_pw3_last_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw3_last_block_col_r, &inExt);
    cl::Buffer buffer_pw4_first_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw4_first_block_col_r, &inExt);
    cl::Buffer buffer_pw4_last_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw4_last_block_col_r, &inExt);
    cl::Buffer buffer_pw5_first_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw5_first_block_col_r, &inExt);
    cl::Buffer buffer_pw5_last_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw5_last_block_col_r, &inExt);
    cl::Buffer buffer_pw6_first_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw6_first_block_col_r, &inExt);
    cl::Buffer buffer_pw6_last_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw6_last_block_col_r, &inExt);
    cl::Buffer buffer_pw7_first_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw7_first_block_col_r, &inExt);
    cl::Buffer buffer_pw7_last_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw7_last_block_col_r, &inExt);
    cl::Buffer buffer_pw8_first_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw8_first_block_col_r, &inExt);
    cl::Buffer buffer_pw8_last_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw8_last_block_col_r, &inExt);
    cl::Buffer buffer_pw9_first_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw9_first_block_col_r, &inExt);
    cl::Buffer buffer_pw9_last_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw9_last_block_col_r, &inExt);
    cl::Buffer buffer_pw10_first_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw10_first_block_col_r, &inExt);
    cl::Buffer buffer_pw10_last_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw10_last_block_col_r, &inExt);
    cl::Buffer buffer_pw11_first_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw11_first_block_col_r, &inExt);
    cl::Buffer buffer_pw11_last_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw11_last_block_col_r, &inExt);
    cl::Buffer buffer_pw12_first_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw12_first_block_col_r, &inExt);
    cl::Buffer buffer_pw12_last_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw12_last_block_col_r, &inExt);
    cl::Buffer buffer_pw13_first_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw13_first_block_col_r, &inExt);
    cl::Buffer buffer_pw13_last_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw13_last_block_col_r, &inExt);
    cl::Buffer buffer_pw14_first_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw14_first_block_col_r, &inExt);
    cl::Buffer buffer_pw14_last_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw14_last_block_col_r, &inExt);
    cl::Buffer buffer_pw15_first_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw15_first_block_col_r, &inExt);
    cl::Buffer buffer_pw15_last_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw15_last_block_col_r, &inExt);
    cl::Buffer buffer_pw16_first_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw16_first_block_col_r, &inExt);
    cl::Buffer buffer_pw16_last_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw16_last_block_col_r, &inExt);
    cl::Buffer buffer_pw17_first_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw17_first_block_col_r, &inExt);
    cl::Buffer buffer_pw17_last_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_pw17_last_block_col_r, &inExt);
    cl::Buffer buffer_conv_1x1_block_col_r(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size_in_bytes_conv_1x1_block_col_r, &inExt);


    //We then need to map our OpenCL buffers to get the pointers
    uint512 *image_raw = (uint512 *) q.enqueueMapBuffer (buffer_image_raw , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_image_raw);
    uint512 *sparse_1x1_weight_raw = (uint512 *) q.enqueueMapBuffer (buffer_1x1 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_1x1);
    uint512 *block_c_raw = (uint512 *) q.enqueueMapBuffer (buffer_block_c , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_block_c);
    uint512 *block_c_16_raw = (uint512 *) q.enqueueMapBuffer (buffer_block_c_16 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_block_c_16);
    uint512 *bn_raw = (uint512 *) q.enqueueMapBuffer (buffer_bn , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_bn);
    uint512 *weight_3x3_raw = (uint512 *) q.enqueueMapBuffer (buffer_3x3 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_3x3);
    uint512 *weight_3x3_17_raw = (uint512 *) q.enqueueMapBuffer (buffer_3x3_17 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_3x3_17);
    uint512 *conv_last_1x1_weight_raw_11 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_11 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_12 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_12 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_13 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_13 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_14 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_14 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_21 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_21 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_22 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_22 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_23 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_23 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_24 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_24 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_31 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_31 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_32 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_32 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_33 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_33 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_34 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_34 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_41 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_41 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_42 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_42 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_43 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_43 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_44 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_44 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_51 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_51 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_52 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_52 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_53 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_53 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_54 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_54 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_61 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_61 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_62 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_62 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_63 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_63 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_64 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_64 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_weight_raw_7 = (uint512 *) q.enqueueMapBuffer (buffer_last_1x1_7 , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_conv_last_1x1);
    uint512 *conv_last_1x1_bias_raw = (uint512 *) q.enqueueMapBuffer (buffer_bias , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_bias);
    uint512 *feature_map_raw = (uint512 *) q.enqueueMapBuffer (buffer_fm , CL_TRUE , CL_MAP_READ | CL_MAP_WRITE , 0, size_in_bytes_fm);
    uint512 *network_output_raw = (uint512 *) q.enqueueMapBuffer (buffer_output , CL_TRUE , CL_MAP_READ , 0, size_in_bytes_output);
    FIX_F *fm_16_1_raw = (FIX_F *) q.enqueueMapBuffer (buffer_fm_16_1 , CL_TRUE , CL_MAP_READ | CL_MAP_WRITE, 0, size_in_bytes_fm_16_1);
    FIX_F *feature_map_2_raw = (FIX_F *) q.enqueueMapBuffer (buffer_fm_2 , CL_TRUE , CL_MAP_READ | CL_MAP_WRITE, 0, size_in_bytes_fm_2);
    uint16 *pw1_last_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw1_last_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw1_last_block_col_r);
    uint16 *pw2_first_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw2_first_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw2_first_block_col_r);
    uint16 *pw2_last_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw2_last_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw2_last_block_col_r);
    uint16 *pw3_first_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw3_first_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw3_first_block_col_r);
    uint16 *pw3_last_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw3_last_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw3_last_block_col_r);
    uint16 *pw4_first_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw4_first_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw4_first_block_col_r);
    uint16 *pw4_last_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw4_last_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw4_last_block_col_r);
    uint16 *pw5_first_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw5_first_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw5_first_block_col_r);
    uint16 *pw5_last_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw5_last_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw5_last_block_col_r);
    uint16 *pw6_first_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw6_first_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw6_first_block_col_r);
    uint16 *pw6_last_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw6_last_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw6_last_block_col_r);
    uint16 *pw7_first_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw7_first_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw7_first_block_col_r);
    uint16 *pw7_last_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw7_last_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw7_last_block_col_r);
    uint16 *pw8_first_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw8_first_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw8_first_block_col_r);
    uint16 *pw8_last_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw8_last_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw8_last_block_col_r);
    uint16 *pw9_first_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw9_first_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw9_first_block_col_r);
    uint16 *pw9_last_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw9_last_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw9_last_block_col_r);
    uint16 *pw10_first_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw10_first_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw10_first_block_col_r);
    uint16 *pw10_last_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw10_last_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw10_last_block_col_r);
    uint16 *pw11_first_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw11_first_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw11_first_block_col_r);
    uint16 *pw11_last_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw11_last_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw11_last_block_col_r);
    uint16 *pw12_first_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw12_first_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw12_first_block_col_r);
    uint16 *pw12_last_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw12_last_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw12_last_block_col_r);
    uint16 *pw13_first_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw13_first_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw13_first_block_col_r);
    uint16 *pw13_last_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw13_last_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw13_last_block_col_r);
    uint16 *pw14_first_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw14_first_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw14_first_block_col_r);
    uint16 *pw14_last_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw14_last_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw14_last_block_col_r);
    uint16 *pw15_first_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw15_first_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw15_first_block_col_r);
    uint16 *pw15_last_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw15_last_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw15_last_block_col_r);
    uint16 *pw16_first_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw16_first_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw16_first_block_col_r);
    uint16 *pw16_last_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw16_last_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw16_last_block_col_r);
    uint16 *pw17_first_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw17_first_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw17_first_block_col_r);
    uint16 *pw17_last_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_pw17_last_block_col_r , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes_pw17_last_block_col_r);
    uint16 *conv_1x1_block_col_r_raw = (uint16 *) q.enqueueMapBuffer (buffer_conv_1x1_block_col_r, CL_TRUE, CL_MAP_WRITE , 0, size_in_bytes_conv_1x1_block_col_r);



//----------------------------------------------------------------------//
    // The host prefers `BSC_DATA_DIR`, then `data/runtime/`, then `data/samples/`,
    // and finally falls back to the current working directory for legacy flows.
    // load image from file;
    std::ifstream fileimg = open_required_file("img1.txt");
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
    std::ifstream file1x1 = open_required_file("1x1_sparse_weight.txt");
    for(int i = 0; i < 32*33198; i++){
        FIX_W n;
        file1x1 >> n;
        sparse_weight_1x1_all[i] = n;
    }
    file1x1.close();

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
    std::ifstream file3x3 = open_required_file("3x3_weight.txt");
    for(int i = 0; i < 226; i++){
        for(int j = 0; j < 32; j++){
            for(int k = 0; k < 3; k++){
                for(int l = 0; l < 3; l++){
                    FIX_W n;
                    file3x3 >> n;
                    weight_3x3_all[i*32*3*3 + j*3*3 + k*3 + l] = n;
                }
            }
        }
    }
    file3x3.close();

    /*FILE* fp1;
    fp1 = fopen("weight_3x3.txt","w");
	for(int i = 0; i < 3*3*3*32; i++){
		fprintf(fp1, "%3.14f \n", weight_3x3_all[i].to_float());
	}

	fclose(fp1);*/

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
				}
				weight_3x3_17_raw[(i*3*3 + m*3 + n)].range(511, 0) = DATA.range(511, 0);
    		}
    	}
    }

    std::cout << "3*3 512bit done" << std::endl;

    // load bias from file;
    std::ifstream fileb = open_required_file("bias.txt");
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
    std::ifstream filei1 = open_required_file("block_c.txt");
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

    std::cout << "block_c 512bit done" << std::endl;


    // load bn from file;
    std::ifstream filebn = open_required_file("sqrt_bn.txt");
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

    std::cout << "bn 512bit done" << std::endl;


    // load the last conv weight from file;
    std::ifstream filelast1 = open_required_file("classifier_1.txt");
    for(int i = 0; i < 160*1280; i++){
        FIX_W n;
        filelast1 >> n;
        conv_last_1x1_weight_1[i] = n;
    }
    filelast1.close();

    std::ifstream filelast2 = open_required_file("classifier_2.txt");
    for(int i = 0; i < 160*1280; i++){
        FIX_W n;
        filelast2 >> n;
        conv_last_1x1_weight_2[i] = n;
    }
    filelast2.close();

    std::ifstream filelast3 = open_required_file("classifier_3.txt");
    for(int i = 0; i < 160*1280; i++){
        FIX_W n;
        filelast3 >> n;
        conv_last_1x1_weight_3[i] = n;
    }
    filelast3.close();

    std::ifstream filelast4 = open_required_file("classifier_4.txt");
    for(int i = 0; i < 160*1280; i++){
        FIX_W n;
        filelast4 >> n;
        conv_last_1x1_weight_4[i] = n;
    }
    filelast4.close();

    std::ifstream filelast5 = open_required_file("classifier_5.txt");
    for(int i = 0; i < 160*1280; i++){
        FIX_W n;
        filelast5 >> n;
        conv_last_1x1_weight_5[i] = n;
    }
    filelast5.close();

    std::ifstream filelast6 = open_required_file("classifier_6.txt");
    for(int i = 0; i < 160*1280; i++){
        FIX_W n;
        filelast6 >> n;
        conv_last_1x1_weight_6[i] = n;
    }
    filelast6.close();

    std::ifstream filelast7 = open_required_file("classifier_7.txt");
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

    std::cout << "conv_last 512bit done" << std::endl;

//----------------------------------------------------------------------//

//----------------------------------------------------------------------//

    std::ifstream file0 = open_required_file("features.1.conv.3.block.col.r.txt");
    std::ifstream file1 = open_required_file("features.2.conv.0.block.col.r.txt");
    std::ifstream file2 = open_required_file("features.2.conv.6.block.col.r.txt");
    std::ifstream file3 = open_required_file("features.3.conv.0.block.col.r.txt");
    std::ifstream file4 = open_required_file("features.3.conv.6.block.col.r.txt");
    std::ifstream file5 = open_required_file("features.4.conv.0.block.col.r.txt");
    std::ifstream file6 = open_required_file("features.4.conv.6.block.col.r.txt");
    std::ifstream file7 = open_required_file("features.5.conv.0.block.col.r.txt");
    std::ifstream file8 = open_required_file("features.5.conv.6.block.col.r.txt");
    std::ifstream file9 = open_required_file("features.6.conv.0.block.col.r.txt");
    std::ifstream file10 = open_required_file("features.6.conv.6.block.col.r.txt");
    std::ifstream file11 = open_required_file("features.7.conv.0.block.col.r.txt");
    std::ifstream file12 = open_required_file("features.7.conv.6.block.col.r.txt");
    std::ifstream file13 = open_required_file("features.8.conv.0.block.col.r.txt");
    std::ifstream file14 = open_required_file("features.8.conv.6.block.col.r.txt");
    std::ifstream file15 = open_required_file("features.9.conv.0.block.col.r.txt");
    std::ifstream file16 = open_required_file("features.9.conv.6.block.col.r.txt");
    std::ifstream file17 = open_required_file("features.10.conv.0.block.col.r.txt");
    std::ifstream file18 = open_required_file("features.10.conv.6.block.col.r.txt");
    std::ifstream file19 = open_required_file("features.11.conv.0.block.col.r.txt");
    std::ifstream file20 = open_required_file("features.11.conv.6.block.col.r.txt");
    std::ifstream file21 = open_required_file("features.12.conv.0.block.col.r.txt");
    std::ifstream file22 = open_required_file("features.12.conv.6.block.col.r.txt");
    std::ifstream file23 = open_required_file("features.13.conv.0.block.col.r.txt");
    std::ifstream file24 = open_required_file("features.13.conv.6.block.col.r.txt");
    std::ifstream file25 = open_required_file("features.14.conv.0.block.col.r.txt");
    std::ifstream file26 = open_required_file("features.14.conv.6.block.col.r.txt");
    std::ifstream file27 = open_required_file("features.15.conv.0.block.col.r.txt");
    std::ifstream file28 = open_required_file("features.15.conv.6.block.col.r.txt");
    std::ifstream file29 = open_required_file("features.16.conv.0.block.col.r.txt");
    std::ifstream file30 = open_required_file("features.16.conv.6.block.col.r.txt");
    std::ifstream file31 = open_required_file("features.17.conv.0.block.col.r.txt");
    std::ifstream file32 = open_required_file("features.17.conv.6.block.col.r.txt");
    std::ifstream file33 = open_required_file("features.18.0.block.col.r.txt");

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

    std::cout << "block_col_r done" << std::endl;


//----------------------------------------------------------------------//
    

    //set the kernel Arguments
    int narg=0;
    krnl_network.setArg(narg++, buffer_image_raw);
    krnl_network.setArg(narg++, buffer_1x1);
    krnl_network.setArg(narg++, buffer_block_c);
    krnl_network.setArg(narg++, buffer_block_c_16);
    krnl_network.setArg(narg++, buffer_bn);
    krnl_network.setArg(narg++, buffer_3x3);
    krnl_network.setArg(narg++, buffer_3x3_17);
    krnl_network.setArg(narg++, buffer_last_1x1_11);
    krnl_network.setArg(narg++, buffer_last_1x1_12);
    krnl_network.setArg(narg++, buffer_last_1x1_13);
    krnl_network.setArg(narg++, buffer_last_1x1_14);
    krnl_network.setArg(narg++, buffer_last_1x1_21);
    krnl_network.setArg(narg++, buffer_last_1x1_22);
    krnl_network.setArg(narg++, buffer_last_1x1_23);
    krnl_network.setArg(narg++, buffer_last_1x1_24);
    krnl_network.setArg(narg++, buffer_last_1x1_31);
    krnl_network.setArg(narg++, buffer_last_1x1_32);
    krnl_network.setArg(narg++, buffer_last_1x1_33);
    krnl_network.setArg(narg++, buffer_last_1x1_34);
    krnl_network.setArg(narg++, buffer_last_1x1_41);
    krnl_network.setArg(narg++, buffer_last_1x1_42);
    krnl_network.setArg(narg++, buffer_last_1x1_43);
    krnl_network.setArg(narg++, buffer_last_1x1_44);
    krnl_network.setArg(narg++, buffer_last_1x1_51);
    krnl_network.setArg(narg++, buffer_last_1x1_52);
    krnl_network.setArg(narg++, buffer_last_1x1_53);
    krnl_network.setArg(narg++, buffer_last_1x1_54);
    krnl_network.setArg(narg++, buffer_last_1x1_61);
    krnl_network.setArg(narg++, buffer_last_1x1_62);
    krnl_network.setArg(narg++, buffer_last_1x1_63);
    krnl_network.setArg(narg++, buffer_last_1x1_64);
    krnl_network.setArg(narg++, buffer_last_1x1_7);
    krnl_network.setArg(narg++, buffer_bias);
    krnl_network.setArg(narg++, buffer_fm);
    krnl_network.setArg(narg++, buffer_output);
    krnl_network.setArg(narg++, buffer_fm_16_1);
    krnl_network.setArg(narg++, buffer_fm_2);
    krnl_network.setArg(narg++, buffer_pw1_last_block_col_r);
    krnl_network.setArg(narg++, buffer_pw2_first_block_col_r);
    krnl_network.setArg(narg++, buffer_pw2_last_block_col_r);
    krnl_network.setArg(narg++, buffer_pw3_first_block_col_r);
    krnl_network.setArg(narg++, buffer_pw3_last_block_col_r);
    krnl_network.setArg(narg++, buffer_pw4_first_block_col_r);
    krnl_network.setArg(narg++, buffer_pw4_last_block_col_r);
    krnl_network.setArg(narg++, buffer_pw5_first_block_col_r);
    krnl_network.setArg(narg++, buffer_pw5_last_block_col_r);
    krnl_network.setArg(narg++, buffer_pw6_first_block_col_r);
    krnl_network.setArg(narg++, buffer_pw6_last_block_col_r);
    krnl_network.setArg(narg++, buffer_pw7_first_block_col_r);
    krnl_network.setArg(narg++, buffer_pw7_last_block_col_r);
    krnl_network.setArg(narg++, buffer_pw8_first_block_col_r);
    krnl_network.setArg(narg++, buffer_pw8_last_block_col_r);
    krnl_network.setArg(narg++, buffer_pw9_first_block_col_r);
    krnl_network.setArg(narg++, buffer_pw9_last_block_col_r);
    krnl_network.setArg(narg++, buffer_pw10_first_block_col_r);
    krnl_network.setArg(narg++, buffer_pw10_last_block_col_r);
    krnl_network.setArg(narg++, buffer_pw11_first_block_col_r);
    krnl_network.setArg(narg++, buffer_pw11_last_block_col_r);
    krnl_network.setArg(narg++, buffer_pw12_first_block_col_r);
    krnl_network.setArg(narg++, buffer_pw12_last_block_col_r);
    krnl_network.setArg(narg++, buffer_pw13_first_block_col_r);
    krnl_network.setArg(narg++, buffer_pw13_last_block_col_r);
    krnl_network.setArg(narg++, buffer_pw14_first_block_col_r);
    krnl_network.setArg(narg++, buffer_pw14_last_block_col_r);
    krnl_network.setArg(narg++, buffer_pw15_first_block_col_r);
    krnl_network.setArg(narg++, buffer_pw15_last_block_col_r);
    krnl_network.setArg(narg++, buffer_pw16_first_block_col_r);
    krnl_network.setArg(narg++, buffer_pw16_last_block_col_r);
    krnl_network.setArg(narg++, buffer_pw17_first_block_col_r);
    krnl_network.setArg(narg++, buffer_pw17_last_block_col_r);
    krnl_network.setArg(narg++, buffer_conv_1x1_block_col_r);


    // Migrate data to kernel space
    q.enqueueMigrateMemObjects({buffer_image_raw},0/* 0 means from host*/);
    q.enqueueMigrateMemObjects({buffer_1x1},0/* 0 means from host*/);
    q.enqueueMigrateMemObjects({buffer_block_c},0/* 0 means from host*/);
    q.enqueueMigrateMemObjects({buffer_block_c_16},0/* 0 means from host*/);
    q.enqueueMigrateMemObjects({buffer_bn},0);
    q.enqueueMigrateMemObjects({buffer_3x3},0);
    q.enqueueMigrateMemObjects({buffer_3x3_17},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_11},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_12},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_13},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_14},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_21},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_22},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_23},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_24},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_31},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_32},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_33},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_34},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_41},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_42},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_43},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_44},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_51},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_52},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_53},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_54},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_61},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_62},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_63},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_64},0);
    q.enqueueMigrateMemObjects({buffer_last_1x1_7},0);
    q.enqueueMigrateMemObjects({buffer_bias},0);
    q.enqueueMigrateMemObjects({buffer_fm},0);
    q.enqueueMigrateMemObjects({buffer_output},0);
    q.enqueueMigrateMemObjects({buffer_fm_16_1},0);
    q.enqueueMigrateMemObjects({buffer_fm_2},0);
    q.enqueueMigrateMemObjects({buffer_pw1_last_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw2_first_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw2_last_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw3_first_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw3_last_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw4_first_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw4_last_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw5_first_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw5_last_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw6_first_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw6_last_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw7_first_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw7_last_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw8_first_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw8_last_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw9_first_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw9_last_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw10_first_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw10_last_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw11_first_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw11_last_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw12_first_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw12_last_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw13_first_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw13_last_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw14_first_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw14_last_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw15_first_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw15_last_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw16_first_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw16_last_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw17_first_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_pw17_last_block_col_r},0);
    q.enqueueMigrateMemObjects({buffer_conv_1x1_block_col_r},0);

    //Launch the Kernel
    q.enqueueTask(krnl_network);

    // Migrate FPGA data back to host
    q.enqueueMigrateMemObjects({buffer_fm},CL_MIGRATE_MEM_OBJECT_HOST);
    q.flush();
    q.finish();
    q.enqueueMigrateMemObjects({buffer_fm_16_1},CL_MIGRATE_MEM_OBJECT_HOST);
    q.flush();
    q.finish();
    q.enqueueMigrateMemObjects({buffer_fm_2},CL_MIGRATE_MEM_OBJECT_HOST);
    q.flush();
    q.finish();
    q.enqueueMigrateMemObjects({buffer_output},CL_MIGRATE_MEM_OBJECT_HOST);
    q.flush();
    q.finish();


    // Generate Output file


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

    // Free memory and garbage collect
    q.enqueueUnmapMemObject(buffer_image_raw , image_raw);
    q.enqueueUnmapMemObject(buffer_1x1, sparse_1x1_weight_raw);
    q.enqueueUnmapMemObject(buffer_block_c, block_c_raw);
    q.enqueueUnmapMemObject(buffer_block_c_16, block_c_16_raw);
    q.enqueueUnmapMemObject(buffer_bn, bn_raw);
    q.enqueueUnmapMemObject(buffer_3x3, weight_3x3_raw);
    q.enqueueUnmapMemObject(buffer_3x3_17, weight_3x3_17_raw);
    q.enqueueUnmapMemObject(buffer_last_1x1_11, conv_last_1x1_weight_raw_11);
    q.enqueueUnmapMemObject(buffer_last_1x1_12, conv_last_1x1_weight_raw_12);
    q.enqueueUnmapMemObject(buffer_last_1x1_13, conv_last_1x1_weight_raw_13);
    q.enqueueUnmapMemObject(buffer_last_1x1_14, conv_last_1x1_weight_raw_14);
    q.enqueueUnmapMemObject(buffer_last_1x1_21, conv_last_1x1_weight_raw_21);
    q.enqueueUnmapMemObject(buffer_last_1x1_22, conv_last_1x1_weight_raw_22);
    q.enqueueUnmapMemObject(buffer_last_1x1_23, conv_last_1x1_weight_raw_23);
    q.enqueueUnmapMemObject(buffer_last_1x1_24, conv_last_1x1_weight_raw_24);
    q.enqueueUnmapMemObject(buffer_last_1x1_31, conv_last_1x1_weight_raw_31);
    q.enqueueUnmapMemObject(buffer_last_1x1_32, conv_last_1x1_weight_raw_32);
    q.enqueueUnmapMemObject(buffer_last_1x1_33, conv_last_1x1_weight_raw_33);
    q.enqueueUnmapMemObject(buffer_last_1x1_34, conv_last_1x1_weight_raw_34);
    q.enqueueUnmapMemObject(buffer_last_1x1_41, conv_last_1x1_weight_raw_41);
    q.enqueueUnmapMemObject(buffer_last_1x1_42, conv_last_1x1_weight_raw_42);
    q.enqueueUnmapMemObject(buffer_last_1x1_43, conv_last_1x1_weight_raw_43);
    q.enqueueUnmapMemObject(buffer_last_1x1_44, conv_last_1x1_weight_raw_44);
    q.enqueueUnmapMemObject(buffer_last_1x1_51, conv_last_1x1_weight_raw_51);
    q.enqueueUnmapMemObject(buffer_last_1x1_52, conv_last_1x1_weight_raw_52);
    q.enqueueUnmapMemObject(buffer_last_1x1_53, conv_last_1x1_weight_raw_53);
    q.enqueueUnmapMemObject(buffer_last_1x1_54, conv_last_1x1_weight_raw_54);
    q.enqueueUnmapMemObject(buffer_last_1x1_61, conv_last_1x1_weight_raw_61);
    q.enqueueUnmapMemObject(buffer_last_1x1_62, conv_last_1x1_weight_raw_62);
    q.enqueueUnmapMemObject(buffer_last_1x1_63, conv_last_1x1_weight_raw_63);
    q.enqueueUnmapMemObject(buffer_last_1x1_64, conv_last_1x1_weight_raw_64);
    q.enqueueUnmapMemObject(buffer_last_1x1_7, conv_last_1x1_weight_raw_7);
    q.enqueueUnmapMemObject(buffer_bias, conv_last_1x1_bias_raw);
    q.enqueueUnmapMemObject(buffer_fm, feature_map_raw);
    q.enqueueUnmapMemObject(buffer_output, network_output_raw);
    q.enqueueUnmapMemObject(buffer_fm_16_1, fm_16_1_raw);
    q.enqueueUnmapMemObject(buffer_fm_2, feature_map_2_raw);
    q.enqueueUnmapMemObject(buffer_pw1_last_block_col_r, pw1_last_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw2_first_block_col_r, pw2_first_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw2_last_block_col_r, pw2_last_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw3_first_block_col_r, pw3_first_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw3_last_block_col_r, pw3_last_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw4_first_block_col_r, pw4_first_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw4_last_block_col_r, pw4_last_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw5_first_block_col_r, pw5_first_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw5_last_block_col_r, pw5_last_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw6_first_block_col_r, pw6_first_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw6_last_block_col_r, pw6_last_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw7_first_block_col_r, pw7_first_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw7_last_block_col_r, pw7_last_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw8_first_block_col_r, pw8_first_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw8_last_block_col_r, pw8_last_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw9_first_block_col_r, pw9_first_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw9_last_block_col_r, pw9_last_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw10_first_block_col_r, pw10_first_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw10_last_block_col_r, pw10_last_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw11_first_block_col_r, pw11_first_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw11_last_block_col_r, pw11_last_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw12_first_block_col_r, pw12_first_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw12_last_block_col_r, pw12_last_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw13_first_block_col_r, pw13_first_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw13_last_block_col_r, pw13_last_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw14_first_block_col_r, pw14_first_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw14_last_block_col_r, pw14_last_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw15_first_block_col_r, pw15_first_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw15_last_block_col_r, pw15_last_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw16_first_block_col_r, pw16_first_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw16_last_block_col_r, pw16_last_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw17_first_block_col_r, pw17_first_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_pw17_last_block_col_r, pw17_last_1x1_block_col_r_raw);
    q.enqueueUnmapMemObject(buffer_conv_1x1_block_col_r, conv_1x1_block_col_r_raw);
    q.finish();


    std::ifstream file_g = open_required_file("golden_data1.txt");
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


    //std::cout << "TEST " << (retval ? "FAILED" : "PASSED") << std::endl;
    //return (retval ? EXIT_FAILURE :  EXIT_SUCCESS);

}
