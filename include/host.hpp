#include <ap_fixed.h>
#include <ap_int.h>
//#include <vector>
#include <CL/cl2.hpp>


typedef ap_fixed<16, 4, AP_RND, AP_SAT> FIX_W;
typedef ap_ufixed<32, 4> FIX_bn;
typedef ap_fixed<16, 11, AP_RND, AP_SAT> FIX_F;
typedef ap_uint<10> uint9;
typedef ap_uint<16> uint16;
typedef ap_uint<512> uint512;

//#define NO_SYNTH


#ifdef NO_SYNTH

	FIX_F *image = (FIX_F*)malloc(224*224*3*sizeof(FIX_F));
	FIX_W *sparse_weight_1x1_all = (FIX_W*)malloc(32*33198*sizeof(FIX_W));
	FIX_W *weight_3x3_all = (FIX_W*)malloc(226*32*3*3*sizeof(FIX_W));
	FIX_W *bias_all = (FIX_W*)malloc(1000*sizeof(FIX_W));
	uint9 *block_c_all = (uint9*)malloc(265584*sizeof(uint9));
	FIX_W *bn_all = (FIX_W*)malloc(32*2132*sizeof(FIX_W));
	FIX_W *conv_last_1x1_weight_1 = (FIX_W*)malloc(160*1280*sizeof(FIX_W));
	FIX_W *conv_last_1x1_weight_2 = (FIX_W*)malloc(160*1280*sizeof(FIX_W));
	FIX_W *conv_last_1x1_weight_3 = (FIX_W*)malloc(160*1280*sizeof(FIX_W));
	FIX_W *conv_last_1x1_weight_4 = (FIX_W*)malloc(160*1280*sizeof(FIX_W));
	FIX_W *conv_last_1x1_weight_5 = (FIX_W*)malloc(160*1280*sizeof(FIX_W));
	FIX_W *conv_last_1x1_weight_6 = (FIX_W*)malloc(160*1280*sizeof(FIX_W));
	FIX_W *conv_last_1x1_weight_7 = (FIX_W*)malloc(40*1280*sizeof(FIX_W));

	FIX_F *hls_output = (FIX_F*)malloc(1000*sizeof(FIX_F));
	float *golden_data = (float*)malloc(1000*sizeof(float));

#else

	FIX_F image[224*224*3];
	FIX_W sparse_weight_1x1_all[32*33198];
	FIX_W weight_3x3_all[226*32*3*3];
	FIX_W bias_all[1000];
	uint9 block_c_all[265584];
	FIX_W bn_all[32*2132];
	FIX_W conv_last_1x1_weight_1[160*1280];
	FIX_W conv_last_1x1_weight_2[160*1280];
	FIX_W conv_last_1x1_weight_3[160*1280];
	FIX_W conv_last_1x1_weight_4[160*1280];
	FIX_W conv_last_1x1_weight_5[160*1280];
	FIX_W conv_last_1x1_weight_6[160*1280];
	FIX_W conv_last_1x1_weight_7[40*1280];

	FIX_F hls_output[1000];
    float golden_data[1000];

#endif

//Customized buffer allocation for 4K boundary alignment
template <typename T>
struct aligned_allocator
{
  using value_type = T;
  T* allocate(std::size_t num)
  {
    void* ptr = nullptr;
    if (posix_memalign(&ptr,4096,num*sizeof(T)))
      throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t num)
  {
    free(p);
  }
};

namespace xcl {
std::vector<cl::Device> get_xil_devices();
std::vector<cl::Device> get_devices(const std::string& vendor_name);
/* find_xclbin_file
 *
 *
 * Description:
 *   Find precompiled program (as commonly created by the Xilinx OpenCL
 *   flow). Using search path below.
 *
 *   Search Path:
 *      $XCL_BINDIR/<name>.<target>.<device>.xclbin
 *      $XCL_BINDIR/<name>.<target>.<device_versionless>.xclbin
 *      $XCL_BINDIR/binary_container_1.xclbin
 *      $XCL_BINDIR/<name>.xclbin
 *      xclbin/<name>.<target>.<device>.xclbin
 *      xclbin/<name>.<target>.<device_versionless>.xclbin
 *      xclbin/binary_container_1.xclbin
 *      xclbin/<name>.xclbin
 *      ../<name>.<target>.<device>.xclbin
 *      ../<name>.<target>.<device_versionless>.xclbin
 *      ../binary_container_1.xclbin
 *      ../<name>.xclbin
 *      ./<name>.<target>.<device>.xclbin
 *      ./<name>.<target>.<device_versionless>.xclbin
 *      ./binary_container_1.xclbin
 *      ./<name>.xclbin
 *
 * Inputs:
 *   _device_name - Targeted Device name
 *   xclbin_name - base name of the xclbin to import.
 *
 * Returns:
 *   An opencl program Binaries object that was created from xclbin_name file.
 */
std::string find_binary_file(const std::string& _device_name, const std::string& xclbin_name);
cl::Program::Binaries import_binary_file(std::string xclbin_file_name);
bool is_emulation();
bool is_hw_emulation();
bool is_xpr_device(const char* device_name);
}

/*typedef struct{
    unsigned flags;
    void *obj;
    void *param;
  } cl_mem_ext_ptr_t;*/