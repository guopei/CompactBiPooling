#include "luaT.h"
#include "THC/THC.h"
#include <cufft.h>

// extract fft(real to complex) and ifft(complex to real) from spectral-lib code 
// and register them to a new lib.

static int fft1d_r2c(lua_State *L) {
    THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");
    THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");

    luaL_argcheck(L, input->nDimension == 2, 2, "input should be 2D real tensor [nLines x N]");
    luaL_argcheck(L, output->nDimension == 3, 2, "output should be 2D complex tensor [nLines x (N/2+1) x 2]");
    
    long nInputLines = input->size[0];
    long N = input->size[1];

    // argument check
    luaL_argcheck(L, output->size[0] == nInputLines, 0, "input and output should have the same number of lines");
    luaL_argcheck(L, (N % 2) == 0, 0, "N should be multiple of 2");
    luaL_argcheck(L, output->size[1] == N/2+1, 0, "output should be N/2+1");
    luaL_argcheck(L, output->size[2] == 2, 0, "output should be complex");
    luaL_argcheck(L, THCudaTensor_isContiguous(NULL,input), 2, "input must be contiguous");
    luaL_argcheck(L, THCudaTensor_isContiguous(NULL,output), 2, "output must be contiguous");
    

    // raw pointers 
    float *input_data = THCudaTensor_data(NULL,input);
    cuComplex *output_data = (cuComplex*)THCudaTensor_data(NULL,output);
    
    // execute FFT
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_R2C, nInputLines);
    cufftExecR2C(plan, (cufftReal*)input_data, (cufftComplex*)output_data);

    // clean up
    cufftDestroy(plan);

    return 0;
}

static int fft1d_c2r(lua_State *L) {
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");	
	THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");

	luaL_argcheck(L, output->nDimension == 2, 2, "output should be 2D real tensor [nLines x N]");
   	luaL_argcheck(L, input->nDimension == 3, 2, "input should be 2D complex tensor [nLines x (N/2+1) x 2]");
	
	long nInputLines = input->size[0];
	long N = output->size[1];

	// argument check
	luaL_argcheck(L, output->size[0] == nInputLines, 0, "input and output should have the same number of lines");
	luaL_argcheck(L, (N % 2) == 0, 0, "N should be multiple of 2");
	luaL_argcheck(L, input->size[1] == N/2+1, 0, "input should be N/2+1");
	luaL_argcheck(L, input->size[2] == 2, 0, "input should be complex");
	luaL_argcheck(L, THCudaTensor_isContiguous(NULL,input), 2, "input must be contiguous");
	luaL_argcheck(L, THCudaTensor_isContiguous(NULL,output), 2, "output must be contiguous");
	

	// raw pointers 
	float *output_data = THCudaTensor_data(NULL,output);
	cuComplex *input_data = (cuComplex*)THCudaTensor_data(NULL,input);
	
	// execute FFT
	cufftHandle plan;
	cufftPlan1d(&plan, N, CUFFT_C2R, nInputLines);
	cufftExecC2R(plan, (cufftComplex*)input_data, (cufftReal*)output_data);

	// clean up
	cufftDestroy(plan);
	return 0;	
}

static const struct luaL_reg cudafft_init [] = {
    {"fft1d_r2c", fft1d_r2c},
    {"fft1d_c2r", fft1d_c2r},
    {NULL, NULL}
};

LUA_EXTERNC int luaopen_libcudafft(lua_State *L) {
    luaL_openlib(L, "cudafft", cudafft_init, 0);
    lua_pop(L,1);
    return 1;
}
