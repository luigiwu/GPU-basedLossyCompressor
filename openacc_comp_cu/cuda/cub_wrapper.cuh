#ifndef CUB_WRAPPER
#define CUB_WRAPPER


#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include </home/yf-wu/cub/cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>

extern "C" 
size_t cub_prefixSum(size_t *d_in, size_t *d_out, size_t ns, void *d_temp, size_t t_size);








#endif