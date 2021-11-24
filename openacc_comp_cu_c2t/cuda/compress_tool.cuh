#ifndef COMPRESS_TOOL
#define COMPRESS_TOOL

#include <cuda.h>
#include <cuda_runtime.h>
#include "cub_wrapper.cuh"

__device__  size_t chk_size;//8;//16;//32;
__device__  size_t len; // 4GB
__device__  size_t num_chk, totLeadNums, totBit2ByteNum, totMidByteNum, compBufSize, midByteOffsetPacked;
__device__  double pwrpr;
__device__  size_t t_size ; //1024: 12800 ; 16: 710000 4GB: 355000
__device__  size_t block_size;
// __managed__  size_t grid_size = num_chk/block_size;
__device__  short reqLen;
__device__  int reqBytesLen,resiBitsLen;
__device__  size_t midByteOffset;
__device__  double last_number;
extern "C"
void set_GPUID(int id);
extern "C"
void initialize_value(size_t* h_len,size_t* h_chk_size, size_t* h_num_chk, double* h_pwrpr, size_t* h_t_size, short* h_reqLen,\
     int* h_reqBytesLen, int* h_resiBitsLen, size_t* h_totLeadNums, size_t* h_totBit2ByteNum, size_t *h_totMidByteNum,\
     size_t* h_compBufSize,size_t* h_midByteOffset, size_t* h_midByteOffsetPacked,size_t* h_block_size);
extern "C"
void HostMemAlloc(void ** pHost, size_t size);
extern "C"
void DeviceMemAlloc(void ** dHost, size_t size);
extern "C"
void HostMemFree(void * pHost);
extern "C"
void DeviceMemFree(void * dHost);
extern "C"
float Host2Device(void* dst, const void* src, size_t count);
extern "C"
float Device2Host(void* dst, const void* src, size_t count);
extern "C"
float comp_p1(double* uncompBuffer, unsigned char* compBuffer, size_t* prefixSum, size_t* prefSums,\
     unsigned char* midBytes4Packing,const size_t grid_size, const size_t block_size);
extern "C"
float comp_p2(unsigned char* compBuffer, size_t* prefixSum, size_t* prefSums,\
     unsigned char* midBytes4Packing,const size_t grid_size, const size_t block_size);
extern "C"
float decomp_p1(unsigned char* compBuffer, size_t* prefixSum, double* local_prefSum,\
    const size_t grid_size, const size_t block_size);
extern "C"
float decomp_p2(double* uncompBuffer, unsigned char* compBuffer, size_t* prefSums,\
     const size_t grid_size, const size_t block_size);

#endif