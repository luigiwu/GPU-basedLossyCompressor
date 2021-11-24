#include "cub_wrapper.cuh"



size_t cub_prefixSum(size_t *d_in, size_t *d_out, size_t ns, void *d_temp, size_t t_size){

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


size_t last_nb = 0;
size_t tot = 0;
void *dtmp = NULL;
size_t tsb = 0;

//void     *d_temp_storage = NULL;
//size_t   temp_storage_bytes = 0;
cub::DeviceScan::ExclusiveSum(dtmp, tsb, d_in, d_out, ns);
// Allocate temporary storage
cudaMalloc(&dtmp, tsb);
printf("tsb= %zu\n",tsb);
cub::DeviceScan::ExclusiveSum(dtmp, tsb, d_in, d_out, ns);

// cub::DeviceScan::ExclusiveSum(d_temp, t_size, d_in, d_out, ns);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cub prefixsum time=%.6f\n", milliseconds/1000.0 );



return last_nb + tot;
}
