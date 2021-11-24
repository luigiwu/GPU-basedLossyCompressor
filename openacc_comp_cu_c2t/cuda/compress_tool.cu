#include "compress_tool.cuh"


// const size_t grid_size2 = reqBytesLen * len/block_size;

typedef union ldouble
{
    double value;
    unsigned long lvalue;
    unsigned char byte[8];
} ldouble;

__device__ inline short getPrecisionReqLength_double(double precision)
{
	ldouble lbuf;
	lbuf.value = precision;
	long lvalue = lbuf.lvalue;
	
	int expValue = (int)((lvalue & 0x7FF0000000000000) >> 52);
	expValue -= 1023;

	return (short)expValue;
}

__device__ inline short computeReqLength_double_MSST19(double realPrecision)
{
	short reqExpo = getPrecisionReqLength_double(realPrecision);
	return 12-reqExpo;
}

__global__ void set_value(){
    size_t K = 1024;
    size_t GB_num = 1;
    size_t byte_num = 8;
    len = GB_num*K/byte_num*K*K;
    chk_size = 32;
    num_chk = len/ chk_size;
    pwrpr = 1E-5;
    t_size = 699903;
    reqLen =  computeReqLength_double_MSST19(pwrpr);
    reqBytesLen = reqLen/8;
	resiBitsLen = reqLen%8;	
    totLeadNums = chk_size;
    if(totLeadNums%4==0)
        totLeadNums = totLeadNums*2/byte_num;
	else
        totLeadNums = totLeadNums*2/byte_num+1;

    totLeadNums *= num_chk;
    // totBit2ByteNum = len;
    totBit2ByteNum = ((resiBitsLen * (chk_size))/byte_num + 1) * (num_chk);
    totMidByteNum = reqBytesLen * (len);
    compBufSize = totLeadNums + totBit2ByteNum + totMidByteNum;
    midByteOffsetPacked = totLeadNums + totBit2ByteNum;
    midByteOffset = 0;
    block_size = 128;
}
void initialize_value(size_t* h_len,size_t* h_chk_size, size_t* h_num_chk, double* h_pwrpr, size_t* h_t_size, short* h_reqLen,\
    int* h_reqBytesLen, int* h_resiBitsLen, size_t* h_totLeadNums, size_t* h_totBit2ByteNum, size_t *h_totMidByteNum,\
    size_t* h_compBufSize,size_t* h_midByteOffset, size_t* h_midByteOffsetPacked,size_t* h_block_size){
    
    set_value<<<1,1>>>();

    // *h_chk_size = 32;
    cudaMemcpyFromSymbol(h_len, len, sizeof(size_t),0, cudaMemcpyDeviceToHost);
    // *h_len = len;
    cudaMemcpyFromSymbol(h_chk_size, chk_size, sizeof(size_t),0, cudaMemcpyDeviceToHost);
    // *h_chk_size = chk_size;
    // cudaDeviceSynchronize();
    
    cudaMemcpyFromSymbol(h_num_chk, num_chk, sizeof(size_t),0, cudaMemcpyDeviceToHost);
    // printf("len = %zu\n",*h_len);

    // *h_num_chk = num_chk;
    cudaMemcpyFromSymbol(h_pwrpr, pwrpr, sizeof(double),0, cudaMemcpyDeviceToHost);
    // *h_pwrpr = pwrpr;
    cudaMemcpyFromSymbol(h_t_size, t_size, sizeof(size_t),0, cudaMemcpyDeviceToHost);
    // *h_t_size = t_size;
    
    
    // cudaMemcpy(h_reqLen, &reqLen, sizeof(short), cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(h_reqLen, reqLen, sizeof(short),0, cudaMemcpyDeviceToHost);

    // *h_reqLen = reqLen;
    
    cudaMemcpyFromSymbol(h_reqBytesLen, reqBytesLen, sizeof(int),0, cudaMemcpyDeviceToHost);  
    // *h_reqBytesLen = reqBytesLen;
    cudaMemcpyFromSymbol(h_resiBitsLen, resiBitsLen, sizeof(int),0, cudaMemcpyDeviceToHost);  
    // *h_resiBitsLen = resiBitsLen;
    cudaMemcpyFromSymbol(h_totLeadNums, totLeadNums, sizeof(size_t),0, cudaMemcpyDeviceToHost);  
    // *h_totLeadNums = totLeadNums;
    cudaMemcpyFromSymbol(h_totBit2ByteNum, totBit2ByteNum, sizeof(size_t),0, cudaMemcpyDeviceToHost); 
    // *h_totBit2ByteNum = totBit2ByteNum;
    cudaMemcpyFromSymbol(h_totMidByteNum, totMidByteNum, sizeof(size_t),0, cudaMemcpyDeviceToHost);  
    // *h_totMidByteNum = totMidByteNum;
    cudaMemcpyFromSymbol(h_compBufSize, compBufSize, sizeof(size_t),0, cudaMemcpyDeviceToHost);  
    // *h_compBufSize = compBufSize;
    cudaMemcpyFromSymbol(h_midByteOffset, midByteOffset, sizeof(size_t),0, cudaMemcpyDeviceToHost); 
    // *h_midByteOffset = midByteOffset;
    cudaMemcpyFromSymbol(h_midByteOffsetPacked, midByteOffsetPacked, sizeof(size_t),0, cudaMemcpyDeviceToHost); 
    // *h_midByteOffsetPacked = midByteOffsetPacked; 
    cudaMemcpyFromSymbol(h_block_size, block_size, sizeof(size_t),0, cudaMemcpyDeviceToHost); 
    // *h_block_size = block_size;
}


void HostMemAlloc(void ** pHost, size_t size){
    cudaHostAlloc(pHost, size, cudaHostAllocDefault);
}

void DeviceMemAlloc(void ** dHost, size_t size){
    cudaError_t a = cudaMalloc(dHost, size);
    if (a!=CUDA_SUCCESS){
        printf("Error happens with device memory malloc\n");
        printf("%s\n",cudaGetErrorString(a));
        exit(1);
    }
}

void HostMemFree(void * pHost){
    cudaFreeHost(pHost); 
}

void DeviceMemFree(void * dHost){
    cudaFree(dHost); 
}

float Host2Device(void* dst, const void* src, size_t count){
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMemcpy(dst,src,count,cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds/1000.0;
}

float Device2Host(void* dst, const void* src, size_t count){
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMemcpy(dst,src,count,cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds/1000.0;
}

__global__ void comp_p1_kernal(double* uncompBuffer, unsigned char* compBuffer, size_t* prefixSum, size_t* prefSums, unsigned char* midBytes4Packing){
    

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < num_chk){
    size_t strtChkMidBytes = midByteOffset+i*reqBytesLen*len/num_chk;
    size_t strtLdNum = i*totLeadNums/num_chk;
    size_t strtResibits = i*totBit2ByteNum/num_chk + totLeadNums;
    size_t packedNum = 0;
    unsigned char mbNbs[4]; // shared memo or registers?
    for(size_t j=0;j<chk_size;j+=4) // sequential
    {
         for (int k=0;k<4;k++)
         {
            if(i*chk_size+j+k == len-1)
            {
                last_number = uncompBuffer[i*chk_size+j+k];
                break;
            }
             double prevVal = (j+k)>0?uncompBuffer[i*chk_size+j+k-1]:0.;
            // double prevVal = (j+k)>0?uncompBuffer[i*chk_size]:0.;
             double currVal = uncompBuffer[i*chk_size+j+k];
             char *ptr0 = NULL, *ptr1 = NULL; 
             int ldNum = 0;
             ptr0 = (char *)&prevVal;
             ptr1 = (char *)&currVal;
             
             /* record number of mid-bytes different from previous */
             if((j+k)!=0)
             {
                for(int l=0;l<reqBytesLen;l++)
                {
                    if (*(ptr0+7-l) == *(ptr1+7-l))
                       ldNum++;
                    else
                        break;
                }
             }
             mbNbs[k] = reqBytesLen - ldNum;
             prefixSum[i] += reqBytesLen - ldNum;

             /* record mid-bytes different from previous */
             for(int l=ldNum;l<reqBytesLen;l++) 
             {
                 //midBytes4Packing[strtChkMidBytes] = *(ptr1+7-l);    
                 midBytes4Packing[strtChkMidBytes] = *(ptr1+7-l);    
                 strtChkMidBytes++;
             }

             /* record and pack residue bits */
            //  unsigned char resiBits = *(ptr1+7-reqBytesLen);
            //  resiBits = (resiBits >> (8-resiBitsLen)) << (8-resiBitsLen);
             
            //  compBuffer[strtResibits+j + k] =  resiBits;
            unsigned char resiBits = *(ptr1+7-reqBytesLen);
             resiBits = (resiBits >> (8-resiBitsLen)) << (8-resiBitsLen);
             size_t posByBits = packedNum * resiBitsLen;
             size_t headByteId = posByBits/8;
             size_t tailByteId = (posByBits+resiBitsLen)/8;
             int headBits = 8-posByBits%8;  // 
             headBits = headBits > resiBitsLen? resiBitsLen: headBits;

             compBuffer[strtResibits+headByteId] = compBuffer[strtResibits+headByteId] | (resiBits >> (posByBits%8));
             compBuffer[strtResibits+tailByteId] = compBuffer[strtResibits+tailByteId] | (resiBits << headBits);
             packedNum++;
         }

         /* pack numbers of mid-bytes */
         int tmp = 0;
         int ctr = 0;
         for(int k = 0;k<4;k++)
         {
            unsigned int type = mbNbs[k];
            switch(type)
            {
                case 0: 
                    break;
                case 1:
                    tmp = (tmp | (1 << (6-ctr*2)));
                    break;
                case 2:
                    tmp = (tmp | (2 << (6-ctr*2)));
                    break;
                case 3:
                    tmp = (tmp | (3 << (6-ctr*2)));
                    break;
                default:
                    ;
            }
            ctr++;
        }
        compBuffer[strtLdNum] = (unsigned char)tmp;    
        strtLdNum++;
    } // end intra-chunk (sequential) for-loop
    }
}

__global__ void comp_p2_kernal(unsigned char* compBuffer, size_t* prefixSum, size_t* prefSums, unsigned char* midBytes4Packing){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i<totMidByteNum){
        size_t real_idx = i/chk_size/reqBytesLen;
	    size_t inchk_ofs = i%(chk_size*reqBytesLen);
	    if(inchk_ofs<prefixSum[real_idx])
	    { 
		    real_idx = prefSums[real_idx];
		    real_idx += inchk_ofs;
		    compBuffer[midByteOffsetPacked+real_idx] = midBytes4Packing[i];
	    }
    }
}

__global__ void comp_p2_kernal_v2(unsigned char* compBuffer, size_t* prefixSum, size_t* prefSums, unsigned char* midBytes4Packing){
    size_t z = blockIdx.x * blockDim.x + threadIdx.x;
    size_t local_chk_size = 2;
    size_t g = z * local_chk_size;
    if(g<totMidByteNum){
        for (size_t i = g;i<g+local_chk_size;i++){
            size_t real_idx = i/chk_size/reqBytesLen;
	        size_t inchk_ofs = i%(chk_size*reqBytesLen);
	        if(inchk_ofs<prefixSum[real_idx])
	        { 
		        real_idx = prefSums[real_idx];
		        real_idx += inchk_ofs;
		        compBuffer[midByteOffsetPacked+real_idx] = midBytes4Packing[i];
            
                // printf("%x ",compBuffer[midByteOffsetPacked+real_idx] );
            
	        }
        }
    }
}

float comp_p1(double* uncompBuffer, unsigned char* compBuffer, size_t* prefixSum, size_t* prefSums, \
    unsigned char* midBytes4Packing,const size_t grid_size, const size_t block_size){
    
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    printf("grid_size  = %zu\n",grid_size);
    comp_p1_kernal<<<grid_size,block_size>>>(uncompBuffer,compBuffer,prefixSum,prefSums,midBytes4Packing);
    
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    return milliseconds/1000.0;
}

float comp_p2(unsigned char* compBuffer, size_t* prefixSum, size_t* prefSums, unsigned char* midBytes4Packing,\
    const size_t grid_size, const size_t block_size){
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    comp_p2_kernal<<<grid_size,block_size>>>(compBuffer,prefixSum,prefSums,midBytes4Packing);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds/1000.0;
}

__global__ void decomp_p1_kernal(unsigned char* compBuffer, size_t* prefixSum, double* local_prefSum){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    
    if(i < num_chk){
        size_t strtLdNum = i*totLeadNums/num_chk;
        
        
        size_t result_preSum = 0;
        for(size_t j=0;j<chk_size;j+=4) // sequential
        {
            unsigned char tmp = compBuffer[strtLdNum];
            for (int k=0;k<4;k++)
            {
                size_t ldNum = 0;
                ldNum = (tmp>>(6-k*2)) & 0x0003;
                local_prefSum[i*chk_size + j + k] = result_preSum;  
                result_preSum += ldNum;              
            }
            strtLdNum++;
        }
        prefixSum[i] =  result_preSum; 

    }

}

__global__ void decomp_p1_kernal_v2(unsigned char* compBuffer, size_t* prefixSum, double* local_prefSum){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    
    
    if(i < num_chk){
        size_t strtLdNum = i*totLeadNums/num_chk;
        
        
        size_t result_preSum = 0;
        for(size_t j=0;j<chk_size;j+=4) // sequential
        {
            unsigned char tmp = compBuffer[strtLdNum];
            for (int k=0;k<4;k++)
            {
                size_t ldNum = 0;
                ldNum = (tmp>>(6-k*2)) & 0x0003;
                local_prefSum[i*chk_size+j+k] = result_preSum;  
                result_preSum += ldNum;              
            }
            strtLdNum++;
        }
        prefixSum[i] =  result_preSum;
    }

}

__global__ void decomp_p2_kernal(double* uncompBuffer, unsigned char* compBuffer, size_t* prefSums){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    
    if( (i < num_chk)){
        size_t strtResibits = i*totBit2ByteNum/num_chk + totLeadNums;
        size_t strMidBytes = midByteOffsetPacked; 
        double prev_num = 0.0;
        size_t packedNum = 0;
        // if(i == num_chk -1)
        //     prefSums[i+1] = prefSums[i] + prefixSum[i];
        for(size_t j=0;j<chk_size;j+=4) // sequential
        {
            for (int k=0;k<4;k++)
            {
                if(i*chk_size+j+k == len-1)
                {
                    uncompBuffer[i*chk_size+j+k] = last_number;
                    break;
                }
                size_t local_id = i * chk_size + j + k;
                int prev_len = 1;
                char resiBits = 0;
                size_t posByBits = packedNum * resiBitsLen;
                size_t headByteId = posByBits/8;
                size_t tailByteId = (posByBits+resiBitsLen)/8;
                int headBits = 8-posByBits%8;  // 
                int head_str_point = posByBits%8;
                headBits = headBits > resiBitsLen? resiBitsLen: headBits;
                int tailBits = resiBitsLen - headBits;
                char headPart = compBuffer[strtResibits+headByteId];
                headPart = headPart>>(8-headBits-head_str_point)<<(8-headBits);
                char tailPart = compBuffer[strtResibits+tailByteId]>> (8-tailBits)<<(8-tailBits-headBits);
                resiBits = resiBits | headPart;
                resiBits = resiBits | tailPart;
                packedNum++;
                int conc_len = ((j+k)==chk_size-1)?prefSums[i+1]-prefSums[i]- uncompBuffer[local_id]:uncompBuffer[local_id+1] - uncompBuffer[local_id];
                size_t conc_offset = prefSums[i] + uncompBuffer[local_id];
                int conc_same_bytes = reqBytesLen -conc_len ;
                char conc_value[8];
                char* ptr_z;
                ptr_z = (char*) &prev_num;
                for(int z = 0; z< 8;z++){
                    conc_value[z] = *(ptr_z+z);
                }
                for(int z = conc_same_bytes ; z< reqBytesLen;z++){
                    conc_value[7-z] = compBuffer[strMidBytes + conc_offset + z - conc_same_bytes];
                }
                conc_value[7-reqBytesLen] = resiBits;
                double* ptr = NULL;
                double tmp;
                ptr = (double*)conc_value;
                tmp = *ptr;
                prev_num = tmp; 
                
                uncompBuffer[i * chk_size+j+k] = tmp;
                
            }
        }
    }

}



float decomp_p1(unsigned char* compBuffer, size_t* prefixSum, double* local_prefSum,\
    const size_t grid_size, const size_t block_size){
        
        float milliseconds = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        decomp_p1_kernal<<<grid_size,block_size>>>(compBuffer,prefixSum,local_prefSum);
    
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        return milliseconds/1000.0;
}



float decomp_p2(double* uncompBuffer, unsigned char* compBuffer, size_t* prefSums,\
    const size_t grid_size, const size_t block_size){

       
        float milliseconds = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        decomp_p2_kernal<<<grid_size,block_size>>>(uncompBuffer,compBuffer,prefSums);
        
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        return milliseconds/1000.0;
}
void set_GPUID(int id){
    cudaSetDevice ( id ) ;
}

