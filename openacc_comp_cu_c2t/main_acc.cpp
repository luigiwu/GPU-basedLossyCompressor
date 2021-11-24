#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>  
#include <time.h>

#include <omp.h>

void read_file(char *filename,unsigned char* oriData, size_t len)//len is in  double
{
 
    
   
    FILE *fd = fopen(filename,"rb"); 
    if(fd == NULL)
    {
        perror("open failed!");
        exit(1);        
    }
  
    fread(oriData,sizeof(unsigned char)*len*8,1,fd);
    
    // double* z = NULL;
    // z = (double*) oriData;
    // for (int i = 0 ; i< len/8;i++){
    //     (i+1)%32==0?printf("%lf\n",*z):printf("%lf ",*z);
    //     z = z+1;
        
    // }

    fclose(fd);

}



typedef union ldouble
{
    double value;
    unsigned long lvalue;
    unsigned char byte[8];
} ldouble;

inline short getPrecisionReqLength_double(double precision)
{
	ldouble lbuf;
	lbuf.value = precision;
	long lvalue = lbuf.lvalue;
	
	int expValue = (int)((lvalue & 0x7FF0000000000000) >> 52);
	expValue -= 1023;

	return (short)expValue;
}

inline short computeReqLength_double_MSST19(double realPrecision)
{
	short reqExpo = getPrecisionReqLength_double(realPrecision);
	return 12-reqExpo;
}

int prefixSum_cpu(size_t *prefixes, size_t *prefSums, size_t plen)
{   
    size_t prev_idx = 0;
    size_t prev_len = 0;
    size_t totNbs = 0; 
    for(size_t i=0;i<plen;i++)
    {
        totNbs += prefixes[i];
        prefSums[i] = prev_idx + prev_len;
        prev_len = prefixes[i];
        prev_idx = prefSums[i];
    }
    return totNbs;
}

void write2bin(char *filename, unsigned char *bytes, size_t len)
{
	FILE *write_ptr = fopen(filename, "wb");
        fwrite(bytes, len, 1, write_ptr);
	fclose(write_ptr);	
}


extern "C"
void set_GPUID(int id);

extern "C"
void initialize_value(size_t* h_len,size_t* h_chk_size, size_t* h_num_chk, double* h_pwrpr, size_t* h_t_size, short* h_reqLen,\
     int* h_reqBytesLen, int* h_resiBitsLen, size_t* h_totLeadNums, size_t* h_totBit2ByteNum, size_t *h_totMidByteNum,\
     size_t* h_compBufSize,size_t* h_midByteOffset, size_t* h_midByteOffsetPacked,size_t* h_block_size);
extern "C"
size_t cub_prefixSum(size_t *d_in, size_t *d_out, size_t n_items, void *d_tmp, size_t tmp_size);
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

int main()
{
    double *oriData;
    size_t len, chk_size, num_chk, t_size, totLeadNums, totBit2ByteNum, totMidByteNum, compBufSize, midByteOffset,\
     midByteOffsetPacked; 
    double pwrpr;
    short reqLen;
    int reqBytesLen, resiBitsLen;
    size_t block_size;
    set_GPUID(1);
    initialize_value(&len, &chk_size, &num_chk, &pwrpr, &t_size, &reqLen, &reqBytesLen, &resiBitsLen, &totLeadNums,\
     &totBit2ByteNum, &totMidByteNum, &compBufSize, &midByteOffset, &midByteOffsetPacked, &block_size);
   
    printf("OrigData = %zu GBs ; CompBuf = %zu GBs\n", len*sizeof(double)/1024/1024/1024,compBufSize/1024/1024/1024);
    printf("Each value requires %d bytes and %d bits\n",reqBytesLen, resiBitsLen);

    printf("OrigData = %.3f GBs ; CompBuf = %.3f GBs\n", len*sizeof(double)/1024./1024/1024,compBufSize/1024./1024/1024);

    /* Allocate Host Memory*/
    HostMemAlloc((void**)&oriData, sizeof(double)*len);
    
    //oriData = (double*) malloc(len*sizeof(double));


    unsigned char *compBuffer_dummy;
    size_t *prefixSum_dummy, *prefSums_dummy;
    unsigned char *midBytes4Packing_dummy;
    
    // const size_t grid_size1 = num_chk/block_size;
    const size_t grid_size1 = num_chk / block_size ;
    const size_t grid_size2 =totMidByteNum/block_size;

    
    HostMemAlloc((void**)&compBuffer_dummy, sizeof(unsigned char)*compBufSize);
    HostMemAlloc((void**)&prefixSum_dummy, sizeof(size_t)*num_chk);
    HostMemAlloc((void**)&prefSums_dummy, sizeof(size_t)*num_chk);
    HostMemAlloc((void**)&midBytes4Packing_dummy, sizeof(unsigned char)*totMidByteNum);
    
    
	    //compBuffer_dummy = (unsigned char*) malloc(sizeof(unsigned char)*compBufSize);
    //prefixSum_dummy = (size_t*) malloc(sizeof(size_t)*num_chk);
    //prefSums_dummy = (size_t*) malloc(sizeof(size_t)*num_chk);
    srand(1234ULL);
	 // init
     
    // double init_tab[2023];
    // double init_tab1[2024];
    // double init_tab2[2025];
    
    // for(int i=0;i<2023;i++) // all magic numbers
    // {	
	//   init_tab[i] = (double)rand() / (double)RAND_MAX;
	//   init_tab1[i] = (double)rand() / (double)RAND_MAX;
	//   init_tab2[i] = (double)rand() / (double)RAND_MAX;
    // }	 
    // init_tab1[2023] = (double)rand() / (double)RAND_MAX;
    // init_tab2[2023] = (double)rand() / (double)RAND_MAX;
    // init_tab2[2024] = (double)rand() / (double)RAND_MAX;

    
     	 
    // init_tab1[2023] = (double)rand() / (double)RAND_MAX;
    // init_tab2[2023] = (double)rand() / (double)RAND_MAX;
    // init_tab2[2024] = (double)rand() / (double)RAND_MAX;

    // #pragma omp parallel for
    // for(size_t i =0;i<len;i++) {
	//     oriData[i] = (init_tab[i%2023] + init_tab1[i%2024] + init_tab2[i%2025])/3;
    // }
    
    unsigned char* ss = (unsigned char*) oriData; 
    read_file("/home/yf-wu/experimentData/QFTPhase_1GB.bin",ss,len );
    
    unsigned char *peek_ptr;
    peek_ptr = (unsigned char *) oriData;
    
    int peek_num = 16>chk_size?chk_size:16, peek_num1=10;

    printf("peek_num = %d\n",peek_num);
    for (size_t i = 0;i<peek_num;i++)
    {
        printf("val%d=%.6f: %x %x %x %x %x %x %x %x\n",i, oriData[i], *(peek_ptr+i*8),*(peek_ptr+i*8+1),*(peek_ptr+i*8+2),*(peek_ptr+i*8+3),\
                                            *(peek_ptr+i*8+4),*(peek_ptr+i*8+5),*(peek_ptr+i*8+6),*(peek_ptr+i*8+7));
    }

    // for (size_t i = 0;i<peek_num;i++)
    // {
    //     printf("val%d=%.6f: %x %x %x %x %x %x %x %x\n",i, *(peek_ptr+i), *(peek_ptr+i*8),*(peek_ptr+i*8+1),*(peek_ptr+i*8+2),*(peek_ptr+i*8+3),\
    //                                         *(peek_ptr+i*8+4),*(peek_ptr+i*8+5),*(peek_ptr+i*8+6),*(peek_ptr+i*8+7));
    // }

    printf("..........................\n");

    for (size_t i = len-chk_size;i<len-chk_size+peek_num;i++)
    {
        printf("val%d=%.6f: %x %x %x %x %x %x %x %x\n",i, oriData[i], *(peek_ptr+i*8),*(peek_ptr+i*8+1),*(peek_ptr+i*8+2),*(peek_ptr+i*8+3),\
                                            *(peek_ptr+i*8+4),*(peek_ptr+i*8+5),*(peek_ptr+i*8+6),*(peek_ptr+i*8+7));
			}

    printf("..........................\n");
    double *uncompBuffer;
 
    DeviceMemAlloc((void **) &uncompBuffer,sizeof(double)*len);
    unsigned char *compBuffer;
    DeviceMemAlloc((void **)&compBuffer,sizeof(unsigned char)*compBufSize); // stationary
    size_t *prefixSum;
    DeviceMemAlloc((void**)&prefixSum, (size_t)sizeof(size_t)*num_chk);
    size_t *prefSums;
    DeviceMemAlloc((void**)&prefSums, (size_t)sizeof(size_t)*num_chk + 1);
    unsigned char *midBytes4Packing;
    DeviceMemAlloc((void **)&midBytes4Packing, sizeof(unsigned char)*totMidByteNum); // pack them to compBuffer afterward



    void *d_tm;
    DeviceMemAlloc(&d_tm, t_size); 	  

    
    // Host2Device(compBuffer,compBuffer_dummy,sizeof(unsigned char)*compBufSize);
    // Host2Device(prefixSum,prefixSum_dummy,sizeof(size_t)*num_chk);
    // Host2Device(prefSums,prefSums,sizeof(size_t)*num_chk);
    // Host2Device(midBytes4Packing,midBytes4Packing_dummy,sizeof(unsigned char)*totMidByteNum);
    
   

    Host2Device(uncompBuffer,oriData,sizeof(double)*len);	
    float time_p1, time_trans, time_p2;
    
    printf("num of chunks = %d\n",num_chk);
    time_p1 = comp_p1( uncompBuffer,  compBuffer,  prefixSum,  prefSums,\
      midBytes4Packing, grid_size1,  block_size);
    
    
    DeviceMemFree(uncompBuffer); 
    
    // printf("trans start\n");
    time_trans = Device2Host(prefixSum_dummy,prefixSum,sizeof(size_t)*num_chk);
    // printf("trans end\n");
    
    
    printf("prefixes: ");
    //for(int i=0;i<num_chk;i++)
    //    printf("%d ",prefixSum_dummy[i]);
    //printf("\n");
    //size_t totMbytes = prefixSum_cpu(prefixSum_dummy,prefSums_dummy,num_chk);
    double time_cub_str=omp_get_wtime(); 
    size_t totMbytes = cub_prefixSum(prefixSum, prefSums, num_chk, d_tm, t_size);
    double time_cub_end=omp_get_wtime(); 
    // printf("cub prefixsum time=%.6f\n", time_cub_end -time_cub_str );

    Device2Host(prefSums_dummy,prefSums,sizeof(size_t)*num_chk);
    totMbytes = prefixSum_dummy[num_chk-1] + prefSums_dummy[num_chk-1];
    printf("last_nb=%ld last_sum=%ld ", prefixSum_dummy[num_chk-1] , prefSums_dummy[num_chk-1]);
    printf("tot=%ld\n",totMbytes);

    
    size_t midbStrt = 0;
    /* packing midBytes */

    

    
    time_p2 = comp_p2(compBuffer,  prefixSum,  prefSums,\
      midBytes4Packing, grid_size2,  block_size);


    printf("packing midbytes time=%.6f \n", time_p2 );
    Device2Host(compBuffer_dummy,compBuffer,sizeof(unsigned char)*compBufSize);
    printf("totunpack=%ld totpack=%ld\n", totMidByteNum, totMbytes);


  
    //Device2Host(midBytes4Packing_dummy,midBytes4Packing,sizeof(unsigned char)*totMidByteNum);   

    /* write to binary */
    char *fname = "compress.bin";
    write2bin(fname, compBuffer_dummy, compBufSize-totMidByteNum+totMbytes);

    /* validation */
    printf("Packed midBytes: \n");
    for(int i =0;i<peek_num;i++)
    {
        printf("%x ",compBuffer_dummy[midByteOffsetPacked+i]);
    }
    printf("...................\n");

    for(int i =totMbytes-peek_num; i< totMbytes;i++)
    {
        printf("%x ",compBuffer_dummy[midByteOffsetPacked+i]);
    }

    for(int i =0;i<1;i++)
    {
        printf("chunk%d midByte Num: ",i);
        for (int k=0;k<10; k++)
        {
            printf("[%d] %x ",i*totLeadNums/num_chk+k, compBuffer_dummy[i*totLeadNums/num_chk+k]);
            
        }
        printf(".................\n");
    }

    for(int i =num_chk-1;i<num_chk;i++)
    {
        printf("chunk%d midByte Num: ",i);
        for (int k=0;k<10; k++)
        {
            printf("[%d] %x ",i*totLeadNums/num_chk+k, compBuffer_dummy[i*totLeadNums/num_chk+k]);
            
        }
        printf("...........\n");
    }

    size_t resibStrt = totLeadNums;
    for(int i =0;i<1;i++)
    {
        printf("chunk%d residue bits: ",i);
        for (int k=0;k<10; k++)
        {
            printf("%x ",compBuffer_dummy[resibStrt+i*totBit2ByteNum/num_chk+k]);
            
        }
        printf("...............\n");
    }

    for(int i =num_chk-1;i<num_chk;i++)
    {
        printf("chunk%d residue bits: ",i);
        for (int k=0;k<10; k++)
        {
            printf("%x ",compBuffer_dummy[resibStrt+i*totBit2ByteNum/num_chk+k]);
            
        }
        printf("...............\n");
    }
    printf("OrigData = %.3f GBs; CompressedData = %.3f GBs, compression time=%.6f \n", len*sizeof(double)/1024./1024/1024, (compBufSize-totMidByteNum+totMbytes)/1024./1024/1024, time_p1); 
    printf("Prefix cost = %.6f GBs\n", (num_chk*sizeof(size_t))/1024./1024/1024); 

    printf("%ld\n",compBufSize-totMidByteNum+totMbytes);
    DeviceMemFree(midBytes4Packing);
    
    float time_dp1, time_dp2;
    DeviceMemAlloc((void **) &uncompBuffer,sizeof(double)*len);
    time_dp1 = decomp_p1(compBuffer,  prefixSum,  uncompBuffer,  grid_size1,  block_size);
    printf("Decompression time for part one=%.6f\n", time_dp1 );
    time_trans = Device2Host(prefixSum_dummy,prefixSum,sizeof(size_t)*num_chk);
    totMbytes = cub_prefixSum(prefixSum, prefSums, num_chk, d_tm, t_size);
    Device2Host(prefSums_dummy,prefSums,sizeof(size_t)*num_chk);
    totMbytes = prefixSum_dummy[num_chk-1] + prefSums_dummy[num_chk-1];
    printf("last_nb=%ld last_sum=%ld ", prefixSum_dummy[num_chk-1] , prefSums_dummy[num_chk-1]);
    printf("tot=%ld\n",totMbytes);
    DeviceMemFree(prefixSum);
    DeviceMemFree(d_tm);
    time_dp2 = decomp_p2(uncompBuffer,compBuffer,prefSums,grid_size1,block_size);
    printf("Decompression time for part two=%.6f\n", time_dp2 );
    time_trans = Device2Host(oriData,uncompBuffer,sizeof(double)*len);
    
    for (size_t i = 0;i<peek_num;i++)
    {
        printf("val%d=%.6f: %x %x %x %x %x %x %x %x\n",i, oriData[i], *(peek_ptr+i*8),*(peek_ptr+i*8+1),*(peek_ptr+i*8+2),*(peek_ptr+i*8+3),\
                                            *(peek_ptr+i*8+4),*(peek_ptr+i*8+5),*(peek_ptr+i*8+6),*(peek_ptr+i*8+7));
    }
    printf("..........................\n");
    for (size_t i = len-chk_size;i<len-chk_size+peek_num;i++)
    {
        printf("val%d=%.6f: %x %x %x %x %x %x %x %x\n",i, oriData[i], *(peek_ptr+i*8),*(peek_ptr+i*8+1),*(peek_ptr+i*8+2),*(peek_ptr+i*8+3),\
                                            *(peek_ptr+i*8+4),*(peek_ptr+i*8+5),*(peek_ptr+i*8+6),*(peek_ptr+i*8+7));
	}
    HostMemFree(oriData);  
    HostMemFree(compBuffer_dummy); 
    HostMemFree(prefixSum_dummy);   
    HostMemFree(prefSums_dummy);  
    HostMemFree(midBytes4Packing_dummy);

    DeviceMemFree(compBuffer);

    DeviceMemFree(prefSums);
    DeviceMemFree(uncompBuffer); 

    return 0;
}
