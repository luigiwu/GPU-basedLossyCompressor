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

short getPrecisionReqLength_double(double precision)
{
	ldouble lbuf;
	lbuf.value = precision;
	long lvalue = lbuf.lvalue;
	
	int expValue = (int)((lvalue & 0x7FF0000000000000) >> 52);
	expValue -= 1023;

	return (short)expValue;
}

short computeReqLength_double_MSST19(double realPrecision)
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



int main()
{
    double *oriData;
    size_t K = 1024;
    size_t GB_num = 1;
    size_t byte_num = 8;
    size_t len = GB_num*K/byte_num*K*K; // 4GB
    size_t chk_size = 32;//8;//16;//32;
    size_t num_chk = len/chk_size; 
    double pwrpr = 1E-5;
    size_t t_size = 355000; //1024: 12800 ; 16: 710000

    short reqLen = computeReqLength_double_MSST19(pwrpr);
    int reqBytesLen = reqLen/8;
	int resiBitsLen = reqLen%8;	
    printf("Each value requires %d bytes and %d bits\n",reqBytesLen, resiBitsLen);


    size_t totLeadNums = chk_size;

    if(totLeadNums%4==0)
		totLeadNums = totLeadNums*2/8;
	else
		totLeadNums = totLeadNums*2/8+1;

    totLeadNums *= num_chk;
    size_t totBit2ByteNum = ((resiBitsLen * chk_size)/8 + 1) * num_chk;
    // size_t totBit2ByteNum = len;
    size_t totMidByteNum = reqBytesLen * len;
    
    size_t compBufSize = totLeadNums + totBit2ByteNum + totMidByteNum;
    
    printf("OrigData = %.3f GBs ; CompBuf = %.3f GBs\n", len*sizeof(double)/1024./1024/1024,compBufSize/1024./1024/1024);

    /* Allocate Host Memory*/
    oriData = (double*)malloc(sizeof(double)*len);
    

    //oriData = (double*) malloc(len*sizeof(double));


    unsigned char *compBuffer_dummy;
    size_t *prefixSum_dummy, *prefSums_dummy;
    unsigned char *midBytes4Packing_dummy;

    compBuffer_dummy = (unsigned char *)malloc(sizeof(unsigned char)*compBufSize); 
    prefixSum_dummy = (size_t*)malloc(sizeof(size_t)*num_chk);
    prefSums_dummy = (size_t*)malloc(sizeof(size_t)*(num_chk+1));
    midBytes4Packing_dummy = (unsigned char *)malloc(sizeof(unsigned char)*totMidByteNum); 
    
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

    printf("..........................\n");
    
    
    size_t midByteOffset = 0;
    size_t midByteOffsetPacked = totLeadNums + totBit2ByteNum;


    size_t *h_tm;
    h_tm = (size_t*)malloc(sizeof(size_t)*(num_chk+1));  
    /* map and transfer host data to device buffer */
   
    
    
    double time1, time2, time3;
    time1 = omp_get_wtime();
    printf("num of chunks = %d\n",num_chk);
    #pragma omp parallel for
    for(size_t i = 0;i<num_chk;i++) // coarse-grained parallelism: chunk-level
    {
        size_t strtChkMidBytes = midByteOffset+i*reqBytesLen*len/num_chk;
        size_t strtLdNum = i*totLeadNums/num_chk;
        size_t strtResibits = i*totBit2ByteNum/num_chk + totLeadNums;
        size_t packedNum = 0;
        unsigned char mbNbs[4]; // shared memo or registers?
        for(size_t j=0;j<chk_size;j+=4) // sequential
        {
             for (int k=0;k<4;k++)
             {
                 double prevVal = (j+k)>0?oriData[i*chk_size+j+k-1]:0.;
                // double prevVal = (j+k)>0?oriData[i*chk_size]:0.;
                 double currVal = oriData[i*chk_size+j+k];
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
                 prefixSum_dummy[i] += reqBytesLen - ldNum;

                 /* record mid-bytes different from previous */
                 for(int l=ldNum;l<reqBytesLen;l++) 
                 {
                     midBytes4Packing_dummy[strtChkMidBytes] = *(ptr1+7-l);    
                     strtChkMidBytes++;
                 }

                 /* record and pack residue bits */
                //  unsigned char resiBits = *(ptr1+7-reqBytesLen);
                //  resiBits = (resiBits >> (8-resiBitsLen)) << (8-resiBitsLen);
                 
                //  compBuffer_dummy[strtResibits+j + k] =  resiBits;
                unsigned char resiBits = *(ptr1+7-reqBytesLen);
                 resiBits = (resiBits >> (8-resiBitsLen)) << (8-resiBitsLen);
                 size_t posByBits = packedNum * resiBitsLen;
                 size_t headByteId = posByBits/8;
                 size_t tailByteId = (posByBits+resiBitsLen)/8;
                 int headBits = 8-posByBits%8;  // 
                 headBits = headBits > resiBitsLen? resiBitsLen: headBits;

                 compBuffer_dummy[strtResibits+headByteId] = compBuffer_dummy[strtResibits+headByteId] | (resiBits >> (posByBits%8));
                 compBuffer_dummy[strtResibits+tailByteId] = compBuffer_dummy[strtResibits+tailByteId] | (resiBits << headBits);
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
		    compBuffer_dummy[strtLdNum] = (unsigned char)tmp;    
            strtLdNum++;
        } // end intra-chunk (sequential) for-loop
    } // end inter-chunk (parallel) for-loop
    
    time2=omp_get_wtime(); 
   
    
    
    printf("prefixes: ");
    //for(int i=0;i<num_chk;i++)
    //    printf("%d ",prefixSum_dummy[i]);
    //printf("\n");
    //size_t totMbytes = prefixSum_cpu(prefixSum_dummy,prefSums_dummy,num_chk);
    size_t totMbytes = prefixSum_cpu(prefixSum_dummy, prefSums_dummy, num_chk);
    time3 = omp_get_wtime();
    printf("CPU prefixsum time=%.6f\n", time3-time2);

    
    totMbytes = prefixSum_dummy[num_chk-1] + prefSums_dummy[num_chk-1];
    printf("last_nb=%ld last_sum=%ld ", prefixSum_dummy[num_chk-1] , prefSums_dummy[num_chk-1]);
    printf("tot=%ld\n",totMbytes);

    
    size_t midbStrt = 0;
    /* packing midBytes */

   double time4 = omp_get_wtime();

    #pragma omp parallel for
    for(size_t i = 0;i<totMidByteNum;i++) 
    {
        
    	size_t real_idx = i/chk_size/reqBytesLen;
	    size_t inchk_ofs = i%(chk_size*reqBytesLen);
	    if(inchk_ofs<prefixSum_dummy[real_idx])
	    { 
		    real_idx = prefSums_dummy[real_idx];
		    real_idx += inchk_ofs;
		    compBuffer_dummy[midByteOffsetPacked+real_idx] = midBytes4Packing_dummy[i];
	    }
    }
    printf("packing midbytes time=%.6f \n", omp_get_wtime()-time4);
    
    printf("totunpack=%ld totpack=%ld\n", totMidByteNum, totMbytes);
    
     

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
    printf("OrigData = %.3f GBs; CompressedData = %.3f GBs, compression time=%.6f \n", len*sizeof(double)/1024./1024/1024, (compBufSize-totMidByteNum+totMbytes)/1024./1024/1024, time2-time1); 
    printf("Prefix cost = %.6f GBs\n", (num_chk*sizeof(size_t))/1024./1024/1024); 

    printf("%ld\n",compBufSize-totMidByteNum+totMbytes);

    double time_dp1 = omp_get_wtime();
    #pragma omp parallel for
    for(size_t i = 0;i<num_chk;i++) // coarse-grained parallelism: chunk-level
    {
        
        size_t strtLdNum = i*totLeadNums/num_chk;
        
        
        size_t result_preSum = 0;
        for(size_t j=0;j<chk_size;j+=4) // sequential
        {
            unsigned char tmp = compBuffer_dummy[strtLdNum];
            for (int k=0;k<4;k++)
            {
                size_t ldNum = 0;
                ldNum = (tmp>>(6-k*2)) & 0x0003;
                oriData[i*chk_size + j + k] = result_preSum;  
                result_preSum += ldNum;              
            }
            strtLdNum++;
        }
        prefixSum_dummy[i] =  result_preSum; 
    }
    double time_dp2 = omp_get_wtime();
    totMbytes = prefixSum_cpu(prefixSum_dummy, prefSums_dummy, num_chk);
    printf("CPU prefixsum time=%.6f\n", omp_get_wtime()-time_dp2);
    printf("last_nb=%ld last_sum=%ld \n", prefixSum_dummy[num_chk-1] , prefSums_dummy[num_chk-1]);
    printf("Decompression time for part one=%.6f\n \n", time_dp2 - time_dp1);

    time_dp1 = omp_get_wtime();
    #pragma omp parallel for
    for(size_t i = 0;i<num_chk;i++) // coarse-grained parallelism: chunk-level
    {
        size_t strtResibits = i*totBit2ByteNum/num_chk + totLeadNums;
        size_t strMidBytes = midByteOffsetPacked; 
        double prev_num = 0.0;
        size_t packedNum = 0;
        if(i == num_chk -1)
            prefSums_dummy[i+1] = prefSums_dummy[i] + prefixSum_dummy[i];
        for(size_t j=0;j<chk_size;j+=4) // sequential
        {
            for (int k=0;k<4;k++)
            {
                
                int local_id = i * chk_size + j + k;
                int prev_len = 1;
                char resiBits = 0;
                size_t posByBits = packedNum * resiBitsLen;
                size_t headByteId = posByBits/8;
                size_t tailByteId = (posByBits+resiBitsLen)/8;
                int headBits = 8-posByBits%8;  // 
                int head_str_point = posByBits%8;
                headBits = headBits > resiBitsLen? resiBitsLen: headBits;
                int tailBits = resiBitsLen - headBits;
                char headPart = compBuffer_dummy[strtResibits+headByteId];
                headPart = headPart>>(8-headBits-head_str_point)<<(8-headBits);
                char tailPart = compBuffer_dummy[strtResibits+tailByteId]>> (8-tailBits)<<(8-tailBits-headBits);
                resiBits = resiBits | headPart;
                resiBits = resiBits | tailPart;
                packedNum++;
                int conc_len = ((j+k)==chk_size-1)?prefSums_dummy[i+1]-prefSums_dummy[i]- oriData[local_id]:oriData[local_id+1] - oriData[local_id];
                size_t conc_offset = prefSums_dummy[i] + oriData[local_id];
                int conc_same_bytes = reqBytesLen -conc_len ;
                char conc_value[8];
                char* ptr_z;
                ptr_z = (char*) &prev_num;
                for(int z = 0; z< 8;z++){
                    conc_value[z] = *(ptr_z+z);
                }
                for(int z = conc_same_bytes ; z< reqBytesLen;z++){
                    conc_value[7-z] = compBuffer_dummy[strMidBytes + conc_offset + z - conc_same_bytes];
                }
                conc_value[7-reqBytesLen] = resiBits;
                double* ptr = NULL;
                double tmp;
                ptr = (double*)conc_value;
                tmp = *ptr;
                prev_num = tmp; 
                
                oriData[i * chk_size+j+k] = tmp;
                
            }
        }
        
    }
    time_dp2 = omp_get_wtime();
    printf("Decompression time for part two=%.6f\n \n", time_dp2 - time_dp1);
    

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


    
    free(oriData);
    free(compBuffer_dummy);
    free(prefixSum_dummy);
    free(prefSums_dummy);
    free(h_tm);
    free(midBytes4Packing_dummy);

    return 0;
}
