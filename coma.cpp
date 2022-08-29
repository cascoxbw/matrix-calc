#include <stdio.h>
#include <immintrin.h>
#include <string.h>
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include<numeric>       // std::accumulate


typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;

#define MAX_SIZE            64*64
#define DATA512Float_LOOP   16
#define DATA512Double_LOOP  8
#define NUM_CASE            25
#define NUM_LOOP            10000

int32_t gCycleCount[NUM_CASE][NUM_LOOP];
double gResistO3[NUM_CASE][NUM_LOOP];
double gResistO3_1 = 0;
double gResistO3_2 = 0;
double gResistO3_3 = 0;
double gResistO3_4 = 0;

void display(int32_t caseid)
{
    std::vector<int32_t> v0(gCycleCount[caseid],gCycleCount[caseid]+NUM_LOOP);
    std::sort(v0.begin(), v0.end());
    
    int32_t sum = std::accumulate(v0.begin(), v0.end(), 0);  
    int32_t avg =  sum / v0.size(); 

    auto maxPosition = max_element(v0.begin(), v0.end());
    auto minPosition = min_element(v0.begin(), v0.end());
    
    
    printf(" case %d: cycle avg=%d, cycle max=%d, cycle min=%d\n", caseid, avg, *maxPosition, *minPosition);
}

void vec_single_mul512_conj(float single_value_re,float single_value_im, float* input_vec_re,float* input_vec_im, float* output_vec_re, float* output_vec_im, int32_t len)
{
   // printf("single re is %f, single im is %f\n",single_value_re,single_value_im);
    __m512 single_vec_re = _mm512_set1_ps(single_value_re);
    __m512 single_vec_im = _mm512_set1_ps(single_value_im);

    if (len <= DATA512Float_LOOP)
    {
        __m512 re = _mm512_add_ps(_mm512_mul_ps(single_vec_re,_mm512_load_ps(input_vec_re)),_mm512_mul_ps(single_vec_im,_mm512_load_ps(input_vec_im)));
        __m512 im = _mm512_sub_ps(_mm512_mul_ps(single_vec_re,_mm512_load_ps(input_vec_im)),_mm512_mul_ps(single_vec_im,_mm512_load_ps(input_vec_re)));
        *(__m512*)(output_vec_re) = re;
        *(__m512*)(output_vec_im) = im;
    }
    else
    {
        int32_t loop = len / DATA512Float_LOOP;
        for(int32_t i = 0; i < loop ; i++)
        {
            __m512 re = _mm512_add_ps(_mm512_mul_ps(single_vec_re,_mm512_load_ps(input_vec_re + i * DATA512Float_LOOP)),_mm512_mul_ps(single_vec_im,_mm512_load_ps(input_vec_im + i * DATA512Float_LOOP)));
            __m512 im = _mm512_sub_ps(_mm512_mul_ps(single_vec_re,_mm512_load_ps(input_vec_im + i * DATA512Float_LOOP)),_mm512_mul_ps(single_vec_im,_mm512_load_ps(input_vec_re + i * DATA512Float_LOOP)));
            *(__m512*)(output_vec_re + i * DATA512Float_LOOP) = re;
            *(__m512*)(output_vec_im + i * DATA512Float_LOOP) = im;
        }
    }
}

void vec_single_mul512_conj_double(double single_value_re,double single_value_im, double* input_vec_re,double* input_vec_im, double* output_vec_re, double* output_vec_im, int32_t len)
{
   // printf("single re is %f, single im is %f\n",single_value_re,single_value_im);
    __m512d single_vec_re = _mm512_set1_pd(single_value_re);
    __m512d single_vec_im = _mm512_set1_pd(single_value_im);
    
    if (len <= DATA512Double_LOOP)
    {
        __m512d re = _mm512_add_pd(_mm512_mul_pd(single_vec_re,_mm512_load_pd(input_vec_re)),_mm512_mul_pd(single_vec_im,_mm512_load_pd(input_vec_im)));
        __m512d im = _mm512_sub_pd(_mm512_mul_pd(single_vec_re,_mm512_load_pd(input_vec_im)),_mm512_mul_pd(single_vec_im,_mm512_load_pd(input_vec_re)));
        *(__m512d*)(output_vec_re) = re;
        *(__m512d*)(output_vec_im) = im;
    }
    else
    {
        int32_t loop = len / DATA512Double_LOOP;
        for(int32_t i = 0; i < loop ; i++)
        {
            __m512d re = _mm512_add_pd(_mm512_mul_pd(single_vec_re,_mm512_load_pd(input_vec_re + i * DATA512Double_LOOP)),_mm512_mul_pd(single_vec_im,_mm512_load_pd(input_vec_im + i * DATA512Double_LOOP)));
            __m512d im = _mm512_sub_pd(_mm512_mul_pd(single_vec_re,_mm512_load_pd(input_vec_im + i * DATA512Double_LOOP)),_mm512_mul_pd(single_vec_im,_mm512_load_pd(input_vec_re + i * DATA512Double_LOOP)));
            *(__m512d*)(output_vec_re + i * DATA512Double_LOOP) = re;
            *(__m512d*)(output_vec_im + i * DATA512Double_LOOP) = im;
        }
    }
}

void coma(int32_t len, float* out_re,float* out_im,float* in_re,float* in_im)
{
    float* re = in_re;
    float* im = in_im;
    float* coma_re = out_re;
    float* coma_im = out_im;
    for(int32_t i = 0; i < len; i++)
    {
        vec_single_mul512_conj(*(re+i),*(im+i),re,im,coma_re+i*len,coma_im+i*len,len);
        gResistO3_1 += *(re+i) + *(im+i) + *re + *im + *coma_re + *coma_im;
    }
}

void coma_double(int32_t len, double* out_re,double* out_im,double* in_re,double* in_im)
{
    double* re = in_re;
    double* im = in_im;
    double* coma_re = out_re;
    double* coma_im = out_im;
    for(int32_t i = 0; i < len; i++)
    {
        vec_single_mul512_conj_double(*(re+i),*(im+i),re,im,coma_re+i*len,coma_im+i*len,len);
        gResistO3_1 += *(re+i) + *(im+i) + *re + *im + *coma_re + *coma_im;
    }
}

void calc_coma_avx512_float(int32_t caseid, int32_t N)
{   
    float out_re[MAX_SIZE] = {0};
    float out_im[MAX_SIZE] = {0};
    float in_re[MAX_SIZE]  = {0};
    float in_im[MAX_SIZE]  = {0};

    uint64_t avg = 0;
    for (int32_t i=0;i<NUM_LOOP;i++)
    {
        int32_t ran = rand()%50;
        for (int32_t k=0;k<MAX_SIZE;k++)
        {
            in_re[k]  = k + i + ran;
            in_im[k]  = k + i + ran;
        }
        
        uint64_t t1 = __rdtsc();
        coma(N,out_re,out_im,in_re,in_im);
        uint64_t t2 = __rdtsc();

        for (int32_t k=0;k<MAX_SIZE;k++)
        {
            gResistO3[caseid][i] = out_re[k]/NUM_LOOP + out_im[k]/NUM_LOOP + gResistO3_1/NUM_LOOP;
        }
        gCycleCount[caseid][i] = t2-t1;
        avg += t2-t1;
    }

    //avg /= NUM_LOOP;
    
    printf(" case %d: calc_coma_%d_avx512_float, cycle total=%lu\n", caseid, N, avg);
    display(caseid);
}

void calc_coma_avx512_double(int32_t caseid, int32_t N)
{   
    double out_re_d[MAX_SIZE] = {0};
    double out_im_d[MAX_SIZE] = {0};
    double in_re_d[MAX_SIZE]  = {0};
    double in_im_d[MAX_SIZE]  = {0};

    uint64_t avg = 0;
    for (int32_t i=0;i<NUM_LOOP;i++)
    {
        int32_t ran = rand()%50;
        for (int32_t k=0;k<MAX_SIZE;k++)
        {
            in_re_d[k]  = k + i + ran;
            in_im_d[k]  = k + i + ran;
        }
        
        uint64_t t1 = __rdtsc();
        coma_double(N,out_re_d,out_im_d,in_re_d,in_im_d);
        uint64_t t2 = __rdtsc();
        
        for (int32_t k=0;k<MAX_SIZE;k++)
        {
            gResistO3[caseid][i] = out_re_d[k]/NUM_LOOP + out_im_d[k]/NUM_LOOP + gResistO3_1/NUM_LOOP;
        }
        gCycleCount[caseid][i] = t2-t1;
        avg += t2-t1;
    }
    //avg /= NUM_LOOP;
    printf(" case %d: calc_coma_%d_avx512_double, cycle total=%lu\n", caseid, N, avg);
    display(caseid);

}

void kron(float* in_re1, float* in_im1,float* in_re2, float* in_im2, int32_t len)
{
    len *= 2;
    if(len*2<=DATA512Float_LOOP)
    {
        *(__m512*)(in_re2) = _mm512_load_ps(in_re1);
        *(__m512*)(in_re2+len) = _mm512_setzero_ps();
        *(__m512*)(in_re2+2*len) = _mm512_setzero_ps();
        *(__m512*)(in_re2+3*len) = _mm512_load_ps(in_re1);
        *(__m512*)(in_im2) = _mm512_load_ps(in_im1);
        *(__m512*)(in_im2+len) = _mm512_setzero_ps();
        *(__m512*)(in_im2+2*len) = _mm512_setzero_ps();
        *(__m512*)(in_im2+3*len) = _mm512_load_ps(in_im1);
        gResistO3_2 += *(in_re2) + *(in_re2+3*len) + 
                       *(in_im2) + *(in_im2+3*len);
    }
    else
    {
        int32_t loop = len/DATA512Float_LOOP;
        for(int32_t i = 0; i < loop; i++)
        {
            *(__m512*)(in_re2+i*DATA512Float_LOOP) = _mm512_load_ps(in_re1+i*16);
            *(__m512*)(in_re2+i*DATA512Float_LOOP+len) = _mm512_setzero_ps();
            *(__m512*)(in_re2+i*DATA512Float_LOOP+2*len) = _mm512_setzero_ps();
            *(__m512*)(in_re2+i*DATA512Float_LOOP+3*len) = _mm512_load_ps(in_re1+i*16);
            *(__m512*)(in_im2+i*DATA512Float_LOOP) = _mm512_load_ps(in_im1+i*16);
            *(__m512*)(in_im2+i*DATA512Float_LOOP+len) = _mm512_setzero_ps();
            *(__m512*)(in_im2+i*DATA512Float_LOOP+2*len) = _mm512_setzero_ps();
            *(__m512*)(in_im2+i*DATA512Float_LOOP+3*len) = _mm512_load_ps(in_im1+i*16);
            gResistO3_2 += *(in_re2+i*DATA512Float_LOOP) + *(in_re2+i*DATA512Float_LOOP+3*len) + 
                           *(in_im2+i*DATA512Float_LOOP) + *(in_im2+i*DATA512Float_LOOP+3*len);
        }
    }
}

void kron_double(double* in_re1, double* in_im1,double* in_re2, double* in_im2, int32_t len)
{
    len *= 2;
    /*if(len<=DATA512Double_LOOP)
    {
        *(__m512d*)(in_re2)       = _mm512_load_pd(in_re1);
        *(__m512d*)(in_re2+len)   = _mm512_setzero_pd();
        *(__m512d*)(in_re2+2*len) = _mm512_setzero_pd();
        *(__m512d*)(in_re2+3*len) = _mm512_load_pd(in_re1);
        *(__m512d*)(in_im2)       = _mm512_load_pd(in_im1);
        *(__m512d*)(in_im2+len)   = _mm512_setzero_pd();
        *(__m512d*)(in_im2+2*len) = _mm512_setzero_pd();
        *(__m512d*)(in_im2+3*len) = _mm512_load_pd(in_im1);
        gResistO3_2 += *in_re2 + *(in_re2+3*len) + *in_im2 + *(in_im2+3*len);
    }
    else*/
    {
        int32_t loop = len/DATA512Double_LOOP;
        for(int32_t i = 0; i < loop; i++)
        {
            *(__m512d*)(in_re2+i*DATA512Double_LOOP) = _mm512_load_pd(in_re1+i*8);
            *(__m512d*)(in_re2+i*DATA512Double_LOOP+len) = _mm512_setzero_pd();
            *(__m512d*)(in_re2+i*DATA512Double_LOOP+2*len) = _mm512_setzero_pd();
            *(__m512d*)(in_re2+i*DATA512Double_LOOP+3*len) = _mm512_load_pd(in_re1+i*8);
            *(__m512d*)(in_im2+i*DATA512Double_LOOP) = _mm512_load_pd(in_im1+i*8);
            *(__m512d*)(in_im2+i*DATA512Double_LOOP+len) = _mm512_setzero_pd();
            *(__m512d*)(in_im2+i*DATA512Double_LOOP+2*len) = _mm512_setzero_pd();
            *(__m512d*)(in_im2+i*DATA512Double_LOOP+3*len) = _mm512_load_pd(in_im1+i*8);
            gResistO3_2 += *(in_re2+i*DATA512Double_LOOP) + *(in_re2+i*DATA512Double_LOOP+3*len) + 
                           *(in_im2+i*DATA512Double_LOOP) + *(in_im2+i*DATA512Double_LOOP+3*len);
        }
    }
}

void calc_kron_avx512_float(int32_t caseid, int32_t N)
{   
    float in_re1[MAX_SIZE] = {0};
    float in_im1[MAX_SIZE] = {0};
    float in_re2[MAX_SIZE] = {0};
    float in_im2[MAX_SIZE] = {0};

    uint64_t avg = 0;
    for (int32_t i=0;i<NUM_LOOP;i++)
    {
        int32_t ran = rand()%50;
        for (int32_t k=0;k<MAX_SIZE;k++)
        {
            in_re1[k] = k + i + ran;
            in_im1[k] = k + i + ran;
        }
        
        uint64_t t1 = __rdtsc();
        kron(in_re1, in_im1, in_re2, in_im2, N);
        uint64_t t2 = __rdtsc();
        
        for (int32_t k=0;k<MAX_SIZE;k++)
        {
            gResistO3[caseid][i] = in_re2[k]/NUM_LOOP + in_im2[k]/NUM_LOOP + gResistO3_2/NUM_LOOP;
        }
        gCycleCount[caseid][i] = t2-t1;
        avg += t2-t1;
    }
    //avg /= NUM_LOOP;
    printf(" case %d: calc_kron_%d_avx512_float, cycle total=%lu\n", caseid, N, avg);
    display(caseid);

}

void calc_kron_avx512_double(int32_t caseid, int32_t N)
{   
    double in_re1_d[MAX_SIZE] = {0};
    double in_im1_d[MAX_SIZE] = {0};
    double in_re2_d[MAX_SIZE] = {0};
    double in_im2_d[MAX_SIZE] = {0};

    uint64_t avg = 0;
    for (int32_t i=0;i<NUM_LOOP;i++)
    {
        int32_t ran = rand()%50;
        for (int32_t k=0;k<MAX_SIZE;k++)
        {
            in_re1_d[k] = k + i + ran;
            in_im1_d[k] = k + i + ran;
        }
        
        uint64_t t1 = __rdtsc();
        kron_double(in_re1_d, in_im1_d, in_re2_d, in_im2_d, N);
        uint64_t t2 = __rdtsc();
        
        for (int32_t k=0;k<MAX_SIZE;k++)
        {
            gResistO3[caseid][i] = in_re2_d[k]/NUM_LOOP + in_im2_d[k]/NUM_LOOP + gResistO3_2/NUM_LOOP;
        }
        gCycleCount[caseid][i] = t2-t1;
        avg += t2-t1;
    }
    //avg /= NUM_LOOP;
    printf(" case %d: calc_kron_%d_avx512_double, cycle total=%lu\n", caseid, N, avg);
    display(caseid);

}

void coma_avg(int32_t len, float* out_re,float* out_im,float* in_re1,float* in_im1,
                              float* in_re2,float* in_im2,float r1,float r2)
{
    len *= 2;
    if (len*2<= DATA512Float_LOOP)
    {
        __m512 re = _mm512_add_ps(_mm512_mul_ps(_mm512_set1_ps(r1),_mm512_load_ps(in_re1)),_mm512_mul_ps(_mm512_set1_ps(r2),_mm512_load_ps(in_re2)));
        __m512 im = _mm512_add_ps(_mm512_mul_ps(_mm512_set1_ps(r1),_mm512_load_ps(in_im1)),_mm512_mul_ps(_mm512_set1_ps(r2),_mm512_load_ps(in_im2)));
        *(__m512*)(out_re) = re;
        *(__m512*)(out_im) = im;
        gResistO3_3 += *(float*)(in_re1) + *(float*)(in_im1) + *(float*)(in_re2) + *(float*)(in_im2);
    }
    else
    {
        int32_t loop = len / DATA512Float_LOOP;
        for(int32_t i = 0; i < loop ; i++)
        {
            __m512 re = _mm512_add_ps(_mm512_mul_ps(_mm512_set1_ps(r1),_mm512_load_ps(in_re1 + i * DATA512Float_LOOP)),_mm512_mul_ps(_mm512_set1_ps(r2),_mm512_load_ps(in_re2 + i * DATA512Float_LOOP)));
            __m512 im = _mm512_add_ps(_mm512_mul_ps(_mm512_set1_ps(r1),_mm512_load_ps(in_im1 + i * DATA512Float_LOOP)),_mm512_mul_ps(_mm512_set1_ps(r2),_mm512_load_ps(in_im2 + i * DATA512Float_LOOP)));
            *(__m512*)(out_re + i * DATA512Float_LOOP) = re;
            *(__m512*)(out_im + i * DATA512Float_LOOP) = im;
            gResistO3_3 += *(in_re1 + i * DATA512Float_LOOP) + *(in_im1 + i * DATA512Float_LOOP) + 
                           *(in_re2 + i * DATA512Float_LOOP) + *(in_im2 + i * DATA512Float_LOOP) + 
                           *(out_re + i * DATA512Float_LOOP) + *(out_im + i * DATA512Float_LOOP);
        }
    }
}
                              
void coma_avg_double(int32_t len, double* out_re,double* out_im,double* in_re1,double* in_im1,
                                       double* in_re2,double* in_im2,double r1,double r2)
{
    len *= 2;
    /*if (len <= DATA512Double_LOOP)
    {
        __m512d re = _mm512_add_pd(_mm512_mul_pd(_mm512_set1_pd(r1),_mm512_load_pd(in_re1)),_mm512_mul_pd(_mm512_set1_pd(r2),_mm512_load_pd(in_re2)));
        __m512d im = _mm512_add_pd(_mm512_mul_pd(_mm512_set1_pd(r1),_mm512_load_pd(in_im1)),_mm512_mul_pd(_mm512_set1_pd(r2),_mm512_load_pd(in_im2)));
        *(__m512d*)(out_re) = re;
        *(__m512d*)(out_im) = im;
        
        gResistO3_3 += *(double*)(in_re1) + *(double*)(in_im1) + *(double*)(in_re2) + *(double*)(in_im2);
    }
    else*/
    {
        int32_t loop = len / DATA512Double_LOOP;
        for(int32_t i = 0; i < loop ; i++)
        {
            __m512d re = _mm512_add_pd(_mm512_mul_pd(_mm512_set1_pd(r1),_mm512_load_pd(in_re1 + i * DATA512Double_LOOP)),_mm512_mul_pd(_mm512_set1_pd(r2),_mm512_load_pd(in_re2 + i * DATA512Double_LOOP)));
            __m512d im = _mm512_add_pd(_mm512_mul_pd(_mm512_set1_pd(r1),_mm512_load_pd(in_im1 + i * DATA512Double_LOOP)),_mm512_mul_pd(_mm512_set1_pd(r2),_mm512_load_pd(in_im2 + i * DATA512Double_LOOP)));
            *(__m512d*)(out_re + i * DATA512Double_LOOP) = re;
            *(__m512d*)(out_im + i * DATA512Double_LOOP) = im;
            gResistO3_3 += *(in_re1 + i * DATA512Double_LOOP) + *(in_im1 + i * DATA512Double_LOOP) + 
                           *(in_re2 + i * DATA512Double_LOOP) + *(in_im2 + i * DATA512Double_LOOP) + 
                           *(out_re + i * DATA512Double_LOOP) + *(out_im + i * DATA512Double_LOOP);
        }
    }
}
                                       
void calc_coma_avg_avx512_float(int32_t caseid, int32_t N)
{   
    float out_re[MAX_SIZE] = {0};
    float out_im[MAX_SIZE] = {0};
    float in_re1[MAX_SIZE] = {0};
    float in_im1[MAX_SIZE] = {0};
    float in_re2[MAX_SIZE] = {0};
    float in_im2[MAX_SIZE] = {0};
    float r1 = 0;
    float r2 = 0;
    
    uint64_t avg = 0;
    for (int32_t i=0;i<NUM_LOOP;i++)
    {
        int32_t ran = rand()%50;
        for (int32_t k=0;k<MAX_SIZE;k++)
        {
            in_re1[k] = k + i + ran;
            in_im1[k] = k + i + ran;
            in_re2[k] = k + i + ran;
            in_im2[k] = k + i + ran;
        }
        
        r1 = i;
        r2 = NUM_LOOP - i;
        
        uint64_t t1 = __rdtsc();
        coma_avg(N, out_re, out_im, in_re1, in_im1, in_re2, in_im2, r1, r2);
        uint64_t t2 = __rdtsc();
        
        for (int32_t k=0;k<MAX_SIZE;k++)
        {
            gResistO3[caseid][i] = out_re[i]/NUM_LOOP + out_im[i]/NUM_LOOP + gResistO3_3/NUM_LOOP;
        }
        gCycleCount[caseid][i] = t2-t1;
        avg += t2-t1;
    }
    //avg /= NUM_LOOP;
    printf(" case %d: calc_coma_avg_%d_avx512_float, cycle total=%lu\n", caseid, N, avg);
    display(caseid);

}    

void calc_coma_avg_avx512_double(int32_t caseid, int32_t N)
{   
    double out_re_d[MAX_SIZE] = {0};
    double out_im_d[MAX_SIZE] = {0};
    double in_re1_d[MAX_SIZE] = {0};
    double in_im1_d[MAX_SIZE] = {0};
    double in_re2_d[MAX_SIZE] = {0};
    double in_im2_d[MAX_SIZE] = {0};
    double r1 = 0;
    double r2 = 0;
    
    uint64_t avg = 0;
    for (int32_t i=0;i<NUM_LOOP;i++)
    {
        int32_t ran = rand()%50;
        for (int32_t k=0;k<MAX_SIZE;k++)
        {
            in_re1_d[k] = k + i + ran;
            in_im1_d[k] = k + i + ran;
            in_re2_d[k] = k + i + ran;
            in_im2_d[k] = k + i + ran;
        }
        
        r1 = i;
        r2 = NUM_LOOP - i;
        
        uint64_t t1 = __rdtsc();
        coma_avg_double(N, out_re_d, out_im_d, in_re1_d, in_im1_d, in_re2_d, in_im2_d, r1, r2);
        uint64_t t2 = __rdtsc();
        
        for (int32_t k=0;k<MAX_SIZE;k++)
        {
            //gResistO3[caseid][i] = out_re_d[i]/NUM_LOOP + out_im_d[i]/NUM_LOOP + gResistO3_3/NUM_LOOP;
        }
        gCycleCount[caseid][i] = t2-t1;
        avg += t2-t1;
    }
    //avg /= NUM_LOOP;
    printf(" case %d: calc_coma_avg_%d_avx512_double, cycle total=%lu\n", caseid, N, avg);
    display(caseid);

}

int main(int argc, char *argv[])
{
    printf("****************************\n");
    printf(" case start \n");
    printf("****************************\n");

    memset(gCycleCount, 0, sizeof(int32_t) * NUM_CASE * NUM_LOOP);
    
    calc_coma_avx512_float(0, 4);
    calc_coma_avx512_float(1, 8);
    calc_coma_avx512_float(2, 32);
    calc_coma_avx512_float(3, 64);
    printf("****************************\n");

    calc_coma_avx512_double(4, 4);
    calc_coma_avx512_double(5, 8);
    calc_coma_avx512_double(6, 32);
    calc_coma_avx512_double(7, 64);
    printf("****************************\n");

    calc_kron_avx512_float(8, 4);
    calc_kron_avx512_float(9, 8);
    calc_kron_avx512_float(10, 32);
    calc_kron_avx512_float(11, 64);
    printf("****************************\n");
        
    calc_kron_avx512_double(12, 4);
    calc_kron_avx512_double(13, 8);
    calc_kron_avx512_double(14, 32);
    calc_kron_avx512_double(15, 64);
    printf("****************************\n");
    
    calc_coma_avg_avx512_float(16, 4);
    calc_coma_avg_avx512_float(17, 8);
    calc_coma_avg_avx512_float(18, 32);
    calc_coma_avg_avx512_float(19, 64);
    printf("****************************\n");

    calc_coma_avg_avx512_double(20, 4);
    calc_coma_avg_avx512_double(21, 8);
    calc_coma_avg_avx512_double(22, 32);
    calc_coma_avg_avx512_double(23, 64);
    
    gResistO3_4 = gResistO3_1 + gResistO3_2 + gResistO3_3;
    printf("****************************\n");
    printf(" case end \n");
    printf("****************************%f\n",gResistO3_4);

    return 0;
}
