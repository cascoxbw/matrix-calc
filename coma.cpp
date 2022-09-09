#include <stdio.h>
#include <immintrin.h>
#include <string.h>
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <numeric>      // std::accumulate
#include <unistd.h>     // sleep


typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;

#define MAX_SIZE            128*128
#define DATA512Float_LOOP   16
#define DATA512Double_LOOP  8
#define NUM_CASE            32
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
    
    double sum = std::accumulate(v0.begin(), v0.end(), 0);  
    double avg =  sum / v0.size(); 
    if (caseid == 31 && NUM_LOOP >= 10000) 
    {
        sum = 0;
        for (int32_t i = 0; i < v0.size(); i++)
        {
            sum += v0[i] / 10;
        }
        avg = sum / (v0.size() / 10);
    }

    //auto maxPosition = max_element(v0.begin(), v0.end());
    //auto minPosition = min_element(v0.begin(), v0.end());
    
    
    printf(" case %d: cycle total=%d, loop num=%zu, cycle avg=%d, cycle 95=%d, cycle 99=%d, cycle 100=%d\n", caseid, (int32_t)sum, v0.size(), (int32_t)avg, v0[NUM_LOOP*0.95], v0[NUM_LOOP*0.99], v0[NUM_LOOP*0.999]);
    //sleep(5);
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
        gResistO3_1 += *(coma_re+i*len) + *(coma_im+i*len);
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
        gResistO3_1 += *(coma_re+i*len) + *(coma_im+i*len);
    }
}

void calc_coma_avx512_float(int32_t caseid, int32_t N)
{   
    float out_re[MAX_SIZE] = {0};
    float out_im[MAX_SIZE] = {0};
    float in_re[MAX_SIZE]  = {0};
    float in_im[MAX_SIZE]  = {0};

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
        gCycleCount[caseid][i] = t2-t1;

    }
    
    printf(" case %d: calc_coma_%d_avx512_float\n", caseid, N);
    display(caseid);
}

void calc_coma_avx512_double(int32_t caseid, int32_t N)
{   
    double out_re_d[MAX_SIZE] = {0};
    double out_im_d[MAX_SIZE] = {0};
    double in_re_d[MAX_SIZE]  = {0};
    double in_im_d[MAX_SIZE]  = {0};

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
        gCycleCount[caseid][i] = t2-t1;
        
    }

    printf(" case %d: calc_coma_%d_avx512_double\n", caseid, N);
    display(caseid);

}

void kron(float* in_re1, float* in_im1,float* in_re2, float* in_im2, int32_t len)
{
    if(len<=DATA512Float_LOOP) //len==4,8
    {
        for(int32_t i = 0; i < len; i++)
        {
            *(__m512*)(in_re2+i*len*2) = _mm512_load_ps(in_re1+i*len);
            *(__m512*)(in_re2+i*len*2+len) = _mm512_setzero_ps();
            *(__m512*)(in_im2+i*len*2) = _mm512_load_ps(in_im1+i*len);
            *(__m512*)(in_im2+i*len*2+len) = _mm512_setzero_ps();
            gResistO3_2 += *(in_re2+i*len) + *(in_re2+i*len+len) + 
                           *(in_im2+i*len) + *(in_im2+i*len+len);
        }
        
        for(int32_t i = 0; i < len; i++)
        {
            *(__m512*)(in_re2+(i+len)*len*2) = _mm512_setzero_ps();
            *(__m512*)(in_re2+(i+len)*len*2+len) = _mm512_load_ps(in_re1+i*len);
            *(__m512*)(in_im2+(i+len)*len*2) = _mm512_setzero_ps();
            *(__m512*)(in_im2+(i+len)*len*2+len) = _mm512_load_ps(in_im1+i*len);
            gResistO3_2 += *(in_re2+(i+len)*len) + *(in_re2+(i+len)*len+len) + 
                           *(in_im2+(i+len)*len) + *(in_im2+(i+len)*len+len);
        }
    }
    else
    {
        int32_t loop = len/DATA512Float_LOOP;
            
        for(int32_t i = 0; i < len; i++)
        {
            for(int32_t j = 0; j < loop; j++)
            *(__m512*)(in_re2+i*len*2+j*DATA512Float_LOOP) = _mm512_load_ps(in_re1+i*len+j*DATA512Float_LOOP);
            for(int32_t j = 0; j < loop; j++)
            *(__m512*)(in_re2+i*len*2+j*DATA512Float_LOOP+len) = _mm512_setzero_ps();
            for(int32_t j = 0; j < loop; j++)
            *(__m512*)(in_im2+i*len*2+j*DATA512Float_LOOP) = _mm512_load_ps(in_im1+i*len+j*DATA512Float_LOOP);
            for(int32_t j = 0; j < loop; j++)
            *(__m512*)(in_im2+i*len*2+j*DATA512Float_LOOP+len) = _mm512_setzero_ps();
            gResistO3_2 += *(in_re2+i*len) + *(in_re2+i*len+len) + 
                           *(in_im2+i*len) + *(in_im2+i*len+len);
        }
         
        for(int32_t i = 0; i < len; i++)
        {
            for(int32_t j = 0; j < loop; j++)
            *(__m512*)(in_re2+(i+len)*len*2+j*DATA512Float_LOOP) = _mm512_setzero_ps();
            for(int32_t j = 0; j < loop; j++)
            *(__m512*)(in_re2+(i+len)*len*2+j*DATA512Float_LOOP+len) = _mm512_load_ps(in_re1+i*len+j*DATA512Float_LOOP);
            for(int32_t j = 0; j < loop; j++)
            *(__m512*)(in_im2+(i+len)*len*2+j*DATA512Float_LOOP) = _mm512_setzero_ps();
            for(int32_t j = 0; j < loop; j++)
            *(__m512*)(in_im2+(i+len)*len*2+j*DATA512Float_LOOP+len) = _mm512_load_ps(in_im1+i*len+j*DATA512Float_LOOP);
            gResistO3_2 += *(in_re2+(i+len)*len) + *(in_re2+(i+len)*len+len) + 
                           *(in_im2+(i+len)*len) + *(in_im2+(i+len)*len+len);
        }
                
    }
}

void kron_double(double* in_re1, double* in_im1,double* in_re2, double* in_im2, int32_t len)
{    
    if(len<=DATA512Double_LOOP) //len==4,8
    {
        for(int32_t i = 0; i < len; i++)
        {
            *(__m512d*)(in_re2+i*len*2) = _mm512_load_pd(in_re1+i*len);
            *(__m512d*)(in_re2+i*len*2+len) = _mm512_setzero_pd();
            *(__m512d*)(in_im2+i*len*2) = _mm512_load_pd(in_im1+i*len);
            *(__m512d*)(in_im2+i*len*2+len) = _mm512_setzero_pd();
            gResistO3_2 += *(in_re2+i*len) + *(in_re2+i*len+len) + 
                           *(in_im2+i*len) + *(in_im2+i*len+len);
        }
        
        for(int32_t i = 0; i < len; i++)
        {
            *(__m512d*)(in_re2+(i+len)*len*2) = _mm512_setzero_pd();
            *(__m512d*)(in_re2+(i+len)*len*2+len) = _mm512_load_pd(in_re1+i*len);
            *(__m512d*)(in_im2+(i+len)*len*2) = _mm512_setzero_pd();
            *(__m512d*)(in_im2+(i+len)*len*2+len) = _mm512_load_pd(in_im1+i*len);
            gResistO3_2 += *(in_re2+(i+len)*len) + *(in_re2+(i+len)*len+len) + 
                           *(in_im2+(i+len)*len) + *(in_im2+(i+len)*len+len);
        }
    }
    else
    {
        int32_t loop = len/DATA512Double_LOOP;
            
        for(int32_t i = 0; i < len; i++)
        {
            for(int32_t j = 0; j < loop; j++)
            *(__m512d*)(in_re2+i*len*2+j*DATA512Double_LOOP) = _mm512_load_pd(in_re1+i*len+j*DATA512Double_LOOP);
            for(int32_t j = 0; j < loop; j++)
            *(__m512d*)(in_re2+i*len*2+j*DATA512Double_LOOP+len) = _mm512_setzero_pd();
            for(int32_t j = 0; j < loop; j++)
            *(__m512d*)(in_im2+i*len*2+j*DATA512Double_LOOP) = _mm512_load_pd(in_im1+i*len+j*DATA512Double_LOOP);
            for(int32_t j = 0; j < loop; j++)
            *(__m512d*)(in_im2+i*len*2+j*DATA512Double_LOOP+len) = _mm512_setzero_pd();
            gResistO3_2 += *(in_re2+i*len) + *(in_re2+i*len+len) + 
                           *(in_im2+i*len) + *(in_im2+i*len+len);
        }
         
        for(int32_t i = 0; i < len; i++)
        {
            for(int32_t j = 0; j < loop; j++)
            *(__m512d*)(in_re2+(i+len)*len*2+j*DATA512Double_LOOP) = _mm512_setzero_pd();
            for(int32_t j = 0; j < loop; j++)
            *(__m512d*)(in_re2+(i+len)*len*2+j*DATA512Double_LOOP+len) = _mm512_load_pd(in_re1+i*len+j*DATA512Double_LOOP);
            for(int32_t j = 0; j < loop; j++)
            *(__m512d*)(in_im2+(i+len)*len*2+j*DATA512Double_LOOP) = _mm512_setzero_pd();
            for(int32_t j = 0; j < loop; j++)
            *(__m512d*)(in_im2+(i+len)*len*2+j*DATA512Double_LOOP+len) = _mm512_load_pd(in_im1+i*len+j*DATA512Double_LOOP);
            gResistO3_2 += *(in_re2+(i+len)*len) + *(in_re2+(i+len)*len+len) + 
                           *(in_im2+(i+len)*len) + *(in_im2+(i+len)*len+len);
        }
    }
}

void calc_kron_avx512_float(int32_t caseid, int32_t N)
{   
    float in_re1[MAX_SIZE] = {0};
    float in_im1[MAX_SIZE] = {0};
    float in_re2[MAX_SIZE] = {0};
    float in_im2[MAX_SIZE] = {0};

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
        gCycleCount[caseid][i] = t2-t1;
        
    }

    printf(" case %d: calc_kron_%d_avx512_float\n", caseid, N);
    display(caseid);

}

void calc_kron_avx512_double(int32_t caseid, int32_t N)
{   
    double in_re1_d[MAX_SIZE] = {0};
    double in_im1_d[MAX_SIZE] = {0};
    double in_re2_d[MAX_SIZE] = {0};
    double in_im2_d[MAX_SIZE] = {0};

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
        gCycleCount[caseid][i] = t2-t1;
        
    }

    printf(" case %d: calc_kron_%d_avx512_double\n", caseid, N);
    display(caseid);

}

void coma_avg(int32_t len, float* out_re,float* out_im,float* in_re1,float* in_im1,
                              float* in_re2,float* in_im2,float r1,float r2)
{
    int32_t loop = (len*len / DATA512Float_LOOP);
   
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
                              
void coma_avg_double(int32_t len, double* out_re,double* out_im,double* in_re1,double* in_im1,
                                       double* in_re2,double* in_im2,double r1,double r2)
{
    int32_t loop = len*len / DATA512Double_LOOP;
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
        gCycleCount[caseid][i] = t2-t1;
        
    }

    printf(" case %d: calc_coma_avg_%d_avx512_float\n", caseid, N);
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
        gCycleCount[caseid][i] = t2-t1;
        
    }

    printf(" case %d: calc_coma_avg_%d_avx512_double\n", caseid, N);
    display(caseid);

}

static __m512 constZero = _mm512_setzero_ps();

// complex multiplication: A * A'
#define GET_AxAH(re, im, out)\
{\
    out = _mm512_add_ps(_mm512_mul_ps(re, re), _mm512_mul_ps(im, im));\
}
#define GET_AxAH_double(re, im, out)\
{\
    out = _mm512_add_pd(_mm512_mul_pd(re, re), _mm512_mul_pd(im, im));\
}

// complex multiplication: A * B'
#define GET_AxBH(are, aim, bre, bim, outre, outim)\
{\
    outre = _mm512_add_ps(_mm512_mul_ps(are, bre), _mm512_mul_ps(aim, bim));\
    outim = _mm512_sub_ps(_mm512_mul_ps(bre, aim), _mm512_mul_ps(are, bim));\
}
#define GET_AxBH_double(are, aim, bre, bim, outre, outim)\
{\
    outre = _mm512_add_pd(_mm512_mul_pd(are, bre), _mm512_mul_pd(aim, bim));\
    outim = _mm512_sub_pd(_mm512_mul_pd(bre, aim), _mm512_mul_pd(are, bim));\
}

// complex multiplication: A' * B
#define GET_AHxB(are, aim, bre, bim, outre, outim)\
{\
    outre = _mm512_add_ps(_mm512_mul_ps(are, bre), _mm512_mul_ps(aim, bim));\
    outim = _mm512_sub_ps(_mm512_mul_ps(are, bim), _mm512_mul_ps(bre, aim));\
}
#define GET_AHxB_double(are, aim, bre, bim, outre, outim)\
{\
    outre = _mm512_add_pd(_mm512_mul_pd(are, bre), _mm512_mul_pd(aim, bim));\
    outim = _mm512_sub_pd(_mm512_mul_pd(are, bim), _mm512_mul_pd(bre, aim));\
}

// complex multiplication: A * B
#define GET_AxB(are, aim, bre, bim, outre, outim)\
{\
    outre = _mm512_sub_ps(_mm512_mul_ps(are, bre), _mm512_mul_ps(aim, bim));\
    outim = _mm512_add_ps(_mm512_mul_ps(bre, aim), _mm512_mul_ps(are, bim));\
}
#define GET_AxB_double(are, aim, bre, bim, outre, outim)\
{\
    outre = _mm512_sub_pd(_mm512_mul_pd(are, bre), _mm512_mul_pd(aim, bim));\
    outim = _mm512_add_pd(_mm512_mul_pd(bre, aim), _mm512_mul_pd(are, bim));\
}

// complex multiplication: A * real(B)
#define GET_AxRealB(are, aim, bre, outre, outim)\
{\
    outre = _mm512_mul_ps(are, bre);\
    outim = _mm512_mul_ps(aim, bre);\
}
#define GET_AxRealB_double(are, aim, bre, outre, outim)\
{\
    outre = _mm512_mul_pd(are, bre);\
    outim = _mm512_mul_pd(aim, bre);\
}

// get G00
#define GET_G00(matGRe, matBRe, matD, matND)\
{\
    matD[0] = _mm512_rsqrt14_ps(matBRe[0][0]);\
    matND[0] = _mm512_sub_ps(constZero, matD[0]);\
}

#define GET_G00_double(matGRe, matBRe, matD, matND)\
{\
    matD[0][0] = _mm512_rsqrt14_pd(matBRe[0][0][0]);\
    matND[0][0] = _mm512_sub_pd(constZero, matD[0][0]);\
    matD[0][1] = _mm512_rsqrt14_pd(matBRe[0][0][1]);\
    matND[0][1] = _mm512_sub_pd(constZero, matD[0][1]);\
}

// get column 0 of matrix G
#define GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, j)\
{\
    matGRe[j][0] = _mm512_mul_ps(matBRe[j][0], matD[0]);\
    matGIm[j][0] = _mm512_mul_ps(matBIm[j][0], matD[0]);\
}

#define GET_G_COL0_double(matGRe, matGIm, matBRe, matBIm, matD, j)\
{\
    matGRe[j][0][0] = _mm512_mul_pd(matBRe[j][0][0], matD[0][0]);\
    matGIm[j][0][0] = _mm512_mul_pd(matBIm[j][0][0], matD[0][0]);\
    matGRe[j][0][1] = _mm512_mul_pd(matBRe[j][0][1], matD[0][1]);\
    matGIm[j][0][1] = _mm512_mul_pd(matBIm[j][0][1], matD[0][1]);\
}

// get G11
#define GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0)\
{\
    GET_AxAH(matGRe[1][0], matGIm[1][0], temp0);\
    matGRe[1][1] = _mm512_sub_ps(matBRe[1][1], temp0);\
    matD[1] = _mm512_rsqrt14_ps(matGRe[1][1]);\
    matND[1] = _mm512_sub_ps(constZero, matD[1]);\
}

#define GET_G11_double(matGRe, matGIm, matBRe, matD, matND, temp0)\
{\
    GET_AxAH_double(matGRe[1][0][0], matGIm[1][0][0], temp0[0]);\
    matGRe[1][1][0] = _mm512_sub_pd(matBRe[1][1][0], temp0[0]);\
    matD[1][0] = _mm512_rsqrt14_pd(matGRe[1][1][0]);\
    matND[1][0] = _mm512_sub_pd(constZero, matD[1][0]);\
    GET_AxAH_double(matGRe[1][0][1], matGIm[1][0][1], temp0[1]);\
    matGRe[1][1][1] = _mm512_sub_pd(matBRe[1][1][1], temp0[1]);\
    matD[1][1] = _mm512_rsqrt14_pd(matGRe[1][1][1]);\
    matND[1][1] = _mm512_sub_pd(constZero, matD[1][1]);\
}

// get column 1 of matrix G
#define GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, j, temp0, temp1)\
{\
    GET_AxBH(matGRe[j][0], matGIm[j][0], matGRe[1][0], matGIm[1][0], temp0, temp1);\
    matGRe[j][1] = _mm512_sub_ps(matBRe[j][1], temp0);\
    matGIm[j][1] = _mm512_sub_ps(matBIm[j][1], temp1);\
    matGRe[j][1] = _mm512_mul_ps(matGRe[j][1], matD[1]);\
    matGIm[j][1] = _mm512_mul_ps(matGIm[j][1], matD[1]);\
}

#define GET_G_COL1_double(matGRe, matGIm, matBRe, matBIm, matD, j, temp0, temp1)\
{\
    GET_AxBH_double(matGRe[j][0][0], matGIm[j][0][0], matGRe[1][0][0], matGIm[1][0][0], temp0[0], temp1[0]);\
    matGRe[j][1][0] = _mm512_sub_pd(matBRe[j][1][0], temp0[0]);\
    matGIm[j][1][0] = _mm512_sub_pd(matBIm[j][1][0], temp1[0]);\
    matGRe[j][1][0] = _mm512_mul_pd(matGRe[j][1][0], matD[1][0]);\
    matGIm[j][1][0] = _mm512_mul_pd(matGIm[j][1][0], matD[1][0]);\
    GET_AxBH_double(matGRe[j][0][1], matGIm[j][0][1], matGRe[1][0][1], matGIm[1][0][1], temp0[1], temp1[1]);\
    matGRe[j][1][1] = _mm512_sub_pd(matBRe[j][1][1], temp0[1]);\
    matGIm[j][1][1] = _mm512_sub_pd(matBIm[j][1][1], temp1[1]);\
    matGRe[j][1][1] = _mm512_mul_pd(matGRe[j][1][1], matD[1][1]);\
    matGIm[j][1][1] = _mm512_mul_pd(matGIm[j][1][1], matD[1][1]);\
}

// get G22
#define GET_G22(matGRe, matGIm, matBRe, matD, matND, temp0, temp1)\
{\
    GET_AxAH(matGRe[2][0], matGIm[2][0], temp0);\
    GET_AxAH(matGRe[2][1], matGIm[2][1], temp1);\
    matGRe[2][2] = _mm512_sub_ps(matBRe[2][2], temp0);\
    matGRe[2][2] = _mm512_sub_ps(matGRe[2][2], temp1);\
    matD[2] = _mm512_rsqrt14_ps(matGRe[2][2]);\
    matND[2] = _mm512_sub_ps(constZero, matD[2]);\
}
#define GET_G22_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1)\
{\
    GET_AxAH_double(matGRe[2][0][0], matGIm[2][0][0], temp0[0]);\
    GET_AxAH_double(matGRe[2][1][0], matGIm[2][1][0], temp1[0]);\
    matGRe[2][2][0] = _mm512_sub_pd(matBRe[2][2][0], temp0[0]);\
    matGRe[2][2][0] = _mm512_sub_pd(matGRe[2][2][0], temp1[0]);\
    matD[2][0] = _mm512_rsqrt14_pd(matGRe[2][2][0]);\
    matND[2][0] = _mm512_sub_pd(constZero, matD[2][0]);\
    GET_AxAH_double(matGRe[2][0][1], matGIm[2][0][1], temp0[1]);\
    GET_AxAH_double(matGRe[2][1][1], matGIm[2][1][1], temp1[1]);\
    matGRe[2][2][1] = _mm512_sub_pd(matBRe[2][2][1], temp0[1]);\
    matGRe[2][2][1] = _mm512_sub_pd(matGRe[2][2][1], temp1[1]);\
    matD[2][1] = _mm512_rsqrt14_pd(matGRe[2][2][1]);\
    matND[2][1] = _mm512_sub_pd(constZero, matD[2][1]);\
}

// get column 2 of matrix G
#define GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, j, temp0, temp1)\
{\
    GET_AxBH(matGRe[j][0], matGIm[j][0], matGRe[2][0], matGIm[2][0], temp0, temp1);\
    matGRe[j][2] = _mm512_sub_ps(matBRe[j][2], temp0);\
    matGIm[j][2] = _mm512_sub_ps(matBIm[j][2], temp1);\
    GET_AxBH(matGRe[j][1], matGIm[j][1], matGRe[2][1], matGIm[2][1], temp0, temp1);\
    matGRe[j][2] = _mm512_sub_ps(matGRe[j][2], temp0);\
    matGIm[j][2] = _mm512_sub_ps(matGIm[j][2], temp1);\
    matGRe[j][2] = _mm512_mul_ps(matGRe[j][2], matD[2]);\
    matGIm[j][2] = _mm512_mul_ps(matGIm[j][2], matD[2]);\
}
#define GET_G_COL2_double(matGRe, matGIm, matBRe, matBIm, matD, j, temp0, temp1)\
{\
    GET_AxBH_double(matGRe[j][0][0], matGIm[j][0][0], matGRe[2][0][0], matGIm[2][0][0], temp0[0], temp1[0]);\
    matGRe[j][2][0] = _mm512_sub_pd(matBRe[j][2][0], temp0[0]);\
    matGIm[j][2][0] = _mm512_sub_pd(matBIm[j][2][0], temp1[0]);\
    GET_AxBH_double(matGRe[j][1][0], matGIm[j][1][0], matGRe[2][1][0], matGIm[2][1][0], temp0[0], temp1[0]);\
    matGRe[j][2][0] = _mm512_sub_pd(matGRe[j][2][0], temp0[0]);\
    matGIm[j][2][0] = _mm512_sub_pd(matGIm[j][2][0], temp1[0]);\
    matGRe[j][2][0] = _mm512_mul_pd(matGRe[j][2][0], matD[2][0]);\
    matGIm[j][2][0] = _mm512_mul_pd(matGIm[j][2][0], matD[2][0]);\
    GET_AxBH_double(matGRe[j][0][1], matGIm[j][0][1], matGRe[2][0][1], matGIm[2][0][1], temp0[1], temp1[1]);\
    matGRe[j][2][1] = _mm512_sub_pd(matBRe[j][2][1], temp0[1]);\
    matGIm[j][2][1] = _mm512_sub_pd(matBIm[j][2][1], temp1[1]);\
    GET_AxBH_double(matGRe[j][1][1], matGIm[j][1][1], matGRe[2][1][1], matGIm[2][1][1], temp0[1], temp1[1]);\
    matGRe[j][2][1] = _mm512_sub_pd(matGRe[j][2][1], temp0[1]);\
    matGIm[j][2][1] = _mm512_sub_pd(matGIm[j][2][1], temp1[1]);\
    matGRe[j][2][1] = _mm512_mul_pd(matGRe[j][2][1], matD[2][1]);\
    matGIm[j][2][1] = _mm512_mul_pd(matGIm[j][2][1], matD[2][1]);\
}


// get G33
#define GET_G33(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2)\
{\
    GET_AxAH(matGRe[3][0], matGIm[3][0], temp0);\
    GET_AxAH(matGRe[3][1], matGIm[3][1], temp1);\
    GET_AxAH(matGRe[3][2], matGIm[3][2], temp2);\
    matGRe[3][3] = _mm512_sub_ps(matBRe[3][3], temp0);\
    matGRe[3][3] = _mm512_sub_ps(matGRe[3][3], temp1);\
    matGRe[3][3] = _mm512_sub_ps(matGRe[3][3], temp2);\
    matD[3] = _mm512_rsqrt14_ps(matGRe[3][3]);\
    matND[3] = _mm512_sub_ps(constZero, matD[3]);\
}
#define GET_G33_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2)\
{\
    GET_AxAH_double(matGRe[3][0][0], matGIm[3][0][0], temp0[0]);\
    GET_AxAH_double(matGRe[3][1][0], matGIm[3][1][0], temp1[0]);\
    GET_AxAH_double(matGRe[3][2][0], matGIm[3][2][0], temp2[0]);\
    matGRe[3][3][0] = _mm512_sub_pd(matBRe[3][3][0], temp0[0]);\
    matGRe[3][3][0] = _mm512_sub_pd(matGRe[3][3][0], temp1[0]);\
    matGRe[3][3][0] = _mm512_sub_pd(matGRe[3][3][0], temp2[0]);\
    matD[3][0] = _mm512_rsqrt14_pd(matGRe[3][3][0]);\
    matND[3][0] = _mm512_sub_pd(constZero, matD[3][0]);\
    GET_AxAH_double(matGRe[3][0][1], matGIm[3][0][1], temp0[1]);\
    GET_AxAH_double(matGRe[3][1][1], matGIm[3][1][1], temp1[1]);\
    GET_AxAH_double(matGRe[3][2][1], matGIm[3][2][1], temp2[1]);\
    matGRe[3][3][1] = _mm512_sub_pd(matBRe[3][3][1], temp0[1]);\
    matGRe[3][3][1] = _mm512_sub_pd(matGRe[3][3][1], temp1[1]);\
    matGRe[3][3][1] = _mm512_sub_pd(matGRe[3][3][1], temp2[1]);\
    matD[3][1] = _mm512_rsqrt14_pd(matGRe[3][3][1]);\
    matND[3][1] = _mm512_sub_pd(constZero, matD[3][1]);\
}

// get column 3 of matrix G
#define GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, j, temp0, temp1)\
{\
    GET_AxBH(matGRe[j][0], matGIm[j][0], matGRe[3][0], matGIm[3][0], temp0, temp1);\
    matGRe[j][3] = _mm512_sub_ps(matBRe[j][3], temp0);\
    matGIm[j][3] = _mm512_sub_ps(matBIm[j][3], temp1);\
    GET_AxBH(matGRe[j][1], matGIm[j][1], matGRe[3][1], matGIm[3][1], temp0, temp1);\
    matGRe[j][3] = _mm512_sub_ps(matGRe[j][3], temp0);\
    matGIm[j][3] = _mm512_sub_ps(matGIm[j][3], temp1);\
    GET_AxBH(matGRe[j][2], matGIm[j][2], matGRe[3][2], matGIm[3][2], temp0, temp1);\
    matGRe[j][3] = _mm512_sub_ps(matGRe[j][3], temp0);\
    matGIm[j][3] = _mm512_sub_ps(matGIm[j][3], temp1);\
    matGRe[j][3] = _mm512_mul_ps(matGRe[j][3], matD[3]);\
    matGIm[j][3] = _mm512_mul_ps(matGIm[j][3], matD[3]);\
}
#define GET_G_COL3_double(matGRe, matGIm, matBRe, matBIm, matD, j, temp0, temp1)\
{\
    GET_AxBH_double(matGRe[j][0][0], matGIm[j][0][0], matGRe[3][0][0], matGIm[3][0][0], temp0[0], temp1[0]);\
    matGRe[j][3][0] = _mm512_sub_pd(matBRe[j][3][0], temp0[0]);\
    matGIm[j][3][0] = _mm512_sub_pd(matBIm[j][3][0], temp1[0]);\
    GET_AxBH_double(matGRe[j][1][0], matGIm[j][1][0], matGRe[3][1][0], matGIm[3][1][0], temp0[0], temp1[0]);\
    matGRe[j][3][0] = _mm512_sub_pd(matGRe[j][3][0], temp0[0]);\
    matGIm[j][3][0] = _mm512_sub_pd(matGIm[j][3][0], temp1[0]);\
    GET_AxBH_double(matGRe[j][2][0], matGIm[j][2][0], matGRe[3][2][0], matGIm[3][2][0], temp0[0], temp1[0]);\
    matGRe[j][3][0] = _mm512_sub_pd(matGRe[j][3][0], temp0[0]);\
    matGIm[j][3][0] = _mm512_sub_pd(matGIm[j][3][0], temp1[0]);\
    matGRe[j][3][0] = _mm512_mul_pd(matGRe[j][3][0], matD[3][0]);\
    matGIm[j][3][0] = _mm512_mul_pd(matGIm[j][3][0], matD[3][0]);\
    GET_AxBH_double(matGRe[j][0][1], matGIm[j][0][1], matGRe[3][0][1], matGIm[3][0][1], temp0[1], temp1[1]);\
    matGRe[j][3][1] = _mm512_sub_pd(matBRe[j][3][1], temp0[1]);\
    matGIm[j][3][1] = _mm512_sub_pd(matBIm[j][3][1], temp1[1]);\
    GET_AxBH_double(matGRe[j][1][1], matGIm[j][1][1], matGRe[3][1][1], matGIm[3][1][1], temp0[1], temp1[1]);\
    matGRe[j][3][1] = _mm512_sub_pd(matGRe[j][3][1], temp0[1]);\
    matGIm[j][3][1] = _mm512_sub_pd(matGIm[j][3][1], temp1[1]);\
    GET_AxBH_double(matGRe[j][2][1], matGIm[j][2][1], matGRe[3][2][1], matGIm[3][2][1], temp0[1], temp1[1]);\
    matGRe[j][3][1] = _mm512_sub_pd(matGRe[j][3][1], temp0[1]);\
    matGIm[j][3][1] = _mm512_sub_pd(matGIm[j][3][1], temp1[1]);\
    matGRe[j][3][1] = _mm512_mul_pd(matGRe[j][3][1], matD[3][1]);\
    matGIm[j][3][1] = _mm512_mul_pd(matGIm[j][3][1], matD[3][1]);\
}

// get G44
#define GET_G44(matGRe, matGIm, matBRe, matD, matND, temp0, temp1)\
{\
    GET_AxAH(matGRe[4][0], matGIm[4][0], temp0);\
    GET_AxAH(matGRe[4][1], matGIm[4][1], temp1);\
    matGRe[4][4] = _mm512_sub_ps(matBRe[4][4], temp0);\
    matGRe[4][4] = _mm512_sub_ps(matGRe[4][4], temp1);\
    GET_AxAH(matGRe[4][2], matGIm[4][2], temp0);\
    GET_AxAH(matGRe[4][3], matGIm[4][3], temp1);\
    matGRe[4][4] = _mm512_sub_ps(matGRe[4][4], temp0);\
    matGRe[4][4] = _mm512_sub_ps(matGRe[4][4], temp1);\
    matD[4] = _mm512_rsqrt14_ps(matGRe[4][4]);\
    matND[4] = _mm512_sub_ps(constZero, matD[4]);\
}
#define GET_G44_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1)\
{\
    GET_AxAH_double(matGRe[4][0][0], matGIm[4][0][0], temp0[0]);\
    GET_AxAH_double(matGRe[4][1][0], matGIm[4][1][0], temp1[0]);\
    matGRe[4][4][0] = _mm512_sub_pd(matBRe[4][4][0], temp0[0]);\
    matGRe[4][4][0] = _mm512_sub_pd(matGRe[4][4][0], temp1[0]);\
    GET_AxAH_double(matGRe[4][2][0], matGIm[4][2][0], temp0[0]);\
    GET_AxAH_double(matGRe[4][3][0], matGIm[4][3][0], temp1[0]);\
    matGRe[4][4][0] = _mm512_sub_pd(matGRe[4][4][0], temp0[0]);\
    matGRe[4][4][0] = _mm512_sub_pd(matGRe[4][4][0], temp1[0]);\
    matD[4][0] = _mm512_rsqrt14_pd(matGRe[4][4][0]);\
    matND[4][0] = _mm512_sub_pd(constZero, matD[4][0]);\
    GET_AxAH_double(matGRe[4][0][1], matGIm[4][0][1], temp0[1]);\
    GET_AxAH_double(matGRe[4][1][1], matGIm[4][1][1], temp1[1]);\
    matGRe[4][4][1] = _mm512_sub_pd(matBRe[4][4][1], temp0[1]);\
    matGRe[4][4][1] = _mm512_sub_pd(matGRe[4][4][1], temp1[1]);\
    GET_AxAH_double(matGRe[4][2][1], matGIm[4][2][1], temp0[1]);\
    GET_AxAH_double(matGRe[4][3][1], matGIm[4][3][1], temp1[1]);\
    matGRe[4][4][1] = _mm512_sub_pd(matGRe[4][4][1], temp0[1]);\
    matGRe[4][4][1] = _mm512_sub_pd(matGRe[4][4][1], temp1[1]);\
    matD[4][1] = _mm512_rsqrt14_pd(matGRe[4][4][1]);\
    matND[4][1] = _mm512_sub_pd(constZero, matD[4][1]);\
}

// get G55
#define GET_G55(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2)\
{\
    GET_AxAH(matGRe[5][0], matGIm[5][0], temp0);\
    GET_AxAH(matGRe[5][1], matGIm[5][1], temp1);\
    matGRe[5][5] = _mm512_sub_ps(matBRe[5][5], temp0);\
    matGRe[5][5] = _mm512_sub_ps(matGRe[5][5], temp1);\
    GET_AxAH(matGRe[5][2], matGIm[5][2], temp0);\
    GET_AxAH(matGRe[5][3], matGIm[5][3], temp1);\
    GET_AxAH(matGRe[5][4], matGIm[5][4], temp2);\
    matGRe[5][5] = _mm512_sub_ps(matGRe[5][5], temp0);\
    matGRe[5][5] = _mm512_sub_ps(matGRe[5][5], temp1);\
    matGRe[5][5] = _mm512_sub_ps(matGRe[5][5], temp2);\
    matD[5] = _mm512_rsqrt14_ps(matGRe[5][5]);\
    matND[5] = _mm512_sub_ps(constZero, matD[5]);\
}
#define GET_G55_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2)\
{\
    GET_AxAH_double(matGRe[5][0][0], matGIm[5][0][0], temp0[0]);\
    GET_AxAH_double(matGRe[5][1][0], matGIm[5][1][0], temp1[0]);\
    matGRe[5][5][0] = _mm512_sub_pd(matBRe[5][5][0], temp0[0]);\
    matGRe[5][5][0] = _mm512_sub_pd(matGRe[5][5][0], temp1[0]);\
    GET_AxAH_double(matGRe[5][2][0], matGIm[5][2][0], temp0[0]);\
    GET_AxAH_double(matGRe[5][3][0], matGIm[5][3][0], temp1[0]);\
    GET_AxAH_double(matGRe[5][4][0], matGIm[5][4][0], temp2[0]);\
    matGRe[5][5][0] = _mm512_sub_pd(matGRe[5][5][0], temp0[0]);\
    matGRe[5][5][0] = _mm512_sub_pd(matGRe[5][5][0], temp1[0]);\
    matGRe[5][5][0] = _mm512_sub_pd(matGRe[5][5][0], temp2[0]);\
    matD[5][0] = _mm512_rsqrt14_pd(matGRe[5][5][0]);\
    matND[5][0] = _mm512_sub_pd(constZero, matD[5][0]);\
    GET_AxAH_double(matGRe[5][0][1], matGIm[5][0][1], temp0[1]);\
    GET_AxAH_double(matGRe[5][1][1], matGIm[5][1][1], temp1[1]);\
    matGRe[5][5][1] = _mm512_sub_pd(matBRe[5][5][1], temp0[1]);\
    matGRe[5][5][1] = _mm512_sub_pd(matGRe[5][5][1], temp1[1]);\
    GET_AxAH_double(matGRe[5][2][1], matGIm[5][2][1], temp0[1]);\
    GET_AxAH_double(matGRe[5][3][1], matGIm[5][3][1], temp1[1]);\
    GET_AxAH_double(matGRe[5][4][1], matGIm[5][4][1], temp2[1]);\
    matGRe[5][5][1] = _mm512_sub_pd(matGRe[5][5][1], temp0[1]);\
    matGRe[5][5][1] = _mm512_sub_pd(matGRe[5][5][1], temp1[1]);\
    matGRe[5][5][1] = _mm512_sub_pd(matGRe[5][5][1], temp2[1]);\
    matD[5][1] = _mm512_rsqrt14_pd(matGRe[5][5][1]);\
    matND[5][1] = _mm512_sub_pd(constZero, matD[5][1]);\
}

// get G66
#define GET_G66(matGRe, matGIm, matBRe, matD, matND, temp0, temp1)\
{\
    GET_AxAH(matGRe[6][0], matGIm[6][0], temp0);\
    GET_AxAH(matGRe[6][1], matGIm[6][1], temp1);\
    matGRe[6][6] = _mm512_sub_ps(matBRe[6][6], temp0);\
    matGRe[6][6] = _mm512_sub_ps(matGRe[6][6], temp1);\
    GET_AxAH(matGRe[6][2], matGIm[6][2], temp0);\
    GET_AxAH(matGRe[6][3], matGIm[6][3], temp1);\
    matGRe[6][6] = _mm512_sub_ps(matGRe[6][6], temp0);\
    matGRe[6][6] = _mm512_sub_ps(matGRe[6][6], temp1);\
    GET_AxAH(matGRe[6][4], matGIm[6][4], temp0);\
    GET_AxAH(matGRe[6][5], matGIm[6][5], temp1);\
    matGRe[6][6] = _mm512_sub_ps(matGRe[6][6], temp0);\
    matGRe[6][6] = _mm512_sub_ps(matGRe[6][6], temp1);\
    matD[6] = _mm512_rsqrt14_ps(matGRe[6][6]);\
    matND[6] = _mm512_sub_ps(constZero, matD[6]);\
}
#define GET_G66_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1)\
{\
    GET_AxAH_double(matGRe[6][0][0], matGIm[6][0][0], temp0[0]);\
    GET_AxAH_double(matGRe[6][1][0], matGIm[6][1][0], temp1[0]);\
    matGRe[6][6][0] = _mm512_sub_pd(matBRe[6][6][0], temp0[0]);\
    matGRe[6][6][0] = _mm512_sub_pd(matGRe[6][6][0], temp1[0]);\
    GET_AxAH_double(matGRe[6][2][0], matGIm[6][2][0], temp0[0]);\
    GET_AxAH_double(matGRe[6][3][0], matGIm[6][3][0], temp1[0]);\
    matGRe[6][6][0] = _mm512_sub_pd(matGRe[6][6][0], temp0[0]);\
    matGRe[6][6][0] = _mm512_sub_pd(matGRe[6][6][0], temp1[0]);\
    GET_AxAH_double(matGRe[6][4][0], matGIm[6][4][0], temp0[0]);\
    GET_AxAH_double(matGRe[6][5][0], matGIm[6][5][0], temp1[0]);\
    matGRe[6][6][0] = _mm512_sub_pd(matGRe[6][6][0], temp0[0]);\
    matGRe[6][6][0] = _mm512_sub_pd(matGRe[6][6][0], temp1[0]);\
    matD[6][0] = _mm512_rsqrt14_pd(matGRe[6][6][0]);\
    matND[6][0] = _mm512_sub_pd(constZero, matD[6][0]);\
    GET_AxAH_double(matGRe[6][0][1], matGIm[6][0][1], temp0[1]);\
    GET_AxAH_double(matGRe[6][1][1], matGIm[6][1][1], temp1[1]);\
    matGRe[6][6][1] = _mm512_sub_pd(matBRe[6][6][1], temp0[1]);\
    matGRe[6][6][1] = _mm512_sub_pd(matGRe[6][6][1], temp1[1]);\
    GET_AxAH_double(matGRe[6][2][1], matGIm[6][2][1], temp0[1]);\
    GET_AxAH_double(matGRe[6][3][1], matGIm[6][3][1], temp1[1]);\
    matGRe[6][6][1] = _mm512_sub_pd(matGRe[6][6][1], temp0[1]);\
    matGRe[6][6][1] = _mm512_sub_pd(matGRe[6][6][1], temp1[1]);\
    GET_AxAH_double(matGRe[6][4][1], matGIm[6][4][1], temp0[1]);\
    GET_AxAH_double(matGRe[6][5][1], matGIm[6][5][1], temp1[1]);\
    matGRe[6][6][1] = _mm512_sub_pd(matGRe[6][6][1], temp0[1]);\
    matGRe[6][6][1] = _mm512_sub_pd(matGRe[6][6][1], temp1[1]);\
    matD[6][1] = _mm512_rsqrt14_pd(matGRe[6][6][1]);\
    matND[6][1] = _mm512_sub_pd(constZero, matD[6][1]);\
}

// get G77
#define GET_G77(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2)\
{\
    GET_AxAH(matGRe[7][0], matGIm[7][0], temp0);\
    GET_AxAH(matGRe[7][1], matGIm[7][1], temp1);\
    matGRe[7][7] = _mm512_sub_ps(matBRe[7][7], temp0);\
    matGRe[7][7] = _mm512_sub_ps(matGRe[7][7], temp1);\
    GET_AxAH(matGRe[7][2], matGIm[7][2], temp0);\
    GET_AxAH(matGRe[7][3], matGIm[7][3], temp1);\
    matGRe[7][7] = _mm512_sub_ps(matGRe[7][7], temp0);\
    matGRe[7][7] = _mm512_sub_ps(matGRe[7][7], temp1);\
    GET_AxAH(matGRe[7][4], matGIm[7][4], temp0);\
    GET_AxAH(matGRe[7][5], matGIm[7][5], temp1);\
    GET_AxAH(matGRe[7][6], matGIm[7][6], temp2);\
    matGRe[7][7] = _mm512_sub_ps(matGRe[7][7], temp0);\
    matGRe[7][7] = _mm512_sub_ps(matGRe[7][7], temp1);\
    matGRe[7][7] = _mm512_sub_ps(matGRe[7][7], temp2);\
    matD[7] = _mm512_rsqrt14_ps(matGRe[7][7]);\
    matND[7] = _mm512_sub_ps(constZero, matD[7]);\
}
#define GET_G77_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2)\
{\
    GET_AxAH_double(matGRe[7][0][0], matGIm[7][0][0], temp0[0]);\
    GET_AxAH_double(matGRe[7][1][0], matGIm[7][1][0], temp1[0]);\
    matGRe[7][7][0] = _mm512_sub_pd(matBRe[7][7][0], temp0[0]);\
    matGRe[7][7][0] = _mm512_sub_pd(matGRe[7][7][0], temp1[0]);\
    GET_AxAH_double(matGRe[7][2][0], matGIm[7][2][0], temp0[0]);\
    GET_AxAH_double(matGRe[7][3][0], matGIm[7][3][0], temp1[0]);\
    matGRe[7][7][0] = _mm512_sub_pd(matGRe[7][7][0], temp0[0]);\
    matGRe[7][7][0] = _mm512_sub_pd(matGRe[7][7][0], temp1[0]);\
    GET_AxAH_double(matGRe[7][4][0], matGIm[7][4][0], temp0[0]);\
    GET_AxAH_double(matGRe[7][5][0], matGIm[7][5][0], temp1[0]);\
    GET_AxAH_double(matGRe[7][6][0], matGIm[7][6][0], temp2[0]);\
    matGRe[7][7][0] = _mm512_sub_pd(matGRe[7][7][0], temp0[0]);\
    matGRe[7][7][0] = _mm512_sub_pd(matGRe[7][7][0], temp1[0]);\
    matGRe[7][7][0] = _mm512_sub_pd(matGRe[7][7][0], temp2[0]);\
    matD[7][0] = _mm512_rsqrt14_pd(matGRe[7][7][0]);\
    matND[7][0] = _mm512_sub_pd(constZero, matD[7][0]);\
    GET_AxAH_double(matGRe[7][0][1], matGIm[7][0][1], temp0[1]);\
    GET_AxAH_double(matGRe[7][1][1], matGIm[7][1][1], temp1[1]);\
    matGRe[7][7][1] = _mm512_sub_pd(matBRe[7][7][1], temp0[1]);\
    matGRe[7][7][1] = _mm512_sub_pd(matGRe[7][7][1], temp1[1]);\
    GET_AxAH_double(matGRe[7][2][1], matGIm[7][2][1], temp0[1]);\
    GET_AxAH_double(matGRe[7][3][1], matGIm[7][3][1], temp1[1]);\
    matGRe[7][7][1] = _mm512_sub_pd(matGRe[7][7][1], temp0[1]);\
    matGRe[7][7][1] = _mm512_sub_pd(matGRe[7][7][1], temp1[1]);\
    GET_AxAH_double(matGRe[7][4][1], matGIm[7][4][1], temp0[1]);\
    GET_AxAH_double(matGRe[7][5][1], matGIm[7][5][1], temp1[1]);\
    GET_AxAH_double(matGRe[7][6][1], matGIm[7][6][1], temp2[1]);\
    matGRe[7][7][1] = _mm512_sub_pd(matGRe[7][7][1], temp0[1]);\
    matGRe[7][7][1] = _mm512_sub_pd(matGRe[7][7][1], temp1[1]);\
    matGRe[7][7][1] = _mm512_sub_pd(matGRe[7][7][1], temp2[1]);\
    matD[7][1] = _mm512_rsqrt14_pd(matGRe[7][7][1]);\
    matND[7][1] = _mm512_sub_pd(constZero, matD[7][1]);\
}

// get Gii, odd diagonal element
#define GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, i)\
{\
    GET_AxAH(matGRe[i][0], matGIm[i][0], temp0);\
    matGRe[i][i] = _mm512_sub_ps(matBRe[i][i], temp0);\
    for (int32_t i1 = 1; i1 < i; i1+=2) \
    {\
        GET_AxAH(matGRe[i][i1], matGIm[i][i1], temp0);\
        GET_AxAH(matGRe[i][i1+1], matGIm[i][i1+1], temp1);\
        matGRe[i][i] = _mm512_sub_ps(matGRe[i][i], temp0);\
        matGRe[i][i] = _mm512_sub_ps(matGRe[i][i], temp1);\
    }\
    matD[i] = _mm512_rsqrt14_ps(matGRe[i][i]);\
    matND[i] = _mm512_sub_ps(constZero, matD[i]);\
}
#define GET_Gii_ODD_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, i)\
{\
    GET_AxAH_double(matGRe[i][0][0], matGIm[i][0][0], temp0[0]);\
    matGRe[i][i][0] = _mm512_sub_pd(matBRe[i][i][0], temp0[0]);\
    for (int32_t i1 = 1; i1 < i; i1+=2) \
    {\
        GET_AxAH_double(matGRe[i][i1][0], matGIm[i][i1][0], temp0[0]);\
        GET_AxAH_double(matGRe[i][i1+1][0], matGIm[i][i1+1][0], temp1[0]);\
        matGRe[i][i][0] = _mm512_sub_pd(matGRe[i][i][0], temp0[0]);\
        matGRe[i][i][0] = _mm512_sub_pd(matGRe[i][i][0], temp1[0]);\
    }\
    matD[i][0] = _mm512_rsqrt14_pd(matGRe[i][i][0]);\
    matND[i][0] = _mm512_sub_pd(constZero, matD[i][0]);\
    GET_AxAH_double(matGRe[i][0][1], matGIm[i][0][1], temp0[1]);\
    matGRe[i][i][1] = _mm512_sub_pd(matBRe[i][i][1], temp0[1]);\
    for (int32_t i1 = 1; i1 < i; i1+=2) \
    {\
        GET_AxAH_double(matGRe[i][i1][1], matGIm[i][i1][1], temp0[1]);\
        GET_AxAH_double(matGRe[i][i1+1][1], matGIm[i][i1+1][1], temp1[1]);\
        matGRe[i][i][1] = _mm512_sub_pd(matGRe[i][i][1], temp0[1]);\
        matGRe[i][i][1] = _mm512_sub_pd(matGRe[i][i][1], temp1[1]);\
    }\
    matD[i][1] = _mm512_rsqrt14_pd(matGRe[i][i][1]);\
    matND[i][1] = _mm512_sub_pd(constZero, matD[i][1]);\
}

// get Gii, even diagonal element
#define GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, i)\
{\
    GET_AxAH(matGRe[i][0], matGIm[i][0], temp0);\
    GET_AxAH(matGRe[i][1], matGIm[i][1], temp1);\
    matGRe[i][i] = _mm512_sub_ps(matBRe[i][i], temp0);\
    matGRe[i][i] = _mm512_sub_ps(matGRe[i][i], temp1);\
    for (int32_t i1 = 2; i1 < i; i1+=2) \
    {\
        GET_AxAH(matGRe[i][i1], matGIm[i][i1], temp0);\
        GET_AxAH(matGRe[i][i1+1], matGIm[i][i1+1], temp1);\
        matGRe[i][i] = _mm512_sub_ps(matGRe[i][i], temp0);\
        matGRe[i][i] = _mm512_sub_ps(matGRe[i][i], temp1);\
    }\
    matD[i] = _mm512_rsqrt14_ps(matGRe[i][i]);\
    matND[i] = _mm512_sub_ps(constZero, matD[i]);\
}
#define GET_Gii_EVEN_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, i)\
{\
    GET_AxAH_double(matGRe[i][0][0], matGIm[i][0][0], temp0[0]);\
    GET_AxAH_double(matGRe[i][1][0], matGIm[i][1][0], temp1[0]);\
    matGRe[i][i][0] = _mm512_sub_pd(matBRe[i][i][0], temp0[0]);\
    matGRe[i][i][0] = _mm512_sub_pd(matGRe[i][i][0], temp1[0]);\
    for (int32_t i1 = 2; i1 < i; i1+=2) \
    {\
        GET_AxAH_double(matGRe[i][i1][0], matGIm[i][i1][0], temp0[0]);\
        GET_AxAH_double(matGRe[i][i1+1][0], matGIm[i][i1+1][0], temp1[0]);\
        matGRe[i][i][0] = _mm512_sub_pd(matGRe[i][i][0], temp0[0]);\
        matGRe[i][i][0] = _mm512_sub_pd(matGRe[i][i][0], temp1[0]);\
    }\
    matD[i][0] = _mm512_rsqrt14_pd(matGRe[i][i][0]);\
    matND[i][0] = _mm512_sub_pd(constZero, matD[i][0]);\
    GET_AxAH_double(matGRe[i][0][1], matGIm[i][0][1], temp0[1]);\
    GET_AxAH_double(matGRe[i][1][1], matGIm[i][1][1], temp1[1]);\
    matGRe[i][i][1] = _mm512_sub_pd(matBRe[i][i][1], temp0[1]);\
    matGRe[i][i][1] = _mm512_sub_pd(matGRe[i][i][1], temp1[1]);\
    for (int32_t i1 = 2; i1 < i; i1+=2) \
    {\
        GET_AxAH_double(matGRe[i][i1][1], matGIm[i][i1][1], temp0[1]);\
        GET_AxAH_double(matGRe[i][i1+1][1], matGIm[i][i1+1][1], temp1[1]);\
        matGRe[i][i][1] = _mm512_sub_pd(matGRe[i][i][1], temp0[1]);\
        matGRe[i][i][1] = _mm512_sub_pd(matGRe[i][i][1], temp1[1]);\
    }\
    matD[i][1] = _mm512_rsqrt14_pd(matGRe[i][i][1]);\
    matND[i][1] = _mm512_sub_pd(constZero, matD[i][1]);\
}

// get odd column n of matrix G, j is row index, i is col index
#define GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, j, i, temp0, temp1)\
{\
    GET_AxBH(matGRe[j][0], matGIm[j][0], matGRe[i][0], matGIm[i][0], temp0, temp1);\
    matGRe[j][i] = _mm512_sub_ps(matBRe[j][i], temp0);\
    matGIm[j][i] = _mm512_sub_ps(matBIm[j][i], temp1);\
    for (int32_t i1 = 1; i1 < i; i1+=2)\
    {\
        GET_AxBH(matGRe[j][i1], matGIm[j][i1], matGRe[i][i1], matGIm[i][i1], temp0, temp1);\
        matGRe[j][i] = _mm512_sub_ps(matGRe[j][i], temp0);\
        matGIm[j][i] = _mm512_sub_ps(matGIm[j][i], temp1);\
        GET_AxBH(matGRe[j][i1+1], matGIm[j][i1+1], matGRe[i][i1+1], matGIm[i][i1+1], temp0, temp1);\
        matGRe[j][i] = _mm512_sub_ps(matGRe[j][i], temp0);\
        matGIm[j][i] = _mm512_sub_ps(matGIm[j][i], temp1);\
    }\
    matGRe[j][i] = _mm512_mul_ps(matGRe[j][i], matD[i]);\
    matGIm[j][i] = _mm512_mul_ps(matGIm[j][i], matD[i]);\
}
#define GET_G_COL_ODD_double(matGRe, matGIm, matBRe, matBIm, matD, j, i, temp0, temp1)\
{\
    GET_AxBH_double(matGRe[j][0][0], matGIm[j][0][0], matGRe[i][0][0], matGIm[i][0][0], temp0[0], temp1[0]);\
    matGRe[j][i][0] = _mm512_sub_pd(matBRe[j][i][0], temp0[0]);\
    matGIm[j][i][0] = _mm512_sub_pd(matBIm[j][i][0], temp1[0]);\
    for (int32_t i1 = 1; i1 < i; i1+=2)\
    {\
        GET_AxBH_double(matGRe[j][i1][0], matGIm[j][i1][0], matGRe[i][i1][0], matGIm[i][i1][0], temp0[0], temp1[0]);\
        matGRe[j][i][0] = _mm512_sub_pd(matGRe[j][i][0], temp0[0]);\
        matGIm[j][i][0] = _mm512_sub_pd(matGIm[j][i][0], temp1[0]);\
        GET_AxBH_double(matGRe[j][i1+1][0], matGIm[j][i1+1][0], matGRe[i][i1+1][0], matGIm[i][i1+1][0], temp0[0], temp1[0]);\
        matGRe[j][i][0] = _mm512_sub_pd(matGRe[j][i][0], temp0[0]);\
        matGIm[j][i][0] = _mm512_sub_pd(matGIm[j][i][0], temp1[0]);\
    }\
    matGRe[j][i][0] = _mm512_mul_pd(matGRe[j][i][0], matD[i][0]);\
    matGIm[j][i][0] = _mm512_mul_pd(matGIm[j][i][0], matD[i][0]);\
    GET_AxBH_double(matGRe[j][0][1], matGIm[j][0][1], matGRe[i][0][1], matGIm[i][0][1], temp0[1], temp1[1]);\
    matGRe[j][i][1] = _mm512_sub_pd(matBRe[j][i][1], temp0[1]);\
    matGIm[j][i][1] = _mm512_sub_pd(matBIm[j][i][1], temp1[1]);\
    for (int32_t i1 = 1; i1 < i; i1+=2)\
    {\
        GET_AxBH_double(matGRe[j][i1][1], matGIm[j][i1][1], matGRe[i][i1][1], matGIm[i][i1][1], temp0[1], temp1[1]);\
        matGRe[j][i][1] = _mm512_sub_pd(matGRe[j][i][1], temp0[1]);\
        matGIm[j][i][1] = _mm512_sub_pd(matGIm[j][i][1], temp1[1]);\
        GET_AxBH_double(matGRe[j][i1+1][1], matGIm[j][i1+1][1], matGRe[i][i1+1][1], matGIm[i][i1+1][1], temp0[1], temp1[1]);\
        matGRe[j][i][1] = _mm512_sub_pd(matGRe[j][i][1], temp0[1]);\
        matGIm[j][i][1] = _mm512_sub_pd(matGIm[j][i][1], temp1[1]);\
    }\
    matGRe[j][i][1] = _mm512_mul_pd(matGRe[j][i][1], matD[i][1]);\
    matGIm[j][i][1] = _mm512_mul_pd(matGIm[j][i][1], matD[i][1]);\
}

// get even column n of matrix G, j is row index, i is col index
#define GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, j, i, temp0, temp1)\
{\
    GET_AxBH(matGRe[j][0], matGIm[j][0], matGRe[i][0], matGIm[i][0], temp0, temp1);\
    matGRe[j][i] = _mm512_sub_ps(matBRe[j][i], temp0);\
    matGIm[j][i] = _mm512_sub_ps(matBIm[j][i], temp1);\
    GET_AxBH(matGRe[j][1], matGIm[j][1], matGRe[i][1], matGIm[i][1], temp0, temp1);\
    matGRe[j][i] = _mm512_sub_ps(matGRe[j][i], temp0);\
    matGIm[j][i] = _mm512_sub_ps(matGIm[j][i], temp1);\
    for (int32_t i1 = 2; i1 < i; i1+=2)\
    {\
        GET_AxBH(matGRe[j][i1], matGIm[j][i1], matGRe[i][i1], matGIm[i][i1], temp0, temp1);\
        matGRe[j][i] = _mm512_sub_ps(matGRe[j][i], temp0);\
        matGIm[j][i] = _mm512_sub_ps(matGIm[j][i], temp1);\
        GET_AxBH(matGRe[j][i1+1], matGIm[j][i1+1], matGRe[i][i1+1], matGIm[i][i1+1], temp0, temp1);\
        matGRe[j][i] = _mm512_sub_ps(matGRe[j][i], temp0);\
        matGIm[j][i] = _mm512_sub_ps(matGIm[j][i], temp1);\
    }\
    matGRe[j][i] = _mm512_mul_ps(matGRe[j][i], matD[i]);\
    matGIm[j][i] = _mm512_mul_ps(matGIm[j][i], matD[i]);\
}
#define GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, j, i, temp0, temp1)\
{\
    GET_AxBH_double(matGRe[j][0][0], matGIm[j][0][0], matGRe[i][0][0], matGIm[i][0][0], temp0[0], temp1[0]);\
    matGRe[j][i][0] = _mm512_sub_pd(matBRe[j][i][0], temp0[0]);\
    matGIm[j][i][0] = _mm512_sub_pd(matBIm[j][i][0], temp1[0]);\
    GET_AxBH_double(matGRe[j][1][0], matGIm[j][1][0], matGRe[i][1][0], matGIm[i][1][0], temp0[0], temp1[0]);\
    matGRe[j][i][0] = _mm512_sub_pd(matGRe[j][i][0], temp0[0]);\
    matGIm[j][i][0] = _mm512_sub_pd(matGIm[j][i][0], temp1[0]);\
    for (int32_t i1 = 2; i1 < i; i1+=2)\
    {\
        GET_AxBH_double(matGRe[j][i1][0], matGIm[j][i1][0], matGRe[i][i1][0], matGIm[i][i1][0], temp0[0], temp1[0]);\
        matGRe[j][i][0] = _mm512_sub_pd(matGRe[j][i][0], temp0[0]);\
        matGIm[j][i][0] = _mm512_sub_pd(matGIm[j][i][0], temp1[0]);\
        GET_AxBH_double(matGRe[j][i1+1][0], matGIm[j][i1+1][0], matGRe[i][i1+1][0], matGIm[i][i1+1][0], temp0[0], temp1[0]);\
        matGRe[j][i][0] = _mm512_sub_pd(matGRe[j][i][0], temp0[0]);\
        matGIm[j][i][0] = _mm512_sub_pd(matGIm[j][i][0], temp1[0]);\
    }\
    matGRe[j][i][0] = _mm512_mul_pd(matGRe[j][i][0], matD[i][0]);\
    matGIm[j][i][0] = _mm512_mul_pd(matGIm[j][i][0], matD[i][0]);\
    GET_AxBH_double(matGRe[j][0][1], matGIm[j][0][1], matGRe[i][0][1], matGIm[i][0][1], temp0[1], temp1[1]);\
    matGRe[j][i][1] = _mm512_sub_pd(matBRe[j][i][1], temp0[1]);\
    matGIm[j][i][1] = _mm512_sub_pd(matBIm[j][i][1], temp1[1]);\
    GET_AxBH_double(matGRe[j][1][1], matGIm[j][1][1], matGRe[i][1][1], matGIm[i][1][1], temp0[1], temp1[1]);\
    matGRe[j][i][1] = _mm512_sub_pd(matGRe[j][i][1], temp0[1]);\
    matGIm[j][i][1] = _mm512_sub_pd(matGIm[j][i][1], temp1[1]);\
    for (int32_t i1 = 2; i1 < i; i1+=2)\
    {\
        GET_AxBH_double(matGRe[j][i1][1], matGIm[j][i1][1], matGRe[i][i1][1], matGIm[i][i1][1], temp0[1], temp1[1]);\
        matGRe[j][i][1] = _mm512_sub_pd(matGRe[j][i][1], temp0[1]);\
        matGIm[j][i][1] = _mm512_sub_pd(matGIm[j][i][1], temp1[1]);\
        GET_AxBH_double(matGRe[j][i1+1][1], matGIm[j][i1+1][1], matGRe[i][i1+1][1], matGIm[i][i1+1][1], temp0[1], temp1[1]);\
        matGRe[j][i][1] = _mm512_sub_pd(matGRe[j][i][1], temp0[1]);\
        matGIm[j][i][1] = _mm512_sub_pd(matGIm[j][i][1], temp1[1]);\
    }\
    matGRe[j][i][1] = _mm512_mul_pd(matGRe[j][i][1], matD[i][1]);\
    matGIm[j][i][1] = _mm512_mul_pd(matGIm[j][i][1], matD[i][1]);\
}

// set value for Lii, diagonal element
#define SET_Lii(matLRe, matLIm, matD, i)\
{\
    matLRe[i][i] = matD[i];\
    matLIm[i][i] = constZero;\
}
#define SET_Lii_double(matLRe, matLIm, matD, i)\
{\
    matLRe[i][i][0] = matD[i][0];\
    matLIm[i][i][0] = constZero;\
    matLRe[i][i][1] = matD[i][1];\
    matLIm[i][i][1] = constZero;\
}

// get element L(i+1, i)
#define GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, j, i)\
{\
    GET_AxRealB(matGRe[j][i], matGIm[j][i], matLRe[i][i],  matLRe[j][i], matLIm[j][i]);\
    matLRe[j][i] = _mm512_mul_ps(matLRe[j][i], matND[j]);\
    matLIm[j][i] = _mm512_mul_ps(matLIm[j][i], matND[j]);\
}
#define GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, j, i)\
{\
    GET_AxRealB_double(matGRe[j][i][0], matGIm[j][i][0], matLRe[i][i][0],  matLRe[j][i][0], matLIm[j][i][0]);\
    matLRe[j][i][0] = _mm512_mul_pd(matLRe[j][i][0], matND[j][0]);\
    matLIm[j][i][0] = _mm512_mul_pd(matLIm[j][i][0], matND[j][0]);\
    GET_AxRealB_double(matGRe[j][i][1], matGIm[j][i][1], matLRe[i][i][1],  matLRe[j][i][1], matLIm[j][i][1]);\
    matLRe[j][i][1] = _mm512_mul_pd(matLRe[j][i][1], matND[j][1]);\
    matLIm[j][i][1] = _mm512_mul_pd(matLIm[j][i][1], matND[j][1]);\
}

// get element Lji, j is larger than i+1
#define GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, j, i, temp0, temp1)\
{\
    GET_AxRealB(matGRe[j][i], matGIm[j][i], matLRe[i][i],  matLRe[j][i], matLIm[j][i]);\
    for (int32_t i1 = i+1; i1 < j; i1++)\
    {\
        GET_AxB(matGRe[j][i1], matGIm[j][i1], matLRe[i1][i], matLIm[i1][i], temp0, temp1);\
        matLRe[j][i] = _mm512_add_ps(matLRe[j][i], temp0);\
        matLIm[j][i] = _mm512_add_ps(matLIm[j][i], temp1);\
    }\
    matLRe[j][i] = _mm512_mul_ps(matLRe[j][i], matND[j]);\
    matLIm[j][i] = _mm512_mul_ps(matLIm[j][i], matND[j]);\
}

#define GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, j, i, temp0, temp1)\
{\
    GET_AxRealB_double(matGRe[j][i][0], matGIm[j][i][0], matLRe[i][i][0],  matLRe[j][i][0], matLIm[j][i][0]);\
    for (int32_t i1 = i+1; i1 < j; i1++)\
    {\
        GET_AxB_double(matGRe[j][i1][0], matGIm[j][i1][0], matLRe[i1][i][0], matLIm[i1][i][0], temp0[0], temp1[0]);\
        matLRe[j][i][0] = _mm512_add_pd(matLRe[j][i][0], temp0[0]);\
        matLIm[j][i][0] = _mm512_add_pd(matLIm[j][i][0], temp1[0]);\
    }\
    matLRe[j][i][0] = _mm512_mul_pd(matLRe[j][i][0], matND[j][0]);\
    matLIm[j][i][0] = _mm512_mul_pd(matLIm[j][i][0], matND[j][0]);\
    GET_AxRealB_double(matGRe[j][i][1], matGIm[j][i][1], matLRe[i][i][1],  matLRe[j][i][1], matLIm[j][i][1]);\
    for (int32_t i1 = i+1; i1 < j; i1++)\
    {\
        GET_AxB_double(matGRe[j][i1][1], matGIm[j][i1][1], matLRe[i1][i][1], matLIm[i1][i][1], temp0[1], temp1[1]);\
        matLRe[j][i][1] = _mm512_add_pd(matLRe[j][i][1], temp0[1]);\
        matLIm[j][i][1] = _mm512_add_pd(matLIm[j][i][1], temp1[1]);\
    }\
    matLRe[j][i][1] = _mm512_mul_pd(matLRe[j][i][1], matND[j][1]);\
    matLIm[j][i][1] = _mm512_mul_pd(matLIm[j][i][1], matND[j][1]);\
}

#define N_4 4
#define N_8 8
#define N_32 32
#define N_64 64
#define MAX_LEN            N_64
__m512 gResistO3_5;
double gResistO3_6 = 0;

void matrix_inv_cholesky_4x4(__m512 matBRe[MAX_LEN][MAX_LEN], __m512 matBIm[MAX_LEN][MAX_LEN],
    __m512 matInvBRe[MAX_LEN][MAX_LEN], __m512 matInvBIm[MAX_LEN][MAX_LEN])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_4][N_4], matGIm[N_4][N_4];
    __m512 matLRe[N_4][N_4], matLIm[N_4][N_4];
    __m512 matD[N_4], matND[N_4];
    __m512 temp0, temp1, temp2;
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00(matGRe, matBRe, matD, matND);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 1);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 2);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 3);

    // Column 1
    GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 2, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);

    // Column 2
    GET_G22(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);

    // Column 3
    GET_G33(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);


    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii(matLRe, matLIm, matD, 0);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 2, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 0, temp0, temp1);

    // Column 1
    SET_Lii(matLRe, matLIm, matD, 1);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 2, 1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 1, temp0, temp1);

    // Column 2
    SET_Lii(matLRe, matLIm, matD, 2);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 3, 2);

    // Column 3
    SET_Lii(matLRe, matLIm, matD, 3);


    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_4; i ++)
    {
        matInvBRe[i][i] = _mm512_mul_ps(matLRe[i][i], matLRe[i][i]);
        for (k = (i+1); k < N_4; k++)
        {
            temp1 = _mm512_add_ps(_mm512_mul_ps(matLRe[k][i], matLRe[k][i]), _mm512_mul_ps(matLIm[k][i], matLIm[k][i]));
            matInvBRe[i][i] = _mm512_add_ps(matInvBRe[i][i], temp1);
        }
        matInvBIm[i][i] = _mm512_setzero_ps();
    }

    for(i = 0; i < N_4; i ++)
    {
        for(j = i+1; j < N_4; j ++)
        {
            matInvBRe[i][j] = _mm512_mul_ps(matLRe[j][i], matLRe[j][j]);
            matInvBIm[i][j] = _mm512_sub_ps(constZero, _mm512_mul_ps(matLIm[j][i], matLRe[j][j]));

            for (k = (j+1); k < N_4; k++)
            {
                GET_AxBH(matLRe[k][j], matLIm[k][j], matLRe[k][i], matLIm[k][i], temp1, temp2);
                matInvBRe[i][j] = _mm512_add_ps(matInvBRe[i][j], temp1);
                matInvBIm[i][j] = _mm512_add_ps(matInvBIm[i][j], temp2);
            }

            // Hermite matrix
            matInvBRe[j][i] = matInvBRe[i][j];
            matInvBIm[j][i] = _mm512_sub_ps(constZero, matInvBIm[i][j]);
        }
    }
    
    gResistO3_5 = _mm512_add_ps(gResistO3_5,_mm512_sub_ps(matInvBRe[0][0], matInvBIm[0][0]));
}

void matrix_inv_cholesky_4x4_double(__m512 matBRe[MAX_LEN][MAX_LEN][2], __m512 matBIm[MAX_LEN][MAX_LEN][2],
    __m512 matInvBRe[MAX_LEN][MAX_LEN][2], __m512 matInvBIm[MAX_LEN][MAX_LEN][2])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_4][N_4][2], matGIm[N_4][N_4][2];
    __m512 matLRe[N_4][N_4][2], matLIm[N_4][N_4][2];
    __m512 matD[N_4][2], matND[N_4][2];
    __m512 temp0[2], temp1[2], temp2[2];
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00_double(matGRe, matBRe, matD, matND);
    GET_G_COL0_double(matGRe, matGIm, matBRe, matBIm, matD, 1);
    GET_G_COL0_double(matGRe, matGIm, matBRe, matBIm, matD, 2);
    GET_G_COL0_double(matGRe, matGIm, matBRe, matBIm, matD, 3);

    // Column 1
    GET_G11_double(matGRe, matGIm, matBRe, matD, matND, temp0);
    GET_G_COL1_double(matGRe, matGIm, matBRe, matBIm, matD, 2, temp0, temp1);
    GET_G_COL1_double(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);

    // Column 2
    GET_G22_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL2_double(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);

    // Column 3
    GET_G33_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);


    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii_double(matLRe, matLIm, matD, 0);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 2, 0, temp0, temp1);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 3, 0, temp0, temp1);

    // Column 1
    SET_Lii_double(matLRe, matLIm, matD, 1);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 2, 1);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 3, 1, temp0, temp1);

    // Column 2
    SET_Lii_double(matLRe, matLIm, matD, 2);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 3, 2);

    // Column 3
    SET_Lii_double(matLRe, matLIm, matD, 3);


    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_4; i ++)
    {
        matInvBRe[i][i][0] = _mm512_mul_pd(matLRe[i][i][0], matLRe[i][i][0]);
        for (k = (i+1); k < N_4; k++)
        {
            temp1[0] = _mm512_add_pd(_mm512_mul_pd(matLRe[k][i][0], matLRe[k][i][0]), _mm512_mul_pd(matLIm[k][i][0], matLIm[k][i][0]));
            matInvBRe[i][i][0] = _mm512_add_pd(matInvBRe[i][i][0], temp1[0]);
        }
        matInvBIm[i][i][0] = _mm512_setzero_pd();
        
        matInvBRe[i][i][1] = _mm512_mul_pd(matLRe[i][i][1], matLRe[i][i][1]);
        for (k = (i+1); k < N_4; k++)
        {
            temp1[1] = _mm512_add_pd(_mm512_mul_pd(matLRe[k][i][1], matLRe[k][i][1]), _mm512_mul_pd(matLIm[k][i][1], matLIm[k][i][1]));
            matInvBRe[i][i][1] = _mm512_add_pd(matInvBRe[i][i][1], temp1[1]);
        }
        matInvBIm[i][i][1] = _mm512_setzero_pd();
    }

    for(i = 0; i < N_4; i ++)
    {
        for(j = i+1; j < N_4; j ++)
        {
            matInvBRe[i][j][0] = _mm512_mul_pd(matLRe[j][i][0], matLRe[j][j][0]);
            matInvBIm[i][j][0] = _mm512_sub_pd(constZero, _mm512_mul_pd(matLIm[j][i][0], matLRe[j][j][0]));

            for (k = (j+1); k < N_4; k++)
            {
                GET_AxBH_double(matLRe[k][j][0], matLIm[k][j][0], matLRe[k][i][0], matLIm[k][i][0], temp1[0], temp2[0]);
                matInvBRe[i][j][0] = _mm512_add_pd(matInvBRe[i][j][0], temp1[0]);
                matInvBIm[i][j][0] = _mm512_add_pd(matInvBIm[i][j][0], temp2[0]);
            }

            // Hermite matrix
            matInvBRe[j][i][0] = matInvBRe[i][j][0];
            matInvBIm[j][i][0] = _mm512_sub_pd(constZero, matInvBIm[i][j][0]);

            
            matInvBRe[i][j][1] = _mm512_mul_pd(matLRe[j][i][1], matLRe[j][j][1]);
            matInvBIm[i][j][1] = _mm512_sub_pd(constZero, _mm512_mul_pd(matLIm[j][i][1], matLRe[j][j][1]));
            
            for (k = (j+1); k < N_4; k++)
            {
                GET_AxBH_double(matLRe[k][j][1], matLIm[k][j][1], matLRe[k][i][1], matLIm[k][i][1], temp1[1], temp2[1]);
                matInvBRe[i][j][1] = _mm512_add_pd(matInvBRe[i][j][1], temp1[1]);
                matInvBIm[i][j][1] = _mm512_add_pd(matInvBIm[i][j][1], temp2[1]);
            }
            
            // Hermite matrix
            matInvBRe[j][i][1] = matInvBRe[i][j][1];
            matInvBIm[j][i][1] = _mm512_sub_pd(constZero, matInvBIm[i][j][1]);
        }
    }
    
    gResistO3_5 = _mm512_add_pd(gResistO3_5,_mm512_sub_pd(matInvBRe[0][0][0], matInvBIm[0][0][0]));
}

void matrix_inv_cholesky_8x8(__m512 matBRe[MAX_LEN][MAX_LEN], __m512 matBIm[MAX_LEN][MAX_LEN],
    __m512 matInvBRe[MAX_LEN][MAX_LEN], __m512 matInvBIm[MAX_LEN][MAX_LEN])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_8][N_8], matGIm[N_8][N_8];
    __m512 matLRe[N_8][N_8], matLIm[N_8][N_8];
    __m512 matD[N_8], matND[N_8];
    __m512 temp0, temp1, temp2;
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00(matGRe, matBRe, matD, matND);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 1);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 2);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 3);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 4);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 5);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 6);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 7);

    // Column 1
    GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 2, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);

    // Column 2
    GET_G22(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);

    // Column 3
    GET_G33(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);

    // Column 4
    GET_G44(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 5, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 6, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 7, 4, temp0, temp1);

    // Column 5
    GET_G55(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 6, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 7, 5, temp0, temp1);

    // Column 6
    GET_G66(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 7, 6, temp0, temp1);

    // Column 7
    GET_G77(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);


    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii(matLRe, matLIm, matD, 0);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 2, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 0, temp0, temp1);

    // Column 1
    SET_Lii(matLRe, matLIm, matD, 1);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 2, 1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 1, temp0, temp1);

    // Column 2
    SET_Lii(matLRe, matLIm, matD, 2);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 3, 2);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 2, temp0, temp1);

    // Column 3
    SET_Lii(matLRe, matLIm, matD, 3);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 4, 3);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 3, temp0, temp1);

    // Column 4
    SET_Lii(matLRe, matLIm, matD, 4);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 5, 4);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 4, temp0, temp1);

    // Column 5
    SET_Lii(matLRe, matLIm, matD, 5);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 6, 5);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 5, temp0, temp1);

    // Column 6
    SET_Lii(matLRe, matLIm, matD, 6);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 7, 6);

    // Column 7
    SET_Lii(matLRe, matLIm, matD, 7);


    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_8; i ++)
    {
        matInvBRe[i][i] = _mm512_mul_ps(matLRe[i][i], matLRe[i][i]);
        for (k = (i+1); k < N_8; k++)
        {
            temp1 = _mm512_add_ps(_mm512_mul_ps(matLRe[k][i], matLRe[k][i]), _mm512_mul_ps(matLIm[k][i], matLIm[k][i]));
            matInvBRe[i][i] = _mm512_add_ps(matInvBRe[i][i], temp1);
        }
        matInvBIm[i][i] = _mm512_setzero_ps();
    }

    for(i = 0; i < N_8; i ++)
    {
        for(j = i+1; j < N_8; j ++)
        {
            matInvBRe[i][j] = _mm512_mul_ps(matLRe[j][i], matLRe[j][j]);
            matInvBIm[i][j] = _mm512_sub_ps(constZero, _mm512_mul_ps(matLIm[j][i], matLRe[j][j]));

            for (k = (j+1); k < N_8; k++)
            {
                GET_AxBH(matLRe[k][j], matLIm[k][j], matLRe[k][i], matLIm[k][i], temp1, temp2);
                matInvBRe[i][j] = _mm512_add_ps(matInvBRe[i][j], temp1);
                matInvBIm[i][j] = _mm512_add_ps(matInvBIm[i][j], temp2);
            }

            // Hermite matrix
            matInvBRe[j][i] = matInvBRe[i][j];
            matInvBIm[j][i] = _mm512_sub_ps(constZero, matInvBIm[i][j]);
        }
    }
    
    gResistO3_5 = _mm512_add_ps(gResistO3_5,_mm512_sub_ps(matInvBRe[0][0], matInvBIm[0][0]));
       
    
}
void matrix_inv_cholesky_8x8_double(__m512 matBRe[MAX_LEN][MAX_LEN][2], __m512 matBIm[MAX_LEN][MAX_LEN][2],
    __m512 matInvBRe[MAX_LEN][MAX_LEN][2], __m512 matInvBIm[MAX_LEN][MAX_LEN][2])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_8][N_8][2], matGIm[N_8][N_8][2];
    __m512 matLRe[N_8][N_8][2], matLIm[N_8][N_8][2];
    __m512 matD[N_8][2], matND[N_8][2];
    __m512 temp0[2], temp1[2], temp2[2];
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00_double(matGRe, matBRe, matD, matND);
    GET_G_COL0_double(matGRe, matGIm, matBRe, matBIm, matD, 1);
    GET_G_COL0_double(matGRe, matGIm, matBRe, matBIm, matD, 2);
    GET_G_COL0_double(matGRe, matGIm, matBRe, matBIm, matD, 3);
    GET_G_COL0_double(matGRe, matGIm, matBRe, matBIm, matD, 4);
    GET_G_COL0_double(matGRe, matGIm, matBRe, matBIm, matD, 5);
    GET_G_COL0_double(matGRe, matGIm, matBRe, matBIm, matD, 6);
    GET_G_COL0_double(matGRe, matGIm, matBRe, matBIm, matD, 7);

    // Column 1
    GET_G11_double(matGRe, matGIm, matBRe, matD, matND, temp0);
    GET_G_COL1_double(matGRe, matGIm, matBRe, matBIm, matD, 2, temp0, temp1);
    GET_G_COL1_double(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL1_double(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL1_double(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL1_double(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL1_double(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);

    // Column 2
    GET_G22_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL2_double(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL2_double(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL2_double(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL2_double(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL2_double(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);

    // Column 3
    GET_G33_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL3_double(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL3_double(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL3_double(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL3_double(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);

    // Column 4
    GET_G44_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, 5, 4, temp0, temp1);
    GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, 6, 4, temp0, temp1);
    GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, 7, 4, temp0, temp1);

    // Column 5
    GET_G55_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL_ODD_double(matGRe, matGIm, matBRe, matBIm, matD, 6, 5, temp0, temp1);
    GET_G_COL_ODD_double(matGRe, matGIm, matBRe, matBIm, matD, 7, 5, temp0, temp1);

    // Column 6
    GET_G66_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, 7, 6, temp0, temp1);

    // Column 7
    GET_G77_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);


    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii_double(matLRe, matLIm, matD, 0);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 2, 0, temp0, temp1);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 3, 0, temp0, temp1);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 4, 0, temp0, temp1);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 5, 0, temp0, temp1);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 6, 0, temp0, temp1);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 7, 0, temp0, temp1);

    // Column 1
    SET_Lii_double(matLRe, matLIm, matD, 1);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 2, 1);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 3, 1, temp0, temp1);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 4, 1, temp0, temp1);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 5, 1, temp0, temp1);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 6, 1, temp0, temp1);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 7, 1, temp0, temp1);

    // Column 2
    SET_Lii_double(matLRe, matLIm, matD, 2);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 3, 2);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 4, 2, temp0, temp1);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 5, 2, temp0, temp1);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 6, 2, temp0, temp1);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 7, 2, temp0, temp1);

    // Column 3
    SET_Lii_double(matLRe, matLIm, matD, 3);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 4, 3);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 5, 3, temp0, temp1);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 6, 3, temp0, temp1);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 7, 3, temp0, temp1);

    // Column 4
    SET_Lii_double(matLRe, matLIm, matD, 4);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 5, 4);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 6, 4, temp0, temp1);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 7, 4, temp0, temp1);

    // Column 5
    SET_Lii_double(matLRe, matLIm, matD, 5);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 6, 5);
    GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, 7, 5, temp0, temp1);

    // Column 6
    SET_Lii_double(matLRe, matLIm, matD, 6);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 7, 6);

    // Column 7
    SET_Lii_double(matLRe, matLIm, matD, 7);


    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_8; i ++)
    {
        matInvBRe[i][i][0] = _mm512_mul_pd(matLRe[i][i][0], matLRe[i][i][0]);
        for (k = (i+1); k < N_8; k++)
        {
            temp1[0] = _mm512_add_pd(_mm512_mul_pd(matLRe[k][i][0], matLRe[k][i][0]), _mm512_mul_pd(matLIm[k][i][0], matLIm[k][i][0]));
            matInvBRe[i][i][0] = _mm512_add_pd(matInvBRe[i][i][0], temp1[0]);
        }
        matInvBIm[i][i][0] = _mm512_setzero_pd();
        
        matInvBRe[i][i][1] = _mm512_mul_pd(matLRe[i][i][1], matLRe[i][i][1]);
        for (k = (i+1); k < N_8; k++)
        {
            temp1[1] = _mm512_add_pd(_mm512_mul_pd(matLRe[k][i][1], matLRe[k][i][1]), _mm512_mul_pd(matLIm[k][i][1], matLIm[k][i][1]));
            matInvBRe[i][i][1] = _mm512_add_pd(matInvBRe[i][i][1], temp1[1]);
        }
        matInvBIm[i][i][1] = _mm512_setzero_pd();
    }

    for(i = 0; i < N_8; i ++)
    {
        for(j = i+1; j < N_8; j ++)
        {
            matInvBRe[i][j][0] = _mm512_mul_pd(matLRe[j][i][0], matLRe[j][j][0]);
            matInvBIm[i][j][0] = _mm512_sub_pd(constZero, _mm512_mul_pd(matLIm[j][i][0], matLRe[j][j][0]));

            for (k = (j+1); k < N_8; k++)
            {
                GET_AxBH_double(matLRe[k][j][0], matLIm[k][j][0], matLRe[k][i][0], matLIm[k][i][0], temp1[0], temp2[0]);
                matInvBRe[i][j][0] = _mm512_add_pd(matInvBRe[i][j][0], temp1[0]);
                matInvBIm[i][j][0] = _mm512_add_pd(matInvBIm[i][j][0], temp2[0]);
            }

            // Hermite matrix
            matInvBRe[j][i][0] = matInvBRe[i][j][0];
            matInvBIm[j][i][0] = _mm512_sub_pd(constZero, matInvBIm[i][j][0]);
            
            matInvBRe[i][j][1] = _mm512_mul_pd(matLRe[j][i][1], matLRe[j][j][1]);
            matInvBIm[i][j][1] = _mm512_sub_pd(constZero, _mm512_mul_pd(matLIm[j][i][1], matLRe[j][j][1]));
            
            for (k = (j+1); k < N_8; k++)
            {
                GET_AxBH_double(matLRe[k][j][1], matLIm[k][j][1], matLRe[k][i][1], matLIm[k][i][1], temp1[1], temp2[1]);
                matInvBRe[i][j][1] = _mm512_add_pd(matInvBRe[i][j][1], temp1[1]);
                matInvBIm[i][j][1] = _mm512_add_pd(matInvBIm[i][j][1], temp2[1]);
            }
            
            // Hermite matrix
            matInvBRe[j][i][1] = matInvBRe[i][j][1];
            matInvBIm[j][i][1] = _mm512_sub_pd(constZero, matInvBIm[i][j][1]);
        }
    }
    
    gResistO3_5 = _mm512_add_pd(gResistO3_5,_mm512_sub_pd(matInvBRe[0][0][0], matInvBIm[0][0][0]));
       
    
}
    
void matrix_inv_cholesky_32x32(__m512 matBRe[MAX_LEN][MAX_LEN], __m512 matBIm[MAX_LEN][MAX_LEN],
    __m512 matInvBRe[MAX_LEN][MAX_LEN], __m512 matInvBIm[MAX_LEN][MAX_LEN])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_32][N_32], matGIm[N_32][N_32];
    __m512 matLRe[N_32][N_32], matLIm[N_32][N_32];
    __m512 matD[N_32], matND[N_32];
    __m512 temp0, temp1, temp2;
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00(matGRe, matBRe, matD, matND);
    for (i=1;i<N_32;i++)GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, i);
    // Column 1
    GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0);
    for (i=2;i<N_32;i++)GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, i, temp0, temp1);
    // Column 2
    GET_G22(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    for (i=3;i<N_32;i++)GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, i, temp0, temp1);
    // Column 3
    GET_G33(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    for (i=4;i<N_32;i++)GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, i, temp0, temp1);
    // Column 4
    GET_G44(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    for (i=5;i<N_32;i++)GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, i, 4, temp0, temp1);
    // Column 5
    GET_G55(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    for (i=6;i<N_32;i++)GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, i, 5, temp0, temp1);
    // Column 6
    GET_G66(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    for (i=7;i<N_32;i++)GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, i, 6, temp0, temp1);
    // Column 7
    GET_G77(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    for (i=8;i<N_32;i++)GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, i, 7, temp0, temp1);
    // Column 8
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 8);
    for (i=9;i<N_32;i++)GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, i, 8, temp0, temp1);
    // Column 9
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 9);
    for (i=10;i<N_32;i++)GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, i, 9, temp0, temp1);
    // Column 10
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 10);
    for (i=11;i<N_32;i++)GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, i, 10, temp0, temp1);
    // Column 11
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 11);
    for (i=12;i<N_32;i++)GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, i, 11, temp0, temp1);
    // Column 12
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 12);
    for (i=13;i<N_32;i++)GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, i, 12, temp0, temp1);
    // Column 13
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 13);
    for (i=14;i<N_32;i++)GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, i, 13, temp0, temp1);
    // Column 14
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 14);
    for (i=15;i<N_32;i++)GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, i, 14, temp0, temp1);
    // Column 15
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 15);
    for (i=16;i<N_32;i++)GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, i, 15, temp0, temp1);
    // Column 16
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 16);
    for (i=17;i<N_32;i++)GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, i, 16, temp0, temp1);
    // Column 17
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 17);
    for (i=18;i<N_32;i++)GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, i, 17, temp0, temp1);
    // Column 18
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 18);
    for (i=19;i<N_32;i++)GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, i, 18, temp0, temp1);
    // Column 19
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 19);
    for (i=20;i<N_32;i++)GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, i, 19, temp0, temp1);
    // Column 20
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 20);
    for (i=21;i<N_32;i++)GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, i, 20, temp0, temp1);
    // Column 21
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 21);
    for (i=22;i<N_32;i++)GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, i, 21, temp0, temp1);
    // Column 22
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 22);
    for (i=23;i<N_32;i++)GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, i, 22, temp0, temp1);
    // Column 23
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 23);
    for (i=24;i<N_32;i++)GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, i, 23, temp0, temp1);
    // Column 24
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 24);
    for (i=25;i<N_32;i++)GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, i, 24, temp0, temp1);
    // Column 25
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 25);
    for (i=26;i<N_32;i++)GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, i, 25, temp0, temp1);
    // Column 26
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 26);
    for (i=27;i<N_32;i++)GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, i, 26, temp0, temp1);
    // Column 27
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 27);
    for (i=28;i<N_32;i++)GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, i, 27, temp0, temp1);
    // Column 28
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 28);
    for (i=29;i<N_32;i++)GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, i, 28, temp0, temp1);
    // Column 29
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 29);
    for (i=30;i<N_32;i++)GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, i, 29, temp0, temp1);
    // Column 30
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 30);
    for (i=31;i<N_32;i++)GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, i, 30, temp0, temp1);
    // Column 31
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 31);

    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii(matLRe, matLIm, matD, 0);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);
    for (i=2;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 0, temp0, temp1);
    // Column 1
    SET_Lii(matLRe, matLIm, matD, 1);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 2, 1);
    for (i=3;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 1, temp0, temp1);
    // Column 2
    SET_Lii(matLRe, matLIm, matD, 2);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 3, 2);
    for (i=4;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 2, temp0, temp1);
    // Column 3
    SET_Lii(matLRe, matLIm, matD, 3);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 4, 3);
    for (i=5;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 3, temp0, temp1);
    // Column 4
    SET_Lii(matLRe, matLIm, matD, 4);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 5, 4);
    for (i=6;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 4, temp0, temp1);
    // Column 5
    SET_Lii(matLRe, matLIm, matD, 5);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 6, 5);
    for (i=7;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 5, temp0, temp1);
    // Column 6
    SET_Lii(matLRe, matLIm, matD, 6);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 7, 6);
    for (i=8;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 6, temp0, temp1);
    // Column 7
    SET_Lii(matLRe, matLIm, matD, 7);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 8, 7);
    for (i=9;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 7, temp0, temp1);
    // Column 8
    SET_Lii(matLRe, matLIm, matD, 8);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 9, 8);
    for (i=10;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 8, temp0, temp1);
    // Column 9
    SET_Lii(matLRe, matLIm, matD, 9);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 10, 9);
    for (i=11;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 9, temp0, temp1);
    // Column 10
    SET_Lii(matLRe, matLIm, matD, 10);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 11, 10);
    for (i=12;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 10, temp0, temp1);
    // Column 11
    SET_Lii(matLRe, matLIm, matD, 11);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 12, 11);
    for (i=13;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 11, temp0, temp1);
    // Column 12
    SET_Lii(matLRe, matLIm, matD, 12);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 13, 12);
    for (i=14;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 12, temp0, temp1);
    // Column 13
    SET_Lii(matLRe, matLIm, matD, 13);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 14, 13);
    for (i=15;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 13, temp0, temp1);
    // Column 14
    SET_Lii(matLRe, matLIm, matD, 14);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 15, 14);
    for (i=16;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 14, temp0, temp1);
    // Column 15
    SET_Lii(matLRe, matLIm, matD, 15);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 16, 15);
    for (i=17;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 15, temp0, temp1);
    // Column 16
    SET_Lii(matLRe, matLIm, matD, 16);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 17, 16);
    for (i=18;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 16, temp0, temp1);
    // Column 17
    SET_Lii(matLRe, matLIm, matD, 17);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 18, 17);
    for (i=19;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 17, temp0, temp1);
    // Column 18
    SET_Lii(matLRe, matLIm, matD, 18);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 19, 18);
    for (i=20;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 18, temp0, temp1);
    // Column 19
    SET_Lii(matLRe, matLIm, matD, 19);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 20, 19);
    for (i=21;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 19, temp0, temp1);
    // Column 20
    SET_Lii(matLRe, matLIm, matD, 20);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 21, 20);
    for (i=22;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 20, temp0, temp1);
    // Column 21
    SET_Lii(matLRe, matLIm, matD, 21);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 22, 21);
    for (i=23;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 21, temp0, temp1);
    // Column 22
    SET_Lii(matLRe, matLIm, matD, 22);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 23, 22);
    for (i=24;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 22, temp0, temp1);
    // Column 23
    SET_Lii(matLRe, matLIm, matD, 23);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 24, 23);
    for (i=25;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 23, temp0, temp1);
    // Column 24
    SET_Lii(matLRe, matLIm, matD, 24);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 25, 24);
    for (i=26;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 24, temp0, temp1);
    // Column 25
    SET_Lii(matLRe, matLIm, matD, 25);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 26, 25);
    for (i=27;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 25, temp0, temp1);
    // Column 26
    SET_Lii(matLRe, matLIm, matD, 26);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 27, 26);
    for (i=28;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 26, temp0, temp1);
    // Column 27
    SET_Lii(matLRe, matLIm, matD, 27);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 28, 27);
    for (i=29;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 27, temp0, temp1);
    // Column 28
    SET_Lii(matLRe, matLIm, matD, 28);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 29, 28);
    for (i=30;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 28, temp0, temp1);
    // Column 29
    SET_Lii(matLRe, matLIm, matD, 29);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 30, 29);
    for (i=31;i<N_32;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, 29, temp0, temp1);
    // Column 30
    SET_Lii(matLRe, matLIm, matD, 30);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 31, 30);
    // Column 31
    SET_Lii(matLRe, matLIm, matD, 31);


    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_32; i ++)
    {
        matInvBRe[i][i] = _mm512_mul_ps(matLRe[i][i], matLRe[i][i]);
        for (k = (i+1); k < N_32; k++)
        {
            temp1 = _mm512_add_ps(_mm512_mul_ps(matLRe[k][i], matLRe[k][i]), _mm512_mul_ps(matLIm[k][i], matLIm[k][i]));
            matInvBRe[i][i] = _mm512_add_ps(matInvBRe[i][i], temp1);
        }
        matInvBIm[i][i] = _mm512_setzero_ps();
    }

    for(i = 0; i < N_32; i ++)
    {
        for(j = i+1; j < N_32; j ++)
        {
            matInvBRe[i][j] = _mm512_mul_ps(matLRe[j][i], matLRe[j][j]);
            matInvBIm[i][j] = _mm512_sub_ps(constZero, _mm512_mul_ps(matLIm[j][i], matLRe[j][j]));

            for (k = (j+1); k < N_32; k++)
            {
                GET_AxBH(matLRe[k][j], matLIm[k][j], matLRe[k][i], matLIm[k][i], temp1, temp2);
                matInvBRe[i][j] = _mm512_add_ps(matInvBRe[i][j], temp1);
                matInvBIm[i][j] = _mm512_add_ps(matInvBIm[i][j], temp2);
            }

            // Hermite matrix
            matInvBRe[j][i] = matInvBRe[i][j];
            matInvBIm[j][i] = _mm512_sub_ps(constZero, matInvBIm[i][j]);
        }
    }
    
    gResistO3_5 = _mm512_add_ps(gResistO3_5,_mm512_sub_ps(matInvBRe[0][0], matInvBIm[0][0]));
}

void matrix_inv_cholesky_32x32_double(__m512 matBRe[MAX_LEN][MAX_LEN][2], __m512 matBIm[MAX_LEN][MAX_LEN][2],
    __m512 matInvBRe[MAX_LEN][MAX_LEN][2], __m512 matInvBIm[MAX_LEN][MAX_LEN][2])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_32][N_32][2], matGIm[N_32][N_32][2];
    __m512 matLRe[N_32][N_32][2], matLIm[N_32][N_32][2];
    __m512 matD[N_32][2], matND[N_32][2];
    __m512 temp0[2], temp1[2], temp2[2];
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00_double(matGRe, matBRe, matD, matND);
    for (i=1;i<N_32;i++)GET_G_COL0_double(matGRe, matGIm, matBRe, matBIm, matD, i);
    // Column 1
    GET_G11_double(matGRe, matGIm, matBRe, matD, matND, temp0);
    for (i=2;i<N_32;i++)GET_G_COL1_double(matGRe, matGIm, matBRe, matBIm, matD, i, temp0, temp1);
    // Column 2
    GET_G22_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    for (i=3;i<N_32;i++)GET_G_COL2_double(matGRe, matGIm, matBRe, matBIm, matD, i, temp0, temp1);
    // Column 3
    GET_G33_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    for (i=4;i<N_32;i++)GET_G_COL3_double(matGRe, matGIm, matBRe, matBIm, matD, i, temp0, temp1);
    // Column 4
    GET_G44_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    for (i=5;i<N_32;i++)GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, i, 4, temp0, temp1);
    // Column 5
    GET_G55_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    for (i=6;i<N_32;i++)GET_G_COL_ODD_double(matGRe, matGIm, matBRe, matBIm, matD, i, 5, temp0, temp1);
    // Column 6
    GET_G66_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    for (i=7;i<N_32;i++)GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, i, 6, temp0, temp1);
    // Column 7
    GET_G77_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    for (i=8;i<N_32;i++)GET_G_COL_ODD_double(matGRe, matGIm, matBRe, matBIm, matD, i, 7, temp0, temp1);
    // Column 8
    GET_Gii_EVEN_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 8);
    for (i=9;i<N_32;i++)GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, i, 8, temp0, temp1);
    // Column 9
    GET_Gii_ODD_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 9);
    for (i=10;i<N_32;i++)GET_G_COL_ODD_double(matGRe, matGIm, matBRe, matBIm, matD, i, 9, temp0, temp1);
    // Column 10
    GET_Gii_EVEN_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 10);
    for (i=11;i<N_32;i++)GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, i, 10, temp0, temp1);
    // Column 11
    GET_Gii_ODD_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 11);
    for (i=12;i<N_32;i++)GET_G_COL_ODD_double(matGRe, matGIm, matBRe, matBIm, matD, i, 11, temp0, temp1);
    // Column 12
    GET_Gii_EVEN_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 12);
    for (i=13;i<N_32;i++)GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, i, 12, temp0, temp1);
    // Column 13
    GET_Gii_ODD_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 13);
    for (i=14;i<N_32;i++)GET_G_COL_ODD_double(matGRe, matGIm, matBRe, matBIm, matD, i, 13, temp0, temp1);
    // Column 14
    GET_Gii_EVEN_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 14);
    for (i=15;i<N_32;i++)GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, i, 14, temp0, temp1);
    // Column 15
    GET_Gii_ODD_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 15);
    for (i=16;i<N_32;i++)GET_G_COL_ODD_double(matGRe, matGIm, matBRe, matBIm, matD, i, 15, temp0, temp1);
    // Column 16
    GET_Gii_EVEN_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 16);
    for (i=17;i<N_32;i++)GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, i, 16, temp0, temp1);
    // Column 17
    GET_Gii_ODD_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 17);
    for (i=18;i<N_32;i++)GET_G_COL_ODD_double(matGRe, matGIm, matBRe, matBIm, matD, i, 17, temp0, temp1);
    // Column 18
    GET_Gii_EVEN_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 18);
    for (i=19;i<N_32;i++)GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, i, 18, temp0, temp1);
    // Column 19
    GET_Gii_ODD_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 19);
    for (i=20;i<N_32;i++)GET_G_COL_ODD_double(matGRe, matGIm, matBRe, matBIm, matD, i, 19, temp0, temp1);
    // Column 20
    GET_Gii_EVEN_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 20);
    for (i=21;i<N_32;i++)GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, i, 20, temp0, temp1);
    // Column 21
    GET_Gii_ODD_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 21);
    for (i=22;i<N_32;i++)GET_G_COL_ODD_double(matGRe, matGIm, matBRe, matBIm, matD, i, 21, temp0, temp1);
    // Column 22
    GET_Gii_EVEN_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 22);
    for (i=23;i<N_32;i++)GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, i, 22, temp0, temp1);
    // Column 23
    GET_Gii_ODD_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 23);
    for (i=24;i<N_32;i++)GET_G_COL_ODD_double(matGRe, matGIm, matBRe, matBIm, matD, i, 23, temp0, temp1);
    // Column 24
    GET_Gii_EVEN_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 24);
    for (i=25;i<N_32;i++)GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, i, 24, temp0, temp1);
    // Column 25
    GET_Gii_ODD_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 25);
    for (i=26;i<N_32;i++)GET_G_COL_ODD_double(matGRe, matGIm, matBRe, matBIm, matD, i, 25, temp0, temp1);
    // Column 26
    GET_Gii_EVEN_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 26);
    for (i=27;i<N_32;i++)GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, i, 26, temp0, temp1);
    // Column 27
    GET_Gii_ODD_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 27);
    for (i=28;i<N_32;i++)GET_G_COL_ODD_double(matGRe, matGIm, matBRe, matBIm, matD, i, 27, temp0, temp1);
    // Column 28
    GET_Gii_EVEN_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 28);
    for (i=29;i<N_32;i++)GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, i, 28, temp0, temp1);
    // Column 29
    GET_Gii_ODD_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 29);
    for (i=30;i<N_32;i++)GET_G_COL_ODD_double(matGRe, matGIm, matBRe, matBIm, matD, i, 29, temp0, temp1);
    // Column 30
    GET_Gii_EVEN_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 30);
    for (i=31;i<N_32;i++)GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, i, 30, temp0, temp1);
    // Column 31
    GET_Gii_ODD_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 31);

    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii_double(matLRe, matLIm, matD, 0);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);
    for (i=2;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 0, temp0, temp1);
    // Column 1
    SET_Lii_double(matLRe, matLIm, matD, 1);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 2, 1);
    for (i=3;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 1, temp0, temp1);
    // Column 2
    SET_Lii_double(matLRe, matLIm, matD, 2);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 3, 2);
    for (i=4;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 2, temp0, temp1);
    // Column 3
    SET_Lii_double(matLRe, matLIm, matD, 3);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 4, 3);
    for (i=5;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 3, temp0, temp1);
    // Column 4
    SET_Lii_double(matLRe, matLIm, matD, 4);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 5, 4);
    for (i=6;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 4, temp0, temp1);
    // Column 5
    SET_Lii_double(matLRe, matLIm, matD, 5);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 6, 5);
    for (i=7;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 5, temp0, temp1);
    // Column 6
    SET_Lii_double(matLRe, matLIm, matD, 6);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 7, 6);
    for (i=8;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 6, temp0, temp1);
    // Column 7
    SET_Lii_double(matLRe, matLIm, matD, 7);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 8, 7);
    for (i=9;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 7, temp0, temp1);
    // Column 8
    SET_Lii_double(matLRe, matLIm, matD, 8);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 9, 8);
    for (i=10;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 8, temp0, temp1);
    // Column 9
    SET_Lii_double(matLRe, matLIm, matD, 9);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 10, 9);
    for (i=11;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 9, temp0, temp1);
    // Column 10
    SET_Lii_double(matLRe, matLIm, matD, 10);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 11, 10);
    for (i=12;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 10, temp0, temp1);
    // Column 11
    SET_Lii_double(matLRe, matLIm, matD, 11);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 12, 11);
    for (i=13;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 11, temp0, temp1);
    // Column 12
    SET_Lii_double(matLRe, matLIm, matD, 12);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 13, 12);
    for (i=14;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 12, temp0, temp1);
    // Column 13
    SET_Lii_double(matLRe, matLIm, matD, 13);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 14, 13);
    for (i=15;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 13, temp0, temp1);
    // Column 14
    SET_Lii_double(matLRe, matLIm, matD, 14);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 15, 14);
    for (i=16;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 14, temp0, temp1);
    // Column 15
    SET_Lii_double(matLRe, matLIm, matD, 15);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 16, 15);
    for (i=17;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 15, temp0, temp1);
    // Column 16
    SET_Lii_double(matLRe, matLIm, matD, 16);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 17, 16);
    for (i=18;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 16, temp0, temp1);
    // Column 17
    SET_Lii_double(matLRe, matLIm, matD, 17);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 18, 17);
    for (i=19;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 17, temp0, temp1);
    // Column 18
    SET_Lii_double(matLRe, matLIm, matD, 18);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 19, 18);
    for (i=20;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 18, temp0, temp1);
    // Column 19
    SET_Lii_double(matLRe, matLIm, matD, 19);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 20, 19);
    for (i=21;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 19, temp0, temp1);
    // Column 20
    SET_Lii_double(matLRe, matLIm, matD, 20);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 21, 20);
    for (i=22;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 20, temp0, temp1);
    // Column 21
    SET_Lii_double(matLRe, matLIm, matD, 21);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 22, 21);
    for (i=23;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 21, temp0, temp1);
    // Column 22
    SET_Lii_double(matLRe, matLIm, matD, 22);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 23, 22);
    for (i=24;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 22, temp0, temp1);
    // Column 23
    SET_Lii_double(matLRe, matLIm, matD, 23);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 24, 23);
    for (i=25;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 23, temp0, temp1);
    // Column 24
    SET_Lii_double(matLRe, matLIm, matD, 24);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 25, 24);
    for (i=26;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 24, temp0, temp1);
    // Column 25
    SET_Lii_double(matLRe, matLIm, matD, 25);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 26, 25);
    for (i=27;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 25, temp0, temp1);
    // Column 26
    SET_Lii_double(matLRe, matLIm, matD, 26);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 27, 26);
    for (i=28;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 26, temp0, temp1);
    // Column 27
    SET_Lii_double(matLRe, matLIm, matD, 27);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 28, 27);
    for (i=29;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 27, temp0, temp1);
    // Column 28
    SET_Lii_double(matLRe, matLIm, matD, 28);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 29, 28);
    for (i=30;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 28, temp0, temp1);
    // Column 29
    SET_Lii_double(matLRe, matLIm, matD, 29);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 30, 29);
    for (i=31;i<N_32;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, 29, temp0, temp1);
    // Column 30
    SET_Lii_double(matLRe, matLIm, matD, 30);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 31, 30);
    // Column 31
    SET_Lii_double(matLRe, matLIm, matD, 31);


    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_32; i ++)
    {
        matInvBRe[i][i][0] = _mm512_mul_pd(matLRe[i][i][0], matLRe[i][i][0]);
        for (k = (i+1); k < N_32; k++)
        {
            temp1[0] = _mm512_add_pd(_mm512_mul_pd(matLRe[k][i][0], matLRe[k][i][0]), _mm512_mul_pd(matLIm[k][i][0], matLIm[k][i][0]));
            matInvBRe[i][i][0] = _mm512_add_pd(matInvBRe[i][i][0], temp1[0]);
        }
        matInvBIm[i][i][0] = _mm512_setzero_pd();
        
        matInvBRe[i][i][1] = _mm512_mul_pd(matLRe[i][i][1], matLRe[i][i][1]);
        for (k = (i+1); k < N_32; k++)
        {
            temp1[1] = _mm512_add_pd(_mm512_mul_pd(matLRe[k][i][1], matLRe[k][i][1]), _mm512_mul_pd(matLIm[k][i][1], matLIm[k][i][1]));
            matInvBRe[i][i][1] = _mm512_add_pd(matInvBRe[i][i][1], temp1[1]);
        }
        matInvBIm[i][i][1] = _mm512_setzero_pd();
    }

    for(i = 0; i < N_32; i ++)
    {
        for(j = i+1; j < N_32; j ++)
        {
            matInvBRe[i][j][0] = _mm512_mul_pd(matLRe[j][i][0], matLRe[j][j][0]);
            matInvBIm[i][j][0] = _mm512_sub_pd(constZero, _mm512_mul_pd(matLIm[j][i][0], matLRe[j][j][0]));

            for (k = (j+1); k < N_32; k++)
            {
                GET_AxBH_double(matLRe[k][j][0], matLIm[k][j][0], matLRe[k][i][0], matLIm[k][i][0], temp1[0], temp2[0]);
                matInvBRe[i][j][0] = _mm512_add_pd(matInvBRe[i][j][0], temp1[0]);
                matInvBIm[i][j][0] = _mm512_add_pd(matInvBIm[i][j][0], temp2[0]);
            }

            // Hermite matrix
            matInvBRe[j][i][0] = matInvBRe[i][j][0];
            matInvBIm[j][i][0] = _mm512_sub_pd(constZero, matInvBIm[i][j][0]);
            
            matInvBRe[i][j][1] = _mm512_mul_pd(matLRe[j][i][1], matLRe[j][j][1]);
            matInvBIm[i][j][1] = _mm512_sub_pd(constZero, _mm512_mul_pd(matLIm[j][i][1], matLRe[j][j][1]));
            
            for (k = (j+1); k < N_32; k++)
            {
                GET_AxBH_double(matLRe[k][j][1], matLIm[k][j][1], matLRe[k][i][1], matLIm[k][i][1], temp1[1], temp2[1]);
                matInvBRe[i][j][1] = _mm512_add_pd(matInvBRe[i][j][1], temp1[1]);
                matInvBIm[i][j][1] = _mm512_add_pd(matInvBIm[i][j][1], temp2[1]);
            }
            
            // Hermite matrix
            matInvBRe[j][i][1] = matInvBRe[i][j][1];
            matInvBIm[j][i][1] = _mm512_sub_pd(constZero, matInvBIm[i][j][1]);
        }
    }
    
    gResistO3_5 = _mm512_add_pd(gResistO3_5,_mm512_sub_pd(matInvBRe[0][0][0], matInvBIm[0][0][0]));
}
    
void matrix_inv_cholesky_64x64(__m512 matBRe[MAX_LEN][MAX_LEN], __m512 matBIm[MAX_LEN][MAX_LEN],
    __m512 matInvBRe[MAX_LEN][MAX_LEN], __m512 matInvBIm[MAX_LEN][MAX_LEN])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_64][N_64], matGIm[N_64][N_64];
    __m512 matLRe[N_64][N_64], matLIm[N_64][N_64];
    __m512 matD[N_64], matND[N_64];
    __m512 temp0, temp1, temp2;
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00(matGRe, matBRe, matD, matND);
    for (i=1;i<N_64;i++)GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, i);
    // Column 1
    GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0);
    for (i=2;i<N_64;i++)GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, i, temp0, temp1);
    // Column 2
    GET_G22(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    for (i=3;i<N_64;i++)GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, i, temp0, temp1);
    // Column 3
    GET_G33(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    for (i=4;i<N_64;i++)GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, i, temp0, temp1);
    // Column 4
    GET_G44(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    for (i=5;i<N_64;i++)GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, i, 4, temp0, temp1);
    // Column 5
    GET_G55(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    for (i=6;i<N_64;i++)GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, i, 5, temp0, temp1);
    // Column 6
    GET_G66(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    for (i=7;i<N_64;i++)GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, i, 6, temp0, temp1);
    // Column 7
    GET_G77(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    for (i=8;i<N_64;i++)GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, i, 7, temp0, temp1);

    for (j=8;j<N_64-1;j++) 
    {
        if (j%2==0)
        {
            // Column j even
            GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, j);
            for (i=j+1;i<N_64;i++)GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, i, j, temp0, temp1);
        }
        else if (j%2==1)
        {
            // Column j odd
            GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, j);
            for (i=j+1;i<N_64;i++)GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, i, j, temp0, temp1);
        }
    }
    // Column 63
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 63);

    /////////////////////////////////// get L, L=invG
    for (j=0;j<N_64-2;j++) 
    {
        // Column j
        SET_Lii(matLRe, matLIm, matD, j);
        GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, j+1, j);
        for (i=j+2;i<N_64;i++)GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, i, j, temp0, temp1);
    }
    // Column 62
    SET_Lii(matLRe, matLIm, matD, 62);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 63, 62);
    // Column 63
    SET_Lii(matLRe, matLIm, matD, 63);


    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_64; i ++)
    {
        matInvBRe[i][i] = _mm512_mul_ps(matLRe[i][i], matLRe[i][i]);
        for (k = (i+1); k < N_64; k++)
        {
            temp1 = _mm512_add_ps(_mm512_mul_ps(matLRe[k][i], matLRe[k][i]), _mm512_mul_ps(matLIm[k][i], matLIm[k][i]));
            matInvBRe[i][i] = _mm512_add_ps(matInvBRe[i][i], temp1);
        }
        matInvBIm[i][i] = _mm512_setzero_ps();
    }

    for(i = 0; i < N_64; i ++)
    {
        for(j = i+1; j < N_64; j ++)
        {
            matInvBRe[i][j] = _mm512_mul_ps(matLRe[j][i], matLRe[j][j]);
            matInvBIm[i][j] = _mm512_sub_ps(constZero, _mm512_mul_ps(matLIm[j][i], matLRe[j][j]));

            for (k = (j+1); k < N_64; k++)
            {
                GET_AxBH(matLRe[k][j], matLIm[k][j], matLRe[k][i], matLIm[k][i], temp1, temp2);
                matInvBRe[i][j] = _mm512_add_ps(matInvBRe[i][j], temp1);
                matInvBIm[i][j] = _mm512_add_ps(matInvBIm[i][j], temp2);
            }

            // Hermite matrix
            matInvBRe[j][i] = matInvBRe[i][j];
            matInvBIm[j][i] = _mm512_sub_ps(constZero, matInvBIm[i][j]);
        }
    }
    
    gResistO3_5 = _mm512_add_ps(gResistO3_5,_mm512_sub_ps(matInvBRe[0][0], matInvBIm[0][0]));
}

void matrix_inv_cholesky_64x64_double(__m512 matBRe[MAX_LEN][MAX_LEN][2], __m512 matBIm[MAX_LEN][MAX_LEN][2],
    __m512 matInvBRe[MAX_LEN][MAX_LEN][2], __m512 matInvBIm[MAX_LEN][MAX_LEN][2])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_64][N_64][2], matGIm[N_64][N_64][2];
    __m512 matLRe[N_64][N_64][2], matLIm[N_64][N_64][2];
    __m512 matD[N_64][2], matND[N_64][2];
    __m512 temp0[2], temp1[2], temp2[2];
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00_double(matGRe, matBRe, matD, matND);
    for (i=1;i<N_64;i++)GET_G_COL0_double(matGRe, matGIm, matBRe, matBIm, matD, i);
    // Column 1
    GET_G11_double(matGRe, matGIm, matBRe, matD, matND, temp0);
    for (i=2;i<N_64;i++)GET_G_COL1_double(matGRe, matGIm, matBRe, matBIm, matD, i, temp0, temp1);
    // Column 2
    GET_G22_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    for (i=3;i<N_64;i++)GET_G_COL2_double(matGRe, matGIm, matBRe, matBIm, matD, i, temp0, temp1);
    // Column 3
    GET_G33_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    for (i=4;i<N_64;i++)GET_G_COL3_double(matGRe, matGIm, matBRe, matBIm, matD, i, temp0, temp1);
    // Column 4
    GET_G44_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    for (i=5;i<N_64;i++)GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, i, 4, temp0, temp1);
    // Column 5
    GET_G55_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    for (i=6;i<N_64;i++)GET_G_COL_ODD_double(matGRe, matGIm, matBRe, matBIm, matD, i, 5, temp0, temp1);
    // Column 6
    GET_G66_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    for (i=7;i<N_64;i++)GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, i, 6, temp0, temp1);
    // Column 7
    GET_G77_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    for (i=8;i<N_64;i++)GET_G_COL_ODD_double(matGRe, matGIm, matBRe, matBIm, matD, i, 7, temp0, temp1);

    for (j=8;j<N_64-1;j++) 
    {
        if (j%2==0)
        {
            // Column j even
            GET_Gii_EVEN_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, j);
            for (i=j+1;i<N_64;i++)GET_G_COL_EVEN_double(matGRe, matGIm, matBRe, matBIm, matD, i, j, temp0, temp1);
        }
        else if (j%2==1)
        {
            // Column j odd
            GET_Gii_ODD_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, j);
            for (i=j+1;i<N_64;i++)GET_G_COL_ODD_double(matGRe, matGIm, matBRe, matBIm, matD, i, j, temp0, temp1);
        }
    }
    // Column 63
    GET_Gii_ODD_double(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 63);

    /////////////////////////////////// get L, L=invG
    for (j=0;j<N_64-2;j++) 
    {
        // Column j
        SET_Lii_double(matLRe, matLIm, matD, j);
        GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, j+1, j);
        for (i=j+2;i<N_64;i++)GET_L_ji_double(matLRe, matLIm, matGRe, matGIm, matND, i, j, temp0, temp1);
    }
    // Column 62
    SET_Lii_double(matLRe, matLIm, matD, 62);
    GET_L_i1i_double(matLRe, matLIm, matGRe, matGIm, matND, 63, 62);
    // Column 63
    SET_Lii_double(matLRe, matLIm, matD, 63);


    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_64; i ++)
    {
        matInvBRe[i][i][0] = _mm512_mul_pd(matLRe[i][i][0], matLRe[i][i][0]);
        for (k = (i+1); k < N_64; k++)
        {
            temp1[0] = _mm512_add_pd(_mm512_mul_pd(matLRe[k][i][0], matLRe[k][i][0]), _mm512_mul_pd(matLIm[k][i][0], matLIm[k][i][0]));
            matInvBRe[i][i][0] = _mm512_add_pd(matInvBRe[i][i][0], temp1[0]);
        }
        matInvBIm[i][i][0] = _mm512_setzero_pd();
        
        matInvBRe[i][i][1] = _mm512_mul_pd(matLRe[i][i][1], matLRe[i][i][1]);
        for (k = (i+1); k < N_64; k++)
        {
            temp1[1] = _mm512_add_pd(_mm512_mul_pd(matLRe[k][i][1], matLRe[k][i][1]), _mm512_mul_pd(matLIm[k][i][1], matLIm[k][i][1]));
            matInvBRe[i][i][1] = _mm512_add_pd(matInvBRe[i][i][1], temp1[1]);
        }
        matInvBIm[i][i][1] = _mm512_setzero_pd();
    }

    for(i = 0; i < N_64; i ++)
    {
        for(j = i+1; j < N_64; j ++)
        {
            matInvBRe[i][j][0] = _mm512_mul_pd(matLRe[j][i][0], matLRe[j][j][0]);
            matInvBIm[i][j][0] = _mm512_sub_pd(constZero, _mm512_mul_pd(matLIm[j][i][0], matLRe[j][j][0]));

            for (k = (j+1); k < N_64; k++)
            {
                GET_AxBH_double(matLRe[k][j][0], matLIm[k][j][0], matLRe[k][i][0], matLIm[k][i][0], temp1[0], temp2[0]);
                matInvBRe[i][j][0] = _mm512_add_pd(matInvBRe[i][j][0], temp1[0]);
                matInvBIm[i][j][0] = _mm512_add_pd(matInvBIm[i][j][0], temp2[0]);
            }

            // Hermite matrix
            matInvBRe[j][i][0] = matInvBRe[i][j][0];
            matInvBIm[j][i][0] = _mm512_sub_pd(constZero, matInvBIm[i][j][0]);
            
            matInvBRe[i][j][1] = _mm512_mul_pd(matLRe[j][i][1], matLRe[j][j][1]);
            matInvBIm[i][j][1] = _mm512_sub_pd(constZero, _mm512_mul_pd(matLIm[j][i][1], matLRe[j][j][1]));
            
            for (k = (j+1); k < N_64; k++)
            {
                GET_AxBH_double(matLRe[k][j][1], matLIm[k][j][1], matLRe[k][i][1], matLIm[k][i][1], temp1[1], temp2[1]);
                matInvBRe[i][j][1] = _mm512_add_pd(matInvBRe[i][j][1], temp1[1]);
                matInvBIm[i][j][1] = _mm512_add_pd(matInvBIm[i][j][1], temp2[1]);
            }
            
            // Hermite matrix
            matInvBRe[j][i][1] = matInvBRe[i][j][1];
            matInvBIm[j][i][1] = _mm512_sub_pd(constZero, matInvBIm[i][j][1]);
        }
    }
    
    gResistO3_5 = _mm512_add_pd(gResistO3_5,_mm512_sub_pd(matInvBRe[0][0][0], matInvBIm[0][0][0]));
}

void calc_cholesky_avx512_float(int32_t caseid, int32_t N)
{   
    float in_re[MAX_SIZE] = {0};
    float in_im[MAX_SIZE] = {0};
    
    for (int32_t i=0;i<NUM_LOOP;i++)
    {
        int32_t ran = rand()%50;
        for (int32_t k=0;k<MAX_SIZE;k++)
        {
            in_re[k] = k + i + ran;
            in_im[k] = k + i + ran;
        }
        
        __m512 matBRe[MAX_LEN][MAX_LEN];
        __m512 matBIm[MAX_LEN][MAX_LEN];
        __m512 matInvBRe[MAX_LEN][MAX_LEN];
        __m512 matInvBIm[MAX_LEN][MAX_LEN];
        
        for (int32_t i=0;i<MAX_LEN;i++)
        {
            int32_t ran = rand()%50;
            for (int32_t k=0;k<MAX_LEN;k++)
            {
                matBRe[i][k] = _mm512_load_ps(in_re + i*ran + k*ran);
                matBIm[i][k] = _mm512_load_ps(in_im + i*ran + k*ran);
            }
        }
        uint64_t t1 = __rdtsc();
        switch(N)
        {
            case 4:
            {   
                matrix_inv_cholesky_4x4(matBRe, matBIm, matInvBRe, matInvBIm);
                break;
            }
            case 8:
            {   
                matrix_inv_cholesky_8x8(matBRe, matBIm, matInvBRe, matInvBIm);
                break;
            }
            case 32:
            {   
                matrix_inv_cholesky_32x32(matBRe, matBIm, matInvBRe, matInvBIm);
                break;
            }
            case 64:
            {   
                matrix_inv_cholesky_64x64(matBRe, matBIm, matInvBRe, matInvBIm);
                break;
            }
        }
        uint64_t t2 = __rdtsc();
        gCycleCount[caseid][i] = t2-t1;
        
        gResistO3_6 += _mm512_reduce_add_ps(gResistO3_5);
        
    }

    printf(" case %d: calc_cholesky_%d_avx512_float\n", caseid, N);
    display(caseid);

}    

void calc_cholesky_avx512_double(int32_t caseid, int32_t N)
{   
    double in_re[MAX_SIZE] = {0};
    double in_im[MAX_SIZE] = {0};
    
    for (int32_t i=0;i<NUM_LOOP;i++)
    {
        int32_t ran = rand()%50;
        for (int32_t k=0;k<MAX_SIZE;k++)
        {
            in_re[k] = k + i + ran;
            in_im[k] = k + i + ran;
        }
        
        __m512 matBRe[MAX_LEN][MAX_LEN][2];
        __m512 matBIm[MAX_LEN][MAX_LEN][2];
        __m512 matInvBRe[MAX_LEN][MAX_LEN][2];
        __m512 matInvBIm[MAX_LEN][MAX_LEN][2];
        
        for (int32_t i=0;i<MAX_LEN;i++)
        {
            int32_t ran = rand()%50;
            for (int32_t k=0;k<MAX_LEN;k++)
            {
                matBRe[i][k][0] = _mm512_load_pd(in_re + i*ran + k*ran);
                matBIm[i][k][0] = _mm512_load_pd(in_im + i*ran + k*ran);
                matBRe[i][k][1] = _mm512_load_pd(in_re + i*ran + k*ran);
                matBIm[i][k][1] = _mm512_load_pd(in_im + i*ran + k*ran);
            }
        }
        uint64_t t1 = __rdtsc();
        switch(N)
        {
            case 4:
            {   
                matrix_inv_cholesky_4x4_double(matBRe, matBIm, matInvBRe, matInvBIm);
                break;
            }
            case 8:
            {   
                matrix_inv_cholesky_8x8_double(matBRe, matBIm, matInvBRe, matInvBIm);
                break;
            }
            case 32:
            {   
                matrix_inv_cholesky_32x32_double(matBRe, matBIm, matInvBRe, matInvBIm);
                break;
            }
            case 64:
            {   
                matrix_inv_cholesky_64x64_double(matBRe, matBIm, matInvBRe, matInvBIm);
                break;
            }
        }
        uint64_t t2 = __rdtsc();
        gCycleCount[caseid][i] = t2-t1;

        gResistO3_6 += _mm512_reduce_add_pd(gResistO3_5);
        
    }

    printf(" case %d: calc_cholesky_%d_avx512_double\n", caseid, N);
    display(caseid);

}    

int main(int argc, char *argv[])
{
    printf("**************************************************************************************************************\n");
    printf(" case start \n");
    printf("**************************************************************************************************************\n");

    memset(gCycleCount, 0, sizeof(int32_t) * NUM_CASE * NUM_LOOP);
    
    calc_coma_avx512_float(0, 4);
    calc_coma_avx512_float(1, 8);
    calc_coma_avx512_float(2, 32);
    calc_coma_avx512_float(3, 64);
    printf("**************************************************************************************************************\n");

    calc_coma_avx512_double(4, 4);
    calc_coma_avx512_double(5, 8);
    calc_coma_avx512_double(6, 32);
    calc_coma_avx512_double(7, 64);
    printf("**************************************************************************************************************\n");

    calc_kron_avx512_float(8, 4);
    calc_kron_avx512_float(9, 8);
    calc_kron_avx512_float(10, 32);
    calc_kron_avx512_float(11, 64);
    printf("**************************************************************************************************************\n");
        
    calc_kron_avx512_double(12, 4);
    calc_kron_avx512_double(13, 8);
    calc_kron_avx512_double(14, 32);
    calc_kron_avx512_double(15, 64);
    printf("**************************************************************************************************************\n");
    
    calc_coma_avg_avx512_float(16, 4);
    calc_coma_avg_avx512_float(17, 8);
    calc_coma_avg_avx512_float(18, 32);
    calc_coma_avg_avx512_float(19, 64);
    printf("**************************************************************************************************************\n");

    calc_coma_avg_avx512_double(20, 4);
    calc_coma_avg_avx512_double(21, 8);
    calc_coma_avg_avx512_double(22, 32);
    calc_coma_avg_avx512_double(23, 64);
    printf("**************************************************************************************************************\n");

    calc_cholesky_avx512_float(24, 4);
    calc_cholesky_avx512_float(25, 8);
    calc_cholesky_avx512_float(26, 32);
    calc_cholesky_avx512_float(27, 64);
    
    printf("**************************************************************************************************************\n");

    calc_cholesky_avx512_double(28, 4);
    calc_cholesky_avx512_double(29, 8);
    calc_cholesky_avx512_double(30, 32);
    calc_cholesky_avx512_double(31, 64);

    gResistO3_4 = gResistO3_1 - gResistO3_2 + gResistO3_3 - gResistO3_6;
    printf("**************************************************************************************************************\n");
    printf(" case end \n");
    printf("**************************************************************************************************************%f\n",gResistO3_4);

    return 0;
}
