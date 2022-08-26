#include <stdio.h>
#include <immintrin.h>
#include <string.h>

typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;

#define MAX_NUM_COMA        64*64
#define MAX_SIZE            MAX_NUM_COMA*2
#define DATA512Float_LOOP   16
#define DATA512Double_LOOP  8
#define NUM_CASE            20
#define NUM_LOOP            100
#define MASK_16_H0          0xFF00
#define MASK_16_H1          0x00FF
#define MASK_8_H0           0xF0
#define MASK_8_H1           0x0F

int32_t gCycleCount[NUM_CASE][NUM_LOOP];
double gResistO3[NUM_CASE][NUM_LOOP];

void vec_single_mul512_conj(float single_value_re,float single_value_im, float* input_vec_re,float* input_vec_im, float* output_vec_re, float* output_vec_im, int32_t len)
{
   // printf("single re is %f, single im is %f\n",single_value_re,single_value_im);
    __m512 single_vec_re = _mm512_set1_ps(single_value_re);
    __m512 single_vec_im = _mm512_set1_ps(single_value_im);
    __m512 re;
    __m512 im;
    int32_t loop = len / DATA512Float_LOOP;
    for(int32_t i = 0; i < loop ; i++)
    {
        re = _mm512_add_ps(_mm512_mul_ps(single_vec_re,_mm512_load_ps(input_vec_re + i * DATA512Float_LOOP)),_mm512_mul_ps(single_vec_im,_mm512_load_ps(input_vec_im + i * DATA512Float_LOOP)));
        im = _mm512_sub_ps(_mm512_mul_ps(single_vec_re,_mm512_load_ps(input_vec_im + i * DATA512Float_LOOP)),_mm512_mul_ps(single_vec_im,_mm512_load_ps(input_vec_re + i * DATA512Float_LOOP)));
        *(__m512*)(output_vec_re + i * DATA512Float_LOOP) = re;
        *(__m512*)(output_vec_im + i * DATA512Float_LOOP) = im;
    }
}

void vec_single_mul512_conj_double(double single_value_re,double single_value_im, double* input_vec_re,double* input_vec_im, double* output_vec_re, double* output_vec_im, int32_t len)
{
   // printf("single re is %f, single im is %f\n",single_value_re,single_value_im);
    __m512d single_vec_re = _mm512_set1_pd(single_value_re);
    __m512d single_vec_im = _mm512_set1_pd(single_value_im);
    __m512d re;
    __m512d im;
    int32_t loop = len / DATA512Double_LOOP;
    for(int32_t i = 0; i < loop ; i++)
    {
        re = _mm512_add_pd(_mm512_mul_pd(single_vec_re,_mm512_load_pd(input_vec_re + i * DATA512Double_LOOP)),_mm512_mul_pd(single_vec_im,_mm512_load_pd(input_vec_im + i * DATA512Double_LOOP)));
        im = _mm512_sub_pd(_mm512_mul_pd(single_vec_re,_mm512_load_pd(input_vec_im + i * DATA512Double_LOOP)),_mm512_mul_pd(single_vec_im,_mm512_load_pd(input_vec_re + i * DATA512Double_LOOP)));
        *(__m512d*)(output_vec_re + i * DATA512Double_LOOP) = re;
        *(__m512d*)(output_vec_im + i * DATA512Double_LOOP) = im;
    }
}

void coma(int32_t len, float* inst_com_re,float* inst_com_im,float* ce_re,float* ce_im)
{
    float* inst_CE_re_1 = ce_re;
    float* inst_CE_im_1 = ce_im;
    //float* inst_CE_re_2 = ce_re+len;
    //float* inst_CE_im_2 = ce_im+len;
    float* inst_coma_re_1 = inst_com_re;
    float* inst_coma_im_1 = inst_com_im;
    //float* inst_coma_re_2 = inst_com_re+len*len;
    //float* inst_coma_im_2 = inst_com_im+len*len;
    // matrix_mul512_conj(inst_CE_re_1,inst_CE_im_1,inst_CE_re_1,inst_CE_im_1,len,1,1,len,inst_coma_re_1,inst_coma_im_1);
    // matrix_mul512_conj(inst_CE_re_2,inst_CE_im_2,inst_CE_re_2,inst_CE_im_2,len,1,1,len,inst_coma_re_2,inst_coma_im_2);
    for(int32_t i = 0; i < len; i++)
    {
        vec_single_mul512_conj(*(inst_CE_re_1+i),*(inst_CE_im_1+i),inst_CE_re_1,inst_CE_im_1,inst_coma_re_1+i*len,inst_coma_im_1+i*len,len);
        //vec_single_mul512_conj(*(inst_CE_re_2+i),*(inst_CE_im_2+i),inst_CE_re_2,inst_CE_im_2,inst_coma_re_2+i*len,inst_coma_im_2+i*len,len);
    }
}

void coma_double(int32_t len, double* inst_com_re,double* inst_com_im,double* ce_re,double* ce_im)
{
    double* inst_CE_re_1 = ce_re;
    double* inst_CE_im_1 = ce_im;
    //double* inst_CE_re_2 = ce_re+len;
    //double* inst_CE_im_2 = ce_im+len;
    double* inst_coma_re_1 = inst_com_re;
    double* inst_coma_im_1 = inst_com_im;
    //double* inst_coma_re_2 = inst_com_re+len*len;
    //double* inst_coma_im_2 = inst_com_im+len*len;
    // matrix_mul512_conj(inst_CE_re_1,inst_CE_im_1,inst_CE_re_1,inst_CE_im_1,len,1,1,len,inst_coma_re_1,inst_coma_im_1);
    // matrix_mul512_conj(inst_CE_re_2,inst_CE_im_2,inst_CE_re_2,inst_CE_im_2,len,1,1,len,inst_coma_re_2,inst_coma_im_2);
    for(int32_t i = 0; i < len; i++)
    {
        vec_single_mul512_conj_double(*(inst_CE_re_1+i),*(inst_CE_im_1+i),inst_CE_re_1,inst_CE_im_1,inst_coma_re_1+i*len,inst_coma_im_1+i*len,len);
        //vec_single_mul512_conj_double(*(inst_CE_re_2+i),*(inst_CE_im_2+i),inst_CE_re_2,inst_CE_im_2,inst_coma_re_2+i*len,inst_coma_im_2+i*len,len);
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
        for (int32_t k=0;k<MAX_SIZE;k++)
        {
            out_re[k] = k%N + i + 1;
            out_im[k] = k%N + i + 2;
            in_re[k]  = k%N + i + 3;
            in_im[k]  = k%N + i + 4;
        }
        
        uint64_t t1 = __rdtsc();
        coma(N,out_re,out_im,in_re,in_im);
        uint64_t t2 = __rdtsc();

        gResistO3[caseid][i] = out_re[i];
        gCycleCount[caseid][i] = t2-t1;
        avg += t2-t1;
    }

    //avg /= NUM_LOOP;
    
    printf(" case %d: calc_coma_%d_avx512_float, avg cycle=%lu\n", caseid, N, avg);

}

void calc_coma_avx512_double(int32_t caseid, int32_t N)
{   
    double out_re[MAX_SIZE] = {0};
    double out_im[MAX_SIZE] = {0};
    double in_re[MAX_SIZE]  = {0};
    double in_im[MAX_SIZE]  = {0};

    uint64_t avg = 0;
    for (int32_t i=0;i<NUM_LOOP;i++)
    {
        for (int32_t k=0;k<MAX_SIZE;k++)
        {
            out_re[k] = k%N + i + 1;
            out_im[k] = k%N + i + 2;
            in_re[k]  = k%N + i + 3;
            in_im[k]  = k%N + i + 4;
        }
        
        uint64_t t1 = __rdtsc();
        coma_double(N,out_re,out_im,in_re,in_im);
        uint64_t t2 = __rdtsc();
        
        gResistO3[caseid][i] = out_re[i];
        gCycleCount[caseid][i] = t2-t1;
        avg += t2-t1;
    }
    //avg /= NUM_LOOP;
    printf(" case %d: calc_coma_%d_avx512_double, avg cycle=%lu\n", caseid, N, avg);

}

void kron(float* in_re1, float* in_im1,float* in_re2, float* in_im2, int32_t len)
{
    if(len*2<=DATA512Float_LOOP)
    {
        *(__m512*)(in_re2) = _mm512_maskz_load_ps((__mmask16)(MASK_16_H1),in_re1);
        *(__m512*)(in_re2+2*len) = _mm512_maskz_load_ps((__mmask16)(MASK_16_H0),in_re1);
        *(__m512*)(in_im2) = _mm512_maskz_load_ps((__mmask16)(MASK_16_H1),in_im1);
        *(__m512*)(in_im2+2*len) = _mm512_maskz_load_ps((__mmask16)(MASK_16_H0),in_im1);
    }
    else
    {
        int32_t loop = len/DATA512Float_LOOP;
        for(int32_t i = 0; i < loop; i++)
        {
            *(__m512*)(in_re2+i*DATA512Float_LOOP) = _mm512_load_ps(in_re1);
            *(__m512*)(in_re2+i*DATA512Float_LOOP+len) = _mm512_setzero_ps();
            *(__m512*)(in_re2+i*DATA512Float_LOOP+2*len) = _mm512_setzero_ps();
            *(__m512*)(in_re2+i*DATA512Float_LOOP+2*len+len) = _mm512_load_ps(in_re1);
            *(__m512*)(in_im2+i*DATA512Float_LOOP) = _mm512_load_ps(in_im1);
            *(__m512*)(in_im2+i*DATA512Float_LOOP+len) = _mm512_setzero_ps();
            *(__m512*)(in_im2+i*DATA512Float_LOOP+2*len) = _mm512_setzero_ps();
            *(__m512*)(in_im2+i*DATA512Float_LOOP+2*len+len) = _mm512_load_ps(in_im1);
        }
    }
}

void kron_double(double* in_re1, double* in_im1,double* in_re2, double* in_im2, int32_t len)
{
    if(len*2<=DATA512Double_LOOP)
    {
        *(__m512d*)(in_re2) = _mm512_maskz_load_pd((__mmask8)(MASK_8_H1),in_re1);
        *(__m512d*)(in_re2+2*len) = _mm512_maskz_load_pd((__mmask8)(MASK_8_H0),in_re1);
        *(__m512d*)(in_im2) = _mm512_maskz_load_pd((__mmask8)(MASK_8_H1),in_im1);
        *(__m512d*)(in_im2+2*len) = _mm512_maskz_load_pd((__mmask8)(MASK_8_H0),in_im1);
    }
    else
    {
        int32_t loop = len/DATA512Double_LOOP;
        for(int32_t i = 0; i < loop; i++)
        {
            *(__m512d*)(in_re2+i*DATA512Double_LOOP) = _mm512_load_pd(in_re1);
            *(__m512d*)(in_re2+i*DATA512Double_LOOP+len) = _mm512_setzero_pd();
            *(__m512d*)(in_re2+i*DATA512Double_LOOP+2*len) = _mm512_setzero_pd();
            *(__m512d*)(in_re2+i*DATA512Double_LOOP+2*len+len) = _mm512_load_pd(in_re1);
            *(__m512d*)(in_im2+i*DATA512Double_LOOP) = _mm512_load_pd(in_im1);
            *(__m512d*)(in_im2+i*DATA512Double_LOOP+len) = _mm512_setzero_pd();
            *(__m512d*)(in_im2+i*DATA512Double_LOOP+2*len) = _mm512_setzero_pd();
            *(__m512d*)(in_im2+i*DATA512Double_LOOP+2*len+len) = _mm512_load_pd(in_im1);
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
        for (int32_t k=0;k<MAX_SIZE;k++)
        {
            in_re1[k] = k%N + i + 1;
            in_im1[k] = k%N + i + 2;
            in_re2[k] = k%N + i + 3;
            in_im2[k] = k%N + i + 4;
        }
        
        uint64_t t1 = __rdtsc();
        kron(in_re1, in_im1, in_re2, in_im2, N);
        uint64_t t2 = __rdtsc();
        
        gResistO3[caseid][i] = in_re2[i] + in_im2[i] + in_re1[i] + in_im1[i];
        gCycleCount[caseid][i] = t2-t1;
        avg += t2-t1;
    }
    //avg /= NUM_LOOP;
    printf(" case %d: calc_kron_%d_avx512_float, avg cycle=%lu\n", caseid, N, avg);

}

void calc_kron_avx512_double(int32_t caseid, int32_t N)
{   
    double in_re1[MAX_SIZE] = {0};
    double in_im1[MAX_SIZE] = {0};
    double in_re2[MAX_SIZE] = {0};
    double in_im2[MAX_SIZE] = {0};

    uint64_t avg = 0;
    for (int32_t i=0;i<NUM_LOOP;i++)
    {
        for (int32_t k=0;k<MAX_SIZE;k++)
        {
            in_re1[k] = k%N + i + 1;
            in_im1[k] = k%N + i + 2;
            in_re2[k] = k%N + i + 3;
            in_im2[k] = k%N + i + 4;
        }
        
        uint64_t t1 = __rdtsc();
        kron_double(in_re1, in_im1, in_re2, in_im2, N);
        uint64_t t2 = __rdtsc();
        
        gResistO3[caseid][i] = in_re2[i] + in_im2[i] + in_re1[i] + in_im1[i];
        gCycleCount[caseid][i] = t2-t1;
        avg += t2-t1;
    }
    //avg /= NUM_LOOP;
    printf(" case %d: calc_kron_%d_avx512_double, avg cycle=%lu\n", caseid, N, avg);

}

int main(int argc, char *argv[])
{
    printf("****************************\n");
    printf(" test start \n");
    printf("****************************\n");

    memset(gCycleCount, 0, sizeof(int32_t) * NUM_CASE * NUM_LOOP);
    
    calc_coma_avx512_float(0, 4);
    calc_coma_avx512_float(1, 8);
    calc_coma_avx512_float(2, 32);
    calc_coma_avx512_float(3, 64);

    calc_coma_avx512_double(4, 4);
    calc_coma_avx512_double(5, 8);
    calc_coma_avx512_double(6, 32);
    calc_coma_avx512_double(7, 64);

    calc_kron_avx512_float(8, 4);
    calc_kron_avx512_float(9, 8);
    calc_kron_avx512_float(10, 32);
    calc_kron_avx512_float(11, 64);
    
    calc_kron_avx512_double(12, 4);
    calc_kron_avx512_double(13, 8);
    calc_kron_avx512_double(14, 32);
    calc_kron_avx512_double(15, 64);
    
    printf("****************************\n");
    printf(" test end \n");
    printf("****************************\n");
    return 0;
}
