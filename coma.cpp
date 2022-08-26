#include <stdio.h>
#include <immintrin.h>
#include <string.h>

typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;

#define MAX_NUM_COMA 64*64
#define DATA512Float_LOOP 16
#define DATA512Double_LOOP 8
#define NUM_CASE 20
#define NUM_LOOP 10

int32_t gCycleCount[NUM_CASE][NUM_LOOP];


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
    float* inst_CE_re_2 = ce_re+len;
    float* inst_CE_im_2 = ce_im+len;
    float* inst_coma_re_1 = inst_com_re;
    float* inst_coma_im_1 = inst_com_im;
    float* inst_coma_re_2 = inst_com_re+len*len;
    float* inst_coma_im_2 = inst_com_im+len*len;
    // matrix_mul512_conj(inst_CE_re_1,inst_CE_im_1,inst_CE_re_1,inst_CE_im_1,len,1,1,len,inst_coma_re_1,inst_coma_im_1);
    // matrix_mul512_conj(inst_CE_re_2,inst_CE_im_2,inst_CE_re_2,inst_CE_im_2,len,1,1,len,inst_coma_re_2,inst_coma_im_2);
    for(int32_t i = 0; i < len; i++)
    {
        vec_single_mul512_conj(*(inst_CE_re_1+i),*(inst_CE_im_1+i),inst_CE_re_1,inst_CE_im_1,inst_coma_re_1+i*len,inst_coma_im_1+i*len,len);
        vec_single_mul512_conj(*(inst_CE_re_2+i),*(inst_CE_im_2+i),inst_CE_re_2,inst_CE_im_2,inst_coma_re_2+i*len,inst_coma_im_2+i*len,len);
    }
}

void coma_double(int32_t len, double* inst_com_re,double* inst_com_im,double* ce_re,double* ce_im)
{
    double* inst_CE_re_1 = ce_re;
    double* inst_CE_im_1 = ce_im;
    double* inst_CE_re_2 = ce_re+len;
    double* inst_CE_im_2 = ce_im+len;
    double* inst_coma_re_1 = inst_com_re;
    double* inst_coma_im_1 = inst_com_im;
    double* inst_coma_re_2 = inst_com_re+len*len;
    double* inst_coma_im_2 = inst_com_im+len*len;
    // matrix_mul512_conj(inst_CE_re_1,inst_CE_im_1,inst_CE_re_1,inst_CE_im_1,len,1,1,len,inst_coma_re_1,inst_coma_im_1);
    // matrix_mul512_conj(inst_CE_re_2,inst_CE_im_2,inst_CE_re_2,inst_CE_im_2,len,1,1,len,inst_coma_re_2,inst_coma_im_2);
    for(int32_t i = 0; i < len; i++)
    {
        vec_single_mul512_conj_double(*(inst_CE_re_1+i),*(inst_CE_im_1+i),inst_CE_re_1,inst_CE_im_1,inst_coma_re_1+i*len,inst_coma_im_1+i*len,len);
        vec_single_mul512_conj_double(*(inst_CE_re_2+i),*(inst_CE_im_2+i),inst_CE_re_2,inst_CE_im_2,inst_coma_re_2+i*len,inst_coma_im_2+i*len,len);
    }
}

void calc_coma_avx512_float(int32_t caseid, int32_t N)
{   
    int32_t len = N*N;
    float *out_re = (float *)calloc(MAX_NUM_COMA*MAX_NUM_COMA*2, sizeof(float));
    float *out_im = (float *)calloc(MAX_NUM_COMA*MAX_NUM_COMA*2, sizeof(float));
    float *in_re  = (float *)calloc(MAX_NUM_COMA*MAX_NUM_COMA*2, sizeof(float));
    float *in_im  = (float *)calloc(MAX_NUM_COMA*MAX_NUM_COMA*2, sizeof(float));

    for (int32_t i=0;i<NUM_LOOP;i++)
    {
        uint64_t t1 = __rdtsc();
        coma(len,out_re,out_im,in_re,in_im);
        uint64_t t2 = __rdtsc();
        gCycleCount[caseid][i] = t2-t1;
    }
    printf(" case %d: calc_coma_%d_avx512_float , end \n", caseid, N);

    free(out_re);
    free(out_im);
    free(in_re);
    free(in_im);
}

void calc_coma_avx512_double(int32_t caseid, int32_t N)
{   
    int32_t len = N*N;
    double *out_re = (double *)calloc(MAX_NUM_COMA*MAX_NUM_COMA*2, sizeof(double));
    double *out_im = (double *)calloc(MAX_NUM_COMA*MAX_NUM_COMA*2, sizeof(double));
    double *in_re  = (double *)calloc(MAX_NUM_COMA*MAX_NUM_COMA*2, sizeof(double));
    double *in_im  = (double *)calloc(MAX_NUM_COMA*MAX_NUM_COMA*2, sizeof(double));

    for (int32_t i=0;i<NUM_LOOP;i++)
    {
        uint64_t t1 = __rdtsc();
        coma_double(len,out_re,out_im,in_re,in_im);
        uint64_t t2 = __rdtsc();
        gCycleCount[caseid][i] = t2-t1;
    }
    printf(" case %d: calc_coma_%d_avx512_double , end \n", caseid, N);

    free(out_re);
    free(out_im);
    free(in_re);
    free(in_im);
}

void vec_mul512(float* m_re, float* m_im, float* n_re, float* n_im, int32_t len, float* out_re, float* out_im)
{
    __m512 Re,Im;
    int32_t N_loop = len/DATA512Float_LOOP;
    float re_temp = 0;
    float im_temp = 0;
    __m512 Re_sum[128],Im_sum[128];
    // Re_sum = _mm512_setzero_ps();
    // Im_sum= _mm512_setzero_ps();
    for(int32_t loop_index = 0; loop_index < N_loop;loop_index++)
    {
        __m512 m_re_512 = _mm512_load_ps(m_re+loop_index*DATA512Float_LOOP);
        __m512 n_re_512 = _mm512_load_ps(n_re+loop_index*DATA512Float_LOOP);
        __m512 m_im_512 = _mm512_load_ps(m_im+loop_index*DATA512Float_LOOP);
        __m512 n_im_512 = _mm512_load_ps(n_im+loop_index*DATA512Float_LOOP);
        *(Re_sum+loop_index) = _mm512_sub_ps(_mm512_mul_ps(m_re_512,n_re_512),_mm512_mul_ps(m_im_512,n_im_512));
        *(Im_sum+loop_index) = _mm512_add_ps(_mm512_mul_ps(m_re_512,n_im_512),_mm512_mul_ps(m_im_512,n_re_512));
        // Re_sum = _mm512_add_ps(Re,Re_sum);
        // Im_sum = _mm512_add_ps(Im,Im_sum);
    }
    Re = _mm512_setzero_ps();
    Im = _mm512_setzero_ps();
    for(int32_t loop_index = 0; loop_index < N_loop; loop_index++)
    {
        Re = _mm512_add_ps(Re_sum[loop_index],Re);
        Im = _mm512_add_ps(Re_sum[loop_index],Im);
    }
    *out_re = _mm512_reduce_add_ps(Re);
    *out_im = _mm512_reduce_add_ps(Im);
}

void matrix_mul512(float* m_re, float* m_im, float* n_re, float* n_im, int32_t m_row, int32_t m_col, int32_t n_row, int32_t n_col, float* out_re, float* out_im)
{
    if(m_col!=n_row)
    {
        printf("error matrix multiplexing dimentions\n");
    }
    float* m_re_start = NULL;
    float* n_re_start = NULL;
    float* m_im_start = NULL;
    float* n_im_start = NULL;
    // the second matrix col
    for(int32_t n_col_index = 0; n_col_index < n_col; n_col_index ++)
    {
        n_re_start = n_re + n_row*n_col_index;
        n_im_start = n_im + n_row*n_col_index;
        for(int32_t m_row_index = 0; m_row_index < m_row; m_row_index ++)
        {
            m_re_start = m_re + m_col*m_row_index;
            m_im_start = m_im + m_col*m_row_index;
            //printf("m_re %f,m_im %f,n_re %f, n_im %f\n",*m_re_start,*m_im_start,*n_re_start,*n_im_start);
            vec_mul512(m_re_start,m_im_start,n_re_start,n_im_start,m_row,out_re+m_row_index*n_col+n_col_index,out_im+m_row_index*n_col+n_col_index);
           // printf("n_col_index is %d,m_col_index is %d,re is %f,im is %f\n",n_col_index,m_row_index,*out_re,*out_im);
        }
    }
}

void calc_kron_avx512_float(int32_t caseid, int32_t N)
{   
    int32_t len = N*N;
    float *out_re = (float *)calloc(MAX_NUM_COMA*MAX_NUM_COMA*2, sizeof(float));
    float *out_im = (float *)calloc(MAX_NUM_COMA*MAX_NUM_COMA*2, sizeof(float));
    float *in_re1  = (float *)calloc(MAX_NUM_COMA*MAX_NUM_COMA*2, sizeof(float));
    float *in_im1  = (float *)calloc(MAX_NUM_COMA*MAX_NUM_COMA*2, sizeof(float));
    float *in_re2  = (float *)calloc(MAX_NUM_COMA*MAX_NUM_COMA*2, sizeof(float));
    float *in_im2  = (float *)calloc(MAX_NUM_COMA*MAX_NUM_COMA*2, sizeof(float));

    for (int32_t i=0;i<NUM_LOOP;i++)
    {
        uint64_t t1 = __rdtsc();
        if (N == 4)
        {
            for(int32_t m_row_index = 0; m_row_index < N; m_row_index ++)
            {
                for(int32_t n_col_index = 0; n_col_index < len*2; n_col_index ++)
                {
                    for(int32_t m_col_index = 0; m_col_index < N; m_col_index ++)
                    {
                        *(out_re+n_col_index+m_row_index*len*2)+=*(in_re1+m_col_index+N*m_row_index)*(*(in_re2+n_col_index+len*2*m_col_index));
                        *(out_im+n_col_index+m_row_index*len*2)+=*(in_im1+m_col_index+N*m_row_index)*(*(in_im2+n_col_index+len*2*m_col_index));
                    }
                }
            }
        }
        else
        {
            matrix_mul512(in_re1,in_im1,in_re2,in_im2,N,N,N,len*2,out_re,out_im);
        }
        uint64_t t2 = __rdtsc();
        gCycleCount[caseid][i] = t2-t1;
    }
    printf(" case %d: calc_kron_%d_avx512_float , end \n", caseid, N);

    free(out_re);
    free(out_im);
    free(in_re1);
    free(in_im1);
    free(in_re2);
    free(in_im2);
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
    
    printf("****************************\n");
    printf(" test end \n");
    printf("****************************\n");
    return 0;
}
