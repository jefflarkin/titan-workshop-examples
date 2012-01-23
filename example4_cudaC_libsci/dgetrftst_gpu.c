#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "libsci_acc.h"

#ifndef min
#define min(a,b)  (((a)<(b))?(a):(b))
#endif
#ifndef max
#define max(a,b)  (((a)<(b))?(b):(a))
#endif



#define BASE_DGETRF 1
#define BASE_DGETRF_CA 0 

#define TEST_DGETRF_CA 0
#define TEST_DGETRF_CRAY 1 

#define NO_PIVOTING 0
#define SHOW_MATRIX 0 

// Flops formula
#define FMULS_GETRF(m, n) ( ((m) < (n)) ? (0.5 * (m) * ((m) * ((n) - (1./3.) * (m) - 1. ) + (n)) + (2. / 3.) * (m)) \
                            :             (0.5 * (n) * ((n) * ((m) - (1./3.) * (n) - 1. ) + (m)) + (2. / 3.) * (n)) )
#define FADDS_GETRF(m, n) ( ((m) < (n)) ? (0.5 * (m) * ((m) * ((n) - (1./3.) * (m)      ) - (n)) + (1. / 6.) * (m)) \
                            :             (0.5 * (n) * ((n) * ((m) - (1./3.) * (n)      ) - (m)) + (1. / 6.) * (n)) )

#define FLOPS(m, n) (      FMULS_GETRF(m, n) +      FADDS_GETRF(m, n) )

extern  double Wall_Time(void);
extern  void    dlarnv_( int *idist, int *iseed, int *n, double *x);
extern  void    dlacpy_( const char *uplo, int *m, int *n,
                         const double *a, int *lda, double *b, int *ldb);
extern  double  dlange_( const char *norm, int *m, int *n,
                         const double *a, int *lda, double *work);
extern  int     dlaswp_( int *n, double *a, int *lda, int *k1,
                         int *k2, int *ipiv, int *incx);


double get_LU_error(int M, int N,
                    double *A,  int lda,
                    double *LU, int *IPIV)
{
    int min_mn = min(M,N);
    int ione   = 1;
    int i, j;
    double alpha = 1.0;
    double beta  = 0.0;
    double *L, *U;
    double work[1], matnorm, residual;

    L = (double *)malloc(sizeof(double)*M*min_mn);
    U = (double *)malloc(sizeof(double)*min_mn*N);
    memset( L, 0, M*min_mn*sizeof(double) );
    memset( U, 0, min_mn*N*sizeof(double) );

    dlaswp_( &N, A, &lda, &ione, &min_mn, IPIV, &ione);
    dlacpy_( "L", &M, &min_mn, LU, &lda, L, &M      );
    dlacpy_( "U", &min_mn, &N, LU, &lda, U, &min_mn );

    for(j=0; j<min_mn; j++)
        L[j+j*M] = 1.0;

    matnorm = dlange_("f", &M, &N, A, &lda, work);

    dgemm_cpu_((char *)"N", (char *)"N", &M, &N, &min_mn,
                  &alpha, L, &M, U, &min_mn, &beta, LU, &lda);

    for( j = 0; j < N; j++ ) {
        for( i = 0; i < M; i++ ) {
            LU[i+j*lda] =  LU[i+j*lda]- A[i+j*lda] ;
        }
    }
    residual = dlange_("f", &M, &N, LU, &lda, work);

    free(L); 
    free(U);
    return residual / (matnorm * N);
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgetrf
*/

char *ACC_LU;
char *NO_PIN;

char PASS[4];
char *SHOW_ERROR;

int main( int argc, char** argv)
{
    libsci_acc_init();

    double       start, end;
    double       flops, gpu_perf, cpu_perf, error;
    double *A, *A2 , *d_A;
    int     *ipiv;

    /* Matrix size */
    int M = 0, N = 0, n2, lda, ldda;
    int size[4] = {1024,1536,2048,2560};

    int i, info, min_mn, maxn, ret;
    int ione     = 1;
    int ISEED[4] = {0,0,0,1};

    float t;

    SHOW_ERROR = getenv("SHOW_ERROR");


    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0)
                N = atoi(argv[++i]);
            else if (strcmp("-M", argv[i])==0)
                M = atoi(argv[++i]);
        }
        if (M>0 && N>0)
            printf("  testing_dgetrf -M %d -N %d\n\n", M, N);
        else
            {
                printf("\nUsage: \n");
                printf("  testing_dgetrf -M %d -N %d\n\n", 1024, 1024);
                exit(1);
            }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_dgetrf_gpu -M %d -N %d\n\n", 1024, 1024);
        M = N = size[3];
    }

    n2     = M * N;
    min_mn = min(M, N);


    /* Allocate host memory for the matrix */
    ipiv = (int *)malloc(sizeof(int)* min_mn);
    A  = (double *)malloc(sizeof(double)*n2);
    A2  = (double *)malloc(sizeof(double)*n2);

    ACC_LU = getenv("LAPACK_ACC_DLU");

    if(ACC_LU){
        if(strcmp(ACC_LU, "CALU") == 0){
                printf("Calling CA version of DGETRF gpu from LIBSCI_ACC testing structure\n");
        }
        else if (strcmp(ACC_LU, "LU") == 0){

                printf("Calling regular version of DGETRF gpu from LIBSCI_ACC testing structure\n");
        }
        else{
                printf("Calling regular version of DGETRF gpu from LIBSCI_ACC testing structure\n");
        }
    }
    else{
         printf("Calling regular version of DGETRF gpu from LIBSCI_ACC testing structure\n");
    }



    printf("\n\n");
    printf("  M     N   CPU GFlop/s    GPU GFlop/s   ||PA-LU||/(||A||*N)\n");
    printf("============================================================\n");
    for(i=0; i<4; i++){
        if (argc == 1){
	    M = N = size[i];
        }
	min_mn= min(M, N);
	lda   = M;
	n2    = lda*N;
	flops = FLOPS( (double)M, (double)N ) / 1000000000;

        /* Initialize the matrix */
        dlarnv_( &ione, ISEED, &n2, A );
        dlacpy_( "A", &M, &N, A, &lda, A2, &lda );

	/* =====================================================================
           Performs operation using LAPACK
           =================================================================== */


        start = Wall_Time();

        dgetrf_cpu_(&M, &N, A, &lda, ipiv, &info);

        end = Wall_Time();

        /* Set A to the original values */

        if (info < 0)
            printf("Argument %d of dgetrf had an illegal value.\n", -info);

        cpu_perf = flops /  (end-start);

           //====================================================================
           //Performs operation using Libsci_acc
           //==================================================================== 
  

        /*  Copy A to the device                  */
        cudaMalloc( &d_A, sizeof(double)*lda*M);
        cublasSetMatrix( M, N, sizeof(double), A2, lda, d_A, lda); 

        /* Calling the accelerator API of dgetrf */
        start = Wall_Time();
        dgetrf_acc_( &M, &N, d_A, &lda, ipiv, &info);
        end = Wall_Time();

       
        /*  Copy A in the device back to the host */
        cublasGetMatrix( M, N, sizeof(double), d_A, lda, A, lda); 
        cudaFree( d_A );

        if (info < 0)
            printf("Argument %d of dgetrf had an illegal value.\n", -info);


        gpu_perf = flops / (end-start);

         //=====================================================================
         //  Check the factorization
         //=================================================================== 
        error = get_LU_error(M, N, A2, lda, A, ipiv);

        if( error < 1e-14 )
                strcpy(PASS, "PASSED"); //PASS[0] = 'P';
        else
                strcpy(PASS, "FAILED"); //PASS[0] = 'F';

	if(SHOW_ERROR)
         printf("%5d %5d  %6.2f         %6.2f         %e\n",
               M, N, cpu_perf, gpu_perf, error);
	else
         printf("%5d %5d  %6.2f         %6.2f         %s\n",
               M, N, cpu_perf, gpu_perf, PASS);


        if (argc != 1)
            break;
    }

    /* Memory clean up */
    free( ipiv );
    free( A );
    free( A2 );

    /* Shutdown */

    libsci_acc_finalize();

}
