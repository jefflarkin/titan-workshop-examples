
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <libsci_acc.h>

extern double dlange_(char *,int *,int *,double *,int *,double *);
extern  double Wall_Time(void);


int main( )
{
   double alpha,beta,*a,*b,*c, *c2,work,error;
   int m,n,k,lda,ldb,ldc;
   int i,j;
   double t2,t1,t_gpu,t_cpu;
   extern double drand48();

   m = 4096;
   k = 4096;
   n = 4096;
   lda = m;
   ldb = k;
   ldc = m;
   alpha =1.0;
   beta=0.0;
   a  = (double *)malloc(sizeof(double)*m*k);
   b  = (double *)malloc(sizeof(double)*n*k);
   c  = (double *)malloc(sizeof(double)*m*n);
   c2 = (double *)malloc(sizeof(double)*m*n);
/*
 * Create random matrixes
 */
   for ( j = 0 ; j < k ; j++ ) {
      for ( i = 0 ; i < m ; i++ ) {
         a[ i+j*lda] = drand48();
      }
   }

   for ( j = 0 ; j < k ; j++ ) {
      for ( i = 0 ; i < n ; i++ ) {
         b[ i+j*ldb] = drand48();
      }
   }

   for ( j = 0 ; j < n ; j++ ) {
      for ( i = 0 ; i < m ; i++ ) {
         c2[ i+j*ldc]= c[ i+j*ldc] = drand48();
      }
   }


/*
 *  Call Libsci-acc mnual API from ACC region
 */
   t1 = Wall_Time();
#pragma acc data region copyin(a[0:lda*k-1],b[0:n*ldb-1]) copy(c[0:ldc*n-1])
   {
      dgemm_acc('n','n',m,n,k,alpha,ad,lda,bd,ldb,beta,cd,ldc);
   }
   t2 = Wall_Time();
   t_gpu =(t2-t1)*1.0e9;
/*
 *  Call Libsci-acc manual API for CPUs
 */
   t1 = Wall_Time();
   dgemm_cpu('n','n',m,n,k,alpha,a,lda,b,ldb,beta,c2,ldc);
   t2 = Wall_Time();
   t_cpu =(t2-t1)*1.0e9;

/*
 *  Check the error
 */
   for( j = 0; j < n; j++ ) {
      for( i = 0; i < m; i++ )
      {
         c[i+j*ldc] -=c2[i+j*ldc];
      }
   }

   error = dlange_("M",&m,&n,c,&ldc,&work);
   printf("Error %e \n",error);
   printf("dgemm_acc %e Gflops\n",2.0*((double)m*(double)n*(double)k)/t_gpu);
   printf("dgemm_cpu %e Gflops\n",2.0*((double)m*(double)n*(double)k)/t_cpu);
   free(a);
   free(b);
   free(c);
   free(c2);
   return 0;
}

