#include <cassert>
#include <cmath>
#include <cstdio>  
#include <cstdlib> 
#include <cstring>
#include <sys/time.h>
#include <omp.h>

#include <string>


const double THRESHOLD = 0.001;
unsigned long long GRAIN = 512*512*512; // recursion base case size
bool  check = false;                    // set to true for checking the result
bool PFSPINWAIT=true;                   // enabling spinWait
int  PFWORKERS=1;                       // parallel_for parallelism degree
int  PFGRAIN  =0;                       // default static scheduling of iterations


void random_init (long M, long N, long P, double *A, double *B, double *C) {
    for (long i = 0; i < M; i++) {  
        for (long j = 0; j < P; j++) {  
            A[i*P+j] = 5.0 - ((double)(rand()%100) / 10.0); 
        }
    }
	
    for (long i = 0; i < P; i++) {  
        for (long j = 0; j < N; j++) {  
            B[i*N+j] = 5.0 - ((double)(rand()%100) / 10.0);
        }     
    }

    for (long i = 0; i < P; i++) {
        for (long j = 0; j < N; j++) {
            C[i*N+j] = 0.0;
        }
    }
}

long parse_arg(const char *string) {
    char *endptr;
    long r = strtol(string, &endptr, 10);
    assert(endptr[0]==0); // used up the whole string
    assert(r>0);          // need a positive integer
    return r;
}
//C99 unsigned long long
unsigned long long parse_argull(const char *string) {
    char *endptr;
    unsigned long long r = strtoull(string, &endptr, 10);
    assert(endptr[0]==0); // used up the whole string
    assert(r>0);          // need a positive integer
    return r;
}

void printarray(const double *A, long m, long n, long N) {
    for (long i=0; i<m; i++) {
	for (long j=0; j<n; j++)
	    printf("%f\t", A[i*N+j]);
	printf("\n");
    }
}

// triple nested loop (ijk) implementation 
inline void seqMatMultPF( long m, long n, long p, 
                       const double* A, const long AN, 
                       const double* B, const long BN, 
                       double* C, const long CN)  {   
#pragma omp parallel for schedule(runtime)
    for(long i=0;i<m;++i) {
        for (long k = 0; k < n; k++) {
            for (long j = 0; j < p; j++) {  
		C[i*CN+j] += A[i*AN+k]*B[k*BN+j];
	    }	
        }  
    }
} 
inline void seqMatMult(long m, long n, long p, 
                       const double* A, const long AN, 
                       const double* B, const long BN, 
                       double* C, const long CN)  {   
    for (long i = 0; i < m; i++)  
        for (long j = 0; j < n; j++) {
            C[i*CN+j] = 0.0;  
            for (long k = 0; k < p; k++)  
                C[i*CN+j] += A[i*AN+k]*B[k*BN+j];  
        }  
} 

// m by n with row stride XN for X YN for Y and CN for C
void mmsum ( const double *X, long XN, const double *Y, long YN,
            double *C, long CN,  long m, long n) {
    for(long i=0;i<m;++i) {
            for (long j=0; j<n; j++) 
                C[i*CN+j] = X[i*XN+j] + Y[i*YN + j];
        }
   
}

// m by n with row stride XN for X YN for Y and CN for C
void mmsub ( const double *X, long XN,  const double *Y, long YN, 
	    double *C, long CN,  long m, long n) {
    for(long i=0;i<m;++i) {
            for (long j=0; j<n; j++)
                C[i*CN+j] = X[i*XN+j] - Y[i*YN + j];
        }

}

/* 
 * Strassen algorithm: 
 *
 *  S1  = A11 + A22
 *  S2  = B11 + B22
 *  P1  = S1 * S2
 *  S3  = A21 + A22
 *  P2  = S3 * B11
 *  S4  = B12 - B22
 *  P3  = A11 * S4
 *  S5  = B21 - B11
 *  P4  = A22 * S5
 *  S6  = A11 + A12
 *  P5  = S6 * B22
 *  S7  = A21 - A11
 *  S8  = B11 + B12
 *  P6  = S7 * S8
 *  S9  = A12 - A22
 *  S10 = B21 + B22
 *  P7  = S9*S10
 *  C11 = P1 + P4 - P5 + P7
 *  C12 = P3 + P5
 *  C21 = P2 + P4
 *  C22 = P1 - P2 + P3 + P6
 *
 */
// Version with reduced memory usage. 
// Instead of using temporary arrays (P2, P3, P6 and P7)
// it uses the C matrices
void DCstrassenMMult ( long m, long n, long p,
                      const double *A, const long AN,  // input:  m by p with row stride AN
                      const double *B, const long BN,  // input:  p by n with row stride BN
                      double *C, const long CN) {      // output: m by n with row stride CN.



    if ( (m==1) || (n==1) || (p==1) ||
         (((unsigned long long)m*n*p) < GRAIN) ) {
        seqMatMultPF(m,n,p, A, AN, B, BN, C, CN);
	} else {

        long m2 = m/2;
        long n2 = n/2;
        long p2 = p/2;    
   
        const double *A11 = &A[0];
        const double *A12 = &A[p2];
        const double *A21 = &A[m2*AN];
        const double *A22 = &A[m2*AN+p2];
        
        const double *B11 = &B[0];
        const double *B12 = &B[n2];
        const double *B21 = &B[p2*BN];
        const double *B22 = &B[p2*BN+n2];
        
        double *C11 = &C[0];
        double *C12 = &C[n2];
        double *C21 = &C[m2*CN];
        double *C22 = &C[m2*CN+n2];
        
        double *P1  = (double*)malloc(m2*n2*sizeof(double));
        double *P2  = (double*)malloc(m2*n2*sizeof(double));
        double *P3  = (double*)malloc(m2*n2*sizeof(double));
        double *P4  = (double*)malloc(m2*n2*sizeof(double));
        double *P5  = (double*)malloc(m2*n2*sizeof(double));
        double *P6  = (double*)malloc(m2*n2*sizeof(double));
        double *P7  = (double*)malloc(m2*n2*sizeof(double));		
        
        double *sumA1= (double*)malloc(m2*p2*sizeof(double));
        double *sumB1= (double*)malloc(p2*n2*sizeof(double));
        double *sumA2= (double*)malloc(m2*p2*sizeof(double));
        double *sumB3= (double*)malloc(p2*n2*sizeof(double));
        double *sumB4= (double*)malloc(p2*n2*sizeof(double));
        double *sumA5= (double*)malloc(m2*p2*sizeof(double));
        double *sumA6= (double*)malloc(m2*p2*sizeof(double));
        double *sumB6= (double*)malloc(p2*n2*sizeof(double));
        double *sumA7= (double*)malloc(m2*p2*sizeof(double));
        double *sumB7= (double*)malloc(p2*n2*sizeof(double));

#pragma omp task
        {
        mmsum( A11, AN, A22, AN, sumA1, p2, m2, p2);               // S1
        mmsum( B11, BN, B22, BN, sumB1, n2, p2, n2);               // S2
        DCstrassenMMult( m2,n2,p2, sumA1, p2, sumB1, n2, P1, n2);  // P1
        }
#pragma omp task
        {
        mmsum( A21, AN, A22, AN, sumA2, p2, m2, p2);               // S3 
        DCstrassenMMult( m2,n2,p2, sumA2, p2, B11, BN, P2, n2);    // P2
        }
#pragma omp task
        {
        mmsub( B12, BN, B22, BN, sumB3, n2, p2, n2);               // S4
        DCstrassenMMult( m2,n2,p2, A11, AN, sumB3, n2, P3, n2);    // P3
        }
#pragma omp task
        {
        mmsub( B21, BN, B11, BN, sumB4, n2, p2, n2);               // S5
        DCstrassenMMult( m2,n2,p2, A22, AN, sumB4, n2, P4, n2);    // P4
        }
#pragma omp task
        {
        mmsum( A11, AN, A12, AN, sumA5, p2, m2, p2);               // S6
        DCstrassenMMult( m2,n2,p2, sumA5, p2, B22, BN, P5, n2);    // P5
        }
#pragma omp task
        {
        mmsub( A21, AN, A11, AN, sumA6, p2, m2, p2);               // S7
        mmsum( B11, BN, B12, BN, sumB6, n2, p2, n2);               // S8
        DCstrassenMMult( m2, n2, p2, sumA6, p2, sumB6, n2, P6, n2); // P6
        }
#pragma omp task
        {
        mmsub( A12, AN, A22, AN, sumA7, p2, m2, p2);               // S9
        mmsum( B21, BN, B22, BN, sumB7, n2, p2, n2);               // S10
        DCstrassenMMult( m2, n2, p2, sumA7, p2, sumB7, n2, P7, n2); // P7
        }
#pragma omp taskwait

#pragma omp parallel for schedule(runtime)
	for(long i=0; i<m2; ++i)
		for(long j=0; j<n2; ++j) {
			C11[i*CN + j] = P1[i*n2 + j] + P4[i*n2 + j] - P5[i*n2 + j] + P7[i*n2 + j];
			C12[i*CN + j] = P3[i*n2 + j] + P5[i*n2 + j];
			C21[i*CN + j] = P2[i*n2 + j] + P4[i*n2 + j];
			C22[i*CN + j] = P1[i*n2 + j] - P2[i*n2 + j] + P3[i*n2 + j] + P6[i*n2 + j];
		}
        
    free(P1); free(P2); free(P3); free(P4); free(P5); free(P6); free(P7);     
    free(sumA1); free(sumB1);
    free(sumA2);
    free(sumB3);
    free(sumB4);
    free(sumA5);
    free(sumA6); free(sumB6);        
    free(sumA7); free(sumB7);
    }
}

void strassenMMult ( long m, long n, long p,
                     const double *A, const long AN,
                     const double *B, const long BN,
                     double *C, const long CN) {     
    #pragma omp parallel
    {
        #pragma omp single
        {
            DCstrassenMMult(m,n,p,A,AN,B,BN,C,CN);
        }
    }
}


long CheckResults(long m, long n, const double *C, const double *C1)
{
	for (long i=0; i<m; i++)
		for (long j=0; j<n; j++) {
			long idx = i*n+j;
			if (fabs(C[idx] - C1[idx]) > THRESHOLD) {
				printf("ERROR %ld,%ld %f != %f\n", i, j, C[idx], C1[idx]);
				return 1;
			}
		}
	printf("OK.\n");
	return 0;
}
 
void get_time( void (*F)( long m, long n, long p, const double* A, long AN, const double* B, long BN, double* C, long CN),
                long m, long n, long p, const double* A, long AN, const double* B, long BN, double *C,
	       const char *descr)
{
	printf("Executing %-40s", descr);
	fflush(stdout);
	struct timeval before,after;
	gettimeofday(&before, NULL);

	F( m, n, p, A, AN, B, BN, C, BN);
	gettimeofday(&after,  NULL);

	double tdiff = after.tv_sec - before.tv_sec + (1e-6)*(after.tv_usec - before.tv_usec);
    printf("\nsecs:%11.6f\n", tdiff);

}

void get_time_and_check(void (*F)( long m, long n, long p, const double* A, long AN, const double* B, long BN, double* C, long CN),
                         long m, long n, long p, const double* A, long AN, const double* B, long BN, const double *C_expected, double *C,
			const char *descr)
{
    get_time(F,  m, n, p, A, AN, B, BN, C, descr);
    CheckResults(m, n, C_expected, C);
}



int main(int argc, char* argv[])  {     
    if (argc < 4) {
        printf("\n\tuse: %s <M> <N> <P> [base_case_size=512*512*512] [pfworkers:pfgrain=1:0] [check=0]\n", argv[0]);
        printf("\t       <-> required argument, [-] optional argument\n");
        printf("\t       A is M by P\n");
        printf("\t       B is P by N\n");
        printf("\t       base_case_size is the base case for the recursion (default 512*512*512)\n");
        printf("\t       pfworkers is the  n. of workers of the ParallelFor pattern (default 1)\n");
        printf("\t       pfgrain is the ParallelFor grain size (0, default static scheduling)\n");
        printf("\t       check!=0 executes also the standard ijk algo for checking the result\n\n");
        printf("\tNOTE: M, N and P should be even.\n\n");
        return -1;
    }
    long M = parse_arg(argv[1]);
    long N = parse_arg(argv[2]);
    long P = parse_arg(argv[3]);
    if (argc >= 5) GRAIN=parse_argull(argv[4]);
    if (argc >= 6) {
        std::string pfarg(argv[5]);
        int n = pfarg.find_first_of(":");
        if (n>0) {
            PFWORKERS = atoi(pfarg.substr(0,n).c_str());
            PFGRAIN   = atoi(pfarg.substr(n+1).c_str());
        } else PFWORKERS = atoi(argv[4]);
    }
    if (argc >= 7) check = (atoi(argv[6])?true:false);

    const double *A = (double*)malloc(M*P*sizeof(double));
    const double *B = (double*)malloc(P*N*sizeof(double));
    assert(A); assert(B);

    double *C       = (double*)malloc(M*N*sizeof(double));
    random_init(M, N, P, const_cast<double*>(A), const_cast<double*>(B), const_cast<double*>(C));
    
    //if (check) {
     //   double *C2   = (double*)malloc(M*N*sizeof(double));
      //  get_time(seqMatMult,  M,N,P, A, P, B, N, C, "Standard ijk algorithm");
       // get_time_and_check(strassenMMult,  M, N, P, A, P, B, N, C, C2, "Strassen");
       // free(C2);
    //} else 
        get_time(strassenMMult,  M, N, P, A, P, B, N, C, "Strassen");
    
    free((void*)A); free((void*)B); free(C);

    return 0;  
} 
