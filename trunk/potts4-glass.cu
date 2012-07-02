/*
 * potts4.cu
 */


#include <stddef.h> // NULL, size_t
#include <math.h> // expf
#include <stdio.h> // printf
#include <time.h> // time
#include <sys/time.h> // gettimeofday
#include <assert.h>
#include <limits.h> // UINT_MAX
#include "cutil.h" // CUDA_SAFE_CALL, CUT_CHECK_ERROR

#include<iostream> /*para los printf*/
#include <iomanip> /*para los printf*/
#include<fstream> /*para los printf*/


// Default parameters
#ifndef Q
#define Q 9 // spins
#endif

#define IS_ISING (Q==2)

#ifndef L
#define L 2048 // linear system size
#endif

#ifndef SAMPLES
#define SAMPLES 1 // number of samples
#endif

#ifndef TEMPERATURE
#define TEMPERATURE 0.70f // temperature
#endif

#ifndef TRAN
#define TRAN 20000 // equilibration time
#endif

#ifndef TMAX
#define TMAX 10000 // measurement time
#endif

#ifndef WAITING
#define WAITING 10000
#endif

#ifndef DELTA_T
#define DELTA_T 1 // sampling period for energy and magnetization
#endif

#define MAYOR_ORDEN ((int) log10((double) TMAX + 0.1))

#ifndef NPOINTS
// #define NPOINTS (MAYOR_ORDEN*9 + 3 + 2)
#define NPOINTS 100
#endif

#define MAX_ORD 9
#define MAX_BIN 20

#define REALIZATIONS 1

// Functions

// maximum
#define MAX(a,b) (((a)<(b))?(b):(a))
// minimum
#define MIN(a,b) (((a)<(b))?(a):(b))
// integer ceiling division
#define DIV_CEIL(a,b) (((a)+(b)-1)/(b))
// highest power of two less than x
// Thanks to Pablo Dal Lago for pointing this out, a minor variation of
// http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
#define ROUND_DOWN_POWER_OF_2(x) (( ( (x)>>0 | (x)>>1 | (x)>>2 | (x)>>3 | (x)>>4 | (x)>>5 | (x)>>6 | (x)>>7 | (x)>>8 | (x)>>9 | (x)>>10 | (x)>>11 | (x)>>12 | (x)>>13 | (x)>>14 | (x)>>15 | (x)>>16 | (x)>>17 | (x)>>18 | (x)>>19 | (x)>>20 ) +1 ) >> 1)

// here we cannot use __CUDA_ARCH__, since it is only available on kernels,
// simply uncomment the desired option

// Hardware parameters for GTX 280 (GT200)
//#define SHARED_PER_BLOCK 16384
//#define WARP_SIZE 32
//#define THREADS_PER_BLOCK 512
//#define BLOCKS_PER_GRID 65535
// Hardware parameters for GTX 470/480 (GF100)
#define SHARED_PER_BLOCK 49152
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 512 // 1024 has 66% occupancy
#define BLOCKS_PER_GRID 65535

// Auto adjusting parameters
// Block size for sumupECUDA, autoadjust to fill up max allowable threads per block
// Block size should be less than THREAD_PER_BLOCK and
// less than the SHARED_PER_BLOCK bytes of shared per block
#define BLOCK_E_TMP (MIN(THREADS_PER_BLOCK, SHARED_PER_BLOCK/sizeof(unsigned int)))
#define BLOCK_E (ROUND_DOWN_POWER_OF_2(BLOCK_E_TMP))
// BLOCKS_PER_GRID limits the amount of linear blocks, we divide in GRID_E
#define GRID_E 64
// block size for sumupMCUDA, autoadjust to fill up shared memory per block
// 16 is a little slack because shared is not purely for __shared__
#define BLOCK_M_TMP (MIN(THREADS_PER_BLOCK, (SHARED_PER_BLOCK-16)/(Q*sizeof(unsigned int))))
#define BLOCK_M (ROUND_DOWN_POWER_OF_2(BLOCK_M_TMP))

// Tweakeable parameters
#define CUDA_DEVICE 1	// card number
#define FRAME 512	// the whole thing is framed for the RNG
#define TILE_X 32	// each block of threads is a tile
#define TILE_Y 8
#undef DETERMINISTIC_WRITE // spin write is deterministic or probabilistic

// Internal definitions and functions
// out vector size, it is +1 since we reach TEMP_MAX
//#define NPOINTS (1+(int)((TEMP_MAX-TEMP_MIN)/DELTA_TEMP))
#define N (L*L) // system size
#define SAFE_PRIMES_FILENAME "../../common/safeprimes_base32.txt"
#define SEED (time(NULL)) // random seed
#define MICROSEC (1E-6)
#define WHITE 0
#define BLACK 1

/* FLOATing point */
typedef float FLOAT;

// cells are bytes
typedef unsigned char BYTE;

// time, E, M
struct statpoint {
	unsigned int t; double e; double m; 
};

static int timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y);

#ifdef PRINT_MAPS
static void writePPMbinaryImage(const char * filename, const unsigned int * vector, const int color);
#endif

texture<BYTE,1,cudaReadModeElementType> t_bond;

using namespace std;

// we frame the grid in FRAME*FRAME/2
#define NUM_THREADS (FRAME*FRAME/2)
// state of the random number generator, last number (x_n), last carry (c_n) packed in 64 bits
__device__ static unsigned long long d_x[NUM_THREADS];
// multipliers (constants)
__device__ static unsigned int d_a[NUM_THREADS];


/***
 * Device functions
 ***/

// RNG: multiply with carry
#include "../../common/CUDAMCMLrng.cu"


__global__ void updateCUDA(const FLOAT temp,
			   const unsigned int color,
			   BYTE* __restrict__ const write, unsigned int* const count,
			   const BYTE* __restrict__ const read, unsigned int const time, BYTE* d_bond) {

	const unsigned int jOriginal = blockIdx.x*TILE_X + threadIdx.x;
	const unsigned int iOriginal = blockIdx.y*TILE_Y + threadIdx.y;
	const unsigned int tid = iOriginal*FRAME + jOriginal;

	int h_before, h_after, delta_E;
	BYTE spin_old, spin_new;
	BYTE spin_neigh_n, spin_neigh_e, spin_neigh_s, spin_neigh_w;

	unsigned int i, bindex;
	// move thread RNG state to registers. Thanks to Carlos Bederián for pointing this out.
	unsigned long long rng_state = d_x[tid];
	const unsigned int rng_const = d_a[tid];
	for (unsigned int iFrame=0; iFrame<(L/FRAME); iFrame++) {
		i = iOriginal + (FRAME/2)*iFrame;
		unsigned int j;
		for (unsigned int jFrame=0; jFrame<(L/FRAME); jFrame++) {
			j = jOriginal + FRAME*jFrame;

			spin_old = write[i*L + j];
			
			//localizing neighbors
			spin_neigh_n = read[i*L + j];
			spin_neigh_e = read[i*L + (j+1)%L];
			spin_neigh_w = read[i*L + (j-1+L)%L];
			spin_neigh_s = read[((i+(2*(color^(j%2))-1)+L/2)%(L/2))*L + j];
			
			//reading bonds
			bindex = 4*(2*(i*L+j)+color);
			BYTE bond0 = d_bond[bindex + 0];
			BYTE bond1 = d_bond[bindex + 1];
			BYTE bond2 = d_bond[bindex + 2];
			BYTE bond3 = d_bond[bindex + 3];
			//BYTE bond0 = tex1Dfetch(t_bond, bindex + 0);
			//BYTE bond1 = tex1Dfetch(t_bond, bindex + 1);
			//BYTE bond2 = tex1Dfetch(t_bond, bindex + 2);
			//BYTE bond3 = tex1Dfetch(t_bond, bindex + 3);

			// computing h_before
			h_before = - ((spin_old==spin_neigh_n)==bond0) - ((spin_old==spin_neigh_e)==bond1)
				   - ((spin_old==spin_neigh_w)==bond2) - ((spin_old==spin_neigh_s)==bond3);

			// new spin
			if (IS_ISING){
			//spin_new = (spin_old+1)%2;
			spin_new = (spin_old==0);
			}else{
			spin_new = (spin_old + (BYTE)(1 + rand_MWC_co(&rng_state, &rng_const)*(Q-1))) % Q;
			}

			// h after taking new spin
			h_after = - ((spin_new==spin_neigh_n)==bond0) - ((spin_new==spin_neigh_e)==bond1)
				  - ((spin_new==spin_neigh_w)==bond2) - ((spin_new==spin_neigh_s)==bond3);

			delta_E = h_after - h_before;
			FLOAT p = rand_MWC_co(&rng_state, &rng_const);
#ifdef DETERMINISTIC_WRITE
			int change = delta_E<=0 || p<=expf(-delta_E/temp);
			write[i*L + j] = (change)*spin_new + (1-change)*spin_old;
			if (time>WAITING) count[i*L + j]+= change;
#else
			if (delta_E<=0 || p<=expf(-delta_E/temp)) {
				write[i*L + j] = spin_new;
 				if (time>WAITING) count[i*L + j]+= 1;
			}
#endif
		}
	}
	d_x[tid] = rng_state; // store RNG state into global again
}


__global__ void calculateCUDA(const BYTE* __restrict__ const white,
			      const BYTE* __restrict__ const black,
			      unsigned int* __restrict__ const Ematrix,
			      unsigned int* __restrict__ const Mmatrix) {
	const unsigned int j = blockIdx.x*TILE_X + threadIdx.x;
	const unsigned int i = blockIdx.y*TILE_Y + threadIdx.y;

	// per-block vector of spin counters
	__shared__ unsigned int M[Q];
	// per block sum of energy
	__shared__ unsigned int E;

	// linear coordinates (threadId, blockId)
	const unsigned int tid = threadIdx.y*TILE_X+threadIdx.x;
	const unsigned int bid = blockIdx.y*(L/TILE_X)+blockIdx.x;

	if (tid==0) { // per-block reset of partial sum of energies
		E = 0;
	}
	if (tid<Q) { // per-block reset of vector of spin counters
		M[tid] = 0;
	}
	__syncthreads();

	BYTE spin;
	uchar4 spin_neigh;

	spin = white[i*L + j];
	spin_neigh.x = black[i*L + j];
	spin_neigh.y = black[i*L + (j+1)%L];
	spin_neigh.z = black[i*L + (j-1+L)%L];
	spin_neigh.w = black[((i+(2*(j%2)-1)+L/2)%(L/2))*L + j];

	// energy is pairwise, is enough to add white.
	atomicAdd(&E, (spin==spin_neigh.x)+(spin==spin_neigh.y)+(spin==spin_neigh.z)+(spin==spin_neigh.w));
	atomicAdd(&(M[spin]), 1);

	spin = black[i*L + j];
	atomicAdd(&(M[spin]), 1);

	// store per-block accumulations
	__syncthreads();
	if (tid<Q) {
		Mmatrix[bid*Q + tid] = M[tid];
	}
	if (tid==0) {
		Ematrix[bid] = E;
	}
}


// Input: vector of BLOCKS_CALCULATE energies
//	where BLOCKS_CALCULATE = DIV_CEIL((L*L/2), (TILE_X*TILE_Y))
// Output: vector of (BLOCKS_CALCULATE/BLOCK_E) partial sums.
__global__ void sumupECUDA(unsigned int* __restrict__ const Ematrix) {
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ unsigned int block_E[BLOCK_E];

	block_E[tid] = Ematrix[bid*BLOCK_E + tid];

	// Kirk&Hwu, Programming Massively Parallel Processors, p.100
	for (unsigned int stride=BLOCK_E/2; 0<stride; stride/=2) {
		__syncthreads();
		if (tid<stride) {
			block_E[tid] += block_E[tid+stride];
		}
	}
	// write result
	if (tid==0) {
		Ematrix[bid] = block_E[0];
	}
}


// Input: vector of Q * BLOCKS_CALCULATE magnetizations: m0,m1,...,mq-1,m0,...
//	where BLOCKS_CALCULATE = DIV_CEIL((L*L/2), (TILE_X*TILE_Y))
// Output: vector of Q * (BLOCKS_CALCULATE/BLOCK_M) partial histogram.
__global__ void sumupMCUDA(unsigned int* __restrict__ const Mmatrix) {
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;

	__shared__ unsigned int block_M[BLOCK_M*Q];

	unsigned int i = 0;
	for (i=0; i<Q; i++) {
		block_M[Q*tid + i] = Mmatrix[Q*BLOCK_M*bid + Q*tid + i];
	}

	// Kirk&Hwu, Programming Massively Parallel Processors, p.100
	for (unsigned int stride=BLOCK_M/2; 0<stride; stride/=2) {
		__syncthreads();
		if (tid<stride) {
			for (i=0; i<Q; i++) {
				block_M[Q*tid + i] += block_M[Q*(tid+stride) + i];
			}
		}
	}
	// write result
	if (tid==0) {
		for (i=0; i<Q; i++) {
			Mmatrix[Q*bid + i] = block_M[i];
		}
	}
}

__global__ void init_randomCUDA(BYTE *a){
	const unsigned int jOriginal = blockIdx.x*TILE_X + threadIdx.x;
	const unsigned int iOriginal = blockIdx.y*TILE_Y + threadIdx.y;
	const unsigned int tid = iOriginal*FRAME + jOriginal;

	unsigned int i;
	unsigned long long rng_state = d_x[tid];
	const unsigned int rng_const = d_a[tid];
	for (unsigned int iFrame=0; iFrame<(L/FRAME); iFrame++) {
		i = iOriginal + (FRAME/2)*iFrame;
		unsigned int j;
		for (unsigned int jFrame=0; jFrame<(L/FRAME); jFrame++) {
			j = jOriginal + FRAME*jFrame;
			int index = i*L + j;
			a[index] = (BYTE) (Q*rand_MWC_co(&rng_state, &rng_const));
		}
	}	
	d_x[tid] = rng_state; // store RNG state into global again		
}

__global__ void fill_bondsCUDA(BYTE* d_bond){

	const unsigned int jOriginal = blockIdx.x*TILE_X + threadIdx.x;
	const unsigned int iOriginal = blockIdx.y*TILE_Y + threadIdx.y;
	const unsigned int tid = iOriginal*FRAME + jOriginal;

	unsigned int i;
	BYTE aux;
	// move thread RNG state to registers.
	unsigned long long rng_state = d_x[tid];
	const unsigned int rng_const = d_a[tid];
	for (unsigned int iFrame=0; iFrame<(L/FRAME); iFrame++) {
		i = iOriginal + (FRAME/2)*iFrame;
		unsigned int j;
		for (unsigned int jFrame=0; jFrame<(L/FRAME); jFrame++) {
			j = jOriginal + FRAME*jFrame;

			unsigned int index0 = 4*(2*(i*L+j)+WHITE)+0;
			unsigned int index1 = 4*(2*(i*L+j)+WHITE)+1;
			unsigned int index2 = 4*(2*(i*L+j)+WHITE)+2;
			unsigned int index3 = 4*(2*(i*L+j)+WHITE)+3;

			aux = (BYTE)(rand_MWC_co(&rng_state, &rng_const) <= 0.5);
			//neighbor0 (north) of index. For neighbor0 index is its south neighbor (which is 0 as well)
			d_bond[index0] = aux;
			d_bond[4*(2*(i*L + j) + BLACK) + 0] = aux;

			aux = (BYTE)(rand_MWC_co(&rng_state, &rng_const) <= 0.5);
			//neighbor1 (east). For neighbor1 index is its west neighbor 
			d_bond[index1] = aux;
			d_bond[4*(2*(i*L + (j+1)%L) + BLACK) + 2] = aux;

			aux = (BYTE)(rand_MWC_co(&rng_state, &rng_const) <= 0.5);
			//neighbor2 (west). For neighbor2 index is its east neighbor 
			d_bond[index2] = aux;
			d_bond[4*(2*(i*L + (j-1+L)%L) + BLACK) + 1] = aux;

			aux = (BYTE)(rand_MWC_co(&rng_state, &rng_const) <= 0.5);
			//neighbor3 (south). For neighbor3 index is its north neighbor (which is 3 as well)
			d_bond[index3] = aux;
			d_bond[4*(2*(((i+(2*(WHITE^(j%2))-1)+L/2)%(L/2))*L + j) + BLACK) + 3] = aux;


			//localizing neighbors
			//spin_neigh_n = read[i*L + j];
			//spin_neigh_e = read[i*L + (j+1)%L];
			//spin_neigh_w = read[i*L + (j-1+L)%L];
			//spin_neigh_s = read[((i+(2*(color^(j%2))-1)+L/2)%(L/2))*L + j];

		}
	}
	d_x[tid] = rng_state; // store RNG state into global again
}


__global__ void calculateMFT_CUDA(unsigned int* semi_matrix_count, unsigned int* mft_distr, unsigned int t){
	const unsigned int jOriginal = blockIdx.x*TILE_X + threadIdx.x;
	const unsigned int iOriginal = blockIdx.y*TILE_Y + threadIdx.y;
	unsigned int tid = threadIdx.y*TILE_X+threadIdx.x;

	__shared__ unsigned int sh_distr[MAX_ORD*MAX_BIN];

	unsigned int k =iOriginal*(L/2)+jOriginal;
	unsigned int nf = semi_matrix_count[k];

	while (tid<MAX_ORD*MAX_BIN){
		sh_distr[tid]=0;
		tid+=TILE_X*TILE_Y;
	}
	__syncthreads();

	if (nf!=0){ 
		FLOAT mftaux = (t-WAITING)/(FLOAT)(nf);
		FLOAT aux = log10f(mftaux);
		int ord = (int) aux;
		int bin = (int) (MAX_BIN*(aux-ord));
		atomicAdd(&sh_distr[ord*MAX_BIN+bin],1);
	}

	__syncthreads();

	tid = threadIdx.y*TILE_X+threadIdx.x;
	while (tid<MAX_ORD*MAX_BIN){
		atomicAdd(&mft_distr[tid],sh_distr[tid]);
		tid+=TILE_X*TILE_Y;
	}
}

/*
__global__ void calculateMFT_CUDA(unsigned int* semi_matrix_count, unsigned int* Pdistr, unsigned int t){
	const unsigned int jOriginal = blockIdx.x*TILE_X + threadIdx.x;
	const unsigned int iOriginal = blockIdx.y*TILE_Y + threadIdx.y;
	
	unsigned int k =iOriginal*(L/2)+jOriginal;
	unsigned int nf = semi_matrix_count[k];
//	int ningunflip = nf==0;

	if (nf!=0){ 
//		FLOAT mftaux = (t-WAITING)/(FLOAT)(nf+ningunflip);
		FLOAT mftaux = (t-WAITING)/(FLOAT)(nf);
		FLOAT aux = log10f(mftaux);
		int ORD = (int) aux;
		int BIN = (int) (MAX_BIN*(aux-ORD));
		atomicAdd(&Pdistr[ORD*MAX_BIN+BIN],1);
	}
}
*/


/***
 * Host functions
 ***/

static void calculateMFT(unsigned int* white_count, unsigned int* black_count, unsigned int* d_mft_distribution, unsigned int time){
	dim3 dimBlock(TILE_X, TILE_Y);
	dim3 dimGrid(L/TILE_X, (L/2)/TILE_Y);

	// #FLIPS update, read from white
	calculateMFT_CUDA<<<dimGrid, dimBlock>>>(white_count, d_mft_distribution, time);
	CUT_CHECK_ERROR("Kernel flip_CUDA(BLACK) execution failed");
	
	// #FLIPS update, read from black
	calculateMFT_CUDA<<<dimGrid, dimBlock>>>(black_count, d_mft_distribution, time);
	CUT_CHECK_ERROR("Kernel flip_CUDA(WHITE) execution failed");
}


void init_random(BYTE* const white, BYTE* const black){
	dim3 dimBlock(TILE_X, TILE_Y);
	dim3 dimGrid(FRAME/TILE_X, (FRAME/2)/TILE_Y);
	assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
	assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);

	init_randomCUDA<<<dimGrid, dimBlock>>>(white);
	CUT_CHECK_ERROR("Kernel init_randomCUDA(WHITE) execution failed");	

	init_randomCUDA<<<dimGrid, dimBlock>>>(black);
	CUT_CHECK_ERROR("Kernel init_randomCUDA(BLACK) execution failed");
};


void print_input1(FLOAT *array, ofstream &fstr){
	fstr << "#distribucion al tiempo" << TMAX << endl;
	for(int j=0;j<MAX_ORD*MAX_BIN;j++){
		if (array[j]!=0) fstr << j/(double)(MAX_BIN) << " " << (MAX_BIN)*array[j] << endl; 
		//fstr << j/(FLOAT)(MAX_BIN) << " " << array[k] << endl; 
	}
	fstr << endl;
}

void print_input2(FLOAT *array, ofstream &fstr){
	fstr << "#distribucion al tiempo" << TMAX << endl;

	for(int j=0;j<MAX_ORD*MAX_BIN;j++){
		if (array[j]!=0) fstr << j/(double)(MAX_BIN) << " " << (MAX_BIN)*(TMAX-WAITING)*log(10.)*array[j]/(double)((MAX_BIN)*pow(10.,j/(double)(MAX_BIN))) << endl; 
		//fstr << j/(FLOAT)(MAX_BIN) << " " << array[k] << endl; 
	}
	fstr << endl;
}


static void update(const FLOAT temp, BYTE* const white, BYTE* const black, unsigned int* white_count, unsigned int* black_count, unsigned int const time, BYTE* d_bond) {
	dim3 dimBlock(TILE_X, TILE_Y);
	dim3 dimGrid(FRAME/TILE_X, (FRAME/2)/TILE_Y);
	assert(0.0f<=temp);
	assert(white!=NULL && black!=NULL);
	assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
	assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);

	// white update, read from black
	updateCUDA<<<dimGrid, dimBlock>>>(temp, WHITE, white, white_count, black, time, d_bond);
	CUT_CHECK_ERROR("Kernel updateCUDA(WHITE) execution failed");

	// black update, read from white
	updateCUDA<<<dimGrid, dimBlock>>>(temp, BLACK, black, black_count, white, time, d_bond);
	CUT_CHECK_ERROR("Kernel updateCUDA(BLACK) execution failed");
}


static double calculate(const BYTE* const white, const BYTE* const black, unsigned int* const M_max) {
	double energy = 0.0;
	assert(white!=NULL && black!=NULL);
	assert(M_max!=NULL);

	const unsigned int numBlocksC = DIV_CEIL(L,TILE_X) * DIV_CEIL(L/2,TILE_Y);

	unsigned int *E_matrix = NULL;
	size_t size = numBlocksC * sizeof(unsigned int);
	CUDA_SAFE_CALL(cudaMalloc((void**) &E_matrix, size));

	unsigned int *M_matrix = NULL;
	size = numBlocksC * Q * sizeof(unsigned int);
	CUDA_SAFE_CALL(cudaMalloc((void**) &M_matrix, size));

	// per block TILE_X*TILE_Y accumulation of Energy and Magnetization
	assert(L%TILE_X==0);
	assert((L/2)%TILE_Y==0);
	dim3 dimBlockC(TILE_X, TILE_Y);
	dim3 dimGridC(L/TILE_X, (L/2)/TILE_Y);
	assert(Q<L/2); // at least one thread per magnetization level (Q)
	assert(dimBlockC.x*dimBlockC.y<=THREADS_PER_BLOCK);
	assert(dimGridC.x<=BLOCKS_PER_GRID && dimGridC.y<=BLOCKS_PER_GRID);
	calculateCUDA<<<dimGridC, dimBlockC>>>(white, black, E_matrix, M_matrix);
	CUT_CHECK_ERROR("Kernel calculateCUDA execution failed");

	// partially sum up the energy matrix in the GPU
	assert(numBlocksC%BLOCK_E==0);
	const unsigned int numBlocksE = DIV_CEIL(numBlocksC, BLOCK_E);
	dim3 dimBlockE(BLOCK_E);
	dim3 dimGridE(numBlocksE);
	assert(dimBlockE.x*dimBlockE.y<=THREADS_PER_BLOCK);
	assert(dimGridE.x<=BLOCKS_PER_GRID && dimGridE.y<=BLOCKS_PER_GRID);
	assert(BLOCK_E*sizeof(unsigned int)<=SHARED_PER_BLOCK); // shared memory per block
	sumupECUDA<<<dimGridE, dimBlockE>>>(E_matrix);
	CUT_CHECK_ERROR("Kernel sumupECUDA execution failed");
	// device to host of energy
	static unsigned int E_blocks[numBlocksE];
	CUDA_SAFE_CALL(cudaMemcpy(E_blocks, E_matrix, numBlocksE*sizeof(unsigned int),
				  cudaMemcpyDeviceToHost));
	// CPU accumulating the remaining numBlocksE energies
	unsigned long tmp_energy = 0L; // for L=32768, max energy is 32768*16384*4=2^31, almost overflowing unsigned int
	for (unsigned int i=0; i<numBlocksE; i++) {
		tmp_energy += E_blocks[i];
	}
	energy = -((double)tmp_energy);

	// partially sum up the magnetization matrix in the GPU
	assert(numBlocksC%BLOCK_M==0);
	const unsigned int numBlocksM = DIV_CEIL(numBlocksC, BLOCK_M);
	dim3 dimBlockM(BLOCK_M);
	dim3 dimGridM(numBlocksM);
	assert(dimBlockM.x*dimBlockM.y<=THREADS_PER_BLOCK);
	assert(dimGridM.x<=BLOCKS_PER_GRID && dimGridM.y<=BLOCKS_PER_GRID);
	assert(BLOCK_M*Q*sizeof(unsigned int)<=SHARED_PER_BLOCK); // shared memory per block
	sumupMCUDA<<<dimGridM, dimBlockM>>>(M_matrix);
	CUT_CHECK_ERROR("Kernel sumupMCUDA execution failed");
	// device to host of magnetization
	static unsigned int M_blocks[numBlocksM*Q];
	CUDA_SAFE_CALL(cudaMemcpy(M_blocks, M_matrix, numBlocksM*Q*sizeof(unsigned int),
				  cudaMemcpyDeviceToHost));
	// CPU accumulating the remaining Q*numBlocksM magnetizations
	unsigned int M[Q] = {0};
	for (unsigned int i=0; i<numBlocksM*Q; i++) {
		M[i%Q] += M_blocks[i];
	}

	// compute maximum magnetization
	*M_max = 0;
	for (unsigned int i=0; i<Q; i++) {
		*M_max = MAX(*M_max, M[i]);
	}

	CUDA_SAFE_CALL(cudaFree(E_matrix));
	CUDA_SAFE_CALL(cudaFree(M_matrix));

	return energy;
}

static void sample(BYTE* const white, BYTE* const black, unsigned int* const white_count, unsigned int* const black_count, 
		   struct statpoint stats[], BYTE* d_bond) {
	assert(white!=NULL && black!=NULL);

	// set the device matrix to 0
	//const size_t size = L*L/2*sizeof(BYTE);
	//CUDA_SAFE_CALL(cudaMemset(white, 0, size));
	//CUDA_SAFE_CALL(cudaMemset(black, 0, size));

	//init random
	init_random(white,black);

	double temp = TEMPERATURE;

/*	// equilibrating phase
	for (unsigned int t=0; t<TRAN; t++) {
		update(temp, white, black, white_count, black_count);
	}
*/
	//estado inicial
	double energy = 0.0, mag = 0.0;
	unsigned int M_max = 0;
	energy = calculate(white, black, &M_max);
	mag = (Q*M_max/(1.0*N) - 1) / (double)(Q-1);
	
	int index = 0;
	assert(index<NPOINTS);
	stats[index].t = 0;
	stats[index].e += energy;
	stats[index].m += mag;
	index++;

	int delta_time = DELTA_T;
	for (unsigned int t=1; t<TMAX; t++) {
		update(temp, white, black, white_count, black_count, t, d_bond);

		//if (t%DELTA_T==0) {
		if (t%delta_time==0) {
			if (t==delta_time*10) delta_time *= 10;
			double energy = 0.0, mag = 0.0;
			unsigned int M_max = 0;
			energy = calculate(white, black, &M_max);
			mag = (Q*M_max/(1.0*N) - 1) / (double)(Q-1);
			assert(index<NPOINTS);
			stats[index].t = t;
			stats[index].e += energy;
			stats[index].m += mag;
			index++;
		}	
	}

	//mido también el último punto
	energy = 0.0, mag = 0.0;
	M_max = 0;
	energy = calculate(white, black, &M_max);
	mag = (Q*M_max/(1.0*N) - 1) / (double)(Q-1);
	assert(index<NPOINTS);
	stats[index].t = TMAX;
	stats[index].e += energy;
	stats[index].m += mag;
}


static void fill_bonds(BYTE* d_bond){

	dim3 dimBlock(TILE_X, TILE_Y);
	dim3 dimGrid(FRAME/TILE_X, (FRAME/2)/TILE_Y);
	assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
	assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);

	// white update, read from black
	fill_bondsCUDA<<<dimGrid, dimBlock>>>(d_bond);
	CUT_CHECK_ERROR("Kernel fill_bondsCUDA(WHITE) execution failed");
}

#ifdef PRINT_MAPS
static void print_mft_maps(unsigned int *white_count, unsigned int *black_count){

	unsigned int *h_tmp = NULL;
	unsigned int *h_matrix = NULL;
	const size_t size40 = L * L * sizeof(unsigned int);
	h_tmp=(unsigned int*)malloc(size40);
	h_matrix=(unsigned int*)malloc(size40);

	/* Copy result from device memory to host memory */
	CUDA_SAFE_CALL(cudaMemcpy(h_tmp, white_count, L*L/2*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(&h_tmp[L*L/2], black_count, L*L/2*sizeof(unsigned int), cudaMemcpyDeviceToHost));

	/* Move red then black to redblack */
	for (int i=0; i<L; i++) {
		for(int j=0; j<L; j++) {
			int in = 2*i - L*(i>=L/2) + ((i>=L/2)+j)%2;
			assert(0<=in && in<L);
			h_matrix[in*L+j] = h_tmp[i*L+j];
		}
	}

	char picturename [120], picturenameppm[120], cmd[120], cmd2[120];
	sprintf(picturename, "frame-glass[CUDA,%i,%i,%f,%i]", Q, L, TEMPERATURE, TMAX);
	sprintf(picturenameppm, "%s.ppm", picturename);
	writePPMbinaryImage(picturenameppm, h_matrix,0);
	sprintf(cmd, "convert %s.ppm %s.png", picturename, picturename);
	system(cmd);
	sprintf(cmd2, "rm %s.ppm", picturename);
	system(cmd2);

	sprintf(picturename, "frame-glass[CUDA,%i,%i,%f,%i]-color", Q, L, TEMPERATURE, TMAX);
	sprintf(picturenameppm, "%s.ppm", picturename);
	writePPMbinaryImage(picturenameppm, h_matrix,1);
	sprintf(cmd, "convert %s.ppm %s.png", picturename, picturename);
	system(cmd);
	sprintf(cmd2, "rm %s.ppm", picturename);
	system(cmd2);
	///////////////////////////////////////

}
#endif


static int ConfigureRandomNumbers(void) {
	// Allocate memory for RNG's
	unsigned long long h_x[NUM_THREADS];
	unsigned int h_a[NUM_THREADS];
	unsigned long long seed = (unsigned long long) SEED;

	// Init RNG's
	int error = init_RNG(h_x, h_a, NUM_THREADS, SAFE_PRIMES_FILENAME, seed);

	if (!error) {
		size_t size_x = NUM_THREADS * sizeof(unsigned long long);
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_x, h_x, size_x));

		size_t size_a = NUM_THREADS * sizeof(unsigned int);
		assert(size_a<size_x);
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_a, h_a, size_a));
	}

	return error;
}


int main(void)
{
	// the lattice
	BYTE *white = NULL, *black = NULL;
	// the counter lattice
	unsigned int *white_count=NULL;
	unsigned int *black_count=NULL;
	FLOAT *mft_distribution_av=NULL;
	unsigned int *h_mft_distribution=NULL;
	unsigned int *d_mft_distribution=NULL;	

	BYTE *d_bond = NULL;
	
	// the stats
 	struct statpoint stat[NPOINTS] = { {0, 0.0, 0.0} };
 	//struct statpoint *stat = calloc(NPOINTS, sizeof(struct statpoint));

	double secs = 0.0;
	struct timeval start = {0L,0L}, end = {0L,0L}, elapsed = {0L,0L};

	// parameters checking
	assert(2<=Q); // at least Ising
	assert(Q<(1<<(sizeof(BYTE)*8))); // do not overflow the representation
	assert(0<DELTA_T && DELTA_T<TMAX); // at least one calculate()
	assert(TMAX%DELTA_T==0); // take equidistant calculate()
	assert(L%2==0); // we can halve height
	//assert(512<=FRAME); // TODO: get rid of this
	assert(__builtin_popcount(FRAME)==1); // FRAME=2^k
	assert(L%FRAME==0); // we can frame the grid
	assert(FRAME%2==0); // frames could be halved
	assert((FRAME/2)%TILE_X==0); // half-frames could be tiled
	assert((FRAME/2)%TILE_Y==0); // half-frames could be tiled
	assert((L*L/2)*4L<UINT_MAX); // max energy, that is all spins are the same, fits into a ulong

	// set the GPGPU computing device
	CUDA_SAFE_CALL(cudaSetDevice(CUDA_DEVICE));

	// print header
	printf("# Q: %i\n", Q);
	printf("# L: %i\n", L);
	printf("# Number of Samples: %i\n", SAMPLES);
	printf("# Minimum Temperature: %f\n", TEMPERATURE);
	printf("# Equilibration Time: %i\n", TRAN);
	printf("# Measurement Time: %i\n", TMAX);
	printf("# Data Acquiring Step (on each OoM): %i\n", DELTA_T);
	printf("# Number of Points: %i\n", NPOINTS);
	printf("# Number of Orders: %i\n", MAX_ORD);
	printf("# Number of Bins: %i\n", MAX_BIN);
	
	// start timer
	gettimeofday(&start, NULL);

	if (ConfigureRandomNumbers()) {
		return 1;
	}

	// stop timer
	gettimeofday(&end, NULL);
	timeval_subtract(&elapsed, &end, &start);
	secs = (double)elapsed.tv_sec + ((double)elapsed.tv_usec*MICROSEC);
	printf("# Configure RNG Time (sec): %lf\n", secs);

	// start timer
	gettimeofday(&start, NULL);

	//cudaMalloc de memoria para el sistema
	const size_t size = L * L/2 * sizeof(BYTE);
	CUDA_SAFE_CALL(cudaMalloc((void**) &white, size));
	CUDA_SAFE_CALL(cudaMalloc((void**) &black, size));

	//cudaMalloc de memoria para los contadores de flips
        const size_t size2 = L * L/2 * sizeof(unsigned int);
        CUDA_SAFE_CALL(cudaMalloc((void**) &white_count, size2));
        CUDA_SAFE_CALL(cudaMalloc((void**) &black_count, size2))
	// set the device counter matrix to 0
	CUDA_SAFE_CALL(cudaMemset(white_count, 0, size2));
	CUDA_SAFE_CALL(cudaMemset(black_count, 0, size2));


	//Malloc de memoria para d_mft_distribution
	const size_t size4 = MAX_ORD *MAX_BIN * sizeof(unsigned int);
        CUDA_SAFE_CALL(cudaMalloc((void**) &d_mft_distribution, size4));
	// set the device vector counter to 0
	CUDA_SAFE_CALL(cudaMemset(d_mft_distribution, 0, size4));

	//Malloc de memoria para h_mft_distribution
	h_mft_distribution = (unsigned int *)malloc(size4);	
	mft_distribution_av = (FLOAT *)malloc(MAX_ORD*MAX_BIN*sizeof(FLOAT));
	for (int j=0; j<MAX_ORD*MAX_BIN; j++){
		mft_distribution_av[j] = 0.0;
		h_mft_distribution[j]=0;
	}

	//cudaMalloc para d_bond
	//d_bond is a matrix of 2 * L/L/2 * 4
	//d_bond[color*L*L/2 + 4*(i*L+j) + neighbor]	
	const size_t sizeb = L*L*4*sizeof(BYTE);
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_bond, sizeb));
	
	for (unsigned int i=0; i<SAMPLES; i++) {

		if(i%REALIZATIONS==0){
			//CUDA_SAFE_CALL(cudaUnbindTexture(t_bond));
			fill_bonds(d_bond);
			//CUDA_SAFE_CALL(cudaBindTexture(NULL, t_bond, d_bond, sizeb));
		}

		// set the device counter matrix to 0
		CUDA_SAFE_CALL(cudaMemset(white_count, 0, size2));
		CUDA_SAFE_CALL(cudaMemset(black_count, 0, size2));

		sample(white, black, white_count, black_count, stat, d_bond);

		// set the device vector counter to 0
		CUDA_SAFE_CALL(cudaMemset(d_mft_distribution, 0, size4));
		calculateMFT(white_count, black_count, d_mft_distribution, TMAX);

		// device to host of mft distribution
		CUDA_SAFE_CALL(cudaMemcpy(h_mft_distribution, d_mft_distribution, size4, cudaMemcpyDeviceToHost));
		unsigned long long totalflips=0;
		for (int j=0; j<MAX_ORD*MAX_BIN; j++) 
			totalflips+=h_mft_distribution[j];
		for (int j=0; j<MAX_ORD*MAX_BIN; j++){
			mft_distribution_av[j]+= h_mft_distribution[j]/(FLOAT)(totalflips);
			//mft_distribution_av[j]= h_mft_distribution[j]/(FLOAT)(L*L);
		}

	}

////////////////////
	unsigned int *reference_distribution=NULL;
	reference_distribution = (unsigned int *)malloc(size4);

	for (int j=0; j<MAX_ORD*MAX_BIN; j++){
		reference_distribution[j]=0;
	}

	for (int k=1; k<(TMAX+1-WAITING); k++){
		FLOAT mftaux = (TMAX-WAITING)/(FLOAT)(k);
		FLOAT aux = log10f(mftaux);
		int ORD = (int) aux;
		int BIN = (int) (MAX_BIN*(aux-ORD));
		reference_distribution[ORD*MAX_BIN+BIN]+=1;
	}

/*
	FLOAT aux=0.0;
	for (int j=0; j<MAX_ORD*MAX_BIN; j++){
		aux+=j*reference_distribution[j];
	}
*/

/////////////////

	char mft_file [120];

	for (int j=0; j<MAX_ORD*MAX_BIN; j++){
		mft_distribution_av[j] = mft_distribution_av[j]/(FLOAT)(SAMPLES);
	}
	sprintf(mft_file, "mft[CUDA,%i,%i,%i,%f,%i,%i,%i,%i]-sin.dat", Q, L, SAMPLES, TEMPERATURE, TRAN, TMAX, WAITING, DELTA_T);
	ofstream resultado1(mft_file);
	print_input1(mft_distribution_av, resultado1);

	for (int j=0; j<MAX_ORD*MAX_BIN; j++){
		if (reference_distribution[j]!=0) mft_distribution_av[j] = mft_distribution_av[j]/(FLOAT)(reference_distribution[j]);
	}
	sprintf(mft_file, "mft[CUDA,%i,%i,%i,%f,%i,%i,%i,%i]-con.dat", Q, L, SAMPLES, TEMPERATURE, TRAN, TMAX, WAITING, DELTA_T);
	ofstream resultado2(mft_file);
	print_input2(mft_distribution_av, resultado2);

#ifdef PRINT_MAPS
	print_mft_maps(white_count, black_count);
#endif

	cudaFree(black);
	cudaFree(white);
	cudaFree(black_count);
	cudaFree(white_count);
	cudaFree(d_mft_distribution);
	free(h_mft_distribution);
	free(mft_distribution_av);
	
	// stop timer
	CUDA_SAFE_CALL(cudaThreadSynchronize()); // ensure all threads are done
	gettimeofday(&end, NULL);
	timeval_subtract(&elapsed, &end, &start);
	secs = (double)elapsed.tv_sec + ((double)elapsed.tv_usec*MICROSEC);
	printf("# Total Simulation Time (sec): %lf\n", secs);

// 	printf("OK hasta antes del ultimo free\n");	
	
	printf("# time\tE\tM\n");
	for (unsigned int i=0; i<NPOINTS; i++) {
		printf ("%i\t%.10lf\t%.10lf\n",
			stat[i].t,
			stat[i].e/((double)N*SAMPLES),
			stat[i].m/SAMPLES);
	}
	return 0;
}


/*
 * http://www.gnu.org/software/libtool/manual/libc/Elapsed-Time.html
 * Subtract the `struct timeval' values X and Y,
 * storing the result in RESULT.
 * return 1 if the difference is negative, otherwise 0.
 */
static int timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y) {
	/* Perform the carry for the later subtraction by updating y. */
	if (x->tv_usec < y->tv_usec) {
		int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
		y->tv_usec -= 1000000 * nsec;
		y->tv_sec += nsec;
	}
	if (x->tv_usec - y->tv_usec > 1000000) {
		int nsec = (x->tv_usec - y->tv_usec) / 1000000;
		y->tv_usec += 1000000 * nsec;
		y->tv_sec -= nsec;
	}

	/* Compute the time remaining to wait. tv_usec is certainly positive. */
	result->tv_sec = x->tv_sec - y->tv_sec;
	result->tv_usec = x->tv_usec - y->tv_usec;

	/* Return 1 if result is negative. */
	return x->tv_sec < y->tv_sec;
}



/* Write to a PPM image
 * http://en.wikipedia.org/wiki/Portable_Gray_Map
 * http://netpbm.sourceforge.net/doc/ppm.html
 */

/* RGB formulae.
 * http://gnuplot-tricks.blogspot.com/2009/06/comment-on-phonged-surfaces-in-gnuplot.html
 * TODO: cannot find what to do with over/under flows, currently saturate in [0,255]
 */
#define RGBf03 (x)
#define RGBf05 (x*x*x)
#define RGBf06 (x*x*x*x)
#define RGBf07 (sqrt(x))
#define RGBf11 (abs(x-0.5))
#define RGBf13 (sin(M_PI*x))
#define RGBf15 (sin(2*M_PI*x))
#define RGBf21 (3*x)
#define RGBf22 (3*x-1)
#define RGBf23 (3*x-2)
#define RGBf28 (abs((3*x-1)/2.))
#define RGBf30 (x/0.32-0.78125)
#define RGBf31 (2*x-0.84)
#define RGBf32 (x/0.08-11.5)

#define MAX_COMPONENT_VALUE 255
void writePPMbinaryImage(const char * filename, const unsigned int * vector, int color)
{
	FILE *f = NULL;
	unsigned int i = 0, j = 0;
	unsigned char RGB[3] = {0,0,0}; /* use 3 first bytes for BGR */ 
	f = fopen(filename, "w");
	assert(f);
	fprintf(f, "P6\n"); /* Portable colormap, binary */
	fprintf(f, "%d %d\n", L, L); /* Image size */
	fprintf(f, "%d\n", MAX_COMPONENT_VALUE); /* Max component value */
	FLOAT x;
	for (i=0; i<L; i++) {
		for (j=0; j<L; j++) {
			/* ColorMaps
			 * http://mainline.brynmawr.edu/Courses/cs120/spring2008/Labs/ColorMaps/colorMap.py
			 */
			//FLOAT x = vector[i*L+j]/(FLOAT) (TMAX/16.); /* [0,1] */
			if (color==0) x = (FLOAT) ( log10((TMAX-WAITING)/(FLOAT) vector[i*L+j]) <= 1); /* [0,1] */
			if (color==1) x = (int) log(vector[i*L+j]) / log(TMAX-WAITING); /* [0,1] */
			//FLOAT x = (FLOAT) ((log(vector[i*L+j]) / log(TMAX-WAITING)) <= 0.5); /* [0,1] */
			RGB[0] = (unsigned char) MIN(255, MAX(0, 256*RGBf21));
			RGB[1] = (unsigned char) MIN(255, MAX(0, 256*RGBf22));
			RGB[2] = (unsigned char) MIN(255, MAX(0, 256*RGBf23));
			//assert(0<=RGB[0] && RGB[0]<256);
			//assert(0<=RGB[1] && RGB[1]<256);
			//assert(0<=RGB[2] && RGB[2]<256);
			fwrite((void *)RGB, 3, 1, f);
		}
	}
	fclose(f); f = NULL;
	return;
}


