/*
 * potts4-glass.cu
 */

/*
 * potts4-glass is a CUDA implementation of the +/-J q-state Potts-glass model.
 * It is based on potts3 ((C) 2010 Ferrero, De Francesco, Wolovick, Cannas) presented in [1].
 * For an L*L system, the code chooses randomly a (+1) or (-1) bond for each nearest neighbors interaction.
 * It sets a initial disordered (random) configuration for the blacks/whites cells of the lattice.
 * It fixes the temperature of the system to TEMPERATURE (quench from infinite temperature to TEMPERATURE)
 * and run TMAX Monte Carlo steps (MCS) measuring physical quantities at selected times (logarithmically distributed).
 * Starting at time TWAIT and until time TMAX the number of flips experienced by each spin is counted.
 * An histogram of mean fliping time distribution (MFTD) is constructed.
 * The whole loop is repeated SAMPLES times to average over different disorder configurations.
 * The outputs are the samples averaged energy as a function of time and MFTD.
 * Copyright (C) 2012 Ezequiel E. Ferrero, Federico Romá, Sebastián Bustingorry and Pablo M. Gleiser.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * This code was originally implemented for: "Dynamical heterogeneities as
 * fingerprints of a backbone structure in Potts models",
 * E. E. Ferrero, F. Romá, S. Bustingorry and P. M. Gleiser
 * http://arxiv.org/abs/1209.1743
 * to be published in Physical Review E
 * Please cite when appropriate.
 */

/* References
 * [1] "q-state Potts model metastability study using optimized GPU-based Monte Carlo algorithms",
 * Ezequiel E. Ferrero, Juan Pablo De Francesco, Nicolás Wolovick, Sergio A. Cannas
 * Computer Physics Communications 183 (2012) 1578–1587
 * http://arxiv.org/abs/1101.0876
 */


#include <stddef.h> // NULL, size_t
#include <math.h> // expf
#include <stdio.h> // printf
#include <time.h> // time
#include <sys/time.h> // gettimeofday
#include <assert.h>
#include <limits.h> // UINT_MAX
#include <iostream> // for print stuff
#include <iomanip> //for print stuff
#include <fstream> // for print stuff

#include "cutil.h" // CUDA_SAFE_CALL, CUT_CHECK_ERROR form CUDA SDK

// Default parameters
#ifndef Q
#define Q 9 // spins
#endif

#define IS_ISING (Q==2)

#ifndef L
#define L 4096 // linear system size
#endif

#ifndef SAMPLES
#define SAMPLES 1 // number of samples
#endif

#ifndef TEMPERATURE
#define TEMPERATURE 0.70 // temperature
#endif

#ifndef TMAX
#define TMAX 200000 // measurement time
#endif

#ifndef TWAIT
#define TWAIT 100000 // waiting time (for the MFTD)
#endif

#ifndef DELTA_T
#define DELTA_T 1 // starting sampling period for the energy (then it increases logaritmically)
#endif

//#define GREATEST_OoM ((int) log10((double) TMAX + 0.1))
// #define NPOINTS (GREATEST_OoM*9 + 3 + 2)
#ifndef NPOINTS
#define NPOINTS 100 // This should be big enought
#endif

// Parameters for the MFTD arrays
#define MAX_ORD 9
#define MAX_BIN 20

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

// simply uncomment the desired option
// Hardware parameters for GTX 470/480 (GF100)
#define SHARED_PER_BLOCK 49152
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 512
#define BLOCKS_PER_GRID 65535

// Auto adjusting parameters
// Block size for SumupECUDA, autoadjust to fill up max allowable threads per block
// Block size should be less than THREAD_PER_BLOCK and
// less than the SHARED_PER_BLOCK bytes of shared per block
#define BLOCK_E_TMP (MIN(THREADS_PER_BLOCK, SHARED_PER_BLOCK/sizeof(unsigned int)))
#define BLOCK_E (ROUND_DOWN_POWER_OF_2(BLOCK_E_TMP))
// BLOCKS_PER_GRID limits the amount of linear blocks, we divide in GRID_E
#define GRID_E 64

// Tweakeable parameters
#define CUDA_DEVICE 0	// card number
#define FRAME 512	// the whole thing is framed for the RNG
#define TILE_X 32	// each block of threads is a tile
#define TILE_Y 8
#undef DETERMINISTIC_WRITE // spin write operation is deterministic or probabilistic, only for profiling purposes.

// Internal definitions and functions
#define N (L*L) // system size
#define SAFE_PRIMES_FILENAME "../../../common/safeprimes_base32.txt"
#define SEED (time(NULL)) // random seed otherwise choose any big integer
#define MICROSEC (1E-6)
#define WHITE 0
#define BLACK 1

/* FLOATing point */
#ifndef FULL_DOUBLE_PRESICION
typedef float FLOAT;
#else
typedef double FLOAT;
#endif

// cells are bytes
typedef unsigned char BYTE;

// time, E, M
struct statpoint {
	unsigned int t; double e; double e2; 
};

static int timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y);

#ifdef PRINT_MAPS
static void writePPMbinaryImage(const char * filename, const unsigned int * vector, const int color);
#endif

#ifdef USE_TEXTURE
texture<BYTE,1,cudaReadModeElementType> t_bond;
#endif

using namespace std;

// we frame the grid in FRAME*FRAME/2
#define NUM_THREADS (FRAME*FRAME/2)
// state of the random number generator, last number (x_n), last carry (c_n) packed in 64 bits
__device__ static unsigned long long d_x[NUM_THREADS];
// multipliers (constants)
__device__ static unsigned int d_a[NUM_THREADS];

// RNG: multiply with carry
#include "../../../common/CUDAMCMLrng.cu"

/***
 * Device functions
 ***/

__global__ void CalculateMftdCUDA(unsigned int* semi_matrix_count, unsigned int* mft_distr, unsigned int t){
	const unsigned int jOriginal = blockIdx.x*TILE_X + threadIdx.x;
	const unsigned int iOriginal = blockIdx.y*TILE_Y + threadIdx.y;
	unsigned int tid = threadIdx.y*TILE_X+threadIdx.x;

	// We construct the histogram in parallel.
	// First each block reduces a partial histogram in shared memory and then we atomicAdd everything.

	__shared__ unsigned int sh_distr[MAX_ORD*MAX_BIN];

	unsigned int k =iOriginal*(L/2)+jOriginal;
	unsigned int nf = semi_matrix_count[k];

	while (tid<MAX_ORD*MAX_BIN){
		sh_distr[tid]=0;
		tid+=TILE_X*TILE_Y;
	}
	__syncthreads();

	if (nf!=0){ 
		FLOAT mftaux = (t-TWAIT)/(FLOAT)(nf);
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

__global__ void UpdateCUDA(const FLOAT temp,
			   const unsigned int color,
			   BYTE* __restrict__ const write, unsigned int* const count,
			   const BYTE* __restrict__ const read, unsigned int const time, const BYTE* d_bond) {

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
			bindex = 2*N*color + 4*(i*L+j);
			#ifndef USE_TEXTURE
			BYTE bond0 = d_bond[bindex + 0];
			BYTE bond1 = d_bond[bindex + 1];
			BYTE bond2 = d_bond[bindex + 2];
			BYTE bond3 = d_bond[bindex + 3];
			#else
			BYTE bond0 = tex1Dfetch(t_bond, bindex + 0);
			BYTE bond1 = tex1Dfetch(t_bond, bindex + 1);
			BYTE bond2 = tex1Dfetch(t_bond, bindex + 2);
			BYTE bond3 = tex1Dfetch(t_bond, bindex + 3);
			#endif

			// computing h_before
			h_before = - ((spin_old==spin_neigh_n)==bond0) - ((spin_old==spin_neigh_e)==bond1)
				   - ((spin_old==spin_neigh_w)==bond2) - ((spin_old==spin_neigh_s)==bond3);

			// new spin
			if (IS_ISING){
			  spin_new = (spin_old==0);
			}else{
			  spin_new = (spin_old + (BYTE)(1 + rand_MWC_co(&rng_state, &rng_const)*(Q-1))) % Q;
			}

			// h after taking new spin
			h_after = - ((spin_new==spin_neigh_n)==bond0) - ((spin_new==spin_neigh_e)==bond1)
				  - ((spin_new==spin_neigh_w)==bond2) - ((spin_new==spin_neigh_s)==bond3);

			delta_E = h_after - h_before;
			FLOAT p = rand_MWC_co(&rng_state, &rng_const);
#ifdef DETERMINISTIC_WRITE //only for profiling porpuses
			int change = delta_E<=0 || p<=expf(-delta_E/temp);
			write[i*L + j] = (change)*spin_new + (1-change)*spin_old;
			if (time>TWAIT) count[i*L + j]+= change;
#else
			if (delta_E<=0 || p<=expf(-delta_E/temp)) {
				write[i*L + j] = spin_new;
				if (time>TWAIT) count[i*L + j]+= 1;
			}
#endif
		}
	}
	d_x[tid] = rng_state; // store RNG state into global again
}

// Input: vector of BLOCKS_CALCULATE energies
//	where BLOCKS_CALCULATE = DIV_CEIL((L*L/2), (TILE_X*TILE_Y))
// Output: vector of (BLOCKS_CALCULATE/BLOCK_E) partial sums.
__global__ void SumupECUDA(unsigned int* __restrict__ const Ematrix) {
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

__global__ void CalculateCUDA(const BYTE* __restrict__ const white,
			      const BYTE* __restrict__ const black,
			      unsigned int* __restrict__ const Ematrix,
			      const BYTE* d_bond) {
	const unsigned int j = blockIdx.x*TILE_X + threadIdx.x;
	const unsigned int i = blockIdx.y*TILE_Y + threadIdx.y;

	// per block sum of energy
	__shared__ unsigned int E;

	// linear coordinates (threadId, blockId)
	const unsigned int tid = threadIdx.y*TILE_X+threadIdx.x;
	const unsigned int bid = blockIdx.y*(L/TILE_X)+blockIdx.x;

	if (tid==0) { // per-block reset of partial sum of energies
		E = 0;
	}
	__syncthreads();

	BYTE spin;
	uchar4 spin_neigh;

	spin = white[i*L + j];
	spin_neigh.x = black[i*L + j];
	spin_neigh.y = black[i*L + (j+1)%L];
	spin_neigh.z = black[i*L + (j-1+L)%L];
	spin_neigh.w = black[((i+(2*(j%2)-1)+L/2)%(L/2))*L + j];

	//reading WHITES bonds
	const unsigned int bindex = 4*(i*L+j);
	BYTE bond0 = d_bond[bindex + 0];
	BYTE bond1 = d_bond[bindex + 1];
	BYTE bond2 = d_bond[bindex + 2];
	BYTE bond3 = d_bond[bindex + 3];

	// energy is pairwise, is enough to add white.
	atomicAdd(&E, (spin==spin_neigh.x)==bond0)+((spin==spin_neigh.y)==bond1)
			+((spin==spin_neigh.z)==bond2)+((spin==spin_neigh.w)==bond3);
	// store per-block accumulations
	__syncthreads();
	if (tid==0) {
		Ematrix[bid] = E;
	}
}

__global__ void InitRandomCUDA(BYTE *a){
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

__global__ void FillBondsCUDA(BYTE* d_bond){

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

			//Index for WHITE cells neighbors.
			//First 4 L*L/2 entries of d_bond stores bonds for WHITES cells, 
			//the second half stores bonds for black cells
			unsigned int index0 = 4*(i*L+j)+0;
			unsigned int index1 = 4*(i*L+j)+1;
			unsigned int index2 = 4*(i*L+j)+2;
			unsigned int index3 = 4*(i*L+j)+3;

			//The real problem, match up bonds
			aux = (BYTE)(rand_MWC_co(&rng_state, &rng_const) <= 0.5);
			//neighbor0 (north) of index. For neighbor0 index is its south neighbor (which is 0 as well)
			d_bond[index0] = aux;
			d_bond[2*N + 4*(i*L + j) + 0] = aux;

			aux = (BYTE)(rand_MWC_co(&rng_state, &rng_const) <= 0.5);
			//neighbor1 (east). For neighbor1 index is its west neighbor 
			d_bond[index1] = aux;
			d_bond[2*N + 4*(i*L + (j+1)%L) + 2] = aux;

			aux = (BYTE)(rand_MWC_co(&rng_state, &rng_const) <= 0.5);
			//neighbor2 (west). For neighbor2 index is its east neighbor 
			d_bond[index2] = aux;
			d_bond[2*N + 4*(i*L + (j-1+L)%L) + 1] = aux;

			aux = (BYTE)(rand_MWC_co(&rng_state, &rng_const) <= 0.5);
			//neighbor3 (south). For neighbor3 index is its north neighbor (which is 3 as well)
			d_bond[index3] = aux;
			d_bond[2*N + 4*(((i+(2*(WHITE^(j%2))-1)+L/2)%(L/2))*L + j) + 3] = aux;

			//localizing neighbors
			//spin_neigh_n = read[i*L + j];
			//spin_neigh_e = read[i*L + (j+1)%L];
			//spin_neigh_w = read[i*L + (j-1+L)%L];
			//spin_neigh_s = read[((i+(2*(color^(j%2))-1)+L/2)%(L/2))*L + j];
		}
	}
	d_x[tid] = rng_state; // store RNG state into global again
}

/***
 * Host functions
 ***/

static void CalculateMftd(unsigned int* white_count, unsigned int* black_count, 
			unsigned int* d_mft_distribution, unsigned int time){
	dim3 dimBlock(TILE_X, TILE_Y);
	dim3 dimGrid(L/TILE_X, (L/2)/TILE_Y);

	// #FLIPS update, read from white
	CalculateMftdCUDA<<<dimGrid, dimBlock>>>(white_count, d_mft_distribution, time);
	CUT_CHECK_ERROR("Kernel flip_CUDA(BLACK) execution failed");
	
	// #FLIPS update, read from black
	CalculateMftdCUDA<<<dimGrid, dimBlock>>>(black_count, d_mft_distribution, time);
	CUT_CHECK_ERROR("Kernel flip_CUDA(WHITE) execution failed");
}

void PrintMftdCorrected(FLOAT *array, ofstream &fstr){
	fstr << "#distribucion al tiempo" << TMAX << endl;

	for(int j=0;j<MAX_ORD*MAX_BIN;j++){
		if (array[j]!=0) fstr << j/(double)(MAX_BIN) << " " << (MAX_BIN)*(TMAX-TWAIT)*log(10.)*array[j]/(double)((MAX_BIN)*pow(10.,j/(double)(MAX_BIN))) << endl; 
	}
	fstr << endl;
}

void PrintMftdRaw(FLOAT *array, ofstream &fstr){
	fstr << "#distribucion al tiempo" << TMAX << endl;
	for(int j=0;j<MAX_ORD*MAX_BIN;j++){
		if (array[j]!=0) fstr << j/(double)(MAX_BIN) << " " << (MAX_BIN)*array[j] << endl; 
	}
	fstr << endl;
}

static void Update(const FLOAT temp, BYTE* const white, BYTE* const black, unsigned int* white_count, unsigned int* black_count, unsigned int const time, BYTE* d_bond) {
	dim3 dimBlock(TILE_X, TILE_Y);
	dim3 dimGrid(FRAME/TILE_X, (FRAME/2)/TILE_Y);
	assert(0.0f<=temp);
	assert(white!=NULL && black!=NULL);
	assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
	assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);

	// white update, read from black
	UpdateCUDA<<<dimGrid, dimBlock>>>(temp, WHITE, white, white_count, black, time, d_bond);
	CUT_CHECK_ERROR("Kernel UpdateCUDA(WHITE) execution failed");

	// black update, read from white
	UpdateCUDA<<<dimGrid, dimBlock>>>(temp, BLACK, black, black_count, white, time, d_bond);
	CUT_CHECK_ERROR("Kernel UpdateCUDA(BLACK) execution failed");
}

static double Calculate(const BYTE* const white, const BYTE* const black, const BYTE* d_bond) {
	double energy = 0.0;
	assert(white!=NULL && black!=NULL);

	const unsigned int numBlocksC = DIV_CEIL(L,TILE_X) * DIV_CEIL(L/2,TILE_Y);

	unsigned int *E_matrix = NULL;
	size_t size = numBlocksC * sizeof(unsigned int);
	CUDA_SAFE_CALL(cudaMalloc((void**) &E_matrix, size));

	// per block TILE_X*TILE_Y accumulation of Energy
	assert(L%TILE_X==0);
	assert((L/2)%TILE_Y==0);
	dim3 dimBlockC(TILE_X, TILE_Y);
	dim3 dimGridC(L/TILE_X, (L/2)/TILE_Y);
	assert(dimBlockC.x*dimBlockC.y<=THREADS_PER_BLOCK);
	assert(dimGridC.x<=BLOCKS_PER_GRID && dimGridC.y<=BLOCKS_PER_GRID);
	CalculateCUDA<<<dimGridC, dimBlockC>>>(white, black, E_matrix, d_bond);
	CUT_CHECK_ERROR("Kernel CalculateCUDA execution failed");

	// partially sum up the energy matrix in the GPU
	assert(numBlocksC%BLOCK_E==0);
	const unsigned int numBlocksE = DIV_CEIL(numBlocksC, BLOCK_E);
	dim3 dimBlockE(BLOCK_E);
	dim3 dimGridE(numBlocksE);
	assert(dimBlockE.x*dimBlockE.y<=THREADS_PER_BLOCK);
	assert(dimGridE.x<=BLOCKS_PER_GRID && dimGridE.y<=BLOCKS_PER_GRID);
	assert(BLOCK_E*sizeof(unsigned int)<=SHARED_PER_BLOCK); // shared memory per block
	SumupECUDA<<<dimGridE, dimBlockE>>>(E_matrix);
	CUT_CHECK_ERROR("Kernel SumupECUDA execution failed");
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

	CUDA_SAFE_CALL(cudaFree(E_matrix));

	return energy;
}

void InitRandom(BYTE* const white, BYTE* const black){
	dim3 dimBlock(TILE_X, TILE_Y);
	dim3 dimGrid(FRAME/TILE_X, (FRAME/2)/TILE_Y);
	assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
	assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);

	InitRandomCUDA<<<dimGrid, dimBlock>>>(white);
	CUT_CHECK_ERROR("Kernel InitRandomCUDA(WHITE) execution failed");

	InitRandomCUDA<<<dimGrid, dimBlock>>>(black);
	CUT_CHECK_ERROR("Kernel InitRandomCUDA(BLACK) execution failed");
};


static void Sample(BYTE* const white, BYTE* const black, unsigned int* const white_count, unsigned int* const black_count, 
		   struct statpoint stats[], BYTE* d_bond) {
	assert(white!=NULL && black!=NULL);

	// Uncomment if ordered initialization is desired
	// set the device lattice to 0
	//const size_t size = L*L/2*sizeof(BYTE);
	//CUDA_SAFE_CALL(cudaMemset(white, 0, size));
	//CUDA_SAFE_CALL(cudaMemset(black, 0, size));

	//init random
	InitRandom(white,black);

	//initial values
	double energy = 0.0;
	energy = Calculate(white, black, d_bond);

	int index = 0;
	assert(index<NPOINTS);
	stats[index].t = 0;
	stats[index].e += energy;
	stats[index].e2 += energy*energy;
	index++;

	int delta_time = DELTA_T;
	FLOAT temp = TEMPERATURE;
	for (unsigned int t=1; t<TMAX; t++) {
		Update(temp, white, black, white_count, black_count, t, d_bond);

		//if (t%DELTA_T==0) { //linear separated points
		if (t%delta_time==0) { //logarithmically separated points
			if (t==delta_time*10) delta_time *= 10;
			double energy = 0.0;
			energy = Calculate(white, black, d_bond);
			assert(index<NPOINTS);
			stats[index].t = t;
			stats[index].e += energy;
			stats[index].e2 += energy*energy;
			index++;
		}
	}

	//be sure to meassure last value
	energy = Calculate(white, black, d_bond);
	assert(index<NPOINTS);
	stats[index].t = TMAX;
	stats[index].e += energy;
	stats[index].e2 += energy*energy;
}


static void FillBonds(BYTE* d_bond){

	dim3 dimBlock(TILE_X, TILE_Y);
	dim3 dimGrid(FRAME/TILE_X, (FRAME/2)/TILE_Y);
	assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
	assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);

	// white update, read from black
	FillBondsCUDA<<<dimGrid, dimBlock>>>(d_bond);
	CUT_CHECK_ERROR("Kernel FillBondsCUDA(WHITE) execution failed");
}

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
	// the spins lattice
	BYTE *white = NULL, *black = NULL;
	//the bonds array
	BYTE *d_bond = NULL;

	// the counter array
	unsigned int *white_count=NULL;
	unsigned int *black_count=NULL;

	//MFTD arrays
	FLOAT *mft_distribution_av=NULL;
	unsigned int *h_mft_distribution=NULL;
	unsigned int *d_mft_distribution=NULL;

	// the energy stats
	struct statpoint stat[NPOINTS] = { {0, 0.0, 0.0} };

	double secs = 0.0;
	struct timeval start = {0L,0L}, end = {0L,0L}, elapsed = {0L,0L};

	// parameters checking
	assert(2<=Q); // at least Ising
	assert(Q<(1<<(sizeof(BYTE)*8))); // do not overflow the representation
	assert(0<DELTA_T && DELTA_T<TMAX); // at least one Calculate()
	assert(L%2==0); // we can halve height
	assert(512<=FRAME); // TODO: get rid of this
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
	printf("# Temperature: %f\n", TEMPERATURE);
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

	//Memory alloc for the system of spins
	const size_t size = L * L/2 * sizeof(BYTE);
	CUDA_SAFE_CALL(cudaMalloc((void**) &white, size));
	CUDA_SAFE_CALL(cudaMalloc((void**) &black, size));

	//Memory alloc for d_bond on the device
	//d_bond is a matrix of N*4
	//d_bond[color*L*L + 4*(i*L+j) + neighbor]
	const size_t sizeb = L*L*4*sizeof(BYTE);
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_bond, sizeb));

	//Memory alloc for the flip counters on the Device
	const size_t size2 = L * L/2 * sizeof(unsigned int);
	CUDA_SAFE_CALL(cudaMalloc((void**) &white_count, size2));
	CUDA_SAFE_CALL(cudaMalloc((void**) &black_count, size2))
	// set the device counter matrix to 0
	CUDA_SAFE_CALL(cudaMemset(white_count, 0, size2));
	CUDA_SAFE_CALL(cudaMemset(black_count, 0, size2));

	//Memory alloc for the MFT distribution on the Device
	const size_t size4 = MAX_ORD *MAX_BIN * sizeof(unsigned int);
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_mft_distribution, size4));
	// set the device vector counter to 0
	CUDA_SAFE_CALL(cudaMemset(d_mft_distribution, 0, size4));

	//Memory alloc for the MFT distribution on the Host
	h_mft_distribution = (unsigned int *)malloc(size4);
	mft_distribution_av = (FLOAT *)malloc(MAX_ORD*MAX_BIN*sizeof(FLOAT));
	// set the host vector counter to 0
	for (int j=0; j<MAX_ORD*MAX_BIN; j++){
		mft_distribution_av[j] = 0.0;
		h_mft_distribution[j]=0;
	}

	for (unsigned int i=0; i<SAMPLES; i++) {

		#ifndef USE_TEXTURE
		FillBonds(d_bond);
		#else
		CUDA_SAFE_CALL(cudaUnbindTexture(t_bond));
		FillBonds(d_bond);
		CUDA_SAFE_CALL(cudaBindTexture(NULL, t_bond, d_bond, sizeb));
		#endif

		// set the device counter matrix to 0
		CUDA_SAFE_CALL(cudaMemset(white_count, 0, size2));
		CUDA_SAFE_CALL(cudaMemset(black_count, 0, size2));

		// run a sample
		Sample(white, black, white_count, black_count, stat, d_bond);

		// calculate sample's MFTD
		CUDA_SAFE_CALL(cudaMemset(d_mft_distribution, 0, size4));
		CalculateMftd(white_count, black_count, d_mft_distribution, TMAX);

		// device to host of mft distribution
		CUDA_SAFE_CALL(cudaMemcpy(h_mft_distribution, d_mft_distribution, size4, cudaMemcpyDeviceToHost));
		unsigned long long totalflips=0;
		for (int j=0; j<MAX_ORD*MAX_BIN; j++) 
			totalflips+=h_mft_distribution[j];
		for (int j=0; j<MAX_ORD*MAX_BIN; j++){
			mft_distribution_av[j]+= h_mft_distribution[j]/(FLOAT)(totalflips);
		}
	}

	char mft_file [120];

	for (int j=0; j<MAX_ORD*MAX_BIN; j++){
		mft_distribution_av[j] = mft_distribution_av[j]/(FLOAT)(SAMPLES);
	}
	sprintf(mft_file, "mft[%i,%i,%i,%f,%i,%i,%i]-raw.dat", Q, L, SAMPLES, TEMPERATURE, TMAX, TWAIT, DELTA_T);
	ofstream filestr1(mft_file);
	PrintMftdRaw(mft_distribution_av, filestr1);

	/*
	The histogram discretization in logarithmic scale can introduce undesired effects.
	Given a MAX_BIN value, not all bins will have a priori the same chance to be populated (besides the physics).
	That is because there are different numbers of integers numbers laying on each bin, as expected, but growing 
	in a non-monotonically way from one bin to the next one.
	To avoid this problem we propose the following correction.
	*/

	/// EXPERIMENTAL ////
	unsigned int *reference_distribution=NULL;
	reference_distribution = (unsigned int *)malloc(size4);

	for (int j=0; j<MAX_ORD*MAX_BIN; j++){
		reference_distribution[j]=0;
	}

	for (int k=1; k<(TMAX+1-TWAIT); k++){
		FLOAT mftaux = (TMAX-TWAIT)/(FLOAT)(k);
		FLOAT aux = log10f(mftaux);
		int ORD = (int) aux;
		int BIN = (int) (MAX_BIN*(aux-ORD));
		reference_distribution[ORD*MAX_BIN+BIN]+=1;
	}

	for (int j=0; j<MAX_ORD*MAX_BIN; j++){
		if (reference_distribution[j]!=0) 
			mft_distribution_av[j] = mft_distribution_av[j]/(FLOAT)(reference_distribution[j]);
	}
	sprintf(mft_file, "mft[%i,%i,%i,%f,%i,%i,%i]-corrected.dat", Q, L, SAMPLES, TEMPERATURE, TMAX, TWAIT, DELTA_T);
	ofstream filestr2(mft_file);
	PrintMftdCorrected(mft_distribution_av, filestr2);

	///--------------///

	//Cleaning issues
	cudaFree(black);
	cudaFree(white);
	cudaFree(black_count);
	cudaFree(white_count);
	cudaFree(d_mft_distribution);
	free(h_mft_distribution);
	free(mft_distribution_av);

	// stop timer
	gettimeofday(&end, NULL);
	timeval_subtract(&elapsed, &end, &start);
	secs = (double)elapsed.tv_sec + ((double)elapsed.tv_usec*MICROSEC);
	printf("# Total Simulation Time (sec): %lf\n", secs);

	printf("# time\tE\tE^2\n");
		printf ("%i\t%.10lf\t%.10lf\n",
			stat[0].t,
			stat[0].e/((double)N*SAMPLES),
			stat[0].e2/((double)N*N*SAMPLES));
	for (unsigned int i=1; i<NPOINTS; i++) {
		if (stat[i].t!=0) printf ("%i\t%.10lf\t%.10lf\n",
			stat[i].t,
			stat[i].e/((double)N*SAMPLES),
			stat[i].e2/((double)N*N*SAMPLES));
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

