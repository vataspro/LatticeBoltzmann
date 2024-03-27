#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <sys/time.h>

/***
 * Print usage
 ***/
void
usage(char *argv[])
{
  fprintf(stderr, "usage: %s N\n", argv[0]);
  return;
}

/***
 * Allocate memory; print error if NULL is returned
 ***/
void *
ualloc(size_t size)
{
  void *ptr = malloc(size);
  if(ptr == NULL) {
    fprintf(stderr, "malloc() returned null; quitting...\n");
    exit(-2);
  }
  return ptr;
}

/***
 * Allocate memory on GPU; print error if not successful
 ***/
void *
gpu_alloc(size_t size)
{
  void *ptr;
  cudaError_t err = cudaMalloc(&ptr, size);
  if(err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc() returned %d; quitting...\n", err);
    exit(-2);
  } 
  return ptr;
}

/***
 * Return a random number in [0, 1)
 ***/
double
urand(void)
{
  double x = (double)rand()/(double)RAND_MAX;
  return x;
}

/***
 * Return seconds elapsed since t0, with t0 = 0 the epoch
 ***/
double
stop_watch(double t0)
{
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (double)t.tv_usec/1e6 - t0;
}

/***
 * Initialise the momentum density
 ***/
void
init_momentum_density(int Nx, int Ny, int NL, double *F)
{
    for(int j=0; j<Ny; j++) {
        for(int i=0; i<Nx; i++) {
            for (int l=0; l<NL; l++){
                F[j*Nx*NL + i*NL + l] = 1.; //urand();
            }
            F[j*Nx*NL + i*NL + 3] += 2 * (1+0.2* cos(2*M_PI*i/Nx*4));
        }
    }
    return;
}

/***
 * Get the density (rho) 
 ***/
void
get_rho(int Nx, int Ny, int Nl, double *F, double *rho)
{
    for(int j=0; j<Ny; j++) {
      for(int i=0; i<Nx; i++) {
        rho[j*Nx + i] = 0.;
          for (int l=0; l<Nl; l++) {
              rho[j*Nx + i] += F[j*Nx*Nl + i*Nl + l];
          }
      }
  }
}

/***
 * Get the density (rho) on the GPU
 ***/
__global__ void
gpu_get_rho(int Nx, int Ny, int Nl, double *F, double *rho)
{
  // Call as <<96, 1, 9>>
  __shared__ double res[96];
  __shared__ double A[96*9];
  
  int idx_loc = threadIdx.x + threadIdx.z*blockDim.x;
  int idx_glob = blockIdx.y * Nx*Nl + idx_loc + (blockIdx.x*blockDim.x)*Nl;

  A[idx_loc] = F[idx_glob];

  if (threadIdx.z == 0) {
    res[threadIdx.x] = 0;
  }

  __syncthreads();
  if (threadIdx.z == 0) {
    for(int k=0; k<Nl; k++) {
      res[threadIdx.x] += A[threadIdx.x * Nl + k];
    }

  rho[blockIdx.y*Nx + threadIdx.x + (blockIdx.x*blockDim.x)] = res[threadIdx.x];
  }
  
  return;

}

/***
 * Normalize the momentum density tensor to set constant mass density
 ***/
void
normalize(int Nx, int Ny, int Nl, double *F, double *rho, double rho0)
{
  //size_t N = Ny*Nx*Nl;
  for (int j=0; j<Ny; j++) {
    for(int i=0; i<Nx; i++) {

      int idx = j*Nx*Nl + i*Nl;
      for(int l=0; l<Nl; l++) {
        F[idx + l] *= rho0 / rho[j*Nx + i];
      }
    }
  }
  return;
}

/***
 * Apply a step of the drift process F[new] = F[x - v*dt]
 ***/
void
drift(int Nx, int Ny, int Nl, double *F, double *Fnew, int *cxs, int *cys) 
{
  int idx;
  //int N = Ny*Nx*Nl; // Total size of F

  /* Apply drift by permuting the momentum density elements */         
  for(int y=0; y<Ny; y++) {
    for(int x=0; x<Nx; x++) {

      idx = (Nx*Nl)*y + Nl*x; // global index
      
      for(int l=0; l<Nl; l++) {
        //int idx_loc = cys[l] * Nx*Nl + cxs[l]*Nl; //...=F[(idx - idx_loc + l + N)%N]
        int idx_new = Nx*Nl*((y-cys[l]+Ny)%Ny) + Nl*((x-cxs[l]+Nx)%Nx);
        Fnew[idx + l] = F[idx_new + l]; // There must be a more elegant way
      }
    }
  }
}

/***
 * Apply a step of the drift process on the GPU
 ***/
__global__ void
gpu_drift(int Nx, int Ny, int Nl, double *F, double *Fnew, int *cxs, int *cys)
{
  /* "... the thread ID of a thread of index (x, y, z)
   is (x + y Dx + z Dx Dy)."                 (z, x, y)
   so we want threadIdx.x -> l
  
   */
  int y = blockIdx.z * blockDim.z + threadIdx.z; //slowest index
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  int l = threadIdx.x;

  int idx = (Nx*Nl)*y + Nl*x; // Global index
  int idx_new = Nx*Nl*((y-cys[l]+Ny)%Ny) + Nl*((x-cxs[l]+Nx)%Nx);

  Fnew[idx + l] = F[idx_new + l]; // Do the drift
  return;
}

/***
 * Defines the cylinder -- returns 1 inside the cylinder
 ***/
int
cylinder(int Nx, int Ny, int x, int y)
{
  return ((x - Nx/4)*(x - Nx/4) + (y - Ny/2)*(y - Ny/2) < (Ny*Ny)/16);
}

/***
 * Defines the cylinder on the GPU
 ***/
__device__ int
gpu_cylinder(int Nx, int Ny, int x, int y)
{
  return ((x - Nx/4)*(x - Nx/4) + (y - Ny/2)*(y - Ny/2) < (Ny*Ny)/16);
}

/***
 * Returns the bounce direction 
 ***/
int
bounce(int n)
{
  int flag0 = (n!=0);
  int flag = (n>4);
  return flag0*(n + 4 + flag)%9;
}

/***
 * Returns the bounce direction on the GPU
 ***/
__device__ int
gpu_bounce(int n)
{
  int flag0 = (n!=0);
  int flag = (n>4);
  return flag0*(n + 4 + flag)%9;
}

/***
 * Bounces the velocities at the cylinder
 ***/
void
applyBoundary(int Nx, int Ny, int Nl, double *F, double *Fnew)
{
  // Could also call (args, ... void (*boundary)() ) and give cylinder

  for (int y=0; y<Ny; y++) { // Loop over space
    for (int x=0; x<Nx; x++) {
      int idx = y*Nl*Nx + x*Nl; // Get the spatial index
      for (int l=0; l<Nl; l++) {
        if (cylinder(Nx, Ny, x, y)) {
        Fnew[idx + l] = F[idx + bounce(l)]; // bounce the velocities
        }
        else {
          Fnew[idx + l] = F[idx + l];
        }
      }
    }
  }
  return;
}

/***
 * Bounces the velocities at the cylinder
 ***/
__global__ void
gpu_applyBoundary(int Nx, int Ny, int Nl, double *F, double *Fnew)
{
  int y = blockIdx.z * blockDim.z + threadIdx.z; //slowest index
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  int l = threadIdx.x;

  int idx = y*Nl*Nx + x*Nl; // Get the spatial index
  if (gpu_cylinder(Nx, Ny, x, y)) {
    Fnew[idx + l] = F[idx + gpu_bounce(l)]; // bounce the velocities
  }

  else {
    Fnew[idx + l] = F[idx + l];
  }
  return;
}

/***
 * Gets mass density and velocities from F
 ***/
void
getFluidVariables(double *rho, double *ux, double *uy,
                  int Nx, int Ny, int Nl,
                   double *F, int *cxs, int *cys)
{
  // Get the mass density
  get_rho(Nx, Ny, Nl, F, rho);

  // Get velocities -> turn into a function?
  for (int y=0; y<Ny; y++) {
    for (int x=0; x<Nx; x++) {

      //ux  = np.sum(F*cxs,2) / rho
      //uy  = np.sum(F*cys,2) / rho
      int idxF = y*Nx*Nl + x*Nl; // Indexes in arrays
      int idxU = y*Nx + x;
      ux[idxU] = 0.; // Initialise velocities
      uy[idxU] = 0.;

      for (int l=0; l<Nl; l++) {
        ux[idxU] += F[idxF + l]*cxs[l]; // Take vector sum of momenta
        uy[idxU] += F[idxF + l]*cys[l];
      }

      ux[idxU] /= rho[idxU]; // Normalise momentum by mass density
      uy[idxU] /= rho[idxU];
    }
  }

  return;
}

/***
 * Get a velocity on the GPU
 ***/
__global__ void
gpu_getVelocity(double *ui,
                       int Nx, int Ny, int Nl,
                       double *F, double *rho, int *cis)
{ 
  // Call as <<32*n, 1, 9>>
  __shared__ double res[96];
  __shared__ double A[96*9];
  
  int idx_loc = threadIdx.x + threadIdx.z*blockDim.x;
  int idx_glob = blockIdx.y * Nx*Nl + idx_loc + (blockIdx.x*blockDim.x)*Nl;

  A[idx_loc] = F[idx_glob];

  if (threadIdx.z == 0) {
    res[threadIdx.x] = 0;
  }

  __syncthreads();
  if (threadIdx.z == 0) {
    for(int k=0; k<Nl; k++) {
      res[threadIdx.x] += A[threadIdx.x * Nl + k]*cis[k];
    }
  int idx_U = blockIdx.y*Nx + threadIdx.x + (blockIdx.x*blockDim.x);
  ui[idx_U] = res[threadIdx.x]/rho[idx_U];
  }
  
  return;
}


/***
 * Apply the collision operator
 ***/
void
applyCollisionOperator(int Nx, int Ny, int Nl,
                        double *F, double tau,
                        double *rho, double *ux, double *uy,
                        int *cxs, int *cys, double *ws) 
{

  //Feq[:,:,i] = rho*w* (1 + 3*(cx*ux+cy*uy) + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2)
 for (int y=0; y<Ny; y++) {
    for (int x=0; x<Nx; x++) {

      int idxF = y*Nx*Nl + x*Nl; // Indexes in arrays
      int idxU = y*Nx + x;

      double ux_ = ux[idxU]; // Get lattice variables
      double uy_ = uy[idxU];
      double u_sq_term = (3/2.)*(ux_*ux_ + uy_*uy_);
      double rho_ = rho[idxU];

      for (int l=0; l<Nl; l++) {
        double u = cxs[l]*ux_ + cys[l]*uy_;
        u = 3.*u +(9/2.)*u*u;
        double Feq = rho_ * ws[l] * (1. + u - u_sq_term);
        F[idxF + l] += - (1.0/tau) * (F[idxF + l] - Feq);
      }
    }
  }
  return;
}

/***
 * Apply the collision operator
 ***/
__global__ void
gpu_applyCollisionOperator(int Nx, int Ny, int Nl,
                        double *F, double tau,
                        double *rho, double *ux, double *uy,
                        int *cxs, int *cys, double *ws)
{
  // Call as <<32*n, 1, 9>>
  int idx_glob = blockIdx.y*Nx*Nl + (blockIdx.x*blockDim.x + threadIdx.x)*Nl + threadIdx.z;
  int idx_U = blockIdx.y*Nx + threadIdx.x + (blockIdx.x*blockDim.x);

  __shared__ double cxs_[9];
  __shared__ double cys_[9];
  __shared__ double ws_[9];
  if (threadIdx.z == 0) {
    cxs_[threadIdx.x%Nl] = cxs[threadIdx.x%Nl];
    cys_[threadIdx.x%Nl] = cys[threadIdx.x%Nl];
    ws_[threadIdx.x%Nl] = ws[threadIdx.x%Nl];
  }
  __syncthreads();
  

  double ux_ = ux[idx_U];
  double uy_ = uy[idx_U];
  double rho_ = rho[idx_U];

  int l = threadIdx.z;
  double u_sq_term = (3/2.)*(ux_*ux_ + uy_*uy_);

  double u = cxs_[l]*ux_ + cys_[l]*uy_;
  u = 3.*u +(9/2.)*u*u;
  double Feq = rho_ * ws_[l] * (1. + u - u_sq_term);

  // F[idxF + l] += - (1.0/tau) * (F[idxF + l] - Feq);
  F[idx_glob] += - (1.0/tau) * (F[idx_glob] - Feq);
  return;
}

/*
 * Compares two arrays of length N
*/
void
compare_arrays(int N, double *F, double *Fnew)
{
    double diff = 0;
    double norm = 0;
    for(int i=0; i<N; i++)
    {
      double d = F[i]-Fnew[i];
      diff += d*d;
      norm += F[i]*F[i];
    }
    printf(" Diff = %e\n", diff/norm);
    return;
}

/***
 * MAIN
 ***/
int
main(void) 
{
    /* Problem parameters */
    int Nx = 96*5, Ny = 64*6, Nl = 9; // Grid and channel size
    double rho0 = 100; // Average density
    double tau = 0.8;

    int Nt = 4000; // Number of time steps

    int cxs[9] = {0, 0, 1, 1, 1, 0,-1,-1,-1}; // Velocity directions
    int cys[9] = {0, 1, 1, 0,-1,-1,-1, 0, 1};

    double weights[9] = {4./9., 1./9., 1./36., // Collision Operator weights
                         1./9., 1./36., 1./9.,
                         1./36., 1./9., 1./36.};

    int *d_cxs = (int *)gpu_alloc(Nl * sizeof(int));
    int *d_cys = (int *)gpu_alloc(Nl * sizeof(int));
    double *d_weights = (double *)gpu_alloc(Nl * sizeof(double));
    cudaMemcpy(d_cxs, cxs, sizeof(int)*Nl, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cys, cys, sizeof(int)*Nl, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, sizeof(double)*Nl, cudaMemcpyHostToDevice);
    
    /* Initialise momentum density tensor */
    double *F = (double*)ualloc(Ny * Nx * Nl * sizeof(double)); // Momentum density
    /* Initialise the GPU tensors and allocate memory */
    double *d_F = (double *)gpu_alloc(Ny * Nx * Nl * sizeof(double));
    double *d_Fnew = (double *)gpu_alloc(Ny * Nx * Nl * sizeof(double));

    /* Velocity tensors */
    double *d_ux = (double *)gpu_alloc(Ny * Nx *sizeof(double));// Velocities
    double *d_uy = (double *)gpu_alloc(Ny * Nx *sizeof(double));

    /* Momentum Density Tensor initialisation */
    init_momentum_density(Nx, Ny, Nl, F); // Initial Condition for F

    /* Initialise mass density */
    double *rho = (double*)ualloc(Ny*Nx*sizeof(double)); // Mass density
    double *d_rho = (double *)gpu_alloc(Ny * Nx *sizeof(double));
    get_rho(Nx, Ny, Nl, F, rho);
    normalize(Nx, Ny, Nl, F, rho, rho0); // Normalize F to set constant mass density

    cudaMemcpy(d_F, F, sizeof(double)*Nx*Ny*Nl, cudaMemcpyHostToDevice); // Copy F to the gpu

    /* RUN ON THE GPU:
    *   GPU
    */
    int nx = 8;
    int ny = 4;
    dim3 thrds(9, nx, ny);// x->l, y->x, z->y to access contiguous elements
    dim3 blcks(1, Nx/nx, Ny/ny);

    int nx_ = 96;
    dim3 thrds_(96, 1, 9);
    dim3 blcks_(Nx/nx_, Ny, 1);

    double t1 = stop_watch(0);
    for(int t=0; t<Nt; t++) {
      // drift and apply boundary
      gpu_drift<<<blcks, thrds>>>(Nx, Ny, Nl, d_F, d_Fnew, d_cxs, d_cys);
      gpu_applyBoundary<<<blcks, thrds>>>(Nx, Ny, Nl, d_Fnew, d_F);
      // gpu_GetFluidVariables
      gpu_get_rho<<<blcks_, thrds_>>>(Nx, Ny, Nl, d_F, d_rho);
      gpu_getVelocity<<<blcks_, thrds_>>>(d_uy, Nx, Ny, Nl, d_F, d_rho, d_cys);
      gpu_getVelocity<<<blcks_, thrds_>>>(d_ux, Nx, Ny, Nl, d_F, d_rho, d_cxs);
      // Apply the collision operator
      gpu_applyCollisionOperator<<<blcks_, thrds_>>>(Nx, Ny, Nl, d_F, tau, d_rho, d_ux, d_uy, // Apply the collision operator
                      d_cxs, d_cys, d_weights);
    }
    t1 = stop_watch(t1);


    cudaMemcpy(F, d_F, sizeof(double)*Nx*Ny*Nl, cudaMemcpyDeviceToHost);
    printf("Running on the GPU complete after %6.4lf sec\n", t1);

    // Print final momentum density tensor
    FILE *f = fopen("F.bin", "wb");
    fwrite(F, sizeof(double), Nx*Ny*Nl, f);
    fclose(f);

    free(F);
    free(rho);
    return 0;

}
