#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

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
 * Return a random number in [0, 1)
 ***/
double
urand(void)
{
  double x = (double)rand()/(double)RAND_MAX;
  return x;
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
 * Defines the cylinder -- returns 1 inside the cylinder
 ***/
int
cylinder(int Nx, int Ny, int x, int y)
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
 * Gets mass density and velocities from F
 ***/
void
getFluidVariables(double *rho, double *ux, double *uy,
                  int Nx, int Ny, int Nl,
                   double *F, int *cxs, int *cys) {

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
 * Apply the collision operator
 ***/
void
applyCollisionOperator(int Nx, int Ny, int Nl,
                        double *F, double tau,
                        double *rho, double *ux, double *uy,
                        int *cxs, int *cys, double *ws) {

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
        float Feq = rho_ * ws[l] * (1. + u - u_sq_term);
        F[idxF + l] += - (1.0/tau) * (F[idxF + l] - Feq);
      }
    }
  }
  return;
}

/***
 * MAIN
 ***/
int
main(void) 
{
    /* Problem parameters */
    int Nx = 96, Ny = 64, Nl = 9; // Grid and channel size
    double rho0 = 100; // Average density
    double tau = 0.6;

    int Nt = 100; // Number of time steps

    int cxs[9] = {0, 0, 1, 1, 1, 0,-1,-1,-1}; // Velocity directions
    int cys[9] = {0, 1, 1, 0,-1,-1,-1, 0, 1};

    double weights[9] = {4./9., 1./9., 1./36., // Collision Operator weights
                         1./9., 1./36., 1./9.,
                         1./36., 1./9., 1./36.};

    /* Initialise momentum density tensor */
    double *F = ualloc(Ny * Nx * Nl * sizeof(double)); // Momentum density
    double *Fnew = ualloc(Ny * Nx * Nl * sizeof(double));

    double *ux = ualloc(Ny*Nx*sizeof(double)); // Velocities
    double *uy = ualloc(Ny*Nx*sizeof(double));

    init_momentum_density(Nx, Ny, Nl, F); // Initial Condition for F

    /* Initialise mass density */
    double *rho = ualloc(Ny*Nx*sizeof(double)); // Mass density
    get_rho(Nx, Ny, Nl, F, rho);
    normalize(Nx, Ny, Nl, F, rho, rho0); // Normalize F to set constant mass density

    /* Evolve for Nt steps */
    for(size_t t=0; t<Nt; t++)
    {
      drift(Nx, Ny, Nl, F, Fnew, &cxs[0], &cys[0]); // Apply drift step
      //memcpy(F, Fnew, sizeof(double)*Ny*Nx*Nl); // Copy state to F
      applyBoundary(Nx, Ny, Nl, Fnew, F); // Apply the collisions with boundaries
      getFluidVariables(rho, ux, uy, Nx, Ny, Nl, F, &cxs[0], &cys[0]); // Get rho and velocities
      applyCollisionOperator(Nx, Ny, Nl, F, tau, rho, ux, uy, // Apply the collision operator
                             &cxs[0], &cys[0], &weights[0]);
    }

    // Print final momentum density tensor
    for(int i=0; i<Nx*Ny*Nl; i++)
    {
        printf("%lf\n", F[i]);
    }
    
    return 0;

}