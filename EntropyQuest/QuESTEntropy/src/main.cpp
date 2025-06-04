#include "../../QuEST/QuEST/include/QuEST.h"
#include <stdio.h>
#include <iostream>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <complex>
#include <cmath>
#include "../eigen-3.4.0/Eigen/Eigenvalues"
#include "../include/qreg.h"
#include <iomanip>

extern "C"
{
  #include "../include/mt19937ar.h"
}

using namespace std;
#define MAX_LINE_LENGTH 256
#define PI 3.14159265358979323846
typedef std::complex<double> C;

int main()
{
  // set seed
  init_genrand(837);  // Equivalent to np.random.seed(837)

  // set detph 
  int depth_min = 0;
  int depth_max = 15;

  // set qubits
  int qubits = 8;
  int dim = pow(2, qubits);

  // angles
  int angles_per_layer = 2*qubits;
  int total_angles = angles_per_layer*depth_max;
  double angles_array[total_angles];

  // data gathering
  double vNd[depth_max];
  double pur[depth_max];
  double R2d[depth_max];

  // get angle values
  for (int i = 0; i < total_angles; i++) {
    double r = 2*PI*genrand_res53();
    //printf("Random[%d] = %.17f\n", i, r);
    angles_array[i] = r;
  }

  // create env and dens mat qreg
  printf("Create Q Environment\n");
  QuESTEnv env = createQuESTEnv();
  printf("Create Q Register\n");
  Qureg density_matrix_qreg = createDensityQureg(qubits, env);

  // Depolarising noise 1 q gate
  float p1 = 0.008;
  p1 = (3 * p1 / 4.0); // scale for quest function

  // Depolarising noise 2q gate
  float p2 = 0.054;
  p2 = (15 * p2 / 16.0); // scale for quest function

  // populate circuit
  int angl_pos = 0;
  for (int i = 0; i < depth_max; i++)
  {
    printf("layer %d \n", i);
    // for each layer add x rotation on all qubits
    for (int j = 0; j < qubits; j++)
    {
      //printf("rx gate: Angle = %.15lf, Applied to q[%d]\n", angles_array[angl_pos], j);
      rotateX(density_matrix_qreg, j, angles_array[angl_pos]);
      mixDepolarising(density_matrix_qreg, j, p1);
      angl_pos++;
    }

    // for each layer add y rotation on all qubits
    for (int j = 0; j < qubits; j++)
    {
      //printf("ry gate: Angle = %.15lf, Applied to q[%d]\n", angles_array[angl_pos], j);
      rotateY(density_matrix_qreg, j, angles_array[angl_pos]);
      mixDepolarising(density_matrix_qreg, j, p1);
      angl_pos++;
    }

    // for each layer add 2x cz layer on nearest neighbour qubits
    for (int g = 0; g < 2; g++)
    {
      for (int j = g; j < qubits-1; j+=2)
      {
        //printf("Cz gate: Applied to control q[%d], and target q[%d]\n", j, j+1);
        controlledPhaseFlip(density_matrix_qreg, j, j+1);
        mixTwoQubitDepolarising(density_matrix_qreg, j, j+1,p2);
      }
    }

    // get matrix to use Eingensolver library
    Eigen::MatrixXcd eing_mat(dim, dim);
    for (int i = 0; i < dim; i++)
    {
      for (int j = 0; j < dim; j++)
      {
        Complex amplitude = getDensityAmp(density_matrix_qreg, i, j);
        eing_mat(i,j) = C(amplitude.real, amplitude.imag);
        //eing_mat(i,j) = C(real, imag);
        //printf("%+.8f%+.8fi ", amplitude.real, amplitude.imag);
        // cout << eing_mat(i,j).real();
      }
      //printf("\n");
    }
    // for each layer, calculate
    // get eingenvalues
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces;
    ces.compute(eing_mat);
    //cout<< "Eingenvalues of matrix are: " << ces.eigenvalues()(0).real();
    complex<double> sum_entropy = 0 ;
    complex<double> sum_eingvals = 0;
    for (int e = 0; e < dim; e++)
    {
      //calculate ln
      complex<double> log_z = log(abs(ces.eigenvalues()(e))) + 1i * arg(ces.eigenvalues()(e));
      complex<double> log2_z = log_z / log(2.0);
      sum_entropy += (-ces.eigenvalues()(e) * log2_z); 
      sum_eingvals += ces.eigenvalues()(e);
    }

    //cout << "sum entropy: " << sum_entropy << "\n";
    //cout << "metric: " << abs(sum_entropy/double(qubits)) <<"\n";

    // Von neumann entropy
    vNd[i] = abs(sum_entropy/double(qubits));
    
    // Purity
    // the trace of density_matrix^2
    pur[i] = (eing_mat*eing_mat).trace().real();

    // R2 entropy
    R2d[i] = -1 * log2(pur[i]) / qubits;
  }
 
  // Print dens matrix for check
  // for (int i = 0; i < pow(2, qubits); i++)
  // {
  //   for (int j = 0; j < pow(2, qubits); j++)
  //   {
  //     Complex amplitude = getDensityAmp(density_matrix_qreg, i, j);
  //     printf("%+.8f%+.8fi ", amplitude.real, amplitude.imag);
  //   }
  //   printf("\n");
  // }
  cout << fixed;
  cout << setprecision(17);

  cout << "VonNeumanE \n";
  for (const auto& e : vNd) {
    cout << e << std::endl;
  }
  
  cout << "Purity \n";
  for (const auto& e : pur) {
    cout << e << std::endl;
  }

  cout << "R2d \n";
  for (const auto& e : R2d) {
    cout << e << std::endl;
  }

  //destroy Quest register and environment
  printf("Destroy Qreg\n");
  destroyQureg(density_matrix_qreg, env);
  printf("Destroy Environment\n");
  destroyQuESTEnv(env);

  printf("%d\n ", angl_pos);

  return 0;
}
