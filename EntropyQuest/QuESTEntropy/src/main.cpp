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
#include <nlohmann/json.hpp>
#include <fstream>
#include <chrono>
using namespace std::chrono;

extern "C"
{
  #include "../include/mt19937ar.h"
}

using json = nlohmann::ordered_json;
using namespace std;
#define MAX_LINE_LENGTH 256
#define PI 3.14159265358979323846
typedef std::complex<double> C;

// function to print density matrix
void print_densmat(int &qubits, Qureg &density_matrix_qreg)
{
  //Print dens matrix for check
  for (int i = 0; i < pow(2, qubits); i++)
  {
    for (int j = 0; j < pow(2, qubits); j++)
    {
      Complex amplitude = getDensityAmp(density_matrix_qreg, i, j);
      printf("%+.8f%+.8fi ", amplitude.real, amplitude.imag);
    }
    printf("\n");
  }
}

//functio to print to the terminal 
void print_to_terminal(vector<double>&vNd, vector<double>&pur, vector<double>&R2d)
{
  // PRINT TO TERMINAL
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
}

void append_to_json(const std::string& filename, double& vNd,  double& pur, double& R2d) 
{
    json j;

    // Read if file already exists
    if (std::filesystem::exists(filename)) {
        std::ifstream in(filename);
        if (in.is_open())
            in >> j;
    }

    // Append your data
    j["all_vNd_diff_n"].push_back(vNd);
    j["all_pur_diff_n"].push_back(pur);
    j["all_R2d_diff_n"].push_back(R2d);

    // Write back
    std::ofstream out(filename);
    if (out.is_open())
        out << std::setw(4) << j << std::endl;
}

void append_to_json_duration(const std::string& filename, auto &duration)
{
  json j;

    // Read if file already exists
    if (std::filesystem::exists(filename)) {
        std::ifstream in(filename);
        if (in.is_open())
            in >> j;
    }

    // Append your data
    j["duration"] = duration;

    // Write back
    std::ofstream out(filename);
    if (out.is_open())
        out << std::setw(4) << j << std::endl;

}

int main()
{
  // set seed
  init_genrand(837);  // Equivalent to np.random.seed(837)

  // set detph 
  int depth_min = 0;
  int depth_max = 15;

  // set qubits
  int qubits_max = 16;

  for (int qubits = 10; qubits <= qubits_max; qubits ++)
  {
    // time round
    auto start = high_resolution_clock::now(); 
    // open file to append to
    string filename = "../../QuESTEntropy/DensityMatrices_metrics/Q" + std::to_string(qubits) + "test_D15.json";

    // loop for different qubit sizes
    int dim = pow(2, qubits);
    // angles
    int angles_per_layer = 2*qubits;
    int total_angles = angles_per_layer*depth_max;
    double angles_array[total_angles];

    // data gathering
    // vector<double> vNd;
    // vector<double> pur;
    // vector<double> R2d;

    // get angle values
    for (int i = 0; i < total_angles; i++) {
      double r = 2*PI*genrand_res53();
      //printf("Random[%d] = %.17f\n", i, r);
      angles_array[i] = r;
    }

    cout<< "Qubits: "<<qubits<< " Depth: " <<depth_max <<endl;
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
    for (int i = 0; i <= depth_max; i++)
    {
      // time each depth
      //auto start = high_resolution_clock::now(); 
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

      // compute metrics
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
      
      // for each layer
      // Von neumann entropy
      // vNd.push_back(abs(sum_entropy/double(qubits)));
      // // Purity - the trace of density_matrix^2
      // pur.push_back((eing_mat*eing_mat).trace().real());
      // // R2 entropy
      // R2d.push_back(-1 * log2(pur[i]) / qubits);

      double vNd = abs(sum_entropy/double(qubits));
      double pur = (eing_mat*eing_mat).trace().real();
      double R2d = -1 * log2(pur) / qubits;

      append_to_json(filename, vNd, pur, R2d);

    }
    auto stop = high_resolution_clock::now();
	  auto duration = duration_cast<nanoseconds>(stop - start).count();
    // SAVE to json file
    // string filename = "../../QuESTEntropy/DensityMatrices_metrics/Q" + std::to_string(qubits) + "_D15.json";
    // ofstream file(filename);
    // json j;
    // j["all_vNd_diff_n"] = {vNd};
    // j["all_pur_diff_n"] = {pur};
    // j["all_R2d_diff_n"] = {R2d};
    // j["duration"] = duration;

    // if (file.is_open())
    // {
    //   file<<std::setw(4) << j; 
    // }
    append_to_json_duration(filename, duration);
    
    //file.close();
    //destroy Quest register and environment
    printf("Destroy Qreg\n");
    destroyQureg(density_matrix_qreg, env);
    printf("Destroy Environment\n");
    destroyQuESTEnv(env);

    //printf("%d\n ", angl_pos);
  }

  return 0;
}
