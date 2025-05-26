#include "../../QuEST/QuEST/include/QuEST.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <complex.h>
#include "../include/qreg.h"
#include "../include/mt19937ar.h"

#define MAX_LINE_LENGTH 256
#define PI 3.14159265358979323846

int main()
{
  // READING from QASM file
  printf("inside main! Test random numb generator: \n");

  init_genrand(837);  // Equivalent to np.random.seed(837)

  for (int i = 0; i < 15; i++) {
    double r = 2*PI*genrand_res53();  // [0, 1], inclusive
    printf("Random[%d] = %.17f\n", i, r);
    
  }

  // READ QASM FILE
  // int depth = 15;
  // for (int i = 0; i <= depth; i++)
  // {
  //   char dir[MAX_LINE_LENGTH] = "C:/Users/maria/Desktop/Entropy_Benchmarking/Qasm_Q5_D15_DensityMatrix/Qasm_qc_Q5_D";
  //   char d_char[20];
  //   sprintf(d_char, "%d", i);;
  //   char extension[] = ".txt";
  //   // set up file to be opened
  //   strcat(dir, d_char);
  //   strcat(dir, extension);
  //   printf("File: %s\n", dir);

  //   printf("Create Q Environment\n");
  //   QuESTEnv env = createQuESTEnv();
  //   int num_qubits = 0;
  //   printf("Init Q Register\n");
  //   Qureg density_matrix_qreg;

  //   densmat_qureg(&density_matrix_qreg, &env, &num_qubits, dir);

    //Save dens mat
  //   char dir_tosave[MAX_LINE_LENGTH] = "C:/Users/maria/Desktop/Entropy_Benchmarking/Entropy_Benchmarking/Entropy_Benchmarking/Code/Quest_Q5_D15_DensityMatrix_NoiseModel/DepolLevelFixed/Data/DensMat_qc_Q5_D";
  //   char extension_tosave[] = ".csv";
  //   char file_dir[MAX_LINE_LENGTH];
  //   //set up file to be opened
  //   strcat(dir_tosave, d_char);
  //   strcat(dir_tosave, extension);
  //   save_densmat(&density_matrix_qreg, &num_qubits, dir_tosave);
  //   printf("Number of qubits in main: %d\n", num_qubits);

  //   //destroy Quest register and environment
  //   printf("Destroy Qreg\n");
  //   destroyQureg(density_matrix_qreg, env);
  //   printf("Destroy Environment\n");
  //   destroyQuESTEnv(env);
  // }

  return 0;
}