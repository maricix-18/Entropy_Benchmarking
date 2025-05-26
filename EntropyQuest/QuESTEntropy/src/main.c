#include "../../QuEST/QuEST/include/QuEST.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <complex.h>
#include "../include/qreg.h"

#define MAX_LINE_LENGTH 256

int main()
{
  // READING from QASM file
  printf("inside main!\n");
  int depth = 15;
  for (int i = 1; i <= depth; i++)
  {
    char dir[MAX_LINE_LENGTH] = "C:/Users/maria/Desktop/Entropy_Benchmarking/Qasm_Q6_D15_DensityMatrix/Qasm_qc_Q6_D";
    char d_char[20];
    sprintf(d_char, "%d", i);;
    char extension[] = ".txt";
    // set up file to be opened
    strcat(dir, d_char);
    strcat(dir, extension);
    printf("File: %s\n", dir);

    printf("Create Q Environment\n");
    QuESTEnv env = createQuESTEnv();
    int num_qubits = 0;
    printf("Init Q Register\n");
    Qureg density_matrix_qreg;

    densmat_qureg(&density_matrix_qreg, &env, &num_qubits, dir);

    // printf("Print dens mat\n");
    // printf("Depth %d\n", i);
    // printf("n qubits: %d\n", num_qubits);
    // int dim = pow(2, num_qubits);
    // printf("dim: %d\n", dim);
    // for (int r = 0; r < dim; r++)
    // {
    //   for (int c = 0; c < dim; c++)
    //     {    
    //       Complex amplitude = getDensityAmp(density_matrix_qreg, r, c);
    //       printf("%+.8f%+.8fi", amplitude.real, amplitude.imag);
    //       if (c < dim - 1) 
    //       {
    //         printf( ", ");
    //       }
    //     }
    //   printf("\n");
    // }

    //Save dens mat
    char dir_tosave[MAX_LINE_LENGTH] = "C:/Users/maria/Desktop/Entropy_Benchmarking/Entropy_Benchmarking/Entropy_Benchmarking/Code/Quest_Q6_D15_DensityMatrix_NoiseModel/Data/DensMat_qc_Q6_D";
    char extension_tosave[] = ".csv";
    char file_dir[MAX_LINE_LENGTH];
    //set up file to be opened
    strcat(dir_tosave, d_char);
    strcat(dir_tosave, extension);
    save_densmat(&density_matrix_qreg, &num_qubits, dir_tosave);
    printf("Number of qubits in main: %d\n", num_qubits);

    //destroy Quest register and environment
    printf("Destroy Qreg\n");
    destroyQureg(density_matrix_qreg, env);
    printf("Destroy Environment\n");
    destroyQuESTEnv(env);
  }

  return 0;
}