#include "QuEST.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <inttypes.h>
#include "../include/qreg.h"

#define MAX_LINE_LENGTH 256

/// @brief Create density matrix from given qasm file
/// @param qureg register for density matrix
/// @param env env
/// @param num_qubits size of register
/// @param file_dir path to file to be read from
/// @return qureg
void densmat_qureg(Qureg *qureg, QuESTEnv *env, int *num_qubits, char *file_dir)
{
    printf("In function file %s\n",file_dir);
    printf("\n");
    FILE* file = fopen(file_dir, "r");
    if (file==NULL) {
      printf("Failed to open file to read from");
    }
    char line[MAX_LINE_LENGTH];

    // Read the file line by line to create the circuit
    while (fgets(line, sizeof(line), file)) 
    {
        // Check for the third line where the qreg is defined
        if (strstr(line, "qreg q[")) {
            // Extract number of qubits from the qreg line
            char *start = strchr(line, '[');
            if (start) {
                *num_qubits = atoi(start + 1); // Convert the number between '[' and ']'
            }
            printf("Number of qubits inside function: %d\n", *num_qubits);
            printf("Create DensityQureg\n");
            *qureg = createDensityQureg(*num_qubits, *env);
            printf("Init zero DensityQureg\n");
            initZeroState(*qureg);
          }
        // Look for lines with rx gates
        else if (strstr(line, "rx(")) {
            double angle;
            int qubit;
            if (sscanf(line, "rx(%lf) q[%d];", &angle, &qubit) == 2) {
                //printf("Found rx gate: Angle = %lf, Applied to q[%d]\n", angle, qubit);
                rotateX(*qureg,(*num_qubits - 1 - qubit),angle);
                // after every gate apply depolarins local noise
                float p1 = 0.008;
                p1 = (3 * p1 / 4.0);
                mixDepolarising(*qureg, (*num_qubits - 1 - qubit), p1);
              }
          }
        // Look for lines with ry gates
        else if (strstr(line, "ry(")) {
            double angle;
            int qubit;
            if (sscanf(line, "ry(%lf) q[%d];", &angle, &qubit) == 2) {
                //printf("Found ry gate: Angle = %lf, Applied to q[%d]\n", angle, qubit);
                rotateY(*qureg, (*num_qubits - 1 - qubit),angle);
                float p1 = 0.008;
                p1 = (3 * p1 / 4.0);
                mixDepolarising(*qureg, (*num_qubits - 1 - qubit), p1);
              }
          }
        // Look for lines with cz gates
        else if (strstr(line, "cz")) {
            int qubit_control;
            int qubit_target;
            if (sscanf(line, "cz q[%d],q[%d];", &qubit_control, &qubit_target) == 2) {
                //printf("Found cz gate: Applied to control q[%d], and target q[%d]\n", qubit_control, qubit_target);
                controlledPhaseFlip(*qureg, *num_qubits-1 -qubit_control, *num_qubits-1-qubit_target);
                float p2 = 0.054;
                p2 = (15 * p2 / 16.0);
                mixTwoQubitDepolarising(*qureg,(*num_qubits - 1 - qubit_control), (*num_qubits - 1 - qubit_target),p2);
              }
          }
      }
    fclose(file);
}

/// @brief save density matrix for plotting
/// @param qureg register for density matrix
/// @param num_qubit size of register
/// @param file_dir_tosave path to file to save density matrix
void save_densmat(Qureg *qureg, int* num_qubit, char* file_dir_tosave)
{
  // Get density matrix
  int dim = pow(2, *num_qubit);
  // allocate space for matrix dim*dim of type Complex
  Complex *matrix = (Complex *)malloc(dim * dim * sizeof(Complex));
  printf("Density matrix:\n");
  printf("dimension: %d\n", dim);

  FILE* file = fopen(file_dir_tosave, "a");
  if (!file) {
    printf("Failed to open file to append to\n");
  }
  
  for (int i = 0; i < dim; i++)
  {
    for (int j = 0; j < dim; j++)
      {
        int offset = i * dim + j;
        Complex amplitude = getDensityAmp(*qureg, i, j);
        matrix[offset] = amplitude;
        fprintf(file, "%+.8f%+.8fi", matrix[offset].real, matrix[offset].imag);
        //printf("%+.8f%+.8fi", matrix[offset].real, matrix[offset].imag);
        if (j < dim - 1)
        {
          fprintf(file, ", ");
          //printf(", ");
        }
      }
       //printf("\n ");
    fprintf(file,"\n");
  }
  //close saving file
  fclose(file);

  // free memory
  free(matrix);
}