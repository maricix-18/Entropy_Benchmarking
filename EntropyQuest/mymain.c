
#include "QuEST.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <complex.h>  

// to run main 
//C:\Users\maria\Desktop\Entropy_Benchmarking\EntropyQuest\QuEST\build>cmake .. -G "MinGW Makefiles" -DUSER_SOURCE=../../mymain.c -DOUTPUT_EXE=mymain
//make

#define MAX_LINE_LENGTH 256

int main()
{
  printf("inside main!\n");

  int depth = 15;
  for (int i=0; i<=depth;i++)
  {
    // for each deoth size read the .txt file
    char file_dir[MAX_LINE_LENGTH];// = "C:/Users/maria/Desktop/Entropy_Benchmarking/Qasm_Q5_D15/Qasm_qc_QC_D";
    //char extension[] = ".txt";
    //char filename[MAX_LINE_LENGTH];
    snprintf(file_dir, sizeof(file_dir), "C:/Users/maria/Desktop/Entropy_Benchmarking/Qasm_Q5_D15/Qasm_qc_Q5_D%d.txt",i);
    printf(file_dir);
    printf("\n");
    FILE* file = fopen(file_dir, "r");
    if (file==NULL) {
      printf("Failed to open file to read from");
      return 1;
    }

    //load QuEST
    printf("Create Qenv\n");
    QuESTEnv env = createQuESTEnv();
    //setNumThreads(1);
    int n_qubits = 0;
    Qureg dens_qreg;
    char line[MAX_LINE_LENGTH];
    int num_qubits = 0;

    // Read the file line by line to create the circuit
    while (fgets(line, sizeof(line), file)) 
      {
        // Check for the third line where the qreg is defined
        if (strstr(line, "qreg q[")) {
            // Extract number of qubits from the qreg line
            char *start = strchr(line, '[');
            if (start) {
                num_qubits = atoi(start + 1); // Convert the number between '[' and ']'
            }
            printf("Number of qubits: %d\n", num_qubits);
            printf("Create DensityQureg\n");
            dens_qreg = createDensityQureg(num_qubits, env);
            printf("Init zero DensityQureg\n");
            initZeroState(dens_qreg);
          }
        // Look for lines with rx gates
        else if (strstr(line, "rx(")) {
            double angle;
            int qubit;
            // Parse the rx gate line
            if (sscanf(line, "rx(%lf) q[%d];", &angle, &qubit) == 2) {
                //printf("Found rx gate: Angle = %lf, Applied to q[%d]\n", angle, qubit);
                rotateX(dens_qreg,(num_qubits - 1 - qubit),angle);
              }
          }
        // Look for lines with ry gates
        else if (strstr(line, "ry(")) {
            double angle;
            int qubit;
            // Parse the rx gate line
            if (sscanf(line, "ry(%lf) q[%d];", &angle, &qubit) == 2) {
                //printf("Found ry gate: Angle = %lf, Applied to q[%d]\n", angle, qubit);
                rotateY(dens_qreg, (num_qubits - 1 - qubit),angle);
              }
          }
        // Look for lines with cz gates
        else if (strstr(line, "cz")) {
            //double angle;
            int qubit_control;
            int qubit_target;
            // Parse the rx gate line
            if (sscanf(line, "cz q[%d],q[%d];", &qubit_control, &qubit_target) == 2) {
                //printf("Found cz gate: Applied to control q[%d], and target q[%d]\n", qubit_control, qubit_target);
                //controlled Z - phase gate with PI
                controlledPhaseFlip(dens_qreg, num_qubits-1 -qubit_control, num_qubits-1-qubit_target);
              }
          }
      }
    // save the density matrices of the current QC with the given Deoth
    int dim = 1 << num_qubits; 
    Complex matrix [dim][dim];
    malloc(dim * sizeof(Complex));
    // save file
    char file_dir_tosave[MAX_LINE_LENGTH];// = "C:/Users/maria/Desktop/Entropy_Benchmarking/Qasm_Q5_D15_results_Quest/Qasm_qc_QC_D";
    //char extension_tosave[] = ".csv";
    //char filename_tosave[MAX_LINE_LENGTH];
    snprintf(file_dir_tosave, sizeof(file_dir_tosave), "C:/Users/maria/Desktop/Entropy_Benchmarking/Qasm_Q5_D15_results_Quest/Qasm_qc_Q5_D%d.csv",i);
    FILE* file_tosave = fopen(file_dir_tosave, "a");  
    if (!file_tosave) {
      perror("Error opening file to output data");
      return 0;
    }
    // Get density matrix
    printf("Density matrix:\n");
    printf("dim: %d", dim);
    for (int i=0; i< dim;i++)
    {
      for (int j=0; j< dim;j++)
        {    
          Complex amplitude = getDensityAmp(dens_qreg,i,j);
          matrix[i][j] = amplitude;
          fprintf(file_tosave, "%+.8f%+.8fi", matrix[i][j].real, matrix[i][j].imag);
          if (j < dim - 1) 
          {
            fprintf(file_tosave, ", ");
          }
        }
      fprintf(file_tosave,"\n");
    }
    //close saving file
    fclose(file_tosave);
    // free memory
    free(matrix);
    
    // close file 
    fclose(file);
    //destroy Quest env and register
    printf("Destroy Qreg\n");
    destroyQureg(dens_qreg, env); 
    printf("Destroy Env\n");
    destroyQuESTEnv(env);   
  }

  //data collection ended
  printf("Data collection eneded.\n");
  return 0;

}
