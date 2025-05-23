#ifndef QREG_H
#define QREG_H

#include "QuEST.h"

void densmat_qureg(Qureg *qureg, QuESTEnv *env, int *num_qubits, char *file_dir);

int swap_Endians(int value);

void save_densmat(Qureg *qureg, int* num_qubit, char* file_dir_tosave);

#endif