/*
    PyInteraph, a software suite to analyze interactions and interaction network in structural ensembles.
    Copyright (C) 2013 Matteo Tiberti <matteo.tiberti@gmail.com>, Gaetano Invernizzi, Yuval Inbar, 
    Matteo Lambrughi, Gideon Schreiber,  Elena Papaleo <elena.papaleo@unimib.it> <elena.papaleo@bio.ku.dk>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

inline int trmI(int row, int col) {  // array index for triangular matrix                                     
  return (row) > (col) ? ((row)+1)*(row)/2+(col) : ((col)+1)*(col)/2+(row);
}

inline int sqmI(int colsn, int row, int col) { // array index for square matrix                               
  return col*colsn + row;
}

//inline double ed(double* coords, int i, int j) {
//  return sqrt( (coords[idxi] - coords[idxj])*(coords[idxi] - coords[idxj]) + (coords[idxi+1] - coords[idxj+1])*//(coords[idxi+1] - coords[idxj+1]) + (coords[idxi+2] - coords[idxj+2])*(coords[idxi+2] - coords[idxj+2]) );
//}

inline double ed(double* coords1, double* coords2, int idxi, int idxj) {
  return sqrt( (coords1[idxi] - coords2[idxj])*(coords1[idxi] - coords2[idxj]) + (coords1[idxi+1] - coords2[idxj+1])*(coords1[idxi+1] - coords2[idxj+1]) + (coords1[idxi+2] - coords2[idxj+2])*(coords1[idxi+2] - coords2[idxj+2]) );
}


int potential_distances(double* coords, int nsets, int set_size, int nframes, double* results) {  
  int i = 0;
  int j = 0;
  int k = 0;
  int l = 0;
  int combinations[8] = {0, 2, 0, 3, 1, 2, 1, 3};
  int natoms = nsets*set_size;
  int ncoords = natoms*3;
  int this_i = 0;
  int this_ij = 0;
  
  for (i=0; i<nframes; i++) {
    this_i = i*ncoords;
    for (j=0; j<nsets; j++) {
      this_ij = this_i + j*set_size*3;
      for (k=0; k<8; k+=2) {
	//printf ("%d %d -> %d\n", this_ij/3+combinations[k], this_ij/3+combinations[k+1], l);
	//printf ("%3f %3f %3f vs %3f %3f %3f\n", coords[this_ij+combinations[k]*3], coords[this_ij+combinations[k]*3+1], coords[this_ij+combinations[k]*3+2], coords[this_ij+combinations[k+1]*3], coords[this_ij+combinations[k+1]*3+1], coords[this_ij+combinations[k+1]*3+2]);
	
	//printf("%f\n", results[l]);
	results[l] = ed(coords, coords, this_ij+combinations[k]*3, this_ij+combinations[k+1]*3);
	l++;
      }
    }
  }
  return 1;
}

int triangular_distmatrix(double* coords, int natoms, int nframes, double co, long* out_mat, double* corrections) {
 
  int i = 0;
  int j = 0;
  int k = 0;

  // Initialize output matrix
  int out_mat_elemsn = natoms*(natoms-1)/2;
  //int* out_mat = (int*) malloc(out_mat_elemsn * sizeof(int));  
  for (i=0; i<out_mat_elemsn; i++) 
    out_mat[i] = 0.0;
  
  int idx_j = 0;
  int idx_k = 0;

  for (i=0; i<nframes; i++) {
    for (j=0; j<natoms; j++) {
      idx_j = i*natoms*3 + j*3;
      for (k=0; k<j; k++) {
	idx_k = i*natoms*3 + k*3;
	//printf("%d %d: %.3f\n", j, k, ed(coords, coords, idx_j, idx_k));
	if (ed(coords, coords, idx_j, idx_k) - corrections[j] - corrections[k] <= co) {
	  out_mat[sqmI(natoms,j,k)] += 1;
	  out_mat[sqmI(natoms,k,j)] += 1;
	}
      }
    }
  }

  return 1;
}

int square_distmatrix(double* coords1, double* coords2, int natoms1, int natoms2, int nframes, double co, long* out_mat) {

  // Initialize output matrix                                                                                     
  int out_mat_elemsn = natoms1 * natoms2;
  int i = 0;
  int j = 0;
  int k = 0;
  //int* out_mat = (int*) malloc(out_mat_elemsn * sizeof(int));

  for (i=0; i<out_mat_elemsn; i++)
    out_mat[i] = 0.0;

  int idx_j = 0;
  int idx_k = 0;

  for (i=0; i<nframes; i++) {
    for (j=0; j<natoms1; j++) {
      for (k=0; k<natoms2; k++) {
	idx_j = i*natoms1 + j*3;
	idx_k = i*natoms2 + k*3;
	if (ed(coords1, coords2, idx_j, idx_k) <= co) {
          out_mat[sqmI(natoms1,j,k)] += 1;	  
	}
      }
    }
  }

  return 1;
}


int triangular_mindist(double* coords, int nframes, int nsets, long* set_sizes, double co, long* out_mat) {
 
  // Initialize output matrix
  int out_mat_elemsn = nsets*(nsets-1)/2;
  //int* out_mat = (int*) malloc(out_mat_elemsn * sizeof(int));
  int i = 0;
  int j = 0;
  int k = 0;
  int l = 0;
  int m = 0;

  for (i=0; i<out_mat_elemsn; i++) 
    out_mat[i] = 0.0;

  int sets_ends[nsets];
  int sets_starts[nsets];

  sets_ends[0] = set_sizes[0]; // USE < NOT <= LATER
  sets_starts[0] = 0;

  for (i=1; i<nsets; i++) {
    //printf("ii %d\n",i);
  //printf("set_sizes[i] %d\n", set_sizes[i]);
    sets_starts[i] = sets_starts[i-1] + set_sizes[i-1];
    sets_ends[i]   = sets_ends[i-1] + set_sizes[i];
  }

  int frame_natoms = sets_ends[nsets-1];
  //printf("sets_ends[0] %d\n", sets_ends[0]);
  //printf("sets_ends[1] %d\n", sets_ends[1]);
  //printf("sets_ends[2] %d\n", sets_ends[2]);
  //printf("sets_ends[nsets-1] %d\n", frame_natoms);
  int frame_ncoords = frame_natoms*3;

  //printf("frame_ncoords %d %d\n", sets_ends[2], sets_starts[2]);

  int idx_j = 0;
  int idx_k = 0;
  int idx_j_end = 0;
  int idx_k_end = 0;
  int frame_i = 0;
 

  for (i=0; i<nframes; i++) {
    frame_i = i*frame_ncoords;
    //printf("frame_i %d\n ", frame_i);
    for (j=0; j<nsets; j++) {
      idx_j = frame_i  + sets_starts[j]*3;
      idx_j_end = frame_i + sets_ends[j]*3;
      for (k=0; k<j; k++) {
	idx_k = frame_i  + sets_starts[k]*3;
	idx_k_end = frame_i + sets_ends[k]*3;
	//printf("    groups %d %d\n", j,k);
	//printf("%d %d: %.3f\n", j, k, ed(coords, coords, idx_j, idx_k));
	for (l=idx_j; l<idx_j_end; l+=3) {
	  for (m=idx_k; m<idx_k_end; m+=3) {
	    //printf("        %d,%d, %.3f\n",l,m,ed(coords, coords, l, m));
	    if (ed(coords, coords, l, m) <= co) {
	      out_mat[sqmI(nsets, j, k)] += 1;
              out_mat[sqmI(nsets, k, j)] += 1;
	      goto next_pair;
	    }
	  }
	}
      next_pair:;
	//printf("out\n");
      }
    }
  }
  return 1;
}

int square_mindist(double* coords1, double* coords2, int nframes, int nsets1, int nsets2, long* set_sizes1, long* set_sizes2, double co, long* out_mat) {
 
  // Initialize output matrix
  int out_mat_elemsn = nsets1*nsets2;
  //int* out_mat = (int*) malloc(out_mat_elemsn * sizeof(int));
  int i=0;
  int j=0;
  int k=0;
  int l=0;
  int m=0;

  for (i=0; i<out_mat_elemsn; i++) 
    out_mat[i] = 0;

  int sets_ends1[nsets1];
  int sets_starts1[nsets1];
  int sets_ends2[nsets2];
  int sets_starts2[nsets2];

  sets_ends1[0] = set_sizes1[0]; // USE < NOT <= LATER
  sets_ends2[0] = set_sizes2[0]; // USE < NOT <= LATER
  sets_starts1[0] = 0;
  sets_starts2[0] = 0;

  for (i=1; i<nsets1; i++) {
    sets_starts1[i] = sets_starts1[i-1] + set_sizes1[i-1];
    sets_ends1[i]   = sets_ends1[i-1] + set_sizes1[i];
  }

  for (i=1; i<nsets2; i++) {
    sets_starts2[i] = sets_starts2[i-1] + set_sizes2[i-1];
    sets_ends2[i]   = sets_ends2[i-1] + set_sizes2[i];
  }

  //printf("set_sizes1 %d %d %d %d %d %d \n", set_sizes1[0], set_sizes1[1], set_sizes1[2], set_sizes1[3], set_sizes1[4], set_sizes1[5]);
  //printf("starts1 %d %d %d\n", sets_starts1[0], sets_starts1[1], sets_starts1[2]);
  //printf("ends1 %d %d %d\n", sets_ends1[0], sets_ends1[1], sets_ends1[2]);

  int frame_natoms1 = sets_ends1[nsets1-1];
  int frame_natoms2 = sets_ends2[nsets2-1];
  int frame_ncoords1 = frame_natoms1*3;
  int frame_ncoords2 = frame_natoms2*3;

  //printf("frame_ncoords %d %d\n", sets_ends[2], sets_starts[2]);

  int idx_j = 0;
  int idx_k = 0;
  int idx_j_end = 0;
  int idx_k_end = 0;
  int frame_i1 = 0;
  int frame_i2 = 0;

  for (i=0; i<nframes; i++) {
    frame_i1 = i*frame_ncoords1;
    frame_i2 = i*frame_ncoords2;
    //printf("frame_i %d\n ", frame_i);
    for (j=0; j<nsets1; j++) {
      idx_j = frame_i1  + sets_starts1[j]*3;
      idx_j_end = frame_i1 + sets_ends1[j]*3;
      for (k=0; k<nsets2; k++) {
	idx_k = frame_i2  + sets_starts2[k]*3;
	idx_k_end = frame_i2 + sets_ends2[k]*3;
	//printf("    groups %d %d\n", j,k);
	//printf("%d %d: %.3f\n", j, k, ed(coords, coords, idx_j, idx_k));
	for (l=idx_j; l<idx_j_end; l+=3) {
	  for (m=idx_k; m<idx_k_end; m+=3) {
	    //printf("        %d,%d, %.3f\n",l,m,ed(coords1, coords2, l, m));
	    if (ed(coords1, coords2, l, m) <= co) {
	      //printf("adding %d %d\n",j,k);
	      out_mat[sqmI(nsets2,k,j)] += 1;
	      goto next_pair;
	    }
	  }
	}
      next_pair:;
	//printf("out\n");
      }
    }
    //printf("%d, %f\n",i,co);
  }
  return 1;
}



//int main() {

  // 2 f * 4 a * 3 c = 24

  //double test_array[24] = {1.0,1.0,1.0,5.0,5.0,5.0,4.0,4.0,4.0,2.0,2.0,2.0,1.0,1.0,1.0,5.0,5.0,5.0,4.0,4.0,4.0,2.0,2.0,2.0};

  //  int nframes = 2;
  //  int natoms = 4;
  //  int ndim = 3;

  //  int* tm = triangular_distmatrix(test_array, natoms, nframes, 10);
  //  printf("%d\n", tm[1]);

  //double test_array2[48] = {1.0,1.0,1.0,5.0,5.0,5.0,4.0,4.0,4.0,2.0,2.0,2.0,1.0,1.0,1.0,5.0,5.0,5.0,4.0,4.0,4.0,2.0,2.0,2.0,1.0,1.0,1.0,5.0,5.0,5.0,4.0,4.0,4.0,2.0,2.0,2.0,1.0,1.0,1.0,5.0,5.0,5.0,4.0,4.0,4.0,2.0,2.0,2.0};
  //double test_array2[48] = {58, 94, 95, 99,  5, 47,  0, 93, 20, 82, 22,  0, 14, 54, 79, 96, 77,
  //			    46, 82, 68, 73, 12, 38, 75, 51, 13, 99, 65, 47, 94, 62, 46, 24, 39,
  //			    97, 64, 13,  6,  0, 16, 91, 89, 86, 45, 50, 28, 95, 41};
  //
  //double test_array2[48] = {48,  19,  25,  52,  60,  10,  98,  61,  76,  70,  13,  81,   7,
//			    63,  56,  98,  48,  27,  78,  74,  85,  88,  77,  35,   3,  42,
//			    69,  30, 100,  27,  71,  66,  92,  45,  87,  62,  41,  20,  43,
//			    26,  38,  68,  84,  83,  26,  24,  74,  52};
//double asd[16];
  // int potential_distances(double* coords, int nsets, int set_size, int nframes, double* results) {

  //potential_distances(test_array2, 2, 4, 2, asd);

  //printf("%f %f\n%f %f\n %f %f\n %f %f\n%f %f\n%f %f\n %f %f\n %f %f\n", asd[0],asd[1],asd[2],asd[3],asd[4],asd[5],asd[6],asd[7],asd[8],asd[9],asd[10],asd[11],asd[12],asd[13],asd[14],asd[15],asd[16]);
  //int* triangular_mindist(double* coords, int natoms, int nframes, int nsets, int* set_sizes, float co) 
  //int set_sizes[3]={2,3,3};
  //int* tm2 = triangular_mindist(test_array2, 8, 2, 3, set_sizes, -1.0);
  
  //double test_arrayA[30] = {1.0,1.0,1.0,5.0,5.0,5.0,4.0,4.0,4.0,2.0,2.0,2.0,1.0,1.0,1.0};
  //double test_arrayB[30] = {5.0,5.0,5.0,4.0,4.0,4.0,2.0,2.0,2.0,5.0,5.0,5.0,4.0,4.0,4.0};
  //int setsA[2] = {2,3};
  //int setsB[2] = {3,2};

  //int* asd = square_mindist(test_arrayA, test_arrayB, 1, 2, 2, setsA, setsB, -1.0);

  //int* square_mindist(double* coords1, double* coords2,  int nframes, int nsets1, int nsets2, int* set_sizes1, int* set_sizes2, float co) {


  //int* triangular_distmatrix(double* coords, int natoms, int nframes, float co) 
//}
