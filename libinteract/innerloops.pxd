cdef extern from "math.h":
     double exp(double)

cdef extern from "clibinteract.h":
     int trmI(int, int)
     int sqmI(int, int, int)
     double ed(double*, double*, int, int)
     int potential_distances(double*, int, int, int, double*)
     int triangular_distmatrix(double*, int, int, double, long*, double*)
     int square_distmatrix(double*, double*, int, int, int, double, long*)
     int triangular_mindist(double*, int, int, long*, double, long*)
     int square_mindist(double*, double*, int, int, int, long*, long*, double, long*)
     
