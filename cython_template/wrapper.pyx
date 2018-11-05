"""
Author: Martin Craig <martin.craig@eng.ox.ac.uk>
Copyright (c) 2013-2015 University of Oxford,  Martin Craig
"""

import numpy as np
cimport numpy as np
import ctypes

cdef extern from "wrapper.h":
    void get_areas(const int *int_array, int N, const float *double_array, int M, float *output_buffer)

def get_areas_wrapper(trig_vertices, vertices):
    num_trigs = trig_vertices.shape[0]
    num_vertices = vertices.shape[0]

    # Make input data flat and contiguous. We need to worry about row-major vs column-major ordering here.
    # 'F' = column major ('Fortran order'), 'C' = row major ('C order')
    cdef np.ndarray[int, ndim=1] ctrig_vertices = trig_vertices.flatten(order='C').astype(ctypes.c_int)
    cdef np.ndarray[np.float32_t, ndim=1] cvertices = vertices.flatten(order='C').astype(np.float32)
    
    # Create data buffer to catch returned data
    cdef np.ndarray[np.float32_t, ndim=1] careas = np.zeros(num_trigs, dtype=np.float32)

    # Run the C code
    get_areas(&ctrig_vertices[0], num_trigs, &cvertices[0], num_vertices, &careas[0])
    
    # Reshape output arrays - this is not necessary here but this is how you'd do it if returning 2D/3D array
    #areas = np.reshape(careas, shape_it_should_be, order='C')

    return careas
    

