
#from ._basic_building_blocks cimport basic_building_blocks
cimport ._basic_building_blocks
cdef class fl:
    """ Convertion between real number and shared float number
        Implemented with numpy array
    
    """
    cdef int k
    cdef int l
    cdef int num_parties
    cdef ._basic_building_blocks.basic_building_blocks bbb
