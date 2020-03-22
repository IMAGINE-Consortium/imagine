***************
Parallelisation
***************

The IMAGINE pipeline was designed with hybrid MPI/OpenMP use on a cluster in
mind:  the Pipeline distributes sampling work *accross different nodes* using
MPI, while Fields and Simulators are assumed to use OpenMP (or similar shared
memory multiprocessing) to run in parallel *within a single multi-core node*.
