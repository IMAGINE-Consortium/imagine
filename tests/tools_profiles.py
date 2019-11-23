import numpy as np
from mpi4py import MPI

from imagine.tools.mpi_helper import mpi_mean, mpi_arrange, mpi_trans, mpi_trace
from imagine.tools.covariance_estimator import oas_mcov
from imagine.tools.timer import Timer

comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
mpirank = comm.Get_rank()

def mpi_mean_timing(ensemble_size, data_size):
    local_ensemble_size = mpi_arrange(ensemble_size)[1] - mpi_arrange(ensemble_size)[0]
    random_data = np.random.rand(local_ensemble_size, data_size)
    tmr = Timer()
    tmr.tick('mpi_mean')
    ensemble_mean = mpi_mean(random_data)
    tmr.tock('mpi_mean')
    if not mpirank:
        print('@ tools_profiles::mpi_mean_timing with '+str(mpisize)+' nodes')
        print('global array shape ('+str(ensemble_size)+','+str(data_size)+')')
        print(str(tmr.record))
    
    
def mpi_trans_timing(ensemble_size, data_size):
    local_ensemble_size = mpi_arrange(ensemble_size)[1] - mpi_arrange(ensemble_size)[0]
    random_data = np.random.rand(local_ensemble_size, data_size)
    tmr = Timer()
    tmr.tick('mpi_trans')
    transed_data = mpi_trans(random_data)
    tmr.tock('mpi_trans')
    if not mpirank:
        print('@ tools_profiles::mpi_trans_timing with '+str(mpisize)+' nodes')
        print('global matrix size ('+str(ensemble_size)+','+str(data_size)+')')
        print(str(tmr.record))
    

def mpi_trace_timing(data_size):
    local_row_size = mpi_arrange(data_size)[1] - mpi_arrange(data_size)[0]
    random_data = np.random.rand(local_row_size, data_size)
    tmr = Timer()
    tmr.tick('mpi_trace')
    trace = mpi_trace(random_data)
    tmr.tock('mpi_trace')
    if not mpirank:
        print('@ tools_profiles::mpi_trace_timing with '+str(mpisize)+' nodes')
        print('global matrix size ('+str(data_size)+','+str(data_size)+')')
        print(str(tmr.record))
   
     
def oas_estimator_timing(data_size):
    local_row_size = mpi_arrange(data_size)[1] - mpi_arrange(data_size)[0]
    random_data = np.random.rand(local_row_size, data_size)
    tmr = Timer()
    tmr.tick('oas_estimator')
    mean, local_cov = oas_mcov(random_data)
    tmr.tock('oas_estimator')
    if not mpirank:
        print('@ tools_profiles::oas_estimator_timing with '+str(mpisize)+' nodes')
        print('global matrix size ('+str(data_size)+','+str(data_size)+')')
        print(str(tmr.record))


if __name__ == '__main__':
    N = 2**13
    mpi_mean_timing(N, N)
    mpi_trans_timing(N, N)
    mpi_trace_timing(N)