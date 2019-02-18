import sys
import mpi4py
import mpi4py_shadow

mpi4py_backup = sys.modules['mpi4py']
sys.modules['mpi4py'] = mpi4py_shadow

import pymultinest

sys.modules['mpi4py'] = mpi4py_backup
