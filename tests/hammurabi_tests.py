"""
this is not a rigorous testing unit
"""
import unittest
import os
import numpy as np

from imagine import Measurements
from imagine import Hammurabi
from imagine import BregLSA
from imagine import BrndES
from imagine import CREAna
from imagine import FEregYMW16


class HammurabiTests(unittest.TestCase):

    def test_hammurabi_init(self):
        """
        check hammurabi init and observable/field register
        :return:
        """
        # mock measuremnts
        arr = np.random.rand(1, 48)
        measuredict = Measurements()
        measuredict.append(('sync', '23', '2', 'I'), arr)  # healpix map
        measuredict.append(('fd', 'nan', '2', 'nan'), arr)
        measuredict.append(('dm', 'nan', '2', 'nan'), arr)
        print('\n this is a non-rigorous test \n',
              'we expect \n',
              'sync/fd/dm at 23 GHz, at nside 2 \n')
        # consider only init, hammurabi executable is not necessary
        xmlpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_ham.xml')
        simer = Hammurabi(measurements=measuredict,xml_path=xmlpath)

        # mock BregLSA field
        paramlist = {'b0': 6.0, 'psi0': 27., 'psi1': 0.9, 'chi0': 25.}
        breg_lsa = BregLSA(paramlist, 23)
        # mock BrndES field
        paramlist = {'rms': 2., 'k0': 0.5, 'a0': 1.7, 'rho': 0.5, 'r0': 8., 'z0': 1.}
        brnd_es = BrndES(paramlist, 23)
        # mock CREAna field
        paramlist = {'alpha': 3.0, 'beta': 0.2, 'theta': 0.1,
                     'r0': 5.6, 'z0': 1.2,
                     'E0': 20.5,
                     'j0': 0.03}
        cre_ana = CREAna(paramlist, 23)
        # mock FEregYMW16 field
        paramlist = dict()
        fereg_ymw16 = FEregYMW16(paramlist, 23)

        # push fields to simulator
        simer.register_fields([breg_lsa, brnd_es, cre_ana, fereg_ymw16])

        # check initialization
        print('\n this is a non-rigorous test \n',
              'we expect \n',
              'lsa with b0: 6., psi0: 27., psi1: 0.9, chi0: 25. \n')
        self.assertEqual(simer.ensemble_size, int(23))
        simer._ham.print_par(['magneticfield', 'regular'])
        simer._ham.print_par(['magneticfield', 'regular', 'lsa'])
        print('\n this is a non-rigorous test \n',
              'we expect \n',
              'es with rms: 2., k0: 0.5, a0: 1.7, rho: 0.5, r0: 8., z0: 1. \n')
        simer._ham.print_par(['magneticfield', 'random'])
        simer._ham.print_par(['magneticfield', 'random', 'global', 'es'])
        print('\n this is a non-rigorous test \n',
              'we expect \n',
              'cre analytic with alpha: 3, beta: 0.2, theta: 0.1 \n'
              'r0: 5.6, z0: 1.2, E0: 20.5, j0: 0.03 \n')
        simer._ham.print_par(['cre'])
        simer._ham.print_par(['cre', 'analytic'])
        print('\n this is a non-rigorous test \n',
              'we expect \n',
              'fereg ymw16 \n')
        simer._ham.print_par(['freeelectron', 'regular'])


if __name__ == '__main__':
    unittest.main()
