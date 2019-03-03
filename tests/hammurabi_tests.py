"""
this is not a rigorous testing unit
"""
import unittest
import os
import numpy as np

from imagine.observables.observable_dict import Measurements
from imagine.simulators.hammurabi.hammurabi import Hammurabi
from imagine.fields.breg_wmap.hamx_field import BregWMAP


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
		# mock fields
		paramlist = {'b0': 6.0, 'psi0': 27.9, 'psi1': 1.3, 'chi0': 24.6}
		breg_wmap = BregWMAP(paramlist, 23)
		simer.register_fields([breg_wmap])
		print('\n this is a non-rigorous test \n',
			  'we expect \n',
			  'wmap with b0: 6, psi0, 27.9, psi1: 1.3, chi0: 24.6 \n')
		self.assertEqual(simer.ensemble_size, int(23))
		simer._ham.print_par(['magneticfield', 'regular'])
		simer._ham.print_par(['magneticfield', 'regular', 'wmap'])


if __name__ == '__main__':
	unittest.main()