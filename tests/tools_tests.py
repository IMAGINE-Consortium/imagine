import unittest
import numpy as np

from imagine.tools.masker import mask_obs, mask_cov
from imagine.tools.random_seed import seed_generator


class TestTools(unittest.TestCase):

    def test_seed(self):
        # test seed gen, in base class
        s1 = seed_generator(0)
        s2 = seed_generator(0)
        self.assertNotEqual(s1, s2)
        s3 = seed_generator(48)
        self.assertEqual(s3, 48)

    def test_mask(self):
        msk_arr = np.array([0., 1., 0., 1., 1., 0.]).reshape(1, 6)
        obs_arr = np.random.rand(1, 6)
        cov_arr = np.random.rand(6, 6)
        # mask by methods
        test_obs = mask_obs(obs_arr, msk_arr)
        test_cov = mask_cov(cov_arr, msk_arr)
        # mask manually
        fid_obs = np.hstack([obs_arr[0, 1], obs_arr[0, 3], obs_arr[0, 4]])
        fid_cov = np.delete(cov_arr, [0, 2, 5], 0)
        fid_cov = np.delete(fid_cov, [0, 2, 5], 1)
        #
        self.assertListEqual(list(test_obs[0]), list(fid_obs))
        for i in range(fid_cov.shape[0]):
            self.assertListEqual(list(test_cov[i]), list(fid_cov[i]))


if __name__ == '__main__':
    unittest.main()
