import nifty8 as ift
import numpy as np


def draw_random(npix, distances, a, s, seed=42, check_divergence=False, adjust_sigmas=False):
    """
    routine to draw a divergence free random magnetic field.
    This is based on the fact that the cross product of two gradient fields is by construction divergence free, i.e.
        `B = \nabla p \cross \nabla q`
    The random scalar fields p and q are defined via power spectra, which are parametrized via an amplitude and a slope.
    Note that several caveats apply:
        1) There is some residual divergence (usually below one promille in relative power, see diagnostic output).
           This is a result of the pointwise product in the crossproduct, which leads to spectral broadening.
           The parts of the sectrum that cannot be represented on the grid are then aliased on smaller modes, leading to
           inconsistencies.
        2) For this reason, setting the slopes of the power spectra to below 4 is not warranted, as the effect is stronger
           the more power is on small scales. Note that the magnetic field will have a spectrum of ~ slope - 2, due to the gradients.
        3) If one chooses non equidistant distance bins, the amplitudes and morphology of the B-components will be different.
           This is a consequence of the divergence free condition.
        4) One should first fix the resolution, as the amplitude parameters are resolution dependent.
        (edited)
    """
    ift.random.push_sseq(seed)
    domain = ift.makeDomain(ift.RGSpace(npix, distances))
    harmonic_domain = ift.makeDomain(domain[0].get_default_codomain())
    ht = ift.HarmonicTransformOperator(harmonic_domain[0], domain[0])

    d_dx = PartialDerivative(harmonic_domain, 0, 1)
    d_dy = PartialDerivative(harmonic_domain, 1, 1)
    d_dz = PartialDerivative(harmonic_domain, 2, 1)

    def pow_spec_q(k):
        if check_divergence:
            import pylab as pl
            pl.ioff()
            pl.plot(np.log10(k + 1.5*k[1]), np.log10(a/(k + 1.5*k[1])**s))
            pl.savefig('pspec')
        return a/(k + 1.5*k[1])**s

    # 1D spectral space on which the power spectrum is defined
    power_domain = ift.PowerSpace(harmonic_domain[0])
    # Mapping to (higher dimensional) harmonic space
    PD = ift.PowerDistributor(harmonic_domain, power_domain)
    # Apply the mapping
    correlation_structure = PD(ift.PS_field(power_domain, pow_spec_q))
    S = ift.makeOp(correlation_structure, sampling_dtype=float)
    position_q = ift.from_random(harmonic_domain, dtype=float)
    position_p = ift.from_random(harmonic_domain, dtype=float)
    q_k = S(position_q)
    p_k = S(position_p)
    grad_q_x = (ht@d_dx)(q_k)
    grad_q_y = (ht@d_dy)(q_k)
    grad_q_z = (ht@d_dz)(q_k)
    grad_p_x = (ht@d_dx)(p_k)
    grad_p_y = (ht@d_dy)(p_k)
    grad_p_z = (ht@d_dz)(p_k)

    b_x = (grad_q_y*grad_p_z - grad_q_z*grad_p_y)
    b_y = (grad_q_z*grad_p_x - grad_q_x*grad_p_z)
    b_z = (grad_q_x*grad_p_y - grad_q_y*grad_p_x)

    if adjust_sigmas:
        print('Checking sigmas: ')
        sigma_b_x = b_x.val.std()
        sigma_b_y = b_y.val.std()
        sigma_b_z = b_z.val.std()
        print('sigma b_x', sigma_b_x)
        print('sigma b_y', sigma_b_y)
        print('sigma b_z', sigma_b_z)

        # producing 'solenoidal' random field with correct variance
        position_sol = ift.from_random(harmonic_domain, dtype=float)
        b_x_sol_k = S(position_sol)
        k_x = d_dx.kfield
        k_y = d_dy.kfield
        k_x_over_k_y = np.zeros(k_x.shape)
        k_x_over_k_y[k_y != 0] = k_x[k_y != 0]/k_y[k_y != 0]
        b_x_sol_k = b_x_sol_k/b_x_sol_k.val.std()
        b_y_sol_k = - ift.Field(harmonic_domain, k_x_over_k_y)*b_x_sol_k
        b_x_sol = ht(b_x_sol_k)
        sigma_correction = np.sqrt(sigma_b_z**2 - sigma_b_x**2)/b_x_sol.val.std()
        b_x_sol = b_x_sol*sigma_correction
        b_y_sol = ht(b_y_sol_k)*sigma_correction
        b_x = b_x + b_x_sol
        b_y = b_y + b_y_sol

        print('Checking adjusted sigmas: ')
        sigma_b_x = b_x.val.std()
        sigma_b_y = b_y.val.std()
        sigma_b_z = b_z.val.std()
        print('sigma b_x', sigma_b_x)
        print('sigma b_y', sigma_b_y)
        print('sigma b_x_sol', b_x_sol.val.std())
        print('sigma b_y_sol', b_y_sol.val.std())
        print('sigma b_z', sigma_b_z)

    if check_divergence:

        div_B = ht((d_dx@ht.adjoint)(b_x) + (d_dy@ht.adjoint)(b_y) + (d_dz@ht.adjoint)(b_z))
        abs_B = (b_x**2 + b_y**2 + b_z**2).sqrt()
        rel_div = div_B/abs_B
        print('Checking divergence: ')
        print('relative mean value: ', rel_div.val.mean())
        print('relative max value: ', rel_div.val.max())
        print('relative min value: ', rel_div.val.min())
        print('Note that some residual divergence is to be expected')

        if adjust_sigmas:
            div_B_sol = ht((d_dx@ht.adjoint)(b_x_sol) + (d_dy@ht.adjoint)(b_y_sol))
            abs_B_sol = (b_x_sol**2 + b_y_sol**2).sqrt()
            rel_div_sol = div_B_sol/abs_B_sol
            print('Checking solenoidal divergence: ')
            print('relative mean value: ', rel_div_sol.val.mean())
            print('relative max value: ', rel_div_sol.val.max())
            print('relative min value: ', rel_div_sol.val.min())
            print('Note that some residual divergence is to be expected')


        import pylab as pl
        pl.ioff()
        pl.imshow(rel_div.val[:, :, 5])
        pl.colorbar()
        pl.savefig('rel_div_xy')
        pl.close()

        pl.imshow(b_x.val[:, :, 5])
        pl.colorbar()
        pl.savefig('b_x_xy_slice')
        pl.close()

        pl.imshow(b_y.val[:, :, 5])
        pl.colorbar()
        pl.savefig('b_y_xy_slice')
        pl.close()

        pl.imshow(b_z.val[:, :, 5])
        pl.colorbar()
        pl.savefig('b_z_xy_slice')
        pl.close()

        pl.imshow(b_x.val[:, 10, :])
        pl.colorbar()
        pl.savefig('b_x_xz_slice')
        pl.close()

        pl.imshow(b_y.val[:, 10, :])
        pl.colorbar()
        pl.savefig('b_y_xz_slice')
        pl.close()

        pl.imshow(b_z.val[:, 10, :])
        pl.colorbar()
        pl.savefig('b_z_xz_slice')
        pl.close()

        pl.imshow(b_x.val[10, :, :])
        pl.colorbar()
        pl.savefig('b_x_yz_slice')
        pl.close()

        pl.imshow(b_y.val[10, :, :])
        pl.colorbar()
        pl.savefig('b_y_yz_slice')
        pl.close()

        pl.imshow(b_z.val[10, :, :])
        pl.colorbar()
        pl.savefig('b_z_yz_slice')
        pl.close()
    return (b_x.val, b_y.val, b_z.val)


class PartialDerivative(ift.EndomorphicOperator):
    def __init__(self,  domain, direction, order):
        self._domain = ift.makeDomain(domain)
        assert self._domain[0].harmonic, 'This operator works in the harmonic domain'
        self.nax = len(self._domain[0].shape)
        assert direction <= self.nax, 'number of spatial dimensions smaller then direction given'
        self._direction = direction
        self.distance = self._domain[0].distances[direction]
        self.co_distance = self._domain[0].get_default_codomain().distances[direction]
        self.co_size = self.co_distance*self._domain[0].shape[direction]
        kfield = np.arange(0, domain.shape[direction])
        idx = kfield > domain.shape[direction]/2
        kfield[idx] = (kfield[idx][::] - domain.shape[direction])
        if len(domain.shape) > 1:
            i = 0
            while i < self._direction:
                kfield = np.repeat(kfield.reshape(1, *kfield.shape), domain.shape[self._direction - i - 1], axis=0)
                i += 1
            i += 1
            while i > self._direction and i < len(domain.shape):
                kfield = np.repeat(kfield.reshape(*kfield.shape, 1), domain.shape[i], axis=-1)
                i += 1

        self.kfield = kfield # * self.distance
        assert isinstance(order, int), 'only non fractional derivatives are supported'
        self.order = order
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        npix = self._domain[0].shape[self._direction]

        sign = 1. if mode == self.TIMES else -1.
        part = x.val * (sign * 2 * np.pi * self.kfield / npix / self.co_distance)**self.order
        if (self.order % 2) == 0 or (npix % 2) == 1:
            return ift.Field(self.domain, part)
        # Fixing the Nyquist frequency for even grids
        part[self.kfield == npix / 2] = 0
        return ift.Field(self.domain, part)


if __name__ == "__main__":
    b_x, b_y, b_z = draw_random((40, 40, 10), (1., 1., .2), .01, 5, seed=42, check_divergence=True, adjust_sigmas=True)
