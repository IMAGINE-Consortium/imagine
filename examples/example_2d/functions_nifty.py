import nifty7 as ift
import numpy as np


def get_mask_op(domain, endpoints):
    mask_array = np.full(domain.shape, 0)
    n_data = domain.shape[0]
    n_depth = domain.shape[1]
    for i in range(n_data):
        ind = np.arange(n_depth) < endpoints[i]
        mask_array[i, ind] = 1
    return ift.makeOp(ift.Field(ift.makeDomain(domain), mask_array))


def get_random_model(domain, fixed_values):
    def fixed_power_spectrum(a0, k0, p):
        return lambda kv: a0 / (1 + (kv/k0) ** (-p))

    harmonic_domain = ift.makeDomain(domain[0].get_default_codomain())
    ht = ift.HarmonicTransformOperator(harmonic_domain[0], domain[0])
    power_space = ift.PowerSpace(harmonic_domain[0])
    pd = ift.PowerDistributor(harmonic_domain[0], power_space)

    if set(fixed_values) != {'a0', 'k0', 'p'}:
        scalar_dom = ift.DomainTuple.scalar_domain()
        expander = ift.VdotOperator(ift.Field.full(ift.makeDomain(power_space), 1.)).adjoint
        ops = {}
        for key in ['a0', 'k0', 'p']:
            if key not in fixed_values:
                ops.update({key: expander@ift.FieldAdapter(scalar_dom, key)})
            else:
                ops.update({key: ift.makeOp(ift.Field.full(ift.makeDomain(power_space), fixed_values['a0']))})
        one_adder = ift.Adder(ift.Field.full(ift.makeDomain(power_space), 1.))
        kv = ift.makeOp(ift.Field(power_space, power_space.k_lenths()))
        ps_model = ops['a0'] @  ((one_adder @  (((ops['k0']**(-1)) @ kv)**ops['p']))**(-1))
        prior_sigma = (pd @ ps_model).ptw('sqrt')
        return ht @ (prior_sigma * ift.ducktape(harmonic_domain, None, 'random_xi'))
    else:
        ps_field = ift.PS_field(power_space, fixed_power_spectrum(fixed_values['a0'],
                                                                  fixed_values['k0'], fixed_values['p']))
        prior_sigma = ift.makeOp((pd(ps_field)).ptw('sqrt'))
        return ht @ (prior_sigma @ ift.ducktape(harmonic_domain, None, 'random_xi'))


def get_spiral_model(domain, absolute, k_key, amp_key):
    def get_phi_and_r_array(dom):
        dom = ift.makeDomain(dom)
        assert len(dom.shape) == 2, 'Spiral can only be defined in 2D'
        i_coords, j_coords = np.meshgrid(np.linspace(- dom.shape[1] * dom[0].distances[1] / 2,
                                                     dom.shape[1] * dom[0].distances[1] / 2, dom.shape[1]),
                                         np.linspace(- dom.shape[0] * dom[0].distances[0] / 2,
                                                     dom.shape[0] * dom[0].distances[0] / 2, dom.shape[0]),
                                         indexing='xy')
        phi_array = np.arctan2(j_coords, i_coords)
        r_array = np.sqrt(i_coords ** 2 + j_coords ** 2)
        return ift.Field(dom, phi_array), \
               ift.Field(dom, np.mod(r_array, np.pi))

    expander = ift.VdotOperator(ift.Field.full(domain, 1)).adjoint

    phi_field, r_field = get_phi_and_r_array(domain)
    r_op = ift.makeOp(r_field)
    phi_op = ift.Adder(-phi_field)

    k = ift.FieldAdapter(ift.DomainTuple.scalar_domain(), k_key).ptw('exp')
    amplitude = ift.FieldAdapter(ift.DomainTuple.scalar_domain(), amp_key).ptw('exp')

    if absolute:
        return (expander @ amplitude)*(phi_op @ r_op @ expander @ k).ptw('sin').ptw('abs')
    return (expander @ amplitude)*(phi_op @ r_op @ expander @ k).ptw('sin')


def get_mask_response(domain, mask_op):
    domain = ift.makeDomain(domain)
    dom_adap = DomainAdapter(domain)
    r = ift.ContractionOperator(dom_adap.target, 1) @ dom_adap
    return r @ mask_op


def get_los_response(domain, starts, end, n_los):
    ends = list()
    ends.append(np.full(shape=(n_los,), fill_value=end[0]))
    ends.append(np.full(shape=(n_los,), fill_value=end[1]))
    return ift.LOSResponse(domain, starts, ends)


def get_mock_truth(model, truth_dict):
    position_dict = {}
    for key, domain in model.domain.items():
        if key in truth_dict:
            position_dict.update({key: ift.Field(ift.makeDomain(ift.DomainTuple.scalar_domain()), truth_dict[key])})
        else:
            position_dict.update({key: ift.from_random(domain)})
    truth = model.force(ift.MultiField.from_dict(position_dict))
    return truth, position_dict


class DomainAdapter(ift.LinearOperator):
    def __init__(self, dom, ):
        self._domain = ift.DomainTuple.make(dom)
        tgt = []
        for dom_ind in range(len(self.domain)):
            typ = self.domain[dom_ind].__class__
            for s, d in zip(list(self.domain[dom_ind].shape), list(self.domain[dom_ind].distances)):
                tgt.append(typ(shape=(s,), distances=(d,)))

        self._target = ift.DomainTuple.make(tuple(tgt))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        return x.cast_domain(self._tgt(mode))