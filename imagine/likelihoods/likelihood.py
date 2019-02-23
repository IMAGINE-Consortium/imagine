'''
Likelihood class defines likelihood posterior function
to be used in Bayesian analysis

members:
._to_global_data
    -- works trivially on list/tuple
__init__
    -- initialisation needs arguments
    a list of measurement dict names
    a dict of measured(observational) data
    with dict entry perfectly matching name list
    a dict of measured(observational) covariance matrices
    with dict entry perfectly matching name list
__call__
    -- running LOG-likelihood calculation, needs argument
    a dict of Observable objects
    with dict entry perfectly matching name list

input argument of __call__ should have form
{observable_name: [observable_data]}

observable name/unit convention:
* ('fd','nan',str(pix/nside),'nan')
    -- Faraday depth (in unit 
* ('dm','nan',str(pix),'nan')
    -- dispersion measure (in unit
* ('sync',str(freq),str(pix),X)
    -- synchrotron emission
    X stands for:
    * 'I'
        -- total intensity (in unit K-cmb)
    * 'Q'
        -- Stokes Q (in unit K-cmb, IAU convention)
    * 'U'
        -- Stokes U (in unit K-cmb, IAU convention)
    * 'PI'
        -- polarisation intensity (in unit K-cmb)
    * 'PA'
        -- polarisation angle (in unit rad, IAU convention)

remarks on observable name:
    -- str(freq), polarisation-related-flag are redundant for Faraday depth and dispersion measure
    so we put 'nan' instead
    -- str(pix/nside) stores either Healpix Nisde, or just number of pixels/points
    we do this for flexibility, in case users have non-Healpix based in/output
'''

from nifty5 import Field

from imagine.observables.observable import Observable

class Likelihood(object):
    
    def __call__(self, observable_names):
        raise NotImplementedError

    # applies to NIFTy5.Field/Observable input
    # acts trivially on normal input
    def _to_global_data(self, data):
        if isinstance(data, (Field,Observable)):
            return data.to_global_data()
        else:
            return data
