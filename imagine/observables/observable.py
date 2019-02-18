from nifty import Field, FieldArray

'''
invoked in observer wrapper and likelihoods
for distributing/manipulating simulated outputs

Observable inherit directly from NIFTy.Field with
two domains' product, which is designed for ensemble
of simulated maps.

#undeciphered
_to/from_hdf5

'''
class Observable(Field):
    
    def __init__(self, domain=None, val=None, dtype=None,
                 distribution_strategy=None, copy=False):
        super(Observable, self).__init__(domain=domain,
                                         val=val,
                                         dtype=dtype,
                                         distribution_strategy=distribution_strategy,
                                         copy=copy)
        assert (len(self.domain) == 2) # prod of two domain types
        assert isinstance(self.domain[0], FieldArray)

    def ensemble_mean(self):
        try:
            self._ensemble_mean
        except AttributeError:
            self._ensemble_mean = self.mean(spaces=0)
        finally:
            return self._ensemble_mean
    
    '''
    # undeciphered
    def _to_hdf5(self, hdf5_group):
        if hasattr(self, '_ensemble_mean'):
            return_dict = {'ensemble_mean': self._ensemble_mean}
        else:
            return_dict = {}
        return_dict.update(super(Observable, self)._to_hdf5(hdf5_group=hdf5_group))
        return return_dict

    @classmethod
    def _from_hdf5(cls, hdf5_group, repository):
        new_field = super(Observable, cls)._from_hdf5(hdf5_group=hdf5_group, repository=repository)
        try:
            observable_mean = repository.get('ensemble_mean', hdf5_group)
            new_field._observable_mean = observable_mean
        except(KeyError):
            pass
        return new_field
    '''
