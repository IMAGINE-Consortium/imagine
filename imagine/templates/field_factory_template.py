from imagine import GeneralFieldFactory

class YourField_Factory(GeneralFieldFactory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field_class = YourFieldClass
        self.default_parameters = {'param_A': param_A_value,
                                   'param_B': param_B_value,
                                   ... }
        self.parameter_ranges = {'param_A': [A_min, A_max],
                                 'param_B': [B_min, B_max],
                                 ... }