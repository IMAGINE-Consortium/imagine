from imagine.fields import DummyField

class DummyFieldTemplate(DummyField):
    """
    Description of the dummy field
    """

    # Class attributes
    NAME = 'name_of_the_dummy_field'

    @property
    def field_checklist(self):
        return {'Parameter_A': 'parameter_A_settings',
                'Parameter_B': None}
    @property
    def simulator_controllist(self):
        return {'simulator_property_A': 'some_setting'}
