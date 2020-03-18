from imagine import MagneticField
    
class DummyFieldTemplate(DummyField):
    """
    Description of the dummy field
    """
    field_name = 'name_of_the_dummy_field'
    
    @property
    def field_checklist(self):
        return {'Parameter_A': None, 'Parameter_B': None, ...}
    