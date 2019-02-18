## IMAGINE brief info

### Prior class

Defines likelihood prior.

Prior is an abstract base class, nothing implemented.

FlatPrior return uniform prior function for each parameter, with parameter logic range [0,1].

### Likelihood class

Defines likelihood functions.

Likelihood is the base class, defines
```
_strip_data
```
for extracting data from Nifty.FieldArray type argument, it is trivial for list and tuple.

Default call of likelihoods requires object of Observable class, with observable_name
matching one of the entries in Observable instance.

### MagneticField class

Defines magnetic field parameter name, default value, prior range, and physical to logic value 
mapping.

```
discriptor_lookup
```
return a dict, with entry as name of parameter, with content as list of xml path for corresponding simulator.
(the default simulator is hammurabiX)

No direct field generation is implemented except in ConstMagneticField class.

Usage of 
```
_to/from_hdf5
```
is unknown.

### Observable class

Simulated outputs are stored in a dict of Observable objects.
```
ensemble_mean
```
calculates mean value in the ensemble of simulated outputs.

Usage of 
```
_to/from_hdf5
```
is unknown.


### Pipeline class
