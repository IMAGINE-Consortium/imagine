# Simulator dependancies
import numpy as np
from imagine.simulators import Simulator
from scipy.special import gamma as gammafunc
import nifty7 as ift

import astropy.units as u
from astropy.coordinates import spherical_to_cartesian
from astropy.coordinates import cartesian_to_spherical
from astropy import constants as cons
MHz   = 1e6 / u.s
GHz   = 1e9 / u.s
me    = cons.m_e.cgs
c     = cons.c.cgs
kb    = cons.k_B.cgs
electron = cons.e.gauss

# Units definitions since astropy doesnt handle [B]_cgs well
gauss_B  = (u.g/u.cm)**(0.5)/u.s
equiv_B  = [(u.G, gauss_B, lambda x: x, lambda x: x)]
ugauss_B = 1e-6 * gauss_B



#%% Define the Simulator class

class SpectralSynchrotronEmissivitySimulator(Simulator):
    """
    Simulator for Galactic synchrotron emissivity.
    
    Requires the user to define observing geometry in the config dictionary    
    
    To run the default __call__ method one requires to input:
    - A 3D magnetic_field instance
    - A cosmic_ray_electron_density instance: 3D scalar density grid and constant spectral_index attribute
    
    Parameters
    ----------
    measurements : imagine.Measurements
        An observables dictionary containing the set of measurements that will be
        used to prepare the mock observables
    sim_config : 
        A dictionary detailing the observing geometry

    Attributes
    ----------
    grid : imagine.Basegrid
        Grid object where the fields were evaluated
    domain : 
        Nifty version of grid, used in the integration
    unitvector_grid : numpy.ndarray
        3D vector field containing the unitvectors pointing towards observer, used 
        for projecting to perpendicular magnetic field component
    fields : dict
        Dictionary containing field types as keys and the sum of evaluated fields
        as values
    observables : list
        List of Observable keys
    output_units : astropy.units.Unit
        Output units used in the simulator
    """

    
    # Class attributes
    SIMULATED_QUANTITIES = ['average_los_brightness']
    REQUIRED_FIELD_TYPES = ['magnetic_field','cosmic_ray_electron_density']
    OPTIONAL_FIELD_TYPES = ['cosmic_ray_electron_spectral_index']
    ALLOWED_GRID_TYPES   = ['cartesian']
    
    def __init__(self,measurements,sim_config={'grid':None,'observer':None,'dist':None,'e_dist':None,'lat':None,'lon':None,'FB':None}):
        
        print("Initializing SynchtrotronEmissivitySimulator")

        # Send the Measurements to the parent class
        super().__init__(measurements) 
        
        # Write assert and test functions to make sure stuf is correct
        # - grid.unit == observer.unit
        # - grid = instance of cartesian grid
        # - all HII regions should be located within the simulation box        
        
        
        # Asses field types and set data-acces function
        #self.get_field_data()        
        
        # Stores class-specific attributes (and for now double definitions)
        for key in measurements.keys(): self.observing_frequency = key[1] * GHz
        grid = sim_config['grid'] # unpack for readability
        
        # Setup unitvector grid used for mapping perpendicular component
        unit = sim_config['observer'].unit 
        unitvectors = []
        for x,y,z in zip(grid.x.ravel()/unit,grid.y.ravel()/unit,grid.z.ravel()/unit):
            v = np.array([x,y,z])-sim_config['observer']/unit
            normv = np.linalg.norm(v)
            if normv == 0: # special case where the observer is inside one of the grid points
                unitvectors.append(v)
            else:
                unitvectors.append(v/normv)
        # Save unitvector grid as class attribute
        self.unitvector_grid = np.reshape(unitvectors, tuple(grid.resolution)+(3,))
    
        # Cast imagine instance of grid into a RGSpace grid  
        cbox = grid.box
        xmax = (cbox[0][1]-cbox[0][0])/unit
        ymax = (cbox[1][1]-cbox[1][0])/unit
        zmax = (cbox[2][1]-cbox[2][0])/unit       
        box  = np.array([xmax,ymax,zmax])
        grid_distances = tuple([b/r for b,r in zip(box, grid.resolution)])
        self.domain = ift.makeDomain(ift.RGSpace(grid.resolution, grid_distances)) # need this later for the los integration 
    
        # Get lines of sight from data
        lat        = sim_config['lat'].to(u.rad)
        lon        = sim_config['lon'].to(u.rad)
        hIIdist    = sim_config['dist'].to(u.kpc)
        if sim_config['e_dist'] != None:
            e_hIIdist  = sim_config['e_dist'].to(u.kpc)
            #print("Simulator distances:\n", hIIdist)
            #print("Simulator rel error:\n", e_hIIdist/hIIdist)
        else:
            e_hIIdist = None
            #print("Working with relative distance error: None")
        behind     = np.where(np.array(sim_config['FB'])=='B')
        nlos       = len(hIIdist)
        
        # Remember the translation
        translation = np.array([xmax,ymax,zmax])/2 * unit
        translated_observer = sim_config['observer'] + translation

        # Cast start and end points in a Nifty compatible format
        starts = []
        for o in translated_observer:
            starts.append(np.full(shape=(nlos,), fill_value=o)*u.kpc)
        start_points = np.vstack(starts).T

        ends = []
        los  = spherical_to_cartesian(r=hIIdist, lat=lat, lon=lon)
        for i,axis in enumerate(los):
            ends.append(axis+translated_observer[i])
        end_points = np.vstack(ends).T

        #print("Simulator translated starts:\n", start_points)
        #print("Simulator translated ends:\n", end_points)

        # Do Front-Behind selection
        deltas = end_points - start_points
        clims  = box * np.sign(deltas) * unit
        clims[clims<0]=0 # if los goes in negative direction clim of xyz=0-plane
        with np.errstate(divide='ignore'):
            all_lambdas = (clims-end_points)/deltas   # possibly divide by zero 
        lambdas = np.min(np.abs(all_lambdas), axis=1) # will drop any inf here
        start_points[behind] = end_points[behind] + np.reshape(lambdas[behind], newshape=(np.size(behind),1))*deltas[behind]     
        
        # Final integration distances
        self.los_distances = np.linalg.norm(end_points-start_points, axis=1)
   
        # convenience for bugfixing:
        behindTF = np.zeros(nlos)*u.kpc
        behindTF[behind] = 1*u.kpc
        #print("Simulator starts after FB (1=behind):\n", np.vstack((np.round(start_points,1).T,behindTF)).T)
        #print("Simulator ends after FB:\n", np.round(end_points,1))
        #print("Simulator distances after FB:\n", self.los_distances)
        #print("Simulator translated_observer\n", translated_observer)
        #print("Simulator start_points\n", start_points)
        #print("Simulator end_points\n", end_points)
        self.start_points = start_points
        self.behindTF = behindTF
        nstarts = self._make_nifty_points(start_points)
        nends   = self._make_nifty_points(end_points)

        self.response = ift.LOSResponse(self.domain, nstarts, nends, sigmas=e_hIIdist, truncation=3.) # domain doesnt know about its units but starts/ends do?
    
    
    def _make_nifty_points(self, points, dim=3):
        rows,cols = np.shape(points)
        if cols != dim: # we want each row to be a coordinate (x,y,z)
            points = points.T
            rows   = cols
        npoints = []
        for d in range(dim):
            npoints.append(np.full(shape=(rows,), fill_value=points[:,d]))
        return npoints
    
    def _spectral_integralF(self, mu):
        return 2**(mu+1)/(mu+2) * gammafunc(mu/2 + 7./3) * gammafunc(mu/2+2./3)    
    
    def _spectral_total_emissivity(self, Bper, ncre):
        # Check constant units and remove them for the calculation
        e     = (cons.e).gauss/(cons.e).gauss.unit
        me    = cons.m_e.cgs/cons.m_e.cgs.unit
        c     = cons.c.cgs/cons.c.cgs.unit
        # Check argument units and remove them for the calculation        
        vobs  = self.observing_frequency.to(1/u.s)*u.s
        ncre  = ncre.to(u.cm**-3)*u.cm**3
        Bper  = Bper.to(u.G)/u.G
        # Handle two spectral index cases:
	# -> fieldlist is not provided on initialization so we opt for a runtime check of alpha type
        try: # alpha is a constant spectral index globally 
            alpha = self.field_parameter_values['cosmic_ray_electron_density']['spectral_index']
        except: pass
        try: # alpha is an instance of a 3D scalar field
            alpha = self.fields['cosmic_ray_electron_spectral_index']
        except: pass
        # Calculate emissivity grid
        fraction1 = (np.sqrt(3)*e**3*ncre/(8*np.pi*me*c**2))
        fraction2 = (4*np.pi*vobs*me*c/(3*e))
        integral  = self._spectral_integralF( (-alpha-3)/2 )
        emissivity = fraction1 * fraction2**((1+alpha)/2) * Bper**((1-alpha)/2) * integral
        assert emissivity.unit == u.dimensionless_unscaled
        # Return emissivty and restore correct units
        return fraction1*fraction2**((1+alpha)/2)*Bper**((1-alpha)/2)*integral * u.kg/(u.m*u.s**2)
    
    def _project_to_perpendicular(self, vectorfield):
        """
        This function takes in a 3D vector field and uses the initialized unitvector_grid
        to project each vector on the unit vector perpendicular to the los to that position.
        """
        v_parallel      = np.zeros(np.shape(vectorfield)) * vectorfield.unit
        amplitudes      = np.sum(vectorfield * self.unitvector_grid, axis=3)
        v_parallel[:,:,:,0]  = amplitudes * self.unitvector_grid[:,:,:,0]
        v_parallel[:,:,:,1]  = amplitudes * self.unitvector_grid[:,:,:,1]
        v_parallel[:,:,:,2]  = amplitudes * self.unitvector_grid[:,:,:,2]
        v_perpendicular      = vectorfield - v_parallel
        v_perp_amplitude     = np.sqrt(np.sum(v_perpendicular*v_perpendicular,axis=3))
        return v_perp_amplitude
    
    def simulate(self, key, coords_dict, realization_id, output_units): 
        # Acces field data
        ncre_grid = self.fields['cosmic_ray_electron_density']  # in units cm^-3
        B_grid    = self.fields['magnetic_field'] # fixing is now done inside emissivity calculation
        #print("Simulator zeros in cre_grid:\n",sum(ncre_grid.flatten()==0))
        # Project to perpendicular component to line of sight
        Bperp_amplitude_grid = self._project_to_perpendicular(B_grid)
        #print("Simulator zeros in Bper_amp_grid\n",sum(Bperp_amplitude_grid.flatten()==0))
        # Calculate grid of emissivity values
        emissivity_grid = self._spectral_total_emissivity(Bperp_amplitude_grid, ncre_grid)
        #emissivity_grid = np.ones(np.shape(emissivity_grid))
        #print("Simulator emissivity_grid:\n", emissivity_grid)
        #print("Simulator number of zero pixels in grid:\n",sum(emissivity_grid.flatten()==0))
        # Do the los integration on the domain defined in init with the new emissivity grid
        HII_LOSemissivities = self.response(ift.Field(self.domain, emissivity_grid)).val_rw()
        HII_LOSemissivities *= emissivity_grid.unit * u.kpc # restore units: domain is assumed to be in kpc
        # Need units to be in K/kpc, average brightness temperature allong the line of sight
        HII_LOSbrightness = c**2/(2*kb*self.observing_frequency**2)*HII_LOSemissivities/self.los_distances
        #print("Simulator final brightness before rounding:\n",HII_LOSbrightness)
        #print("Simulator coordinate-brightness summary:\n", np.vstack((np.round(self.start_points,1).T,self.behindTF,HII_LOSbrightness/(u.K/u.kpc)*u.kpc)).T)
        return HII_LOSbrightness
        #print("Simulator los_disntaces:\n", self.los_distances)
        #return HII_LOSemissivities *u.K/u.kpc

