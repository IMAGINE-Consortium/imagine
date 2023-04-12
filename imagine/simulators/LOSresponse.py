import nifty8 as ift
import numpy as np

from astropy.coordinates import spherical_to_cartesian
from astropy.coordinates import cartesian_to_spherical


def _make_nifty_points(points, dim=3):
    rows,cols = np.shape(points)
    if cols != dim: # we want each row to be a coordinate (x,y,z)
        points = points.T
        rows   = cols
    npoints = []
    for d in range(dim):
        npoints.append(np.full(shape=(rows,), fill_value=points[:,d]))
    return npoints


def build_nifty_los(grid, behind, unit, observer, dist, lon, lat, dist_error):
            # Cast imagine instance of grid into a RGSpace grid  
            
                # Setup unitvector grid used for mapping perpendicular component


    cbox = grid.box
    xmax = (cbox[0][1]-cbox[0][0])/unit
    ymax = (cbox[1][1]-cbox[1][0])/unit
    zmax = (cbox[2][1]-cbox[2][0])/unit       
    box  = np.array([xmax,ymax,zmax])
    grid_distances = tuple([b/r for b,r in zip(box, grid.resolution)])
    domain = ift.makeDomain(ift.RGSpace(grid.resolution, grid_distances)) # need this later for the los integration 

    # Remember the translation
    translation = np.array([xmax,ymax,zmax])/2 * unit
    translated_observer = observer + translation

    # Cast start and end points in a Nifty compatible format
    starts = []
    for o in translated_observer: starts.append(np.full(shape=(len(dist),), fill_value=o)*unit)
    start_points = np.vstack(starts).T

    ends = []
    los  = spherical_to_cartesian(r=dist, lat=lat, lon=lon)
    for i,axis in enumerate(los): ends.append(axis+translated_observer[i])
    end_points = np.vstack(ends).T
    deltas = end_points - start_points
    clims  = box * np.sign(deltas) * unit
    clims[clims<0]=0 # if los goes in negative direction clim of xyz=0-plane
    
    with np.errstate(divide='ignore'):
        all_lambdas = (clims-end_points)/deltas   # possibly divide by zero 
    lambdas = np.min(np.abs(all_lambdas), axis=1) # will drop any inf here

    start_points[behind] = end_points[behind] + np.reshape(lambdas[behind], newshape=(np.size(behind),1))*deltas[behind]     
    
    # los_distances = np.linalg.norm(end_points-start_points, axis=1)

    nstarts = _make_nifty_points(start_points)
    nends   = _make_nifty_points(end_points)
    
    return ift.LOSResponse(domain, nstarts, nends, sigmas=dist_error, truncation=3.)


def apply_response(response, field_val):
    return response(ift.Field(response.domain, field_val)).val_rw()