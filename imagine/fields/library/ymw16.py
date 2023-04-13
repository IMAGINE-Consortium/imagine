import ImagineModels as im
import numpy as np
import astropy.units as u
import astropy as ap

from imagine.fields.base_fields import MagneticField
from imagine.fields.field_factory import FieldFactory

__all__ = ['YMW16',]


class YMW16(MagneticField):
    NAME = 'YMW16'
    STOCHASTIC_FIELD = False
    PARAMETER_NAMES = ['r_warp', 'r0', 't0_gamma_w', 't1_ad', 't1_bd', 't1_n1', 't1_h1', 
                       't2_a2', 't2_b2', 't2_n2', 't2_k2', 't3_b2s', 't3_ka', 't3_aa', 't3_ncn', 
                       't3_wcn', 't3_thetacn', 't3_nsg', 't3_wsg', 't3_thetasg',
    ]
    DEFAULT_UNITS =  { }

    def compute_field(self, seed):
        
        res = self.grid.resolution

        x = self.grid.box[0]#.to_value()
        x = np.linspace(x[0].value, x[1].value, res[0])
        y = self.grid.box[1]#.to_value()
        y = np.linspace(y[0].value, y[1].value, res[1])
        z = self.grid.box[2]#.to_value()
        z = np.linspace(z[0].value, z[1].value, res[2])
        # Creates an empty array to store the result

        model = im.YMW16()

        for key, val in self.parameters.items():
            if hasattr(model, key):
                if isinstance(val, ap.Quantity):
                    val = val.to(self.DEFAULT_UNITS[key]).value
                setattr(model, key, val)

        m = model.on_grid(grid_x=x, grid_y=y, grid_z=z)*self.UNITS

        return m

'''

  // Thick disc


  // spiralarms

  std::array<, 5> t3_rmin{3.35, 3.707, 3.56, 3.670, 8.21};
  std::array<, 5> t3_phimin{44.4, 120.0, 218.6, 330.3, 55.1};
  std::array<, 5> t3_tpitch{11.43, 9.84, 10.38, 10.54, 2.77};
  std::array<, 5> t3_cpitch{11.43, 9.84, 10.38, 10.54, 2.77};
  std::array<, 5> t3_narm{0.135, 0.129, 0.103, 0.116, 0.0057};
  std::array<, 5> t3_warm{300., 500., 300., 500., 300.};

  // Galactic Center
   t4_ngc = 6.2;
   t4_agc = 160.;
   t4_hgc = 35.;

  // gum
   t5_kgn = 1.4;
   t5_ngn = 1.84;
   t5_wgn = 15.1;
   t5_agn = 125.8;

  // local bubble
   t6_j_lb = 0.480;
   t6_nlb1 = 1.094;
   t6_detlb1 = 28.4;
   t6_wlb1 = 14.2;
   t6_hlb1 = 112.9;
   t6_thetalb1 = 195.4;
   t6_nlb2 = 2.33;
   t6_detlb2 = 14.7;
   t6_wlb2 = 15.6;
   t6_hlb2 = 43.6;
   t6_thetalb2 = 278.2;

  // loop
   t7_nli = 1.907;
   t7_rli = 80.;
   t7_wli = 15.;
   t7_detthetali = 30.0;
   t7_thetali = 40.0;


class YMW16Factory(FieldFactory):
    FIELD_CLASS = YMW16
    DEFAULT_PARAMETERS = { }
    PRIORS = {}
'''