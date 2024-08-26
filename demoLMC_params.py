import time, sys

import numpy as np
from sedpy.observate import load_filters

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer

from prospect.models.templates import TemplateLibrary
from prospect.models import priors, sedmodel
from prospect.sources import FastStepBasis

from prospect.utils.obsutils import fix_obs
import pickle

from astropy.cosmology import WMAP9 as cosmo
catalog = pickle.load(open('/home/skleiman27/prospectordir/data/SAGAsatscatalog.p', 'rb'))

# --------------
# Model Definition
# --------------

def build_model(objid=0, add_duste=False, add_neb=False, **extras):
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object.
    :param object_redshift:
        If given, given the model redshift to this value.
    :param add_dust: (optional, default: False)
        Switch to add (fixed) parameters relevant for dust emission.
    :param add_neb: (optional, default: False)
        Switch to add (fixed) parameters relevant for nebular emission, and
        turn nebular emission on.
    :param luminosity_distance: (optional)
        If present, add a `"lumdist"` parameter to the model, and set it's
        value (in Mpc) to this.  This allows one to decouple redshift from
        distance, and fit, e.g., absolute magnitudes (by setting
        luminosity_distance to 1e-5 (10pc))
    """

    # input basic dirichlet SFH
    model_params = TemplateLibrary["dirichlet_sfh"]

    # IMF
    model_params['imf_type'] =  {'N': 1,'isfree': False,'init': 1} #1 = chabrier}

    # Fix the redshift
    # TODO: set this to whatever the galaxy's redshift is, using objid
    model_params["zred"] = {'N': 1,'isfree': False,'init': catalog['redshift'][objid]}

    # modify to increase nbins
    nbins_sfh = 10
    model_params['agebins']['N'] = nbins_sfh
    model_params['mass']['N'] = nbins_sfh
    model_params['z_fraction']['N'] = nbins_sfh-1
                                                                      
    # add redshift scaling to agebins, such that t_max = t_univ
    # note: agebins units are log(yr)
    def zred_to_agebins(zred=None,agebins=None,**extras):
        tuniv = cosmo.age(zred).value[0]*1e9
        tbinmax = (tuniv*0.9)
        agelims = [0.0,7.0] + np.linspace(8.0,np.log10(tbinmax),nbins_sfh-2).tolist() + [np.log10(tuniv)]
        agebins = np.array([agelims[:-1], agelims[1:]])
        return agebins.T
   
    alpha_sfh = 0.7  # desired Dirichlet concentration 
    alpha = np.repeat(alpha_sfh,nbins_sfh-1)
    tilde_alpha = np.array([alpha[i-1:].sum() for i in range(1,nbins_sfh)])
    model_params['agebins']['init'] = zred_to_agebins(np.atleast_1d(model_params["zred"]["init"])) 
    model_params['z_fraction']['init'] = np.array([(i-1)/float(i) for i in range(nbins_sfh, 1, -1)])
    model_params['z_fraction']['prior'] = priors.Beta(mini=0.0, maxi=1.0, alpha=tilde_alpha, beta=alpha)    

    # TODO: change this prior based on the galaxy's actual mass
    model_params['total_mass']['prior'] = priors.LogUniform(mini=1.e8, maxi=1.e11)    
    # TODO: adjust metallicity priors to be consistent with the mass; can use a mass-metallicity relation
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-1.8, maxi=0.79)                            
    
    # assume fixed dust
    model_params["dust_type"] = {'isfree': False, 'init': 2}  # Calzetti (2000) dust attenuation law
    # TODO: may want to change prior for more massive galaxies
    model_params["dust2"] = {'isfree': True, 'init':0.2, 'prior':priors.TopHat(mini=0.0, maxi=0.6)}

    if add_duste:
        # Add dust emission (with fitted dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])
        model_params['duste_qpah'] = {'isfree': False, 'init': 3.5}
        model_params['duste_gamma'] = {'isfree': False, 'init':0.01}
        model_params['duste_umin'] = {'isfree': False, 'init':1.0}


    if add_neb:
        # Add nebular emission (with fitted parameters)
        model_params.update(TemplateLibrary["nebular"])
        model_params['gas_logu']['isfree'] = True
        model_params['gas_logz']['isfree'] = True

   # Now instantiate the model using this new dictionary of parameter specifications
    model = sedmodel.SedModel(model_params)

    print(model)
    return model

# --------------
# Observational Data
# --------------

def build_obs(objid=0, phottable='demo_photometry.dat',
              luminosity_distance=None, **kwargs):
    """Load photometry from an ascii file.  Assumes the following columns:
    `objid`, `filterset`, [`mag0`,....,`magN`] where N >= 11.  The User should
    modify this function (including adding keyword arguments) to read in their
    particular data format and put it in the required dictionary.
    :param objid:
        The object id for the row of the photomotery file to use.  Integer.
        Requires that there be an `objid` column in the ascii file.
    :param phottable:
        Name (and path) of the ascii file containing the photometry.
    :param luminosity_distance: (optional)
        The Johnson 2013 data are given as AB absolute magnitudes.  They can be
        turned into apparent magnitudes by supplying a luminosity distance.
    :returns obs:
        Dictionary of observational data.
    """

    catalog = pickle.load(open('/home/skleiman27/prospectordir/data/SAGAsatscatalog.p', 'rb'))

    # TODO: decide which filters to use
    catalog['mags'][catalog['mags']>30]=np.nan
    mags = catalog['mags'][0:7,objid]
    mags_err = catalog['mags_err'][0:7,objid]
    galex = ['galex_FUV','galex_NUV']
    sdss = ['sdss_{0}0'.format(b) for b in 'grz']
    wise = ['wise_w'+n for n in '12']
    filternames = galex + sdss + wise

    # Build output dictionary.
    obs = {}
    # This is a list of sedpy filter objects.    See the
    # sedpy.observate.load_filters command for more details on its syntax.
    obs['filters'] = load_filters(filternames)
    # This is a list of maggies, converted from mags.  It should have the same
    # order as `filters` above.
    obs['maggies'] = np.squeeze(10**(-mags/2.5))
    # HACK.  You should use real flux uncertainties
    obs['maggies_unc'] = np.sqrt(mags_err**2. * (0.921034*np.exp(-0.921034*mags))**2.)
    # Here we mask out any NaNs or infs
    obs['phot_mask'] = np.isfinite(np.squeeze(mags))
    # We have no spectrum.
    obs['wavelength'] = None
    obs['spectrum'] = None

    # Add unessential bonus info.  This will be stored in output
    #obs['dmod'] = catalog[ind]['dmod']
    obs['objid'] = objid

    # This ensures all required keys are present and adds some extra useful info
    obs = fix_obs(obs)

    return obs

# --------------
# SPS Object
# --------------

def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    sps = FastStepBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    return sps

# -----------------
# Noise Model
# ------------------

def build_noise(**extras):
    return None, None

# -----------
# Everything
# ------------

def build_all(**kwargs):

    return (build_obs(**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))


if __name__ == '__main__':

    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
    parser.add_argument('--object_redshift', type=float, default=0.0,
                        help=("Redshift for the model"))
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--add_duste', action="store_true",
                        help="If set, add dust emission to the model.")
    parser.add_argument('--luminosity_distance', type=float, default=1e-5,
                        help=("Luminosity distance in Mpc. Defaults to 10pc "
                              "(for case of absolute mags)"))
    parser.add_argument('--phottable', type=str, default="demo_photometry.dat",
                        help="Names of table from which to get photometry.")
    parser.add_argument('--objid', type=int, default=0,
                        help="zero-index row number in the table to fit.")

    args = parser.parse_args()
    run_params = vars(args)
    obs, model, sps, noise = build_all(**run_params)

    run_params["sps_libraries"] = sps.ssp.libraries
    run_params["param_file"] = __file__

    print(model)

    if args.debug:
        sys.exit()

    #hfile = setup_h5(model=model, obs=obs, **run_params)
    ts = time.strftime("%y%b%d-%H.%M", time.localtime())
    hfile = "{0}_{1}_LMCresult.h5".format(args.outfile, ts)

    output = fit_model(obs, model, sps, noise, **run_params)

    print("writing to {}".format(hfile))
    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1],
                      sps=sps)

    try:
        hfile.close()
    except(AttributeError):
        pass
