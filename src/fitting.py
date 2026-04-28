import numpy as np
import pandas as pd
import scipy.interpolate
import lmfit, tqdm
import scipy

import argparse
from typing import Dict, Any, List, Tuple, Callable, Optional
import matplotlib.pyplot as plt
import interpolator, corner

import os, glob
from pathlib import Path
import logging
import tarfile
import shutil

from . import likelihoods, sed_util as util, photometry
from . import plotting

def get_split(df : pd.DataFrame, num_tasks : int) -> pd.DataFrame:
	"""split the dataframe by the task id"""
	if num_tasks != 0:
		chunks = np.array_split(df, num_tasks)
		task_id = os.getenv("SGE_TASK_ID")
		task_id = int(task_id) if task_id is not None else 1
		df_segment = chunks[task_id-1]
		return df_segment
	else:
		return df

def read_chainnames(path : str) -> dict:
	files = glob.glob(str(Path(path) / "*.npy"))
	names = [Path(file).stem for file in files]
	return np.array(names, dtype=np.int64)

def fit_mcmc(df : pd.DataFrame,
		fluxdict : dict,
		extinction_vec : np.array,
		interp : scipy.interpolate,
		logg_function : scipy.interpolate,
		use_gravz : bool,
		units : str = "fnu",
		outfile : str = None,
		fix_distance_av : bool = False,
	) -> pd.DataFrame:
	"""fit using MCMC"""
	_ = util.check_valid(df)
	fluxcols, e_fluxcols = map(list,zip(*fluxdict.values()))
	likelihood = interpolator.fit.Likelihood(interp = interp)
	chains = {}
	for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
		print(f"Gaia DR3 {df.gaia_dr3_source_id.values[i].astype(np.int64)}")
		if fix_distance_av:
			theta = np.array([12000, 0.012]) if logg_function is not None \
				else np.array([12000, 0.012, 0.6])
		else:
			theta = np.array([12000, 0.012, 100, 0.05]) if logg_function is not None \
				else np.array([12000, 0.012, 100, 0.05, 0.6])
		av_prior_mean = row.meanAV if row.meanAV != 0 else 0.001
		av_prior_std  = 0.1*row.meanAV if row.meanAV != 0 else 0.0001
		# construct the loss arguments
		loss_args = {
					 'fl' : np.asarray(row[fluxcols].values, dtype=float),
					 'e_fl' : np.asarray(row[e_fluxcols].values, dtype=float),
					 'plx_prior' : (row.parallax, row.parallax_error),
					 'av_prior' : (av_prior_mean, av_prior_std),
					 'likelihood' : likelihood,
					 'ext_vector' : extinction_vec,
					 'units' : units,
					 'logg_function' : logg_function,
					}
		if fix_distance_av:
			loss_args['fixed_distance'] = 1000.0 / row.parallax
			loss_args['fixed_av'] = av_prior_mean

		if use_gravz:
			loss_args['vg_prior'] = (row.gravz, row.gravz_error)
		chain = interpolator.fit.mcmc_fit(likelihoods.mcmc_likelihood, loss_args, theta)
		chains[df.gaia_dr3_source_id.values[i].astype(np.int64)] = chain
		if outfile is not None:
			print(f"\n\nSaving to {outfile}/{df.gaia_dr3_source_id.values[i].astype(np.int64)}.npy\n\n")
			np.save(f"{outfile}/{df.gaia_dr3_source_id.values[i].astype(np.int64)}.npy", chain)
			print(f"mean parameters: {np.mean(chain, axis=0)}")
	return df, chains

def fit_leastsq(df : pd.DataFrame, 
		fluxdict : dict, 
		lambda_eff: np.array,
		extinction_vec : np.array, 
		interp : scipy.interpolate, 
		logg_function : scipy.interpolate,
		units : str = "fnu",
		outfile : str = None,
		savesed: bool = True,
	) -> pd.DataFrame:
	"""fit a dataframe using least squares"""
	_ = util.check_valid(df)

	if savesed:
		os.makedirs("sedfigs", exist_ok=True)

	fluxcols, e_fluxcols = map(list,zip(*fluxdict.values()))	
	# perform the fitting
	params = lmfit.Parameters()
	params.add('teff', value=10000, min=2000, max=50000, vary=True)
	params.add('logg', value=8, min=7.15, max=9.0, vary=True)
	source_ids = [] ; covar = [] ; redchi = []
	teff = [] ; e_teff = [] ; logg = []; e_logg = []
	for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
		res = lmfit.minimize(
				likelihoods.leastsq_likelihood, 
				params, 
				args = (
					row[fluxcols].values * 10**(-0.4*(extinction_vec*row.meanAV)), 
					row[e_fluxcols].values, 
					row.parallax, 
					interp,
					logg_function,
					units,
				),
				method = 'leastsq',
				nan_policy = 'omit'
			)

		if savesed:
			plotting.plot_sed(
				df.gaia_dr3_source_id.values[i].astype(np.int64),
				row[fluxcols].values * 10**(-0.4*(extinction_vec*row.meanAV)), 
				row[e_fluxcols].values,
				[sub[0] for sub in fluxdict.values()],
				lambda_eff,
				res.params['teff'].value, 
				res.params['teff'].stderr,
				res.params['logg'].value,
				res.params['logg'].stderr,
				row.parallax,
				interp, logg_function,
				folder = "sedfigs"
			)

		source_ids.append(df.gaia_dr3_source_id.values[i].astype(np.int64))
		teff.append(res.params['teff'].value)	; e_teff.append(res.params['teff'].stderr)		 
		logg.append(res.params['logg'].value)	; e_logg.append(res.params['logg'].stderr) 
		try:
			covar.append(res.covar[0,1])	
		except:
			covar.append(0)
		#	 print(df.gaia_dr3_source_id.values[i].astype(np.int64), res.covar)		   
		redchi.append(res.redchi)

	data = pd.DataFrame({
		'gaia_dr3_source_id' : source_ids,
		'teff' : teff,
		'e_teff' : e_teff,
		'logg' : logg,
		'e_logg' : e_logg,
		'covar' : covar,
		'redchi' : redchi
	})

	if outfile is not None:
		print(f"\nSaving to {outfile}\n")
		data.to_parquet(f"{outfile}")

	if savesed:
		with tarfile.open("sedfigs.tar", "w") as tar:
			# arcname avoids embedding full absolute paths
			tar.add("sedfigs", arcname=os.path.basename("sedfigs"))

		if os.path.exists("sedfigs"):
			shutil.rmtree("sedfigs")

	return df, data
 

def _process_in_chunks(
		df: pd.DataFrame,
		function: Callable[[pd.DataFrame], pd.DataFrame],
		chunk_size: int = 5000,
		functionkws : dict = {}
	) -> Tuple[List[Any], List[Dict[str, Any]]]:
	"""Split df into chunks of at most chunk_size rows, process each chunk, and collect results.
	Returns:
		all_synphot: list of synphot results, one per chunk
		all_fluxdict: list of fluxdict results, one per chunk
	"""
	all_synphot = []
	all_fluxdict = []
	all_lambda_eff = []
	all_extinction = []
	n = len(df)
	if n <= chunk_size:
		synphot, fluxdict, lambda_eff, extinction_vec = function(df, **functionkws)
		all_synphot.append(synphot)
		all_fluxdict.append(fluxdict)
		all_lambda_eff.append(lambda_eff)
		all_extinction.append(extinction_vec)
	else:
		# Determine split indices
		for start in range(0, n, chunk_size):
			stop = min(start + chunk_size, n)
			chunk = df.iloc[start:stop]
			synphot, fluxdict, lambda_eff, extinction_vec = function(chunk, **functionkws)
			all_synphot.append(synphot)
			all_fluxdict.append(fluxdict)
			all_lambda_eff.append(lambda_eff)
			all_extinction.append(extinction_vec)
	return pd.concat(all_synphot), all_fluxdict[0], all_lambda_eff[0], all_extinction[0]

def pipeline(
		df : pd.DataFrame,
		systems : list[str],
		photometry_func : Callable[[pd.DataFrame], pd.DataFrame],
		logg_function = None,
		mode : str = 'leastsq',
		fixedhe : float = None,
		source_id : str = 'wd_source_id',
		ra : str = 'wd_ra',
		dec : str = 'wd_dec',
		parallax : str = 'wd_parallax',
		parallax_error : str = 'wd_parallax_error',
		meanAV : str = 'wd_meanAV',
		gravz : str = None,
		gravz_error : str = None,
		outfile : str = None,
		numtasks : int = None,
		fix_distance_av : bool = False,
		prebuilt_fluxes : Optional[Path] = None,
	):
	"""run the pipeline and return either a list of chains or the dataframe"""
	assert mode in ['leastsq', 'mcmc'], "Invalid fitting mode!"""

	df = util.extract_data(
		df, 
		source_id, 
		ra, 
		dec, 
		parallax, 
		parallax_error, 
		meanAV, 
		gravz, 
		gravz_error
	)

	units = "fnu"
	use_gravz = True if (gravz is not None) and (gravz_error is not None) else False
	fkws = {'systems' : systems, 'use_gravz' : use_gravz,}

	synphot, fluxdict, lambda_eff, extinction_vec = _process_in_chunks(
		df, photometry_func, chunk_size = 5000, functionkws = fkws
	)
	interp, _= util.make_interpolator(
		util.convert_names(list(fluxdict.keys())), units = units, fixedhe=fixedhe
	)
	pyphot_bands = util.convert_names(list(fluxdict.keys()))

	# drop rows which return zero photometry
	synphot = synphot.dropna(subset=[i for sub in fluxdict.values() for i in sub], thresh=4).reset_index()
	logging.info(f"In total, {len(synphot)} objects with at least two photometric bands")

	if mode == 'leastsq':
		return fit_leastsq(
				synphot, fluxdict, lambda_eff, extinction_vec, interp, logg_function, units = units, outfile = outfile)
	elif mode == 'mcmc':
		return fit_mcmc(synphot, fluxdict, extinction_vec, interp, logg_function, units = units, use_gravz = use_gravz, outfile = outfile, fix_distance_av = fix_distance_av)
	else:
		raise "Invalid input!!"

def main():
	logging.basicConfig(
		level=logging.INFO, # Sets the minimum level to capture
		format='%(asctime)s - %(levelname)s - %(message)s' # Customizes output
	)
	parser = argparse.ArgumentParser(
		description="Fit white dwarf SEDs using least-squares or MCMC.",
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument('inpath',  type=str, help='Path to input parquet file')
	parser.add_argument('outpath', type=str, help='Output parquet file (leastsq) or directory (mcmc)')
	parser.add_argument('mode',    type=str, choices=['leastsq', 'mcmc'], help='Fitting method')
	parser.add_argument('--systems', type=str, default='gaia,sdss',
						help='Comma-separated photometric systems, e.g. gaia,sdss,panstarrs (default: gaia,sdss)')
	parser.add_argument('--sourceid',		 type=str, default='gaia_dr3_source_id')
	parser.add_argument('--ra',				 type=str, default='ra')
	parser.add_argument('--dec',			 type=str, default='dec')
	parser.add_argument('--parallax',		 type=str, default='parallax')
	parser.add_argument('--parallax_error',  type=str, default='parallax_error')
	parser.add_argument('--meanAV',			 type=str, default='meanAV')
	parser.add_argument('--gravz',			 type=str, default=None)
	parser.add_argument('--gravz_error',	 type=str, default=None)
	parser.add_argument('--fixedhe',		 type=float, default=None,
						help='Fixed He abundance (None = pure-H DA atmosphere)')
	parser.add_argument('--fix_distance_av', action='store_true', default=False,
						help='Fix distance and AV at prior means rather than sampling them in MCMC')
	args = parser.parse_args()

	systems = [s.strip() for s in args.systems.split(',')]
	
	if ".csv" in args.inpath:
		dataframe = pd.read_csv(args.inpath)
	else:
		dataframe = pd.read_parquet(args.inpath)
		
	if args.mode == 'mcmc':	
		logg_function = util.get_logg_function()
	else:
		logg_function = util.get_actual_logg_function()

	df, results = pipeline(
		dataframe, systems, photometry.process_dataframe, logg_function,
		mode=args.mode, fixedhe=args.fixedhe,
		source_id=args.sourceid, ra=args.ra, dec=args.dec,
		parallax=args.parallax, parallax_error=args.parallax_error,
		meanAV=args.meanAV, gravz=args.gravz, gravz_error=args.gravz_error,
		fix_distance_av=args.fix_distance_av, outfile=args.outpath,
	)
	if args.mode == 'leastsq':
		results.to_parquet(args.outpath, index=False)


if __name__ == "__main__":
	main()
