# white dwarf sed fitting tool

install using `pip install git+https://github.com/stefanarseneau/wdseds` which will install a command line tool:

```
arseneau@CAS-685COM524:~$ fit-seds -h
usage: fit-seds [-h] [--systems SYSTEMS] [--sourceid SOURCEID] [--ra RA] [--dec DEC] [--parallax PARALLAX] [--parallax_error PARALLAX_ERROR]
                [--meanAV MEANAV] [--gravz GRAVZ] [--gravz_error GRAVZ_ERROR] [--fixedhe FIXEDHE] [--fix_distance_av]
                inpath outpath {leastsq,mcmc}

Fit white dwarf SEDs using least-squares or MCMC.

positional arguments:
  inpath                Path to input parquet file
  outpath               Output parquet file (leastsq) or directory (mcmc)
  {leastsq,mcmc}        Fitting method

options:
  -h, --help            show this help message and exit
  --systems SYSTEMS     Comma-separated photometric systems, e.g. gaia,sdss,panstarrs (default: gaia,sdss)
  --sourceid SOURCEID
  --ra RA
  --dec DEC
  --parallax PARALLAX
  --parallax_error PARALLAX_ERROR
  --meanAV MEANAV
  --gravz GRAVZ
  --gravz_error GRAVZ_ERROR
  --fixedhe FIXEDHE     Fixed He abundance (None = pure-H DA atmosphere; 30 = pure H DB atmosphere)
  --skipplotting        Skip saving the plots?
  --fix_distance_av     Fix distance and AV at prior means rather than sampling them in MCMC
```

to fit the sed, run the following command:

`fit-seds sample_vincent_dz_massive.csv out.csv leastsq --systems sdss,panstarrs,skymapper --skipplotting --fixedhe=30`
