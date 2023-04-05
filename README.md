# optweight

Filter 2D data (e.g. CMB or weak lensing data) on the curved sky in a statically optimal way using the preconditioned conjugate gradient method. Main use is to compute inverse-covariance weighted data or [Wiener-filtered ](https://en.wikipedia.org/wiki/Generalized_Wiener_filter) data. 

#### Features
* Noise models that can be either diagonal in the pixel domain, or diagonal in a wavelet domain to allow for sparially varying noise correlation over pixels. The signal covariance is assumed to be diagonal in the harmonic domain.
* Multiple preconditioners including the pseudo-inverse preconditioner by [Seljebotn et al., A&A 627 A98 (2019)](https://www.aanda.org/articles/aa/abs/2019/07/aa32037-17/aa32037-17.html).

### Dependencies

- Python>=3.7
- Intel MKL library
- a C compiler (tested with gcc and icc)
- [pytest](https://pypi.org/project/pytest/)
- [pixell](https://pypi.org/project/pixell/)

### Installation

Start by making sure the MKL library is loaded in your environment. On most clusters this can be achieved by loading a predefined module. On the Princeton `della` and `tiger` clusters you can use `module load intel-mkl` (see [here](https://researchcomputing.princeton.edu/faq/how-to-build-using-intel-mkl) for more information). On `NERSC` you can use `load intel` (see [here](https://docs-dev.nersc.gov/cgpu/software/math/)). Once you have loaded the module, check if the `MKLROOT` environment variable has been set (`echo $MKLROOT`).

Once the MKL environment has been set, `git clone` this repository, go into the directory and run:
```
$ pip install .
```
Consider adding the `-e` flag (`pip install -e .`) to enable automatic 
updating of code changes when developing.

Run tests:

```
$ cd tests
$ python -m pytest .
```


