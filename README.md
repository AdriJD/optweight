# optweight

![Build](https://github.com/AdriJD/optweight/actions/workflows/python-app.yml/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/AdriJD/optweight/graph/badge.svg?token=I6GNMW49E6)](https://codecov.io/gh/AdriJD/optweight)

`optweight` is a library for filtering two-dimensional data on the sphere (e.g. CMB or weak lensing data) in a statically optimal way. The main use is the computation of [Wiener-filtered](https://en.wikipedia.org/wiki/Generalized_Wiener_filter) data. Additionally, the library can be used to compute inverse-covariance weighted data and draw constrained signal realizations. 

Under the hood, the code solves a large linear system using the conjugate gradient method. Special care has been taken to precondition the linear system. This results in fast convergence even when the data include many masked pixels (e.g. due to Galactic and/or point source masks).


#### Features
* Noise models that can be either diagonal in the pixel domain, or diagonal in a wavelet domain to allow for sparially varying noise correlation over pixels. The signal covariance is assumed to be diagonal in the harmonic domain.
* Multiple preconditioners including the pseudo-inverse and multigrid preconditioners by [Seljebotn et al., A&A 627 A98 (2019)](https://www.aanda.org/articles/aa/abs/2019/07/aa32037-17/aa32037-17.html).
* Delensing using the fast lensing operations from [Reinecke, Belkner and Carron, A&A 678 A165 (2023)](https://arxiv.org/abs/2304.10431) implemented in the [lenspyx](https://github.com/carronj/lenspyx) library.

### Dependencies

- Python>=3.8,<=3.11
- Intel MKL library 
- a C compiler (tested with gcc and icc)
- [ducc0](https://gitlab.mpcdf.mpg.de/mtr/ducc)
- [pytest](https://pypi.org/project/pytest/)
- [pixell](https://pypi.org/project/pixell/)
- [lenspyx](https://github.com/carronj/lenspyx)

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


