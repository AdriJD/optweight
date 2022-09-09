# optweight

Filter 2D data (e.g. CMB or weak lensing data) on the curved sky in a statically optimal way using the preconditioned conjugate gradient method. Main use is to compute inverse-covariance weighted data or [Wiener-filtered ](https://en.wikipedia.org/wiki/Generalized_Wiener_filter) data. 

#### Features
* Noise models that can be either diagonal in the pixel domain, or diagonal in a wavelet domain to allow for sparially varying noise correlation over pixels. The signal covariance is assumed to be diagonal in the harmonic domain.
* Multiple preconditioners including the pseudo-inverse preconditioner by [Seljebotn et al., A&A 627 A98 (2019)](https://www.aanda.org/articles/aa/abs/2019/07/aa32037-17/aa32037-17.html).

### Dependencies

- Python>=3.6
- a C compiler (tested with gcc and icc)
- [pytest](https://pypi.org/project/pytest/)
- [pixell](https://pypi.org/project/pixell/)
- [enlib](https://github.com/amaurea/enlib/)

### Installation


```
$ pip install .
```

Run tests:

```
$ python -m pytest tests
```

Consider adding the `-e` flag to the `pip install` command to enable automatic 
updating of code changes when developing.
