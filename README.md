# optweight

Filter CMB maps in a statically optimal way using the preconditioned conjugate gradient method. Main use is to compute inverse-covariance weighted data or [Wiener-filtered ](https://en.wikipedia.org/wiki/Generalized_Wiener_filter) data. Makes use of the pseudo-inverse preconditioner by [Seljeboth et al., A&A 627 A98 (2019)](https://www.aanda.org/articles/aa/abs/2019/07/aa32037-17/aa32037-17.html).

### Dependencies

- Python>=3.4
- a C compiler (tested with gcc and icc)
- [pytest](https://pypi.org/project/pytest/)
- [pixell](https://pypi.org/project/pixell/)
- [enlib](https://github.com/amaurea/enlib)


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