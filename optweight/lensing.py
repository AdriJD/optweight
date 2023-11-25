'''
Simple wrapper around routines in lenspyx.
'''
import numpy as np
import lenspyx

from optweight import sht, alm_c_utils, mat_utils

class LensAlm():
    '''
    Expose lensing and adjoint lensing operations.

    Parameters
    ----------
    plm : (2, nelem_phi) or (1, nelem_phi) complex array
        Phi (and Omega) spherical harmonic coefficients.
    ainfo_lens : pixell.curvedsky.alm_info object
        Metainfo of plm coefficients.
    ainfo : pixell.curvedsky.alm_info object
        Metainfo of sky alms to be lensed.
    epsilon : float, optional
        Precision of lenspyx's remapping operation.
    inplace : bool, optional
        If set, perform lensing operations inplace.

    Attributes
    ----------
    ainfo : pixell.curvedsky.alm_info object
        Metainfo of sky alms to be lensed.    
    geom : lenspyx.remapping.utils_geom.Geom object
        Metainfo of internal map object.
    deflection : lenspyx.remapping.deflection.deflection object
        Deflection field object.
    nthreads : int
        Number of threads used.
    inplace : bool
        Whether lensing and adjoint lensing operations will be inplace.
    '''
    
    def __init__(self, plm, ainfo_lens, ainfo, epsilon=1e-6, inplace=False):

        lmax_lens = ainfo_lens.lmax
        
        ells = np.arange(lmax_lens + 1)
        p2d = np.sqrt(ells * (ells + 1))

        # Determine whether we just have phi or also curl.
        plm = mat_utils.atleast_nd(plm, 2)
        nphi = plm.shape[0]

        # Lenspyx needs double precision.
        plm = plm.astype(np.complex128, copy=False)
        
        # Convert phi and Omega (if present) to spin-1 deflection fields.
        dlm = alm_c_utils.lmul(plm, p2d, ainfo_lens, inplace=False)
        mmax = None
        
        self.nthreads = sht.get_nthreads()
        self.geom = lenspyx.lensing.get_geom(
            ('gl', {'lmax' : lmax_lens}))
        self.deflection = lenspyx.remapping.deflection(
            self.geom, dlm[0], mmax, numthreads=self.nthreads,
            dclm=dlm[1] if nphi == 2 else None, epsilon=epsilon)
        self.ainfo = ainfo
        self.inplace = inplace        

    @staticmethod
    def get_slices(npol):
        '''
        Return slices into first axes of alm-like array for Stokes
        I and E,B based on number of elements.

        Parameters
        ----------
        npol : int
            Number of elements in first axis of alm array.

        Returns
        -------
        tslice : slice, None
            Slice for T coefficients.
        pslice : slice, None
            Slice of E, B coefficients.

        Raises
        ------
        ValueError
            If npol is not 1, 2 or 3.
        '''
        
        tslice, pslice = None, None
        if npol == 1:
            tslice = slice(0, 1)
        elif npol == 2:
            pslice = slice(0, 2)
        elif npol == 3:
            tslice = slice(0, 1)
            pslice = slice(1, 3)
        else:
            raise ValueError(f'{npol=} not supported')
            
        return tslice, pslice
            
    def lens(self, alm):
        '''
        Compute the lensing operation: alm' = L alm.

        Parameters
        ----------
        alm : (1, nelem) or (2, nelem) or (3, nelem) complex array
            T or EB or TEB spherical harmonic coefficients.

        Returns
        -------
        out : (1, nelem) or (2, nelem) or (3, nelem) complex array
            Lensed input alms.

        Raises
        ------
        ValueError
            If alm shape[0] is not 1, 2 or 3.
            If alm shape does not match ainfo used to init the class.
        '''

        alm = mat_utils.atleast_nd(alm, 2)
        npol = alm.shape[0]

        if not npol in [1, 2, 3]:
            raise ValueError(f'{alm.shape[0]=} not in [1, 2, 3]')
        
        if alm.shape[-1] != self.ainfo.nelem:
            raise ValueError(f'{alm.shape[-1]=} != {self.ainfo.nelem=}')

        tslice, pslice = self.get_slices(npol)

        idtype = alm.dtype        
        if self.inplace:
            # Lenspyx needs double precision.
            if idtype == np.complex64:
                alm_lens = alm.astype(np.complex128, copy=True)                
            else:
                alm_lens = alm
        else:
            alm_lens = alm.astype(np.complex128, copy=True)

        mmax = None
        backwards = False
            
        if tslice is not None:
            omap = self.deflection.gclm2lenmap(
                alm_lens[tslice], mmax, 0, backwards, polrot=True, ptg=None)
            self.geom.adjoint_synthesis(
                omap, 0, self.ainfo.lmax, self.ainfo.lmax, apply_weights=True,
                nthreads=self.nthreads, alm=alm_lens[tslice])

        if pslice is not None:
            omap = self.deflection.gclm2lenmap(
                alm_lens[pslice], mmax, 2, backwards, polrot=True, ptg=None)
            self.geom.adjoint_synthesis(
                omap, 2, self.ainfo.lmax, self.ainfo.lmax, apply_weights=True,
                nthreads=self.nthreads, alm=alm_lens[pslice])

        alm_lens = alm_lens.astype(idtype, copy=False)
        if self.inplace and idtype == np.complex64:
            # In this case we kept the original alm array.
            alm[:] = alm_lens
            alm_lens = alm            
            
        return alm_lens
            
    def lens_adjoint(self, alm):
        '''
        Compute the adjoint lensing operation: alm' = L^H alm.

        Parameters
        ----------
        alm : (1, nelem) or (2, nelem) or (3, nelem) complex array
            T or EB or TEB spherical harmonic coefficients.

        Returns
        -------
        out : (1, nelem) or (2, nelem) or (3, nelem) complex array
            Adjoint-lensed input alms.

        Raises
        ------
        ValueError
            If alm shape[0] is not 1, 2 or 3.
            If alm shape does not match ainfo used to init the class.        
        '''

        alm = mat_utils.atleast_nd(alm, 2)
        npol = alm.shape[0]

        if not npol in [1, 2, 3]:
            raise ValueError(f'{alm.shape[0]=} not in [1, 2, 3]')
        
        if alm.shape[-1] != self.ainfo.nelem:
            raise ValueError(f'{alm.shape[-1]=} != {self.ainfo.nelem=}')

        tslice, pslice = self.get_slices(npol)

        idtype = alm.dtype        
        if self.inplace:
            # Lenspyx needs double precision.
            if idtype == np.complex64:
                alm_lens_adj = alm.astype(np.complex128, copy=True)                
            else:
                alm_lens_adj = alm
        else:
            alm_lens_adj = alm.astype(np.complex128, copy=True)

        mmax = None
        lmax_out = self.ainfo.lmax
                    
        if tslice is not None:
            self.deflection.lensgclm(
                alm[tslice], mmax, 0, lmax_out, mmax, alm_lens_adj[tslice],
                backwards=True, nomagn=False, polrot=True,
                out_sht_mode='STANDARD')

        if pslice is not None:
            self.deflection.lensgclm(
                alm[pslice], mmax, 2, lmax_out, mmax, alm_lens_adj[pslice],
                backwards=True, nomagn=False, polrot=True,
                out_sht_mode='STANDARD')

        alm_lens_adj = alm_lens_adj.astype(idtype, copy=False)
        if self.inplace and idtype == np.complex64:
            # In this case we kept the original alm array.
            alm[:] = alm_lens_adj
            alm_lens_adj = alm            
                        
        return alm_lens_adj
