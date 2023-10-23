/*
 * Apply a weight per ring for a general iso-latitude geometry.
 * 
 * Arguments
 * ---------
 * imap     : (sum(nphi)) input map, will be overwritten!
 * weight   : (ntheta) per-ring weights.
 */

void _apply_ringweight_core_sp(float *imap, const double *weight, const long *nphi,
                               const long *offsets, const long *stride, const long nrow);
void _apply_ringweight_core_dp(double *imap, const double *weight, const long *nphi,
                               const long *offsets, const long *stride, const long nrow);
