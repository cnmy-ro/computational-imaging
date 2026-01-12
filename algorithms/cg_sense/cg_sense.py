"""
CG-SENSE

TODO:
- Implement noise decorrelation
- Extend support for non-Cartesian sampling -- do density correction and gridding
"""

import numpy as np


# ---
# FFTs

def fft2c(image):
    image = np.fft.ifftshift(image, axes=(-2, -1))
    kspace = np.fft.fft2(image, axes=(-2, -1))
    kspace = np.fft.fftshift(kspace, axes=(-2, -1))
    return kspace

def ifft2c(kspace):
    kspace = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(kspace, axes=(-2, -1))
    image = np.fft.fftshift(image, axes=(-2, -1))
    return image


# ---
# Linear operators

class LinOp:

    def __init__(self):
        ...

    def __call__(self, x):
        ...

    def H(self, x):
        ...

class SENSE(LinOp):

    def __init__(self, csm, mask):
        self.csm = csm
        self.mask = mask

    def __call__(self, image):
        coil_images = self.csm * image
        kspace = fft2c(coil_images)
        kspace = self.mask * kspace
        return kspace

    def H(self, kspace):
        kspace = self.mask * kspace
        coil_images = ifft2c(kspace)
        image = np.sum(np.conj(self.csm) * coil_images, axis=0)
        return image

class IntensityCorrection(LinOp):

    def __init__(self, csm):        
        self.weight_map = 1. / (np.sqrt(np.sum(np.abs(csm)**2, axis=0) + 1e-12))

    def __call__(self, image):
        return self.weight_map * image

    def H(self, image):
        return self.weight_map * image

class DensityCorrection(LinOp):
    # TODO
    def __init__(self):
        pass

    def __call__(self, kspace):
        return kspace

    def H(self, kspace):
        return kspace


# ---
# Utils

def conjdot(a, b):
    return np.abs(np.sum(a.conj() * b))


# ---
# CG-SENSE algorithm

def cg_sense(kspace, csm, mask, eps=1e-6):
    
    # Linear operators
    E = SENSE(csm, mask)
    I = IntensityCorrection(csm)
    D = DensityCorrection()

    # Init state
    a = I(E.H(D(kspace)))  # (H, W)
    b = np.zeros_like(a)  # (H, W)
    p = a.copy()          # (H, W)
    r = a.copy()          # (H, W)

    # Precompute
    rdotr_prev = conjdot(r, r)

    # Loop
    while True:

        delta = rdotr_prev / conjdot(a, a)
        if delta < eps:
            break
        
        q = I(E.H(D(E(I(p)))))
        b += (rdotr_prev / conjdot(p, q)) * p
        r -= (rdotr_prev / conjdot(p, q)) * q
        rdotr = conjdot(r, r)
        p = r + (rdotr / rdotr_prev) * p
        
        rdotr_prev = rdotr

    v = I(b)
    return v
