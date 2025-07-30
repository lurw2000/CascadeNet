import pywt 
import numpy as np


def sig2spec(signal, scales, wavelet="mexh"):
    """
    Use Continuous Wavelet Transform to convert a signal (1d array of time series) into a spectrogram (2d array of coefficients)
    """
    coef, freqs = pywt.cwt(
        data=signal, scales=scales, wavelet=wavelet
    )
    return coef, freqs

def spec2sig(coef, scales, wavelet="mexh", clip=False, normalize=False):
    """
    Reconstruct the time series from the spectrogram.

    Note that signal-to-spectrogram conversion is not invertible, and the reconstruction is not perfect.
    """
    mwf = pywt.ContinuousWavelet(wavelet).wavefun()
    y_0 = mwf[0][np.argmin(np.abs(mwf[1]))]

    # integrating over all scales
    r_sum = np.transpose(np.sum(np.transpose(coef)/ scales ** 0.5, axis=-1))
    reconstructed = r_sum * (1 / y_0)
    # normalize reconstructed by first clipping all negative values to 0, then mapping to [0, 1]
    if clip:
        reconstructed[reconstructed < 0] = 0
    if normalize:
        reconstructed = reconstructed / reconstructed.max()

    return reconstructed