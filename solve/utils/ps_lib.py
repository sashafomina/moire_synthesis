from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

__all__ = ['read_image', 'write_image', 'angular_fourier_frequencies', 'resize']

#------------------------------------------------------------------------------

def read_image(path: Union[Path, str]) -> np.ndarray:
    '''
    Read a PNG or JPG image an array of linear RGB radiance values ∈ [0,1].
    '''
    return (np.float32(Image.open(path)) / 255)**2.2


def write_image(path: Union[Path, str], image: np.ndarray) -> None:
    '''
    Write an array of linear RGB radiance values ∈ [0,1] as a PNG or JPG image.
    '''
    Image.fromarray(np.uint8(255 * image.clip(0, 1)**(1/2.2))).save(path)

def angular_fourier_frequencies(height: int, width: int) -> np.ndarray:
    '''
    Return angluar freqeuncy components for each DFT component.
    '''
    col_freqs = np.fft.fftfreq(height)[:, None]
    row_freqs = np.fft.fftfreq(width)[None, :]
    col_freqs[0, :] = col_freqs[1, :]
    row_freqs[:, 0] = row_freqs[:, 1]
    return 2 * np.pi * np.sqrt(row_freqs**2 + col_freqs**2)


def resize(image, size):
    '''
    Convert an image to the specified size (height, width) using bicubic
    interpolation.
    '''
    return _resize_channel(image, size) if image.ndim == 2 else np.dstack([
        _resize_channel(chan, size) for chan in image.transpose(2, 0, 1)
    ])


def _resize_channel(chan, size):
    return np.asarray(Image.fromarray(chan).resize(size[::-1], Image.CUBIC))
