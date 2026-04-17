
from enum import Enum


# ----------------------------------------------------------------------------
# class ResamplingMethod
# ----------------------------------------------------------------------------
class ResamplingMethod(str, Enum):

    '''
    This class represents GDAL's valid resampling methods.  GDAL should
    provide a class like this, but it does not.  This allows us to enforce
    these values at run time.
    '''
    AVERAGE = 'average'
    BILINEAR = 'bilinear'
    CUBIC = 'cubic'
    CUBICSPLINE = 'cubicspline'
    LANCZOS = 'lanczos'
    MODE = 'mode'
    NEAREST = 'nearest'
    RMS = 'rms'
