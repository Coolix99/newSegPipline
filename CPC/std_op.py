import pyclesperanto_prototype as cle
import numpy as np
from scipy.stats import iqr, kurtosis

from CPC.CPC_config import *

def initialize_device(use_gpu=True):
    """
    Select the appropriate device based on the flag.
    If use_gpu is True, it will attempt to select a GPU; otherwise, it will select a CPU.
    
    Args:
    - use_gpu (bool): Whether to use GPU or not. Defaults to True.
    """
    if use_gpu:
        device = cle.select_device(dev_type='gpu')
        print("Using GPU: ", device)

def getProfile(im):
    """
    Calculates a profile of intensity distribution statistics for non-zero pixels in the image.
    This function now includes the 10th and 90th percentiles and the kurtosis of the distribution.

    Args:
    im (numpy.ndarray): The input image for which to calculate the intensity profile.

    Returns:
    numpy.ndarray: A 1D array of calculated features describing the intensity distribution.
    """
    non_zero_pixels = im[im > 0].flatten()

    if non_zero_pixels.size == 0:
        # If there are no non-zero pixels, return a profile of zeros for each statistic
        return np.zeros(8)

    # Calculate statistics
    mean_intensity = np.mean(non_zero_pixels)
    std_dev_intensity = np.std(non_zero_pixels)
    median_intensity = np.median(non_zero_pixels)
    iqr_intensity = iqr(non_zero_pixels)  # Interquartile range
    high_intensity_fraction = np.mean(non_zero_pixels > mean_intensity)  # Fraction of pixels above the mean intensity
    percentile_10th = np.percentile(non_zero_pixels, 10)  # 10th percentile
    percentile_90th = np.percentile(non_zero_pixels, 90)  # 90th percentile
    distribution_kurtosis = kurtosis(non_zero_pixels)  # Kurtosis of the intensity distribution

    # Create a profile vector containing all the statistics
    profile = np.array([
        mean_intensity, std_dev_intensity, median_intensity, 
        iqr_intensity, high_intensity_fraction, percentile_10th, 
        percentile_90th, distribution_kurtosis
    ])

    return profile

def std_scaling(image,scale,interpolate=False):
    factor=scale/target_scaling
    # Use pyclesperanto's resample method to rescale the image
    rescaled_image = cle.resample(image, factor_x=factor[2], factor_y=factor[1], factor_z=factor[0],linear_interpolation=interpolate)
    return rescaled_image

def std_filtering(image):
    image = cle.median_sphere(image, None, 1.0, 1.0, 1.0)
    #image = cle.top_hat_sphere(image, None, 15.0, 15.0, 4.0)
    return image/np.max(image)

def normalize_image_intensities(image):
    """
    Normalize the intensities of an image. Only consider pixels with intensities > 0,
    and normalize based on the 99th highest value among those pixels.

    Args:
    - image (numpy.ndarray): The input image.

    Returns:
    - numpy.ndarray: The normalized image.
    """

    # Create a mask for pixels greater than zero
    mask = image > 0
    
    # Compute the 99th percentile of the selected pixels
    percentile_99 = np.percentile(image[mask], 99)
    
    # Normalize the intensities
    # For pixels > 0, scale based on the 99th percentile; otherwise, leave them as is
    # Use np.minimum to ensure values do not exceed 1 after normalization
    normalized_image = np.zeros_like(image, dtype=np.float32)
    normalized_image[mask] = np.minimum(image[mask] / percentile_99, 1)
    
    return normalized_image

def prepareData(orig_nuclei,orig_scale):
    
    nuclei=std_scaling(orig_nuclei,orig_scale,True)
    nuclei=std_filtering(nuclei).get()
    nuclei=normalize_image_intensities(nuclei)
    profile=getProfile(nuclei)
    return nuclei,profile

def prepareExample(orig_nuclei,orig_masks,orig_scale):
    prepared_nuclei,nuclei_profile=prepareData(orig_nuclei,orig_scale)
    prepared_masks=std_scaling(orig_masks,orig_scale,False).get().astype(int)
    return prepared_nuclei,prepared_masks,nuclei_profile

def std_reverse_scaling(image,scale,interpolate=False):
    factor=target_scaling/scale
    # Use pyclesperanto's resample method to rescale the image
    rescaled_image = cle.resample(image, factor_x=factor[2], factor_y=factor[1], factor_z=factor[0],linear_interpolation=interpolate)
    return rescaled_image