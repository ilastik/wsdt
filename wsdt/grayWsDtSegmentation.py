import vigra
import numpy as np

from utils import binary_seeds_from_distance_transform, iterative_inplace_watershed


def grayWsDtSegmentation(pmap,
        sigmaDt, # TODO rename this parameter
        #minMembraneSize, # TODO implement meaningful size filtering later
        minSegmentSize,
        sigmaMinima, # TODO For now we only use a single sigma
        #sigmaWeights,
        groupSeeds=False,
        #preserve_membrane_pmaps=False,
        out_debug_image_dict=None,
        out=None
        ):
    assert out_debug_image_dict is None or isinstance(out_debug_image_dict, dict)
    assert isinstance(pmap, np.ndarray), \
        "Make sure that pmap is numpy array, instead of: " + str(type(pmap))
    assert pmap.ndim in (2,3), "Input must be 2D or 3D.  shape={}".format( pmap.shape )

    # apply the grayscale distance transform
    # TODO smooth the pmap before ?!
    # TODO properly processing similar to signed_distance_transform
    distance_to_membrane = vigra.filters.multiGrayscaleDilation(pmap, sigmaDt)
    distance_to_membrane = 1. - distance_to_membrane

    binary_seeds = binary_seeds_from_distance_transform(distance_to_membrane, sigmaMinima, out_debug_image_dict)

    if groupSeeds:
        labeled_seeds = group_seeds_by_distance( binary_seeds, distance_to_membrane, out=out )
    else:
        labeled_seeds = vigra.analysis.labelMultiArrayWithBackground(binary_seeds.view('uint8'), out=out)

    del binary_seeds
    #save_debug_image('seeds', labeled_seeds, out_debug_image_dict)

    #if sigmaWeights != 0.0:
    #    vigra.filters.gaussianSmoothing(distance_to_membrane, sigmaWeights, out=distance_to_membrane)
    #    save_debug_image('smoothed DT for watershed', distance_to_membrane, out_debug_image_dict)

    # Invert the DT: Watershed code requires seeds to be at minimums, not maximums
    distance_to_membrane[:] *= -1
    max_label = iterative_inplace_watershed(distance_to_membrane, labeled_seeds, minSegmentSize, out_debug_image_dict)
    return labeled_seeds, max_label
