import numpy
import vigra

import logging
logger = logging.getLogger(__name__)

from utils import log_calls, signed_distance_transform, binary_seeds_from_distance_transform, save_debug_image, iterative_inplace_watershed


@log_calls(logging.INFO)
def wsDtSegmentation(pmap, pmin, minMembraneSize, minSegmentSize, sigmaMinima, sigmaWeights, groupSeeds=True, preserve_membrane_pmaps=False, out_debug_image_dict=None, out=None):
    """A probability map 'pmap' is provided and thresholded using pmin.
    This results in a mask. Every connected component which has fewer pixel
    than 'minMembraneSize' is deleted from the mask. The mask is used to
    calculate the signed distance transformation.

    From this distance transformation the segmentation is computed using
    a seeded watershed algorithm. The seeds are placed on the local maxima
    of the distanceTransform after smoothing with 'sigmaMinima'.

    The weights of the watershed are defined by the inverse of the signed
    distance transform smoothed with 'sigmaWeights'.

    'minSegmentSize' determines how small the smallest segment in the final
    segmentation is allowed to be. If there are smaller ones the corresponding
    seeds are deleted and the watershed is done again.

    If 'groupSeeds' is True, multiple seed points that are clearly in the
    same neuron will be merged with a heuristik that ensures that no seeds of
    two different neurons are merged.

    If preserve_membrane_pmaps is True, then the pixels under the membranes
    (after thresholding) will not be replaced with a negative distance transform.
    Instead, they will be negated, and the watershed on those pixels will flow
    according to the inverted probabilities, not the distance to the threshold boundary.
    In cases of thick membranes whose probability distribution is not symmetric across
    the membrane, this will place the segment boundaries along the membrane probability
    maximum, not the geometric center.

    If 'out_debug_image_dict' is not None, it must be a dict, and this function
    will save intermediate results to the dict as vigra.ChunkedArrayCompressed objects.

    Returns: Label image, uint32.  The label values are guaranteed to be consecutive, 1..N.

    Implementation Note: This algorithm has the potential to use a lot of RAM, so this
                         code goes attempts to operate *in-place* on large arrays whenever
                         possible, and we also delete intermediate results soon
                         as possible, sometimes in the middle of a function.
    """
    assert out_debug_image_dict is None or isinstance(out_debug_image_dict, dict)
    assert isinstance(pmap, numpy.ndarray), \
        "Make sure that pmap is numpy array, instead of: " + str(type(pmap))
    assert pmap.ndim in (2,3), "Input must be 2D or 3D.  shape={}".format( pmap.shape )

    distance_to_membrane = signed_distance_transform(pmap, pmin, minMembraneSize, preserve_membrane_pmaps, out_debug_image_dict)
    binary_seeds = binary_seeds_from_distance_transform(distance_to_membrane, sigmaMinima, out_debug_image_dict)

    if groupSeeds:
        labeled_seeds = group_seeds_by_distance( binary_seeds, distance_to_membrane, out=out )
    else:
        labeled_seeds = vigra.analysis.labelMultiArrayWithBackground(binary_seeds.view(numpy.uint8), out=out)

    del binary_seeds
    save_debug_image('seeds', labeled_seeds, out_debug_image_dict)

    if sigmaWeights != 0.0:
        vigra.filters.gaussianSmoothing(distance_to_membrane, sigmaWeights, out=distance_to_membrane)
        save_debug_image('smoothed DT for watershed', distance_to_membrane, out_debug_image_dict)

    # Invert the DT: Watershed code requires seeds to be at minimums, not maximums
    distance_to_membrane[:] *= -1
    max_label = iterative_inplace_watershed(distance_to_membrane, labeled_seeds, minSegmentSize, out_debug_image_dict)
    return labeled_seeds, max_label
