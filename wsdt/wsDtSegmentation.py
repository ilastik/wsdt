import numpy
import vigra

# This code was adapted from the version in Timo's fork of vigra.
def wsDtSegmentation(pmap, pmin, minMembraneSize, minSegmentSize, sigmaMinima, sigmaWeights, cleanCloseSeeds=True, returnSeedsOnly=False):
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

    If 'cleanCloseSeeds' is True, multiple seed points that are clearly in the
    same neuron will be merged with a heuristik that ensures that no seeds of
    two different neurons are merged.
    """

    assert type(pmap) is numpy.ndarray, "Make sure that pmap is a plain numpy array, instead of: " + str(type(pmap))

    # assert that pmap is 2d or 3d
    assert len( pmap.shape ) == 2 or len( pmap.shape ) == 3, str( pmap.shape )

    (signed_dt, dist_to_mem) = getSignedDt(pmap, pmin, minMembraneSize)
    seeds     = getDtSeeds(signed_dt, sigmaMinima, dist_to_mem, cleanCloseSeeds)

    if returnSeedsOnly:
        return seeds

    weights   = getDtWeights(signed_dt, sigmaWeights)
    segmentation = iterativeWs(weights, seeds, minSegmentSize)

    return segmentation
    #return (segmentation, seeds, weights)

def localMinimaND(image, *args, **kwargs):
    assert image.ndim in (2,3), \
        "Unsupported dimensionality: {}".format( image.ndim )
    if image.ndim == 2:
        return vigra.analysis.localMinima(image, *args, **kwargs)
    if image.ndim == 3:
        return vigra.analysis.localMinima3D(image, *args, **kwargs)

# get the signed distance transform of pmap
def getSignedDt(pmap, pmin, minMembraneSize):

    # get the thresholded pmap
    binary_membranes = numpy.zeros_like(pmap, dtype=numpy.uint8)
    binary_membranes[pmap >= pmin] = 1

    # delete small CCs
    labeled = vigra.analysis.labelMultiArrayWithBackground(binary_membranes)
    remove_wrongly_sized_connected_components(labeled, minMembraneSize, in_place=True)

    # use cleaned binary image as mask
    big_membranes_only = numpy.zeros_like(binary_membranes, dtype = numpy.float32)
    big_membranes_only[labeled > 0] = 1.

    # perform signed dt on mask
    distance_to_membrane    = vigra.filters.distanceTransform(big_membranes_only)
    distance_to_nonmembrane = vigra.filters.distanceTransform(big_membranes_only, background=False)
    distance_to_nonmembrane[distance_to_nonmembrane>0] -= 1
    dtSigned = distance_to_membrane - distance_to_nonmembrane
    dtSigned[:] *= -1
    dtSigned[:] -= dtSigned.min()

    return (dtSigned, distance_to_membrane)


# get the seeds from the signed distance transform
def getDtSeeds(dtSigned, sigmaMinima, distance_to_membrane, cleanCloseSeeds):

    dtSignedSmoothMinima = dtSigned
    if sigmaMinima != 0.0:
        dtSignedSmoothMinima = vigra.filters.gaussianSmoothing(dtSigned, sigmaMinima)

    seedsVolume = localMinimaND(dtSignedSmoothMinima, allowPlateaus=True, allowAtBorder=True)

    if cleanCloseSeeds:
        _cleanCloseSeeds(seedsVolume, distance_to_membrane)

    seedsLabeled = vigra.analysis.labelMultiArrayWithBackground(seedsVolume)
    return seedsLabeled


# get the weights from the signed distance transform
def getDtWeights(dtSigned, sigmaWeights):

    dtSignedSmoothWeights = dtSigned
    if sigmaWeights != 0.0:
        dtSignedSmoothWeights = vigra.filters.gaussianSmoothing(dtSigned, sigmaWeights)

    return dtSignedSmoothWeights


# perform watershed on weights and seeds
def iterativeWs(weights, seedsLabeled, minSegmentSize):

    segmentation = vigra.analysis.watershedsNew(weights, seeds=seedsLabeled)[0]

    if minSegmentSize:
        remove_wrongly_sized_connected_components(segmentation, minSegmentSize, in_place=True)
        segmentation = vigra.analysis.watershedsNew(weights, seeds=segmentation)[0]

    return segmentation



def vigra_bincount(labels):
    """
    A RAM-efficient implementation of numpy.bincount() when you're dealing with uint32 labels.
    If your data isn't int64, numpy.bincount() will copy it internally -- a huge RAM overhead.
    (This implementation may also need to make a copy, but it prefers uint32, not int64.)
    """
    import vigra
    import numpy as np
    labels = labels.astype(np.uint32, copy=False)
    labels = np.ravel(labels, order='K').reshape((-1, 1), order='A')
    # We don't care what the 'image' parameter is, but we have to give something
    image = labels.view(np.float32)
    counts = vigra.analysis.extractRegionFeatures(image, labels, ['Count'])['Count']
    return counts.astype(np.int64)

def remove_wrongly_sized_connected_components(a, min_size, max_size=None, in_place=False, bin_out=False):
    """
    Copied from lazyflow.operators.opFilterLabels.py
    Originally adapted from http://github.com/jni/ray/blob/develop/ray/morpho.py
    (MIT License)
    """
    original_dtype = a.dtype

    if not in_place:
        a = a.copy()
    if min_size == 0 and (max_size is None or max_size > numpy.prod(a.shape)): # shortcut for efficiency
        if (bin_out):
            numpy.place(a,a,1)
        return a

    try:
        component_sizes = vigra_bincount(a)
    except TypeError:
        # On 32-bit systems, must explicitly convert from uint32 to int
        # (This fix is just for VM testing.)
        component_sizes = numpy.bincount( numpy.asarray(a.ravel(), dtype=int) )
    bad_sizes = component_sizes < min_size
    if max_size is not None:
        numpy.logical_or( bad_sizes, component_sizes > max_size, out=bad_sizes )

    bad_locations = bad_sizes[a]
    a[bad_locations] = 0
    if (bin_out):
        # Replace non-zero values with 1
        numpy.place(a,a,1)
    return numpy.array(a, dtype=original_dtype)

def _cleanCloseSeeds(seedsVolume, distance_to_membrane):
    seeds = nonMaximumSuppressionSeeds(volumeToListOfPoints(seedsVolume), distance_to_membrane)
    seedsVolume = numpy.zeros_like(seedsVolume, dtype=numpy.uint32)
    seedsVolume[seeds.T.tolist()] = 1
    return seedsVolume

def cdist(xy1, xy2):
    # influenced by: http://stackoverflow.com/a/1871630
    # FIXME This might lead to a memory overflow for too many seeds!
    d = numpy.zeros((xy1.shape[1], xy1.shape[0], xy1.shape[0]))
    for i in numpy.arange(xy1.shape[1]):
        d[i,:,:] = numpy.square(numpy.subtract.outer(xy1[:,i], xy2[:,i]))
    d = numpy.sum(d, axis=0)
    return numpy.sqrt(d)

def findBestSeedCloserThanMembrane(seeds, distances, distanceTrafo, membraneDistance):
    """ finds the best seed of the given seeds, that is the seed with the highest value distance transformation."""
    closeSeeds = distances <= membraneDistance
    numpy.zeros_like(closeSeeds)
    # iterate over all close seeds
    maximumDistance = -numpy.inf
    mostCentralSeed = None
    for seed in seeds[closeSeeds]:
        if distanceTrafo[tuple(seed)] > maximumDistance:
            maximumDistance = distanceTrafo[tuple(seed)]
            mostCentralSeed = seed
    return mostCentralSeed


def nonMaximumSuppressionSeeds(seeds, distanceTrafo):
    """ removes all seeds that have a neigbour that is closer than the the next membrane

    seeds is a list of all seeds, distanceTrafo is array-like
    return is a list of all seeds that are relevant.

    works only for 3d
    """
    seedsCleaned = set()

    # calculate the distances from each seed to the next seeds.
    distances = cdist(seeds, seeds)
    for i in numpy.arange(len(seeds)):
        membraneDistance = distanceTrafo[tuple(seeds[i])]
        bestAlternative = findBestSeedCloserThanMembrane(seeds, distances[i,:], distanceTrafo, membraneDistance)
        seedsCleaned.add(tuple(bestAlternative))
    return numpy.array(list(seedsCleaned))


def volumeToListOfPoints(seedsVolume, threshold=0.):
    return numpy.array(numpy.where(seedsVolume > threshold)).transpose()
