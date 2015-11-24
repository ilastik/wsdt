import numpy
from vigra import analysis, filters

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

    # assert that pmap is 2d or 3d
    assert len( pmap.shape ) == 2 or len( pmap.shape ) == 3

    if len( pmap.shape ) == 3:
        print "I am 3d"
        (signed_dt, dist_to_mem) = getSignedDt(pmap, pmin, minMembraneSize)
        seeds     = getDtSeeds(signed_dt, sigmaMinima, dist_to_mem, cleanCloseSeeds)

    else:
        print "I am 2d"
        (signed_dt, dist_to_mem) = getSignedDt_2d(pmap, pmin, minMembraneSize)
        seeds     = getDtSeeds_2d(signed_dt, sigmaMinima, dist_to_mem, cleanCloseSeeds)

    if returnSeedsOnly:
        return seeds

    print "Weights"
    weights   = getDtWeights(signed_dt, sigmaWeights)

    if len( pmap.shape ) == 3:
        segmentation = iterativeWs(weights, seeds, minSegmentSize)
    else:
        segmentation = iterativeWs_2d(weights, seeds, minSegmentSize)

    #return segmentation
    return (segmentation, seeds, weights)


# get the signed distance transform of pmap
def getSignedDt(pmap, pmin, minMembraneSize):

    # get the thresholded pmap
    binary_membranes = numpy.zeros_like(pmap, dtype=numpy.uint8)
    binary_membranes[pmap >= pmin] = 1

    # delete small CCs
    labeled = analysis.labelVolumeWithBackground(binary_membranes)
    remove_wrongly_sized_connected_components(labeled, minMembraneSize, in_place=True)

    # use cleaned binary image as mask
    big_membranes_only = numpy.zeros_like(binary_membranes, dtype = numpy.float32)
    big_membranes_only[labeled > 0] = 1.

    # perform signed dt on mask
    distance_to_membrane = numpy.array( filters.distanceTransform3D(big_membranes_only) )
    distance_to_nonmembrane = numpy.array( filters.distanceTransform3D(big_membranes_only, background=False) )
    distance_to_nonmembrane[distance_to_nonmembrane>0] -= 1
    dtSigned = distance_to_membrane - distance_to_nonmembrane
    dtSigned[:] *= -1
    dtSigned[:] -= dtSigned.min()

    return (dtSigned, distance_to_membrane)


# get the signed distance transform of pmap (2d)
def getSignedDt_2d(pmap, pmin, minMembraneSize):

    # get the thresholded pmap
    binary_membranes = numpy.zeros_like(pmap, dtype=numpy.uint8)
    binary_membranes[pmap >= pmin] = 1

    # delete small CCs
    labeled = analysis.labelImageWithBackground(binary_membranes)
    remove_wrongly_sized_connected_components(labeled, minMembraneSize, in_place=True)

    # use cleaned binary image as mask
    big_membranes_only = numpy.zeros_like(binary_membranes, dtype = numpy.float32)
    big_membranes_only[labeled > 0] = 1.

    # perform signed dt on mask
    distance_to_membrane    = numpy.array( filters.distanceTransform2D(big_membranes_only) )
    distance_to_nonmembrane = numpy.array( filters.distanceTransform2D(big_membranes_only, background=False) )
    distance_to_nonmembrane[distance_to_nonmembrane>0] -= 1
    dtSigned = distance_to_membrane - distance_to_nonmembrane
    dtSigned[:] *= -1
    dtSigned[:] -= dtSigned.min()

    return (dtSigned, distance_to_membrane)


# get the seeds from the signed distance transform
def getDtSeeds(dtSigned, sigmaMinima, distance_to_membrane, cleanCloseSeeds):

    dtSignedSmoothMinima = dtSigned.copy()
    if sigmaMinima != 0.0:
        dtSignedSmoothMinima = filters.gaussianSmoothing(dtSigned, sigmaMinima)

    seedsVolume = analysis.localMinima3D(dtSignedSmoothMinima, neighborhood=26, allowPlateaus=True, allowAtBorder=True)

    if cleanCloseSeeds:
        _cleanCloseSeeds(seedsVolume, distance_to_membrane)

    seedsLabeled = analysis.labelVolumeWithBackground(seedsVolume)
    return seedsLabeled


# get the seeds from the signed distance transform (2d)
def getDtSeeds_2d(dtSigned, sigmaMinima, distance_to_membrane, cleanCloseSeeds):

    dtSignedSmoothMinima = dtSigned.copy()
    if sigmaMinima != 0.0:
        dtSignedSmoothMinima = filters.gaussianSmoothing(dtSigned, sigmaMinima)

    seedsArea = analysis.localMinima(dtSignedSmoothMinima, neighborhood=8, allowPlateaus=True, allowAtBorder=True)

    if cleanCloseSeeds:
        _cleanCloseSeeds(seedsArea, distance_to_membrane)

    seedsLabeled = analysis.labelImageWithBackground(seedsArea)
    return seedsLabeled


# get the weights from the signed distance transform
def getDtWeights(dtSigned, sigmaWeights):

    dtSignedSmoothWeights = dtSigned.copy()
    if sigmaWeights != 0.0:
        dtSignedSmoothWeights = filters.gaussianSmoothing(dtSigned, sigmaWeights)

    return dtSignedSmoothWeights


# perform watershed on weights and seeds
def iterativeWs(weights, seedsLabeled, minSegmentSize):

    segmentation = analysis.watershedsNew(weights, seeds=seedsLabeled, neighborhood=26)[0]

    if minSegmentSize:
        remove_wrongly_sized_connected_components(segmentation, minSegmentSize, in_place=True)
        segmentation = analysis.watershedsNew(weights, seeds=segmentation, neighborhood=26)[0]

    return segmentation


# perform watershed on weights and seeds
def iterativeWs_2d(weights, seedsLabeled, minSegmentSize):

    segmentation = analysis.watershedsNew(weights, seeds=seedsLabeled, neighborhood=8)[0]

    if minSegmentSize:
        remove_wrongly_sized_connected_components(segmentation, minSegmentSize, in_place=True)
        segmentation = analysis.watershedsNew(weights, seeds=segmentation, neighborhood=8)[0]

    return segmentation


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
        component_sizes = numpy.bincount( a.ravel() )
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
