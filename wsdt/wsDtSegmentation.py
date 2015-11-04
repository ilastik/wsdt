from vigra import analysis

# This code was adapted from the version in Timo's fork of vigra.
def wsDtSegmentation(pmap, pmin, minMembraneSize, minSegmentSize, sigmaMinima, sigmaWeights, cleanCloseSeeds=True):
    """A probability map 'pmap' is provided and thresholded using pmin.
    This results in a mask. Every connected component which has fewer pixel
    than 'minMembraneSize' is deleted from the mask. The mask is used to
    calculate the signed distance transformation.

    From this distance transformation the segmentation is computed using
    a seeded watershed algorithm. The seeds are placed on the local maxima
    of the distanceTrafo after smoothing with 'sigmaMinima'.

    The weights of the watershed are defined by the inverse of the signed
    distance transform smoothed with 'sigmaWeights'.

    'minSegmentSize' determines how small the smallest segment in the final
    segmentation is allowed to be. If there are smaller ones the corresponding
    seeds are deleted and the watershed is done again.

    If 'cleanCloseSeeds' is True, multiple seed points that are clearly in the
    same neuron will be merged with a heuristik that ensures that no seeds of
    two different neurons are merged.
    """

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
            if distanceTrafo[seed[0], seed[1], seed[2]] > maximumDistance:
                maximumDistance = distanceTrafo[seed[0], seed[1], seed[2]]
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
            membraneDistance = distanceTrafo[seeds[i,0], seeds[i,1], seeds[i,2]]
            bestAlternative = findBestSeedCloserThanMembrane(seeds, distances[i,:], distanceTrafo, membraneDistance)
            seedsCleaned.add(tuple(bestAlternative))
        return numpy.array(list(seedsCleaned))


    def volumeToListOfPoints(seedsVolume, threshold=0.):
        return numpy.array(numpy.where(seedsVolume > threshold)).transpose()


    # get the thresholded pmap
    binary = numpy.zeros_like(pmap, dtype=numpy.uint32)
    binary[pmap >= pmin] = 1

    # delete small CCs
    labeled = analysis.labelVolumeWithBackground(binary)
    remove_wrongly_sized_connected_components(labeled, minMembraneSize)

    # use cleaned binary image as mask
    mask = numpy.zeros_like(binary, dtype = numpy.float32)
    mask[labeled > 0] = 1.

    # perform signed dt on mask
    dt = filters.distanceTransform3D(mask)
    dtInv = filters.distanceTransform3D(mask, background=False)
    dtInv[dtInv>0] -= 1
    dtSigned = dt.max() - dt + dtInv

    dtSignedSmoothMinima = filters.gaussianSmoothing(dtSigned, sigmaMinima)
    dtSignedSmoothWeights = filters.gaussianSmoothing(dtSigned, sigmaWeights)

    seedsVolume = analysis.localMinima3D(dtSignedSmoothMinima, neighborhood=26, allowAtBorder=True)

    if cleanCloseSeeds:
        seeds = nonMaximumSuppressionSeeds(volumeToListOfPoints(seedsVolume), dt)
        seedsVolume = numpy.zeros_like(pmap, dtype=numpy.uint32)
        seedsVolume[seeds.T.tolist()] = 1

    seedsLabeled = analysis.labelVolumeWithBackground(seedsVolume)
    segmentation = analysis.watershedsNew(dtSignedSmoothWeights, seeds = seedsLabeled, neighborhood=26)[0]

    remove_wrongly_sized_connected_components(segmentation, minSegmentSize, in_place=True)

    segmentation = analysis.watershedsNew(dtSignedSmoothWeights, seeds = segmentation, neighborhood=26)[0]

    return segmentation

def remove_wrongly_sized_connected_components(a, min_size, max_size, in_place, bin_out=False):
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

