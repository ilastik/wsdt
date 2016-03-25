import numpy
import vigra
import networkx as nx

# This code was adapted from the version in Timo's fork of vigra.
def wsDtSegmentation(pmap,
                     pmin,
                     minMembraneSize,
                     minSegmentSize,
                     sigmaMinima,
                     sigmaWeights,
                     groupSeeds=True,
                     out_debug_image_dict=None):
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
    
    If 'out_debug_image_dict' is not None, it must be a dict, and this function
    will save intermediate results to the dict as vigra.ChunkedArrayCompressed objects.
    """
    assert out_debug_image_dict is None or isinstance(out_debug_image_dict, dict)
    # FIXME: This shouldn't be...
    assert type(pmap) is numpy.ndarray, \
        "Make sure that pmap is a plain numpy array, instead of: " + str(type(pmap))

    # assert that pmap is 2d or 3d
    assert pmap.ndim in (2,3), "Input must be 2D or 3D.  shape={}".format( pmap.shape )

    distance_to_membrane = getSignedDt(pmap, pmin, minMembraneSize, out_debug_image_dict)
    binary_seeds = getDtBinarySeeds(distance_to_membrane, sigmaMinima, out_debug_image_dict)

    if groupSeeds:
        seedsLabeled = group_seeds_by_distance( binary_seeds, distance_to_membrane )
    else:
        seedsLabeled = vigra.analysis.labelMultiArrayWithBackground(binary_seeds.view(numpy.uint8))

    del binary_seeds
    save_debug_image('seeds', seedsLabeled, out_debug_image_dict)

    if sigmaWeights != 0.0:
        vigra.filters.gaussianSmoothing(distance_to_membrane, sigmaWeights, out=distance_to_membrane)
        save_debug_image('smoothed DT for watershed', distance_to_membrane, out_debug_image_dict)

    distance_to_membrane[:] *= -1
    iterativeWsInplace(distance_to_membrane, seedsLabeled, minSegmentSize, out_debug_image_dict)
    return seedsLabeled

def save_debug_image( name, image, out_debug_image_dict ):
    if out_debug_image_dict is None:
        return
    
    if hasattr(image, 'axistags'):
        axistags=image.axistags
    else:
        axistags = None

    out_debug_image_dict[name] = vigra.ChunkedArrayCompressed(image.shape, dtype=image.dtype, axistags=axistags)
    out_debug_image_dict[name][:] = image

def localMaximaND(image, *args, **kwargs):
    assert image.ndim in (2,3), \
        "Unsupported dimensionality: {}".format( image.ndim )
    if image.ndim == 2:
        return vigra.analysis.localMaxima(image, *args, **kwargs)
    if image.ndim == 3:
        return vigra.analysis.localMaxima3D(image, *args, **kwargs)

# get the signed distance transform of pmap
def getSignedDt(pmap, pmin, minMembraneSize, out_debug_image_dict):
    """
    Performs a threshold on the given image 'pmap' > pmin, and performs
    a distance transform to the threshold border for all pixels outside the
    threshold boundaries (positive distances) and also all pixels inside
    the boundary (negative distances).
    
    The result is a signed distance transform.
    """
    # get the thresholded pmap
    binary_membranes = (pmap >= pmin).view(numpy.uint8)

    # delete small CCs
    labeled = vigra.analysis.labelMultiArrayWithBackground(binary_membranes)
    save_debug_image('thresholded membranes', labeled, out_debug_image_dict)
    del binary_membranes

    remove_wrongly_sized_connected_components(labeled, minMembraneSize, in_place=True)
    save_debug_image('filtered membranes', labeled, out_debug_image_dict)

    # perform signed dt on mask
    distance_to_membrane = vigra.filters.distanceTransform(labeled)

    # Save RAM with a sneaky trick:
    # Use distanceTransform in-place, despite the fact that the input and output don't have the same types!
    # (We can just cast labeled as a float32, since uint32 and float32 are the same size.)
    distance_to_nonmembrane = labeled.view(numpy.float32)
    vigra.filters.distanceTransform(labeled, background=False, out=distance_to_nonmembrane)
    del labeled # Delete this name, not the array

    # Combine the inner/outer distance transforms
    distance_to_nonmembrane[distance_to_nonmembrane>0] -= 1
    distance_to_membrane[:] -= distance_to_nonmembrane

    save_debug_image('distance transform', distance_to_membrane, out_debug_image_dict)
    return distance_to_membrane

# get the seeds from the signed distance transform
def getDtBinarySeeds(distance_to_membrane, sigmaMinima, out_debug_image_dict):
    # Can't work in-place: Not allowed to modify input
    dtSigned = distance_to_membrane.copy()

    if sigmaMinima != 0.0:
        dtSigned = vigra.filters.gaussianSmoothing(dtSigned, sigmaMinima, out=dtSigned)
        save_debug_image('smoothed DT for seeds', dtSigned, out_debug_image_dict)

    localMaximaND(dtSigned, allowPlateaus=True, allowAtBorder=True, marker=numpy.nan, out=dtSigned)
    seedsVolume = numpy.isnan(dtSigned)
    save_debug_image('binary seeds', seedsVolume.view(numpy.uint8), out_debug_image_dict)
    return seedsVolume

# perform watershed on weights and seeds INPLACE on the seeds
def iterativeWsInplace(weights, seedsLabeled, minSegmentSize, out_debug_image_dict):
    vigra.analysis.watershedsNew(weights, seeds=seedsLabeled, out=seedsLabeled)[0]

    if minSegmentSize:
        save_debug_image('initial watershed', seedsLabeled, out_debug_image_dict)
        remove_wrongly_sized_connected_components(seedsLabeled, minSegmentSize, in_place=True)
        vigra.analysis.watershedsNew(weights, seeds=seedsLabeled, out=seedsLabeled)[0]

def vigra_bincount(labels):
    """
    A RAM-efficient implementation of numpy.bincount() when you're dealing with uint32 labels.
    If your data isn't int64, numpy.bincount() will copy it internally -- a huge RAM overhead.
    (This implementation may also need to make a copy, but it prefers uint32, not int64.)
    """
    labels = labels.astype(numpy.uint32, copy=False)
    labels = numpy.ravel(labels, order='K').reshape((-1, 1), order='A')
    # We don't care what the 'image' parameter is, but we have to give something
    image = labels.view(numpy.float32)
    counts = vigra.analysis.extractRegionFeatures(image, labels, ['Count'])['Count']
    return counts.astype(numpy.int64)

def remove_wrongly_sized_connected_components(a, min_size, max_size=None, in_place=False, bin_out=False):
    original_dtype = a.dtype

    if not in_place:
        a = a.copy()
    if min_size == 0 and (max_size is None or max_size > numpy.prod(a.shape)): # shortcut for efficiency
        if (bin_out):
            numpy.place(a,a,1)
        return a

    component_sizes = vigra_bincount(a)
    bad_sizes = component_sizes < min_size
    if max_size is not None:
        numpy.logical_or( bad_sizes, component_sizes > max_size, out=bad_sizes )
    del component_sizes

    bad_locations = bad_sizes[a]
    a[bad_locations] = 0
    del bad_locations
    if (bin_out):
        # Replace non-zero values with 1
        numpy.place(a,a,1)
    return numpy.asarray(a, dtype=original_dtype)

def group_seeds_by_distance(binary_seeds, distance_to_membrane):
    """
    Label seeds in groups, such that every seed in each group is closer to at
    least one other seed in its group than it is to the nearest membrane.
    
    Warning: The RAM needed by this function is proportional to N**2,
             where N is the number of seed points in the image.
             For example, for 50,000 seed points, this function needs more than 10 GB of RAM.
             Consider breaking your image into blocks and processing them sequentially.
    
    Parameters
    ----------
    binary_seeds
        A boolean image indicating where the seeds are
    
    distance_to_membrane
        A float32 image of distances to the membranes

    Returns a label image.
    """
    seed_locations = nonzero_coord_array(binary_seeds)
    assert seed_locations.shape[1] == binary_seeds.ndim
    num_seeds = seed_locations.shape[0]
    
    # Save RAM: shrink the dtype if possible
    if seed_locations.max() < numpy.sqrt(2**31):
        seed_locations = seed_locations.astype( numpy.int32 )

    # Compute the distance of each seed to all other seeds
    # This matrix might be huge (see warning above).
    pairwise_distances = pairwise_euclidean_distances(seed_locations)
    del seed_locations

    # Extract the distance of each seed point to the nearest membrane
    point_distances_to_membrane = distance_to_membrane[binary_seeds]
    
    # Find the seed pairs that are closer to each other either of them is to a membrane.
    close_pairs     = (pairwise_distances < point_distances_to_membrane[:, None])
    close_pairs[:] &= (pairwise_distances < point_distances_to_membrane[None, :])

    # Delete these big arrays now that we're done with them
    del pairwise_distances
    del point_distances_to_membrane

    # Create a graph of the seed points containing only the connections between 'close' seeds as found above.
    # (Note that self->self edges are included in this graph, since that distance is 0.0)
    seed_graph = nx.Graph( iter(nonzero_coord_array(close_pairs)) )
    seed_labels = numpy.zeros( (num_seeds,), dtype=numpy.uint32 )

    # Find the connected components in the graph, and give each CC a unique ID, starting at 1.
    for group_label, grouped_seed_indexes in enumerate(nx.connected_components(seed_graph), start=1):
        for seed_index in grouped_seed_indexes:
            seed_labels[seed_index] = group_label

    # Apply the new labels to the original image
    labeled_seed_img = numpy.zeros( binary_seeds.shape, dtype=numpy.uint32 )
    labeled_seed_img[binary_seeds] = seed_labels
    return labeled_seed_img
    
def pairwise_euclidean_distances( coord_array ):
    """
    For all coordinates in the given array of shape (N, DIM),
    return a symmetric array of shape (N,N) of the distances
    of each item to all others.
    """
    assert numpy.issubdtype(coord_array.dtype, numpy.signedinteger), \
        "The coordinate array dtype must be signed, and large enough "\
        "to hold the square of the maximum coordinate."

    num_points = len(coord_array)
    ndim = coord_array.shape[-1]
    subtracted = numpy.ndarray( (num_points, num_points, ndim), dtype=numpy.float32 )
    for i in range(coord_array.shape[-1]):
        numpy.subtract.outer(coord_array[...,i], coord_array[...,i], out=subtracted[...,i])
    abs_subtracted = numpy.abs(subtracted, out=subtracted)

    squared_distances = numpy.add.reduce(numpy.power(abs_subtracted, 2), axis=-1)
    distances = numpy.sqrt(squared_distances, out=squared_distances)
    assert distances.shape == (num_points, num_points)
    return distances

def nonzero_coord_array(a):
    """
    (Copied from lazyflow.utility.helpers)
    
    Equivalent to np.transpose(a.nonzero()), but much
    faster for large arrays, thanks to a little trick:
    The elements of the tuple returned by a.nonzero() share a common base,
    so we can avoid the copy that would normally be incurred when
    calling transpose() on the tuple.
    """
    base_array = a.nonzero()[0].base
    
    # This is necessary because VigraArrays have their own version
    # of nonzero(), which adds an extra base in the view chain.
    while base_array.base is not None:
        base_array = base_array.base
    return base_array
