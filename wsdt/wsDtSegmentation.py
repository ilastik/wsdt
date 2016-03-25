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
                     cleanCloseSeeds=True,
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

    If 'cleanCloseSeeds' is True, multiple seed points that are clearly in the
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

    (signed_dt, dist_to_mem) = getSignedDt(pmap, pmin, minMembraneSize, out_debug_image_dict)

    if cleanCloseSeeds:
        binary_seeds = getDtBinarySeeds(signed_dt, sigmaMinima, out_debug_image_dict)
        seedsLabeled = group_seeds_by_distance( binary_seeds, dist_to_mem )
    else:
        del dist_to_mem
        binary_seeds = getDtBinarySeeds(signed_dt, sigmaMinima, out_debug_image_dict)
        seedsLabeled = vigra.analysis.labelMultiArrayWithBackground(binary_seeds)

    del binary_seeds
    save_debug_image('seeds', seedsLabeled, out_debug_image_dict)

    if sigmaWeights != 0.0:
        vigra.filters.gaussianSmoothing(signed_dt, sigmaWeights, out=signed_dt)
        save_debug_image('smoothed DT for watershed', signed_dt, out_debug_image_dict)

    iterativeWsInplace(signed_dt, seedsLabeled, minSegmentSize, out_debug_image_dict)
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

def localMinimaND(image, *args, **kwargs):
    assert image.ndim in (2,3), \
        "Unsupported dimensionality: {}".format( image.ndim )
    if image.ndim == 2:
        return vigra.analysis.localMinima(image, *args, **kwargs)
    if image.ndim == 3:
        return vigra.analysis.localMinima3D(image, *args, **kwargs)

# get the signed distance transform of pmap
def getSignedDt(pmap, pmin, minMembraneSize, out_debug_image_dict):
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

    # Combine distance transforms
    distance_to_nonmembrane[distance_to_nonmembrane>0] -= 1

    # Perform this calculation, but in-place to save RAM
    # dtSigned = distance_to_membrane.max() - distance_to_membrane + distance_to_nonmembrane

    dtSigned = distance_to_nonmembrane
    dtSigned[:] -= distance_to_membrane
    dtSigned[:] += distance_to_membrane.max()

    save_debug_image('distance transform', distance_to_nonmembrane, out_debug_image_dict)
    return (dtSigned, distance_to_membrane)

# get the seeds from the signed distance transform
def getDtBinarySeeds(dtSigned, sigmaMinima, out_debug_image_dict):
    # Can't work in-place: Not allowed to modify input
    dtSigned = dtSigned.copy()

    if sigmaMinima != 0.0:
        dtSigned = vigra.filters.gaussianSmoothing(dtSigned, sigmaMinima, out=dtSigned)
        save_debug_image('smoothed DT for seeds', dtSigned, out_debug_image_dict)

    localMinimaND(dtSigned, allowPlateaus=True, allowAtBorder=True, marker=numpy.nan, out=dtSigned)
    seedsVolume = numpy.isnan(dtSigned).view(numpy.uint8)
    save_debug_image('binary seeds', dtSigned, out_debug_image_dict)
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
    import vigra
    import numpy as np
    labels = labels.astype(np.uint32, copy=False)
    labels = np.ravel(labels, order='K').reshape((-1, 1), order='A')
    # We don't care what the 'image' parameter is, but we have to give something
    image = labels.view(np.float32)
    counts = vigra.analysis.extractRegionFeatures(image, labels, ['Count'])['Count']
    return counts.astype(np.int64)

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

    Returns a label image.
    """
    seed_locations = nonzero_coord_array(binary_seeds)
    assert seed_locations.shape[1] == binary_seeds.ndim
    num_seeds = seed_locations.shape[0]

    # Save some RAM by shrinking the dtype now
    if seed_locations.max() < 1<<15:
        seed_locations = seed_locations.astype(numpy.int16)
    else:
        seed_locations = seed_locations.astype(numpy.int32)

    # Compute the distance of each seed to all other seeds
    distances = pairwise_euclidean_distances(seed_locations)

    # Extract the distance of each seed to the nearest membrane
    point_distances_to_membrane = distance_to_membrane[tuple(seed_locations.transpose())]
    
    # Find the seed pairs that are closer to each other than they are to a membrane
    valid_edges = numpy.ones( distances.shape, dtype=numpy.bool_ )
    valid_edges = numpy.logical_and( valid_edges, ( distances < point_distances_to_membrane[:, None] ), out=valid_edges )
    valid_edges = numpy.logical_and( valid_edges, ( distances < point_distances_to_membrane[None, :] ), out=valid_edges )

    # Create a graph where each edge is a valid pair as determined above.
    # (Note that self->self edges are included in this graph, since that distance is 0.0)
    seed_graph = nx.Graph( iter(nonzero_coord_array(valid_edges)) )
    seed_labels = numpy.zeros( (num_seeds,), dtype=numpy.uint32 )

    # Find the connected components in the graph, and give each CC a unique ID, starting at 1.
    for group_label, grouped_seed_indexes in enumerate(nx.connected_components(seed_graph), start=1):
        for seed_index in grouped_seed_indexes:
            seed_labels[seed_index] = group_label

    # Apply the new labels to the original image
    labeled_seed_img = numpy.zeros( binary_seeds.shape, dtype=numpy.uint32 )
    labeled_seed_img[tuple(seed_locations.transpose())] = seed_labels
    return labeled_seed_img
    
def pairwise_euclidean_distances( coord_array ):
    """
    For all coordinates in the given array of shape (N, DIM),
    return a symmetric array of shape (N,N) of the distances
    of each item to all others.
    """
    num_points = len(coord_array)
    ndim = coord_array.shape[-1]
    subtracted = numpy.ndarray( (num_points, num_points, ndim), dtype=numpy.float32 )
    for i in range(coord_array.shape[-1]):
        subtracted[...,i] = numpy.subtract.outer(coord_array[...,i], coord_array[...,i])
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
