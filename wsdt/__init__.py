from .wsDtSegmentation import wsDtSegmentation, iterative_inplace_watershed, binary_seeds_from_distance_transform, group_seeds_by_distance

try:
    from wsdt._version import version as __version__
except ImportError:
    __version__ = "0.0.0dev"
