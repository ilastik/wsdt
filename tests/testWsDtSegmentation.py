import unittest
import numpy as np
import vigra

from wsdt import wsDtSegmentation
from numpy_allocation_tracking.decorators import assert_mem_usage_factor

class TestWsDtSegmentation(unittest.TestCase):

    def _gen_input_data(self, ndim):
        assert ndim in (2,3)

        # Create a volume with 8 sections
        pmap = np.zeros( (101,)*ndim, dtype=np.float32 )

        # Three Z-planes (or Y in 2d)
        pmap[  :4,  :] = 1

        # For a threshold of 0.5, these values are all equivalent,
        # and the watershed boundary will end up in the center under default settings.
        # But if preserve_membrane_pmaps=True, then the watershed
        # boundary will end up on slice 53.
        pmap[47, :] = 0.5
        pmap[48, :] = 0.5
        pmap[49, :] = 0.5
        pmap[50, :] = 0.75
        pmap[51, :] = 0.75
        pmap[52, :] = 0.75
        pmap[53, :] = 1.0
        
        pmap[-4:,   :] = 1

        # Three Y-planes (or X in 2d)
        pmap[:,   :4] = 1

        # See note above about thes values
        pmap[:, 47] = 0.5
        pmap[:, 48] = 0.5
        pmap[:, 49] = 0.5
        pmap[:, 50] = 0.75
        pmap[:, 51] = 0.75
        pmap[:, 52] = 0.75
        pmap[:, 53] = 1.0

        pmap[:, -4:] = 1

        if ndim == 3:
            # Three X-planes
            pmap[:, :, :4] = 1
            pmap[:, :, 47:54] = 1
            pmap[:, :, -4:] = 1
        
        return pmap
        

    def test_simple_3D(self):
        pmap = self._gen_input_data(3)

        debug_results = {}
        ws_output = wsDtSegmentation(pmap, 0.5, 0, 10, 0.1, 0.1, groupSeeds=False, out_debug_image_dict=debug_results)
        seeds = debug_results['seeds'][:]
        assert seeds.max() == 8
        assert ws_output.max() == 8

        # Ensure consecutive label values
        label_list = vigra.analysis.unique(ws_output)
        assert label_list.max() == len(label_list)

        # Expect seeds at (25,25,25), (25,25,75), (25,75,25), etc...
        expected_seed_coords = list(np.ndindex((2,2,2)))
        expected_seed_coords = 50*np.array(expected_seed_coords) + 25

        #print "EXPECTED:\n", expected_seed_coords
        #print "SEEDS:\n", np.array(np.where(seeds)).transpose()

        for seed_coord in expected_seed_coords:
            assert seeds[tuple(seed_coord)], "No seed at: {}".format( seed_coord )

    def test_simple_2D(self):
        pmap = self._gen_input_data(2)

        debug_results = {}
        ws_output = wsDtSegmentation(pmap, 0.5, 0, 10, 0.1, 0.1, groupSeeds=False, out_debug_image_dict=debug_results)
        seeds = debug_results['seeds'][:]
        assert seeds.max() == 4
        assert ws_output.max() == 4

        # Ensure consecutive label values
        label_list = vigra.analysis.unique(ws_output)
        assert label_list.max() == len(label_list)

        # Expect seeds at (25,25,25), (25,25,75), (25,75,25), etc...
        expected_seed_coords = list(np.ndindex((2,2)))
        expected_seed_coords = 50*np.array(expected_seed_coords) + 25

        #print "EXPECTED:\n", expected_seed_coords
        #print "SEEDS:\n", np.array(np.where(seeds)).transpose()

        for seed_coord in expected_seed_coords:
            assert seeds[tuple(seed_coord)], "No seed at: {}".format( seed_coord )

        #vigra.impex.writeImage(ws_output, '/tmp/simple_2d_ws.tiff')
        

    def test_preserve_membrane_pmaps(self):
        pmap = self._gen_input_data(2)

        debug_results = {}
        ws_output = wsDtSegmentation(pmap, 0.5, 0, 10, 0.0, 0.0, groupSeeds=False, preserve_membrane_pmaps=True, out_debug_image_dict=debug_results)
        seeds = debug_results['seeds'][:]
        assert seeds.max() == 4
        assert ws_output.max() == 4

        # Check that pmaps were preserved in the distance transform
        distance_transform = debug_results['distance transform'][:]
        assert (distance_transform[pmap >= 0.5] == -pmap[pmap >= 0.5]).all()

        # Ensure consecutive label values
        label_list = vigra.analysis.unique(ws_output)
        assert label_list.max() == len(label_list)

        # Expect seeds at (25,25,25), (25,25,75), (25,75,25), etc...
        expected_seed_coords = list(np.ndindex((2,2)))
        expected_seed_coords = 50*np.array(expected_seed_coords) + 25

        #print "EXPECTED:\n", expected_seed_coords
        #print "SEEDS:\n", np.array(np.where(seeds)).transpose()

        for seed_coord in expected_seed_coords:
            assert seeds[tuple(seed_coord)], "No seed at: {}".format( seed_coord )

        # Due to the way our test data is constructed (it has plateaus),
        # we don't have guarantees for exactly how wide the final segments will be.
        # But we know that the upper-left segment should be larger than the lower right,
        # since the pmap maximum is off-center in that direction
        upper_left_val = ws_output[25, 25]
        lower_right_val = ws_output[75, 75]
        assert (ws_output == upper_left_val).sum() > (ws_output == lower_right_val).sum()
        
        #vigra.impex.writeImage(ws_output, '/tmp/preserved_2d.tiff')

    def test_min_segment_size(self):
        """
        Verify that small segments get removed properly.

        In this test we'll use input data that looks roughly like the following:
        
            0                101               202               303
          0 +-----------------+-----------------+-----------------+
            |                 |        |        |                 |
            |                 |                 |                 |
            |                 |                 |                 |
            |                                                     |
         50 |        w               x   y               z        |
            |                                                     |
            |                 |                 |                 |
            |                 |                 |                 |
            |                 |        |        |                 |
        101 +-----------------+-----------------+-----------------+

        The markers (wxyz) indicate where seeds will end up.
        
        The x and y segments will be too small to remain, so they'll
        be eaten up by the other segments. Nonetheless, the returned label
        image will have consecutive label values.
        """
        input_data = np.zeros((101, 303), dtype=np.float32)
        
        # Add borders
        input_data[0] = 1
        input_data[-1] = 1
        input_data[:, 0] = 1
        input_data[:, -1] = 1

        # Add vertical notches extending from the upper/lower borders
        input_data[  1:40, 101] = 1
        input_data[-40:-1, 101] = 1
        
        input_data[-10:-1, 151] = 1
        input_data[  1:10, 151] = 1

        input_data[  1:40, 202] = 1
        input_data[-40:-1, 202] = 1

        
        # First, no min segment size
        debug_results = {}
        min_segment_size = 0
        ws_output = wsDtSegmentation(input_data, 0.5, 0, min_segment_size, 0.0, 0.0, groupSeeds=False, out_debug_image_dict=debug_results)
        assert ws_output.max() == 4

        # Ensure consecutive label values
        label_list = vigra.analysis.unique(ws_output)
        assert label_list.max() == len(label_list)
        
        #from wsdt.wsDtSegmentation import vigra_bincount
        #print vigra_bincount( ws_output )

        # Now with a min segment size
        debug_results = {}
        min_segment_size = 90*90
        ws_output = wsDtSegmentation(input_data, 0.5, 0, min_segment_size, 0.0, 0.0, groupSeeds=False, out_debug_image_dict=debug_results)
        assert ws_output.max() == 2
        #print vigra_bincount( ws_output )
        
        # Ensure consecutive label values
        label_list = vigra.analysis.unique(ws_output)
        assert label_list.max() == len(label_list)

    def test_border_seeds(self):
        """
        Check if seeds at the borders are generated.
        """
        # create a volume with membrane evidence everywhere
        pmap = np.ones((50, 50, 50))

        # create funnel without membrane evidence growing larger towards the block border.
        pmap[0, 12:39, 12:39] = 0
        pmap[1:50, 13:38, 13:38] = 0

        debug_results = {}
        _ws_output = wsDtSegmentation(pmap, 0.5, 0, 10, 0.1, 0.1, groupSeeds=False, out_debug_image_dict=debug_results)
        seeds = debug_results['seeds'][:]
        assert seeds.sum() == 1
        assert seeds[0, 25, 25] == 1

        # Ensure consecutive label values
        label_list = vigra.analysis.unique(_ws_output)
        assert label_list.max() == len(label_list)

    def test_memory_usage(self):
        pmap = self._gen_input_data(3)

        # Wrap the segmentation function in this decorator, to verify it's memory usage.
        ws_output = assert_mem_usage_factor(2.7)(wsDtSegmentation)(pmap, 0.5, 0, 10, 0.1, 0.1, groupSeeds=False)
        assert ws_output.max() == 8

        # Ensure consecutive label values
        label_list = vigra.analysis.unique(ws_output)
        assert label_list.max() == len(label_list)

        # Now try again, with groupSeeds=True
        # Note: This is a best-case scenario for memory usage, since the memory 
        #       usage of the seed-grouping function depends on the NUMBER of seeds,
        #       and we have very few seeds in this test.
        ws_output = assert_mem_usage_factor(3.5)(wsDtSegmentation)(pmap, 0.5, 0, 10, 0.1, 0.1, groupSeeds=True)
        assert ws_output.max() == 8

    def test_debug_output(self):
        """
        Just exercise the API for debug images, even though we're not
        really checking the *contents* of the images in this test.
        """
        pmap = self._gen_input_data(3)
        debug_images = {}
        ws_output = wsDtSegmentation(pmap, 0.5, 0, 10, 0.1, 0.1, groupSeeds=False, out_debug_image_dict=debug_images)
        assert ws_output.max() == 8

        # Ensure consecutive label values
        label_list = vigra.analysis.unique(ws_output)
        assert label_list.max() == len(label_list)
        
        assert 'thresholded membranes' in debug_images
        assert debug_images['thresholded membranes'].shape == ws_output.shape

        assert 'filtered membranes' in debug_images
        assert debug_images['filtered membranes'].shape == ws_output.shape

    def test_group_close_seeds(self):
        """
        In this test we'll use input data that looks roughly like the following:
        
            0                101               202               303
          0 +-----------------+-----------------+-----------------+
            |        |        |        |        |                 |
            |                 |                 |                 |
            |                 |                 |                 |
            |                                   |                 |
         50 |      x   x             y   y      |        z        |
            |                                   |                 |
            |                 |                 |                 |
            |                 |                 |                 |
            |        |        |        |        |                 |
        101 +-----------------+-----------------+-----------------+

        The x and y markers indicate where seeds will end up.
        
        With groupSeeds=False, we get 5 seed points and 5 final segments.
        But with groupSeeds=True, the two x points end up in the same segment,
        and the two y points end up in the same segment.
        The lone z point will not be merged with anything.
        """
        
        input_data = np.zeros((101, 303), dtype=np.float32)
        
        # Add borders
        input_data[0] = 1
        input_data[-1] = 1
        input_data[:, 0] = 1
        input_data[:, -1] = 1

        # Add complete divider for the z compartment
        input_data[:, 202] = 1

        # Add notches extending from the upper/lower borders
        input_data[  1:10,  51] = 1
        input_data[  1:40, 101] = 1
        input_data[  1:10, 151] = 1
        input_data[-10:-1,  51] = 1
        input_data[-40:-1, 101] = 1
        input_data[-10:-1, 151] = 1
        
        # First, try without groupSeeds
        debug_results = {}
        ws_output = wsDtSegmentation(input_data, 0.5, 0, 0, 0.0, 0.0, groupSeeds=False, out_debug_image_dict=debug_results)
        assert ws_output.max() == 5

        # Ensure consecutive label values
        label_list = vigra.analysis.unique(ws_output)
        assert label_list.max() == len(label_list)
        
        # Now, with groupSeeds=True, the left-hand seeds should 
        # be merged and the right-hand seeds should be merged
        debug_results = {}
        ws_output = wsDtSegmentation(input_data, 0.5, 0, 0, 0.0, 0.0, groupSeeds=True, out_debug_image_dict=debug_results)
        assert ws_output.max() == 3

        # Ensure consecutive label values
        label_list = vigra.analysis.unique(ws_output)
        assert label_list.max() == len(label_list)
        
        assert (ws_output[:,   0: 90] == ws_output[51,  51]).all()
        assert (ws_output[:, 110:190] == ws_output[51, 151]).all()
        assert (ws_output[:, 210:290] == ws_output[51, 251]).all()
        
        # The segment values are different
        assert ws_output[51,51] != ws_output[51, 151] != ws_output[51, 251]

    def test_group_seeds_ram_usage(self):
        """
        The original implementation of the groupSeeds option needed
        a lot of RAM, scaling with the number of seeds by N**2.
        The new implementation does the work in batches, so it
        doesn't need as much RAM.  
        
        Here we create a test image that will result in lots of seeds,
        and we'll verify that RAM usage stays under control.
        
        The test image looks roughly like this (seeds marked with 'x'):
        
        +-----------------------------------------------------+
        |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
        |                                                     |
        |                                                     |
        |       x  x  x  x  x  x  x  x  x  x  x  x  x  x      |
        |                                                     |
        |                                                     |
        +-----------------------------------------------------+
        """
        input_data = np.zeros((101, 20001), dtype=np.float32)

        # Add borders
        input_data[0] = 1
        input_data[-1] = 1
        input_data[:, 0] = 1
        input_data[:, -1] = 1

        # Add tick marks
        input_data[:10, ::10] = 1
        
        # Sanity check, try without groupSeeds, make sure we've got a lot of segments
        ws_output = wsDtSegmentation(input_data, 0.5, 0, 0, 2.0, 0.0, groupSeeds=False)
        assert ws_output.max() > 1900

        # Ensure consecutive label values
        label_list = vigra.analysis.unique(ws_output)
        assert label_list.max() == len(label_list)
        
        # Now check RAM with groupSeeds=True
        ws_output = assert_mem_usage_factor(3.0)(wsDtSegmentation)(input_data, 0.5, 0, 0, 2.0, 0.0, groupSeeds=True)
        assert ws_output.max() == 1        

        # Ensure consecutive label values
        label_list = vigra.analysis.unique(ws_output)
        assert label_list.max() == len(label_list)

    def test_out_param(self):
        pmap = self._gen_input_data(2)

        debug_results = {}
        preallocated = np.random.randint( 0, 100, pmap.shape ).astype(np.uint32)
        ws_output = wsDtSegmentation(pmap, 0.5, 0, 10, 0.1, 0.1, groupSeeds=False, out_debug_image_dict=debug_results, out=preallocated)
        assert ws_output is preallocated
        seeds = debug_results['seeds'][:]
        assert seeds.max() == 4
        assert ws_output.max() == 4

        # Ensure consecutive label values
        label_list = vigra.analysis.unique(ws_output)
        assert label_list.max() == len(label_list)
        
        # Also with groupSeeds=True
        preallocated = np.random.randint( 0, 100, pmap.shape ).astype(np.uint32)
        ws_output = wsDtSegmentation(pmap, 0.5, 0, 10, 0.1, 0.1, groupSeeds=True, out_debug_image_dict=debug_results, out=preallocated)
        assert ws_output is preallocated
        assert seeds.max() == 4
        assert ws_output.max() == 4

        # Ensure consecutive label values
        label_list = vigra.analysis.unique(ws_output)
        assert label_list.max() == len(label_list)

if __name__ == "__main__":
    import sys
    import logging
    mem_logger = logging.getLogger('numpy_allocation_tracking')
    mem_logger.setLevel(logging.DEBUG)
    mem_logger.addHandler(logging.StreamHandler(sys.stdout))
    unittest.main()
