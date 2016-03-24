import unittest
import numpy as np

from wsdt import wsDtSegmentation
from numpy_allocation_tracking.decorators import assert_mem_usage_factor

class TestWsDtSegmentation(unittest.TestCase):

    def _gen_input_data(self, ndim):
        assert ndim in (2,3)

        # Create a volume with 8 sections
        pmap = np.zeros( (101,)*ndim, dtype=np.float32 )

        # Three Z-planes
        pmap[  :2,  :] = 1
        pmap[49:51, :] = 1
        pmap[-2:,   :] = 1

        # Three Y-planes
        pmap[:,   :2] = 1
        pmap[:, 49:51] = 1
        pmap[:, -2:] = 1

        if ndim == 3:
            # Three X-planes
            pmap[:, :, :2] = 1
            pmap[:, :, 49:51] = 1
            pmap[:, :, -2:] = 1
        
        return pmap
        

    def test_simple_3D(self):
        pmap = self._gen_input_data(3)

        debug_results = {}
        ws_output = wsDtSegmentation(pmap, 0.5, 0, 10, 0.1, 0.1, cleanCloseSeeds=False, out_debug_image_dict=debug_results)
        seeds = debug_results['seeds'][:]
        assert seeds.max() == 8
        assert ws_output.max() == 8

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
        ws_output = wsDtSegmentation(pmap, 0.5, 0, 10, 0.1, 0.1, cleanCloseSeeds=False, out_debug_image_dict=debug_results)
        seeds = debug_results['seeds'][:]
        assert seeds.max() == 4
        assert ws_output.max() == 4

        # Expect seeds at (25,25,25), (25,25,75), (25,75,25), etc...
        expected_seed_coords = list(np.ndindex((2,2)))
        expected_seed_coords = 50*np.array(expected_seed_coords) + 25

        #print "EXPECTED:\n", expected_seed_coords
        #print "SEEDS:\n", np.array(np.where(seeds)).transpose()

        for seed_coord in expected_seed_coords:
            assert seeds[tuple(seed_coord)], "No seed at: {}".format( seed_coord )


    def test_border_seeds(self):
        """
        check if seeds at the borders are generated
        """

        # create a volume with membrane evidence everywhere
        pmap = np.ones((50, 50, 50))

        # create funnel without membrane evidence growing larger towards the block border.
        pmap[0, 12:39, 12:39] = 0
        pmap[1:50, 13:38, 13:38] = 0

        debug_results = {}
        _ws_output = wsDtSegmentation(pmap, 0.5, 0, 10, 0.1, 0.1, cleanCloseSeeds=False, out_debug_image_dict=debug_results)
        seeds = debug_results['seeds'][:]
        assert seeds.sum() == 1
        assert seeds[0, 25, 25] == 1

    def test_memory_usage(self):
        pmap = self._gen_input_data(3)

        # Wrap the segmentation function in this decorator, to verify it's memory usage.
        ws_output = assert_mem_usage_factor(2.5)(wsDtSegmentation)(pmap, 0.5, 0, 10, 0.1, 0.1, cleanCloseSeeds=False)
        assert ws_output.max() == 8

    def test_debug_output(self):
        """
        Just exercise the API for debug images, even though we're not
        really checking the *contents* of the images in this test.
        """
        pmap = self._gen_input_data(3)
        debug_images = {}
        ws_output = wsDtSegmentation(pmap, 0.5, 0, 10, 0.1, 0.1, cleanCloseSeeds=False, out_debug_image_dict=debug_images)
        assert ws_output.max() == 8
        
        assert 'thresholded membranes' in debug_images
        assert debug_images['thresholded membranes'].shape == ws_output.shape

        assert 'filtered membranes' in debug_images
        assert debug_images['filtered membranes'].shape == ws_output.shape

    def test_clean_close_seeds(self):
        """
        In this test we'll use input data that looks roughly like the following:
        
        +------------------+------------------+
        |        |         |        |         |
        |                  |                  |
        |                  |                  |
        |                                     |
        |   x         x         y        y    |
        |                                     |
        |                  |                  |
        |                  |                  |
        |        |         |        |         |
        +------------------+------------------+

        The x and y markers indicate where seeds will end up.
        With cleanCloseSeeds=False, we would have 4 seed points and 4 final segments.
        But with cleanCloseSeeds=True, the two x points will end up in the same segment,
        and the two y points will end up in the same segment.
        """
        
        input_data = np.zeros((100, 200), dtype=np.float32)
        
        # Add borders
        input_data[0] = 1
        input_data[-1] = 1
        input_data[:, 0] = 1
        input_data[:, -1] = 1

        # Add notches on the borders
        input_data[1:5,   50] = 1
        input_data[1:40, 100] = 1
        input_data[1:5,  150] = 1

        input_data[-5:-1,   50] = 1
        input_data[-40:-1, 100] = 1
        input_data[-5:-1,  150] = 1

        # First, try without cleanCloseSeeds
        debug_results = {}
        ws_output = wsDtSegmentation(input_data, 0.5, 0, 0, 0.0, 0.0, cleanCloseSeeds=False, out_debug_image_dict=debug_results)
        assert ws_output.max() == 4

        # Now, with cleanCloseSeeds=True, the left-hand seeds should 
        # be merged and the right-hand seeds should be merged
        debug_results = {}
        ws_output = wsDtSegmentation(input_data, 0.5, 0, 0, 0.0, 0.0, cleanCloseSeeds=True, out_debug_image_dict=debug_results)
        assert ws_output.max() == 2
        assert (ws_output[:, :100] == 1).all()
        assert (ws_output[:, 101:] == 2).all()

if __name__ == "__main__":
    import sys
    import logging
    mem_logger = logging.getLogger('numpy_allocation_tracking')
    mem_logger.setLevel(logging.DEBUG)
    mem_logger.addHandler(logging.StreamHandler(sys.stdout))
    unittest.main()
