import unittest
import numpy as np

from wsdt import wsDtSegmentation

class TestWsDtSegmentation(unittest.TestCase):

    def test1(self):
        # Create a volume with 8 sections
        pmap = np.zeros( (101,101,101), dtype=np.float32 )

        # Three Z-planes
        pmap[:2, :, :] = 1
        pmap[49:51, :, :] = 1
        pmap[-2:, :, :] = 1

        # Three Y-planes
        pmap[:, :2, :] = 1
        pmap[:, 49:51, :] = 1
        pmap[:, -2:, :] = 1

        # Three X-planes
        pmap[:, :, :2] = 1
        pmap[:, :, 49:51] = 1
        pmap[:, :, -2:] = 1

        # First, just check the seeds
        seeds = wsDtSegmentation(pmap, 0.5, 0, 10, 0.1, 0.1, cleanCloseSeeds=False, returnSeedsOnly=True)
        assert seeds.max() == 8

        # Expect seeds at (25,25,25), (25,25,75), (25,75,25), etc...
        expected_seed_coords = list(np.ndindex((2,2,2)))
        expected_seed_coords = 50*np.array(expected_seed_coords) + 25

        #print "EXPECTED:\n", expected_seed_coords
        #print "SEEDS:\n", np.array(np.where(seeds)).transpose()

        for seed_coord in expected_seed_coords:
            assert seeds[tuple(seed_coord)], "No seed at: {}".format( seed_coord )

        # Now check the whole watershed volume
        ws_output = wsDtSegmentation(pmap, 0.5, 0, 10, 0.1, 0.1, cleanCloseSeeds=False)
        assert ws_output.max() == 8


    def test2(self):
        # check if seeds at the borders are generated

        # create a volume with membrane evidence everywhere
        pmap = np.ones((50, 50, 50))

        # create funnel without membrane evidence growing larger towards the block border.
        pmap[0, 12:39, 12:39] = 0
        pmap[1:50, 13:38, 13:38] = 0
        seeds = wsDtSegmentation(pmap, 0.5, 0, 10, 0.1, 0.1, cleanCloseSeeds=False, returnSeedsOnly=True)
        assert seeds.sum() == 1
        assert seeds[0, 25, 25] == 1


if __name__ == "__main__":
    unittest.main()
