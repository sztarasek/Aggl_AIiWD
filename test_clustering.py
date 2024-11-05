import unittest
from agg import AgglomerativeHierarchicalClustering
from measurements import distance, single_link, complete_link, average_link, get_distance_measure

class TestAgglomerativeHierarchicalClustering(unittest.TestCase):

    def setUp(self):
        # Sample data points
        self.data = [
            [1.0, 2.0],
            [2.0, 3.0],
            [5.0, 6.0],
            [8.0, 8.0]
        ]

    def test_init_clusters(self):
        # Initialize clustering with K = 2 and single link measure (M = 0)
        clustering = AgglomerativeHierarchicalClustering(self.data, K=2, M=0)
        clusters = clustering.init_clusters()

        # Test if clusters are initialized correctly
        expected_clusters = {
            0: [[1.0, 2.0]],
            1: [[2.0, 3.0]],
            2: [[5.0, 6.0]],
            3: [[8.0, 8.0]]
        }
        self.assertEqual(clusters, expected_clusters)

    def test_find_closest_clusters_single_link(self):
        clustering = AgglomerativeHierarchicalClustering(self.data, K=2, M=0)
        clustering.clusters = clustering.init_clusters()

        closest_clusters = clustering.find_closest_clusters()

        # The closest pair of points with single-link distance should be (0, 1)
        self.assertEqual(closest_clusters, (0, 1))

    def test_find_closest_clusters_complete_link(self):
        clustering = AgglomerativeHierarchicalClustering(self.data, K=2, M=1)
        clustering.clusters = clustering.init_clusters()

        closest_clusters = clustering.find_closest_clusters()

        # For complete-link distance, (0, 1) should also be closest
        self.assertEqual(closest_clusters, (0, 1))

    def test_merge_and_form_new_clusters(self):
        clustering = AgglomerativeHierarchicalClustering(self.data, K=2, M=0)
        clustering.clusters = clustering.init_clusters()

        # Merge clusters 0 and 1
        new_clusters = clustering.merge_and_form_new_clusters(0, 1)

        expected_clusters = {
            4: [[1.0, 2.0], [2.0, 3.0]],  # New cluster formed by merging 0 and 1
            2: [[5.0, 6.0]],
            3: [[8.0, 8.0]]
        }

        self.assertEqual(new_clusters, expected_clusters)

    def test_run_algorithm(self):
        # Test the full run of the algorithm
        clustering = AgglomerativeHierarchicalClustering(self.data, K=2, M=0)
        clustering.run_algorithm()

        # After running, there should be exactly 2 clusters
        self.assertEqual(len(clustering.clusters), 2)


class TestDistanceMeasures(unittest.TestCase):

    def test_distance(self):
        # Test Euclidean distance between two points
        p = [1.0, 2.0]
        q = [4.0, 6.0]
        self.assertAlmostEqual(distance(p, q), 5.0)

    def test_single_link(self):
        ci = [[1.0, 2.0], [2.0, 3.0]]
        cj = [[5.0, 6.0], [8.0, 8.0]]

        # Closest distance between ci and cj
        self.assertAlmostEqual(single_link(ci, cj), 5.0)

    def test_complete_link(self):
        ci = [[1.0, 2.0], [2.0, 3.0]]
        cj = [[5.0, 6.0], [8.0, 8.0]]

        # Furthest distance between ci and cj
        self.assertAlmostEqual(complete_link(ci, cj), 8.602325267042627)

    def test_average_link(self):
        ci = [[1.0, 2.0], [2.0, 3.0]]
        cj = [[5.0, 6.0], [8.0, 8.0]]

        # Average distance between all points in ci and cj
        self.assertAlmostEqual(average_link(ci, cj), 6.801162633521313)

    def test_get_distance_measure(self):
        # Test that the correct distance measure is returned
        self.assertEqual(get_distance_measure(0), single_link)
        self.assertEqual(get_distance_measure(1), complete_link)
        self.assertEqual(get_distance_measure(2), average_link)


if __name__ == '__main__':
    unittest.main()