import unittest
import numpy as np
import torch
import open3d as o3d
from sonata_integration import run_sonata_inference

class TestSonataIntegration(unittest.TestCase):

    def setUp(self):
        """Set up a sample point cloud for testing."""
        self.sample_ply_path = "sample_point_cloud.ply"
        # Ensure the sample file exists
        pcd = o3d.geometry.PointCloud()
        points = np.random.rand(2048, 3)
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.random.rand(2048, 3))
        pcd.normals = o3d.utility.Vector3dVector(np.random.rand(2048, 3))
        o3d.io.write_point_cloud(self.sample_ply_path, pcd)

    def test_run_sonata_inference(self):
        """
        Test the end-to-end Sonata inference pipeline.
        This test will load the pretrained Sonata model, process a sample
        point cloud, and verify the output features.
        """
        # This test requires a network connection to download the model
        # and may take a moment to run the first time.
        try:
            features = run_sonata_inference(self.sample_ply_path)
            
            # Check that the output is a numpy array
            self.assertIsInstance(features, np.ndarray)
            
            # The number of points in the output should match the input
            pcd = o3d.io.read_point_cloud(self.sample_ply_path)
            self.assertEqual(features.shape[0], len(pcd.points))
            
            # The feature dimension will depend on the Sonata model,
            # but it should be greater than 0.
            self.assertGreater(features.shape[1], 0)

        except Exception as e:
            self.fail(f"run_sonata_inference raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
