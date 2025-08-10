import unittest
import numpy as np
import torch
import open3d as o3d
import os

# This is a bit of a hack to be able to import the script file.
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# We need to mock the sonata package since the real dependencies are not installed.
from unittest.mock import MagicMock, patch

# Mock the sonata package and its modules
sonata_mock = MagicMock()
sys.modules['sonata'] = sonata_mock
sys.modules['sonata.model'] = MagicMock()
sys.modules['sonata.transform'] = MagicMock()
sys.modules['sonata.data'] = MagicMock()

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

    @patch('sonata_integration.load_model')
    def test_run_sonata_inference(self, mock_load_model):
        """
        Test the end-to-end Sonata inference pipeline with mocks.
        """
        # Configure the mock model
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(2048, 256) # Simulate feature output
        mock_load_model.return_value.cuda.return_value = mock_model

        # Run the inference function
        features = run_sonata_inference(self.sample_ply_path)
        
        # Check that the output is a numpy array
        self.assertIsInstance(features, np.ndarray)
        
        # The number of points in the output should match the input
        pcd = o3d.io.read_point_cloud(self.sample_ply_path)
        self.assertEqual(features.shape[0], len(pcd.points))
        
        # The feature dimension should match our mock output
        self.assertEqual(features.shape[1], 256)

if __name__ == '__main__':
    # We need to install open3d for the tests to run
    # pip install open3d
    unittest.main()