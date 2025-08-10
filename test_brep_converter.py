import os
import unittest
import numpy as np
import open3d as o3d
from brep_converter import xyz_to_ply, ply_to_xyz

class TestBrepConverter(unittest.TestCase):

    def setUp(self):
        """Set up test files."""
        self.xyz_file = "test.xyz"
        self.ply_file = "test.ply"
        self.converted_xyz = "converted.xyz"
        self.converted_ply = "converted.ply"

        # Create a sample XYZ file
        points = np.random.rand(10, 3)
        np.savetxt(self.xyz_file, points)

        # Create a sample PLY file
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(self.ply_file, pcd)

    def tearDown(self):
        """Clean up test files."""
        files = [
            self.xyz_file, self.ply_file,
            self.converted_xyz, self.converted_ply
        ]
        for f in files:
            if os.path.exists(f):
                os.remove(f)

    def test_xyz_to_ply(self):
        """Test conversion from XYZ to PLY."""
        xyz_to_ply(self.xyz_file, self.converted_ply)
        self.assertTrue(os.path.exists(self.converted_ply))
        pcd = o3d.io.read_point_cloud(self.converted_ply)
        self.assertEqual(len(pcd.points), 10)

    def test_ply_to_xyz(self):
        """Test conversion from PLY to XYZ."""
        ply_to_xyz(self.ply_file, self.converted_xyz)
        self.assertTrue(os.path.exists(self.converted_xyz))
        points = np.loadtxt(self.converted_xyz)
        self.assertEqual(points.shape, (10, 3))

if __name__ == '__main__':
    unittest.main()
