import pytest
import os
import numpy as np
from unittest.mock import patch, MagicMock

# This is a bit of a hack to be able to import the script file.
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sonata_integration import SonataSegmenter, BRepExtractor, IntegratedPipeline, DEFAULT_CONFIG

@pytest.fixture
def default_config():
    """Return a copy of the default config for modification."""
    return DEFAULT_CONFIG.copy()

@pytest.fixture
def mock_point_cloud():
    """Create a dummy Open3D PointCloud object."""
    pcd = MagicMock()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    return pcd

# --- SonataSegmenter Tests ---

@patch('sonata_integration.torch.cuda.is_available', return_value=True)
def test_segmenter_init_cuda(mock_cuda_available, default_config):
    """Test SonataSegmenter initialization with CUDA."""
    default_config['segmentation']['device'] = 'cuda'
    segmenter = SonataSegmenter(default_config['segmentation'])
    assert segmenter.device.type == 'cuda'

@patch('sonata_integration.torch.cuda.is_available', return_value=False)
def test_segmenter_init_cpu(mock_cuda_available, default_config):
    """Test SonataSegmenter initialization on CPU."""
    default_config['segmentation']['device'] = 'cuda' # Request cuda
    segmenter = SonataSegmenter(default_config['segmentation'])
    assert segmenter.device.type == 'cpu' # But should fall back to cpu

def test_segmenter_simulation(default_config, mock_point_cloud):
    """Test that the simulated segmentation returns labels of the correct shape."""
    segmenter = SonataSegmenter(default_config['segmentation'])
    labels = segmenter.segment(mock_point_cloud)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(mock_point_cloud.points)

# --- BRepExtractor Tests ---

def test_brep_extractor_init(default_config):
    """Test BRepExtractor initialization."""
    extractor = BRepExtractor(default_config['brep_extraction'])
    assert extractor.config is not None

@patch('open3d.geometry.PointCloud.segment_plane')
def test_brep_extractor_fit_plane(mock_segment_plane, default_config):
    """Test the plane fitting logic."""
    # Mock the return value of segment_plane
    mock_segment_plane.return_value = ([1, 2, 3, 4], [0, 1, 2]) # equation, inliers
    
    extractor = BRepExtractor(default_config['brep_extraction'])
    mock_segment = MagicMock()
    mock_segment.get_center.return_value = np.array([0,0,0])
    mock_segment.points = o3d.utility.Vector3dVector(np.random.rand(10,3))

    results = extractor.extract({'1': mock_segment})
    
    assert '1' in results
    assert results['1']['type'] == 'plane'
    assert results['1']['parameters']['equation'] == [1, 2, 3, 4]
    mock_segment_plane.assert_called_once()

# --- IntegratedPipeline Tests ---

@pytest.fixture
def sample_pipeline_files(tmp_path):
    """Create dummy files for pipeline testing."""
    pcd_path = tmp_path / "test.ply"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(600, 3)) # More than min_cloud_points
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    return [pcd_path]

@patch('sonata_integration.SonataSegmenter')
@patch('sonata_integration.BRepExtractor')
def test_pipeline_init(mock_brep_extractor, mock_segmenter, default_config):
    """Test pipeline initialization."""
    pipeline = IntegratedPipeline(default_config)
    assert pipeline.config is not None
    assert pipeline.segmenter is None # Check for lazy loading
    assert pipeline.brep_extractor is None

@patch('sonata_integration.o3d.io.write_point_cloud')
@patch('sonata_integration.json.dump')
def test_pipeline_process_single(mock_json_dump, mock_write_pcd, sample_pipeline_files, default_config):
    """Test the end-to-end processing for a single file."""
    pipeline = IntegratedPipeline(default_config)
    result = pipeline._process_single(sample_pipeline_files[0])

    assert "Success" in result
    # Check that checkpointing functions were called
    mock_write_pcd.assert_called_once()
    mock_json_dump.assert_called_once()

def test_pipeline_process_sparse_cloud(sample_pipeline_files, default_config):
    """Test that sparse clouds are skipped correctly."""
    # Create a sparse cloud
    sparse_pcd = o3d.geometry.PointCloud()
    sparse_pcd.points = o3d.utility.Vector3dVector(np.random.rand(10, 3))
    sparse_path = Path(sample_pipeline_files[0]).parent / "sparse.ply"
    o3d.io.write_point_cloud(str(sparse_path), sparse_pcd)

    pipeline = IntegratedPipeline(default_config)
    result = pipeline._process_single(sparse_path)
    assert "Skipped (Sparse)" in result

@patch('argparse.ArgumentParser.parse_args')
@patch('sonata_integration.IntegratedPipeline.run')
def test_pipeline_main_args(mock_run, mock_parse_args, sample_pipeline_files):
    """Test the main function argument parsing and pipeline execution."""
    mock_parse_args.return_value = MagicMock(
        input_files=sample_pipeline_files,
        output_dir=Path("./checkpoints"),
        no_parallel=True
    )
    
    # We need to mock plt since it's imported in main
    with patch.dict('sys.modules', {'matplotlib.pyplot': MagicMock()}) as mock_plt:
        from sonata_integration import main as pipeline_main
        pipeline_main()

    # Check that the run method was called with the correct arguments
    mock_run.assert_called_once_with(sample_pipeline_files, parallel=False)