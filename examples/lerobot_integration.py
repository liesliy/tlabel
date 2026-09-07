#!/usr/bin/env python3
"""
LeRobot Integration Example

Demonstrates bidirectional conversion between TLabel and LeRobot formats.

Requirements:
    pip install tlabel[lerobot]  # or: pip install pyarrow

Usage:
    python examples/lerobot_integration.py
"""

import json
import tempfile
from pathlib import Path

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("Please install pyarrow: pip install pyarrow")
    exit(1)

from tlabel.converters import lerobot_to_tlabel, tlabel_to_lerobot
from tlabel.core.types import TLabelData, TLabelFrame, TLabelSchemaV2


def create_demo_lerobot_episode(output_dir: Path, num_frames: int = 50):
    """Create a synthetic LeRobot episode for testing"""
    print(f"Creating demo LeRobot episode in {output_dir}...")
    
    # Create directory structure
    data_dir = output_dir / "data"
    meta_dir = output_dir / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic tactile data (simulating a grasp-hold-release cycle)
    np.random.seed(42)
    
    # Tactile features: [contact, force_magnitude, slip_event, object_deformation, confidence]
    tactile_data = []
    for i in range(num_frames):
        # Simulate grasp phases
        if i < 10:  # Approach
            contact = False
            force = 0.0
            slip = False
            deformation = 0.0
        elif i < 30:  # Grasp and hold
            contact = True
            force = 0.5 + 0.3 * np.sin(i * 0.2)
            slip = False
            deformation = 0.1 + 0.05 * np.random.randn()
        else:  # Release
            contact = i < 40
            force = max(0, 0.5 - (i - 30) * 0.05)
            slip = i >= 35 and i < 40
            deformation = max(0, 0.1 - (i - 30) * 0.01)
        
        confidence = 0.9 + 0.1 * np.random.rand()
        tactile_data.append([contact, force, slip, deformation, confidence])
    
    # Create Parquet file
    tactile_array = pa.array(tactile_data, type=pa.list_(pa.float64()))
    timestamps = pa.array([i * 0.033 for i in range(num_frames)])
    actions = pa.array([np.random.randn(7).tolist() for _ in range(num_frames)])
    
    table = pa.table({
        "observation.tactile": tactile_array,
        "action": actions,
        "timestamp": timestamps,
    })
    
    pq.write_table(table, data_dir / "episode_0.parquet")
    
    # Create meta/info.json
    info = {
        "episode_id": "demo_episode_0",
        "task": "Grasp a soft object",
        "robot_type": "demo_robot",
        "features": {
            "observation.tactile": {
                "feature_names": [
                    "contact",
                    "force_magnitude",
                    "slip_event",
                    "object_deformation",
                    "confidence"
                ]
            }
        }
    }
    
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"  Created {num_frames} frames of tactile data")
    print(f"  Features: {info['features']['observation.tactile']['feature_names']}")


def demo_lerobot_to_tlabel():
    """Demonstrate LeRobot → TLabel conversion"""
    print("\n" + "="*60)
    print("Demo 1: LeRobot → TLabel")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        lerobot_dir = tmpdir / "lerobot_episode"
        
        # Create demo LeRobot data
        create_demo_lerobot_episode(lerobot_dir, num_frames=50)
        
        # Convert to TLabel
        print("\nConverting to TLabel...")
        tlabel_data = lerobot_to_tlabel(lerobot_dir)
        
        print(f"\nTLabel Data Summary:")
        print(f"  Sensor: {tlabel_data.sensor_info.get('type', 'unknown')}")
        print(f"  Frames: {len(tlabel_data.frames)}")
        print(f"  Capabilities: {list(tlabel_data.capabilities.keys())}")
        print(f"  Episode: {tlabel_data.episode_info.get('episode_id', 'unknown')}")
        
        # Show first few frames
        print("\nFirst 5 frames:")
        for i, frame in enumerate(tlabel_data.frames[:5]):
            print(f"  Frame {i}: contact={frame.schema_v2.contact}, "
                  f"force={frame.schema_v2.force_magnitude:.3f}, "
                  f"slip={frame.schema_v2.slip_event}")
        
        # Export to JSON
        output_json = tmpdir / "tlabel_output.json"
        tlabel_data.export(str(output_json))
        print(f"\nExported to: {output_json}")
        
        # Show JSON preview
        with open(output_json) as f:
            data = json.load(f)
            print(f"\nJSON structure:")
            print(f"  Keys: {list(data.keys())}")
            print(f"  Frames: {len(data.get('frames', []))}")
            if data.get('frames'):
                print(f"  First frame keys: {list(data['frames'][0].keys())}")


def demo_tlabel_to_lerobot():
    """Demonstrate TLabel → LeRobot conversion"""
    print("\n" + "="*60)
    print("Demo 2: TLabel → LeRobot (Write annotations back)")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        lerobot_dir = tmpdir / "lerobot_episode"
        
        # Create demo LeRobot data
        create_demo_lerobot_episode(lerobot_dir, num_frames=50)
        
        # Convert to TLabel and modify
        print("\nConverting to TLabel...")
        tlabel_data = lerobot_to_tlabel(lerobot_dir)
        
        # Simulate annotation: modify confidence values
        print("\nSimulating annotation (adjusting confidence)...")
        for i, frame in enumerate(tlabel_data.frames):
            if frame.schema_v2.contact:
                # Simulate quality assessment: lower confidence for high force
                if frame.schema_v2.force_magnitude > 0.7:
                    frame.schema_v2.confidence = 0.75
                else:
                    frame.schema_v2.confidence = 0.95
        
        # Export modified TLabel
        tlabel_json = tmpdir / "tlabel_annotated.json"
        tlabel_data.export(str(tlabel_json))
        print(f"Exported annotated TLabel to: {tlabel_json}")
        
        # Convert back to LeRobot (write annotations to existing LeRobot dataset)
        print("\nWriting annotations back to LeRobot format...")
        tlabel_to_lerobot(tlabel_json, lerobot_dir, overwrite=True)
        
        print(f"\nAnnotations written to: {lerobot_dir}")
        
        # Verify output
        parquet_files = list((lerobot_dir / "data").glob("*.parquet"))
        if parquet_files:
            table = pq.read_table(parquet_files[0])
            print(f"\nOutput verification:")
            print(f"  Parquet rows: {table.num_rows}")
            print(f"  Columns: {table.column_names}")
            if "observation.tactile" in table.column_names:
                print(f"  Tactile data: {len(table['observation.tactile'])} frames")
                print(f"  Note: Annotations are written to existing LeRobot dataset")


def demo_end_to_end():
    """Demonstrate complete workflow"""
    print("\n" + "="*60)
    print("Demo 3: End-to-End Workflow")
    print("="*60)
    
    print("\nComplete workflow:")
    print("  1. Load LeRobot data")
    print("  2. Convert to TLabel for annotation")
    print("  3. Export annotated TLabel")
    print("  4. Convert back to LeRobot for training")
    print("\nSee Demo 1 and Demo 2 for implementation details.")
    print("\nKey points:")
    print("  - Bidirectional conversion preserves data integrity")
    print("  - TLabel adds semantic annotations (14 dimensions)")
    print("  - LeRobot format is compatible with training pipeline")
    print("  - No data loss in conversion")


def main():
    print("TLabel × LeRobot Integration Examples")
    print("="*60)
    
    try:
        demo_lerobot_to_tlabel()
        demo_tlabel_to_lerobot()
        demo_end_to_end()
        
        print("\n" + "="*60)
        print("All demos completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("  - Try with your own LeRobot data")
        print("  - Use data.review() for interactive annotation")
        print("  - Export to FTP-1 format for foundation models")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
