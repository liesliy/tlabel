"""
TacQuad adapter test and demo data generator

Creates a minimal TacQuad-compatible dataset structure for testing,
then loads it through the adapter to verify correctness.
"""

import os
import csv
import numpy as np
from pathlib import Path


def create_demo_dataset(root_dir, num_items=2, frames_per_item=5,
                        img_size=(64, 64), with_tac3d=True):
    """Create a minimal TacQuad-compatible demo dataset

    Args:
        root_dir: Root directory for the demo dataset
        num_items: Number of touch items to create
        frames_per_item: Number of frames per sensor per item
        img_size: Image size (H, W)
        with_tac3d: Include Tac3D force field data
    """
    root = Path(root_dir)

    # Create subset directories
    for subset in ['indoor', 'outdoor']:
        data_dir = root / f'data_{subset}'
        csv_path = root / f'contact_{subset}.csv'

        csv_rows = []
        n_items = num_items if subset == 'indoor' else 1

        for item_idx in range(n_items):
            item_name = f'demo_item_{item_idx:03d}'
            item_dir = data_dir / item_name

            # Each sensor has its own frame range
            gelsight_start, gelsight_end = 0, frames_per_item - 1
            digit_start, digit_end = 0, frames_per_item - 1
            duragel_start, duragel_end = 1, frames_per_item  # slightly different

            # Create sensor directories and images
            for sensor_name, start, end in [
                ('gelsight', gelsight_start, gelsight_end),
                ('digit', digit_start, digit_end),
                ('duragel', duragel_start, duragel_end),
            ]:
                sensor_dir = item_dir / sensor_name
                sensor_dir.mkdir(parents=True, exist_ok=True)

                for t in range(start, end + 1):
                    # Create synthetic RGB image with contact pattern
                    np.random.seed(t + int(sensor_name == 'digit') * 100)
                    img = np.random.randint(0, 255, (*img_size, 3), dtype=np.uint8)
                    # Add contact-like pattern in center
                    cy, cx = img_size[0] // 2, img_size[1] // 2
                    r = max(3, frames_per_item - t)
                    y_coords, x_coords = np.mgrid[0:img_size[0], 0:img_size[1]]
                    mask = ((y_coords - cy)**2 + (x_coords - cx)**2) < r**2
                    if t > start:
                        img[mask] = [200, 100, 50]  # contact region

                    # Save as PNG using PIL or cv2
                    try:
                        from PIL import Image
                        pil_img = Image.fromarray(img)
                        pil_img.save(str(sensor_dir / f'{t}.png'))
                    except ImportError:
                        import cv2
                        cv2.imwrite(str(sensor_dir / f'{t}.png'), img)

                # Create paired vision directory
                vision_dir = item_dir / f'img_{sensor_name}'
                vision_dir.mkdir(parents=True, exist_ok=True)
                n_vision = max(2, (end - start + 1) // 2)
                for v in range(n_vision):
                    np.random.seed(v + 1000)
                    vimg = np.random.randint(
                        0, 255, (*img_size, 3), dtype=np.uint8
                    )
                    try:
                        from PIL import Image
                        pil_img = Image.fromarray(vimg)
                        pil_img.save(str(vision_dir / f'{v}.png'))
                    except ImportError:
                        import cv2
                        cv2.imwrite(str(vision_dir / f'{v}.png'), vimg)

            # Create Tac3D force field data (optional)
            if with_tac3d:
                tac3d_dir = item_dir / 'tac3d'
                tac3d_dir.mkdir(parents=True, exist_ok=True)
                max_end = max(gelsight_end, digit_end, duragel_end)
                for t in range(0, max_end + 1):
                    # Synthetic 20x20x3 force field
                    np.random.seed(t + 2000)
                    force = np.random.randn(20, 20, 3).astype(np.float32)
                    force[:, :, 2] = abs(force[:, :, 2]) * (t + 1) * 0.5
                    np.save(str(tac3d_dir / f'{t}.npy'), force)

            csv_rows.append([
                item_name,
                gelsight_start, gelsight_end,
                digit_start, digit_end,
                duragel_start, duragel_end,
            ])

        # Write CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in csv_rows:
                writer.writerow(row)

    print(f"Demo dataset created at: {root}")
    print(f"  Indoor items: {num_items}")
    print(f"  Outdoor items: 1")
    print(f"  Frames per sensor per item: {frames_per_item}")
    print(f"  Image size: {img_size}")
    print(f"  Tac3D included: {with_tac3d}")
    return root


def run_test(demo_root):
    """Run adapter test on demo dataset"""

    # Need to set up minimal tlabel package structure
    # The test imports from tlabel.adapters.tacquad directly

    from tlabel.adapters.tacquad import TacQuadAdapter

    adapter = TacQuadAdapter()

    print("\n=== Test 1: Basic load (gelsight, indoor) ===")
    data = adapter.load(
        str(demo_root),
        sensor='gelsight',
        subset='indoor',
    )
    print(f"  Loaded: {data.num_frames} frames")
    print(f"  Sensor: {data.sensor_type}")
    print(f"  Duration: {data.duration_s:.2f}s")
    print(f"  Capabilities: {len(data.capabilities)} dimensions")
    assert data.num_frames > 0, "Should have frames"
    print("  PASSED")

    print("\n=== Test 2: Load all sensors (both subsets) ===")
    data = adapter.load(
        str(demo_root),
        sensor='all',
        subset='both',
    )
    print(f"  Loaded: {data.num_frames} frames from all sensors")
    print(f"  Episode info: {data.episode_info.get('stats', {})}")
    assert data.num_frames > 0
    print("  PASSED")

    print("\n=== Test 3: Load with Tac3D ===")
    data = adapter.load(
        str(demo_root),
        sensor='digit',
        subset='indoor',
        max_items=1,
    )
    tac3d_frames = sum(
        1 for f in data.frames
        if f.sensor_specific.get('has_tac3d', False)
    )
    print(f"  Tac3D frames loaded: {tac3d_frames}")
    print(f"  Total frames: {data.num_frames}")
    assert data.num_frames > 0
    print("  PASSED")

    print("\n=== Test 4: Verify 14-dim Schema V2 format ===")
    frame = data.frames[0]
    from tlabel.core.schema import SCHEMA_V2_FIELD_NAMES
    sv2_dict = frame.schema_v2.to_dict()
    for key in SCHEMA_V2_FIELD_NAMES:
        assert key in sv2_dict, f"Missing key: {key}"
    print(f"  All {len(SCHEMA_V2_FIELD_NAMES)} dimensions present")
    print("  PASSED")

    print("\n=== Test 5: Phase inference ===")
    phases = set(f.manipulation_phase for f in data.frames)
    print(f"  Unique phases: {phases}")
    assert len(phases) > 0
    print("  PASSED")

    print("\n=== Test 6: Sensor info ===")
    info = adapter.get_sensor_info()
    assert info['type'] == 'vision-based_tactile'
    assert 'sensors' in info
    print(f"  Sensor type: {info['type']}")
    print(f"  Sensors: {list(info['sensors'].keys())}")
    print("  PASSED")

    print("\n=== Test 7: to_dict export ===")
    d = data.to_dict()
    assert 'schema_version' in d
    assert 'frames' in d
    assert 'sensor' in d
    print(f"  Export keys: {list(d.keys())}")
    print(f"  Frames exported: {len(d['frames'])}")
    print("  PASSED")

    print("\n=== All tests passed! ===")


if __name__ == '__main__':
    import tempfile
    demo_dir = tempfile.mkdtemp(prefix='tacquad_demo_')
    print(f"Creating demo dataset in: {demo_dir}")
    root = create_demo_dataset(demo_dir, num_items=2, frames_per_item=5)
    print()
    run_test(root)
