from geoTransformer.GeoTransformer.experiments.roboeye.model import create_model
from geoTransformer.GeoTransformer.experiments.roboeye.utils import (
    create_transfer_learning_cfg,
    freeze_model_components,
)


def inspect_model_structure() -> None:
    """Inspect model structure and parameter distribution"""
    print("MODEL STRUCTURE ANALYSIS")
    print("=" * 50)

    cfg = create_transfer_learning_cfg()
    model = create_model(cfg)

    print("Top-level modules:")
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        has_params = param_count > 0
        status = "🟢 LEARNABLE" if has_params else "ALGORITHMIC"
        print(f"{name}: {type(module).__name__} ({param_count:,} params) {status}")

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: {}".format(total_params))


def test_progressive_freezing() -> None:
    """Test progressive freezing strategy"""
    print("PROGRESSIVE FREEZING TEST")
    print("=" * 50)

    cfg = create_transfer_learning_cfg()
    model = create_model(cfg)

    for phase in range(4):
        # Reset all parameters to trainable
        for param in model.parameters():
            param.requires_grad = True

        # Apply freezing
        freeze_model_components(model, phase)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        percentage = 100 * trainable / total
        print("Phase {}: {}/{} ({:.1f}%) trainable".format(phase, trainable, total, percentage))


def decode_checkpoint_name(checkpoint_path: str) -> None:
    """Decode parameters from checkpoint filename"""
    print("CHECKPOINT NAME DECODER")
    print("=" * 50)
    from pathlib import Path

    filename = Path(checkpoint_path).stem
    print("Filename: {}".format(filename))

    # Parse components
    components = {}
    parts = filename.split("_")

    for part in parts:
        if "voxel" in part and "mm" in part:
            components["voxel_size"] = part.replace("voxel", "").replace("mm", "") + "mm"
        elif "sigma-d" in part and "mm" in part:
            components["sigma_d"] = part.replace("sigma-d", "").replace("mm", "") + "mm"
        elif "sigma-a" in part:
            components["sigma_a"] = part.replace("sigma-a", "")
        elif "angle-k" in part:
            components["angle_k"] = part.replace("angle-k", "")
        elif "scale" in part:
            components["scale_factor"] = part.replace("scale", "")
        elif "phase" in part:
            components["phase"] = part.replace("phase", "")
        elif "epoch" in part:
            components["epoch"] = part.replace("epoch", "")

    print("Decoded parameters:")
    for key, value in components.items():
        print("{}: {}".format(key, value))


def test_checkpoint_naming() -> None:
    """Test checkpoint naming convention"""
    print("CHECKPOINT NAMING PREVIEW")
    print("=" * 50)

    scenarios = [
        {"pretrained_voxel": 0.025, "target_voxel": 0.009, "name": "3DMatch → RoboEye"},
        {"pretrained_voxel": 0.05, "target_voxel": 0.01, "name": "Large → Medium"},
    ]

    for scenario in scenarios:
        print("{}".format(scenario["name"]))
        cfg = create_transfer_learning_cfg(
            scenario["pretrained_voxel"], scenario["target_voxel"], dataset_id=None
        )

        # Generate naming parameters
        voxel_size_mm = int(cfg.backbone.init_voxel_size * 1000)
        sigma_d_mm = int(cfg.geotransformer.sigma_d * 1000)
        sigma_a = int(cfg.geotransformer.sigma_a)
        angle_k = cfg.geotransformer.angle_k
        scale_factor = int(cfg.transfer_learning["scale_factor"] * 1000)

        param_suffix = f"voxel{voxel_size_mm}mm_sigma-d{sigma_d_mm}mm_sigma-a{sigma_a}_angle-k{angle_k}_scale{scale_factor}"

        print("Example: best_checkpoint_phase_1_{}.pth.tar".format(param_suffix))


def detect_learnable_parameters() -> None:
    """Show how to detect learnable vs algorithmic modules"""
    print("PARAMETER DETECTION GUIDE")
    print("=" * 50)

    print("Quick detection methods:")
    print("  # Check parameter count")
    print("  param_count = sum(p.numel() for p in module.parameters())")
    print("  has_params = param_count > 0")
    print()
    print("  # Check for specific attributes")
    print("  has_weights = hasattr(module, 'weight') or hasattr(module, 'bias')")
    print()
    print("  # Find explicit nn.Parameter attributes")
    print("  for name in dir(module):")
    print("      if isinstance(getattr(module, name), nn.Parameter):")
    print("          print(f'Found parameter: {name}')")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="GeoTransformer Debug and Inspection Utilities")
    parser.add_argument("--model", action="store_true", help="Inspect model structure")
    parser.add_argument("--freezing", action="store_true", help="Test progressive freezing")
    parser.add_argument("--naming", action="store_true", help="Test checkpoint naming")
    parser.add_argument("--decode", help="Decode checkpoint filename")
    parser.add_argument("--detection", action="store_true", help="Show parameter detection guide")
    parser.add_argument("--all", action="store_true", help="Run all inspections")

    args = parser.parse_args()

    if args.all or not any([args.model, args.freezing, args.naming, args.decode, args.detection]):
        # Run all by default
        inspect_model_structure()
        test_progressive_freezing()
        test_checkpoint_naming()
        detect_learnable_parameters()
    else:
        if args.model:
            inspect_model_structure()
        if args.freezing:
            test_progressive_freezing()
        if args.naming:
            test_checkpoint_naming()
        if args.decode:
            decode_checkpoint_name(args.decode)
        if args.detection:
            detect_learnable_parameters()


if __name__ == "__main__":
    main()
