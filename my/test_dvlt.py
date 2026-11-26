"""Run pretrained DVLT on three construction views with `python my/test_dvlt.py`."""

import argparse
from contextlib import ExitStack
from os.path import expanduser
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from matplotlib.figure import Figure
from PIL import Image, ImageOps

from dvlt.common.constants import DataField, PredictionField
from dvlt.model.dvlt.model import DVLT
from dvlt.struct.cameras import Cameras
from dvlt.util.preprocess import SEGMENTATION_MASK_FIELD, load_sequence, preprocess_images
from dvlt.viz.scene_plotly import scene_to_plotly


def preprocess_native_images(frames: list[Image.Image], patch_size: int, device: torch.device) -> dict:
    width = max(frame.width for frame in frames)
    height = max(frame.height for frame in frames)
    width = ((width + patch_size - 1) // patch_size) * patch_size
    height = ((height + patch_size - 1) // patch_size) * patch_size

    padded_frames = []
    masks = []
    for frame in frames:
        gap_x, gap_y = width - frame.width, height - frame.height
        left, top = gap_x // 2, gap_y // 2
        padding = (left, top, gap_x - left, gap_y - top)
        padded_frames.append(ImageOps.expand(frame.convert("RGB"), border=padding))
        masks.append(ImageOps.expand(Image.new("L", frame.size, 255), border=padding))

    batch = preprocess_images(padded_frames, max(width, height), patch_size, device, pil_masks=masks)
    batch["gradio_valid_pixels"] = batch.pop(SEGMENTATION_MASK_FIELD)
    return batch


def save_preview(output: Path, names: list[str], arrays: dict[str, np.ndarray]) -> None:
    depth_maps = np.ma.array(arrays["depths"], mask=~arrays["valid_pixels"])
    low, high = np.percentile(depth_maps.compressed(), [2, 98])

    figure = Figure(figsize=(10, 2.7 * len(names)), layout="constrained")
    axes = figure.subplots(len(names), 2, squeeze=False)
    for name, image, depth_map, (image_ax, depth_ax) in zip(names, arrays["images_rgb"], depth_maps, axes, strict=True):
        image_ax.imshow(image)
        image_ax.set_title(name, loc="left")
        image_ax.set_axis_off()
        depth_plot = depth_ax.imshow(depth_map, cmap="viridis", vmin=low, vmax=high)
        depth_ax.set_axis_off()

    axes[0, 1].set_title("Predicted depth", loc="left")
    figure.colorbar(depth_plot, ax=axes[:, 1], label="Depth (model units)", extend="both")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)


def save_viewer(output: Path, scene_name: str, arrays: dict[str, np.ndarray], cameras: Cameras) -> None:
    valid = arrays["valid_pixels"] & (arrays["depths"] > 0)
    points, colors = arrays["world_points"][valid], arrays["images_rgb"][valid]
    if len(points) > 75_000:
        indices = np.random.default_rng(0).choice(len(points), size=75_000, replace=False)
        points, colors = points[indices], colors[indices]

    figure = scene_to_plotly(scene_name, points, colors, cameras, view_coordinates="RDF")
    figure.update_layout(
        title=f"DVLT point cloud · {scene_name}",
        autosize=True,
        width=None,
        height=None,
        margin=dict(l=8, r=8, b=8, t=54),
    )
    figure.write_html(
        output,
        include_plotlyjs=True,
        default_height="95vh",
        config={"responsive": True, "scrollZoom": True, "displaylogo": False},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    work_dir = Path.home() / "work/dvlt"
    parser.add_argument("--images", type=Path, default=work_dir / "images/construction/3")
    parser.add_argument("--image-size", type=int, help="Resize the input long edge (default: native resolution)")
    parser.add_argument("--checkpoint", default="nvidia/dvlt", help="Hugging Face model ID or local checkpoint path")
    parser.add_argument("--inference-steps", type=int, default=16, help="Number of refinement steps (default: 16)")
    parser.add_argument(
        "--use-depth-conf", action="store_true", help="Weight camera fitting by predicted depth confidence"
    )
    parser.add_argument("--output-dir", type=Path, default=work_dir, help="Output directory (default: ~/work/dvlt)")
    parser.add_argument(
        "--preview",
        type=Path,
        help="Override the preview path (default: dvlt_preview.png in --output-dir)",
    )
    args = parser.parse_args()
    if args.inference_steps < 1:
        parser.error("--inference-steps must be at least 1")
    image_dir = args.images.expanduser()
    if not image_dir.is_dir():
        parser.error(f"image directory does not exist: {image_dir}")

    # The Tesla P4 lacks native BF16 support.
    accelerator = Accelerator(mixed_precision="no")
    model = DVLT(
        img_size=504,
        inference_steps=args.inference_steps,
        load_patch_embed_weights=False,
        use_depth_conf_for_pose=args.use_depth_conf,
    )
    if args.image_size is not None and args.image_size < model.patch_size:
        parser.error(f"--image-size must be at least {model.patch_size}")
    _, frames = load_sequence(image_dir)
    with ExitStack() as stack:
        for frame in frames:
            stack.enter_context(frame)
        if len(frames) < 2:
            parser.error(f"expected at least two views in {image_dir}")
        names = [Path(frame.filename).name for frame in frames]
        if args.image_size is None:
            batch = preprocess_native_images(frames, model.patch_size, accelerator.device)
        else:
            batch = preprocess_images(frames, args.image_size, model.patch_size, accelerator.device)

    _, views, _, height, width = batch[DataField.IMAGES].shape
    pose_weights = "depth confidence" if args.use_depth_conf else "uniform weights"
    print(
        f"Running DVLT on {views} images in {image_dir} at {width}x{height} using {accelerator.device} (FP32) "
        f"with {args.inference_steps} refinement steps and {pose_weights} for camera fitting",
        flush=True,
    )
    model.load_pretrained(expanduser(args.checkpoint), strict=True)
    model.setup_test(accelerator)
    with torch.no_grad():
        predictions = model.predict(batch, accelerator)

    depths = predictions[PredictionField.DEPTHS]
    points = predictions[PredictionField.WORLD_POINTS]
    (cameras,) = predictions[PredictionField.CAMERAS]
    camera_to_worlds = cameras.camera_to_worlds
    intrinsics = cameras.get_intrinsics_matrices()
    for name, tensor, expected_shape in (
        ("depths", depths, (1, views, height, width)),
        ("world points", points, (1, views, height, width, 3)),
        ("camera poses", camera_to_worlds, (views, 3, 4)),
        ("camera intrinsics", intrinsics, (views, 3, 3)),
    ):
        if tensor.shape != expected_shape:
            raise RuntimeError(f"{name}: expected shape {expected_shape}, got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{name}: found non-finite values")

    arrays = {
        field.value: value[0].cpu().numpy() for field, value in predictions.items() if isinstance(value, torch.Tensor)
    }
    arrays.update(
        image_names=np.asarray(names),
        images_rgb=(batch[DataField.IMAGES][0].permute(0, 2, 3, 1) * 255).round().byte().cpu().numpy(),
        valid_pixels=batch["gradio_valid_pixels"][0].cpu().numpy(),
        camera_to_worlds=camera_to_worlds.cpu().numpy(),
        intrinsics=intrinsics.cpu().numpy(),
    )
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    preview = args.preview.expanduser() if args.preview else output_dir / "dvlt_preview.png"
    archive = output_dir / "dvlt_predictions.npz"
    viewer = output_dir / "dvlt_point_cloud.html"
    np.savez_compressed(archive, **arrays)
    save_preview(preview, names, arrays)
    save_viewer(viewer, image_dir.name, arrays, cameras)
    print(f"PASS: {views} cameras and {width}x{height} depth maps with finite 3D points")
    print(f"Preview saved to {preview}")
    print(f"Predictions saved to {archive}")
    print(f"Point-cloud viewer saved to {viewer}")


if __name__ == "__main__":
    main()
