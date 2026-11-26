"""Browse saved DVLT point clouds in a local Three.js viewer."""

import argparse
import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from zipfile import BadZipFile

import numpy as np


ASSET_DIR = Path(__file__).with_name("point_cloud_browser")
ARCHIVE_NAME = "dvlt_predictions.npz"
MAX_POINTS = 500_000
POINT_FIELDS = {"projected": "world_points", "direct": "world_points_direct"}
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/viewer.css": ("viewer.css", "text/css; charset=utf-8"),
    "/viewer.js": ("viewer.js", "text/javascript; charset=utf-8"),
    # Vendored from https://registry.npmjs.org/three/-/three-0.180.0.tgz.
    "/vendor/three.module.min.js": ("vendor/three.module.min.js", "text/javascript; charset=utf-8"),
    "/vendor/three.core.min.js": ("vendor/three.core.min.js", "text/javascript; charset=utf-8"),
    "/vendor/OrbitControls.js": ("vendor/OrbitControls.js", "text/javascript; charset=utf-8"),
    "/vendor/LICENSE": ("vendor/LICENSE", "text/plain; charset=utf-8"),
}


def discover_runs(root: Path) -> dict[str, Path]:
    return {
        archive.parent.relative_to(root).as_posix(): archive
        for archive in sorted(root.rglob(ARCHIVE_NAME))
        if archive.is_file() and archive.resolve().is_relative_to(root)
    }


def camera_vertices(pose: np.ndarray, intrinsics: np.ndarray, height: int, width: int, depth: float) -> list:
    pixels = np.array([[0, width, width, 0], [0, 0, height, height], [1, 1, 1, 1]], dtype=float)
    corners = np.linalg.solve(intrinsics, pixels) * depth
    center = pose[:, 3:4]
    vertices = np.concatenate((center, pose[:, :3] @ corners + center), axis=1).T
    vertices[:, 1:] *= -1  # DVLT's OpenCV axes to Three.js Y-up, Z-back.
    return vertices.tolist()


@dataclass
class Predictions:
    points: np.ndarray
    colors: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray | None
    cutoffs: np.ndarray | None
    views: list[dict]
    point_source: str
    has_direct_points: bool


@lru_cache(maxsize=4)
def load_predictions(path: Path, modified_ns: int, changed_ns: int, size: int, point_source: str) -> Predictions:
    """Cache each reconstruction so slider changes do not reopen the archive."""
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("Expected a NumPy .npz prediction archive")
    with archive:
        has_direct_points = POINT_FIELDS["direct"] in archive
        if not has_direct_points:
            point_source = "projected"
        point_field = POINT_FIELDS[point_source]
        points = archive[point_field]
        if points.ndim != 4 or points.shape[-1] != 3 or points.dtype.kind != "f":
            raise ValueError(f"{point_field} must be a floating-point array of shape [views, height, width, 3]")
        views, height, width, _ = points.shape
        arrays = {
            name: archive[name]
            for name in ("images_rgb", "valid_pixels", "depths", "camera_to_worlds", "intrinsics", "image_names")
        }
        confidence = None
        confidence_fields = ("depths_conf", "world_points_direct_conf")
        if point_source == "direct":
            confidence_fields = confidence_fields[::-1]
        for name in confidence_fields:
            if name in archive:
                confidence = archive[name]
                break

    expected = {
        "images_rgb": (points.shape, "u"),
        "valid_pixels": ((views, height, width), "b"),
        "depths": ((views, height, width), "f"),
        "camera_to_worlds": ((views, 3, 4), "f"),
        "intrinsics": ((views, 3, 3), "f"),
        "image_names": ((views,), "U"),
    }
    for name, (shape, kind) in expected.items():
        if arrays[name].shape != shape or arrays[name].dtype.kind != kind:
            raise ValueError(f"Unexpected shape or dtype for {name}")
    if arrays["images_rgb"].dtype != np.uint8:
        raise ValueError("images_rgb must be uint8")
    if not (np.isfinite(arrays["camera_to_worlds"]).all() and np.isfinite(arrays["intrinsics"]).all()):
        raise ValueError("Camera matrices contain nonfinite values")

    depths = arrays["depths"]
    valid = arrays["valid_pixels"] & (depths > 0) & np.isfinite(depths) & np.isfinite(points).all(axis=-1)
    if not valid.any():
        raise ValueError("The run has no valid 3D points")

    cutoffs = None
    if confidence is not None:
        if confidence.shape != depths.shape or confidence.dtype.kind != "f":
            raise ValueError("Confidence must be a floating-point array matching the depth maps")
        valid_scores = confidence[valid & np.isfinite(confidence)]
        if valid_scores.size:
            cutoffs = np.percentile(valid_scores, np.arange(101))

    low, high = np.percentile(points[valid], (5, 95), axis=0)
    frustum_depth = max(float(np.linalg.norm(high - low)) * 0.04, 1e-4)
    view_data = []
    for index, image_name in enumerate(arrays["image_names"]):
        vertices = camera_vertices(
            arrays["camera_to_worlds"][index], arrays["intrinsics"][index], height, width, frustum_depth
        )
        view_data.append({"name": str(image_name), "frustum": vertices})
    return Predictions(
        points, arrays["images_rgb"], valid, confidence, cutoffs, view_data, point_source, has_direct_points
    )


def read_cloud(
    path: Path, percentile: int, view: int, point_source: str = "projected"
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    stat = path.stat()
    prediction = load_predictions(path, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, point_source)
    if view >= len(prediction.views):
        raise ValueError("Source image was not found in this run")
    selection = slice(None) if view == -1 else slice(view, view + 1)
    eligible = prediction.valid[selection]
    available = int(eligible.sum())
    cutoffs = prediction.cutoffs
    scores = prediction.confidence[selection] if cutoffs is not None else None
    if scores is not None and percentile == 100:
        eligible = np.zeros_like(eligible)
    elif scores is not None and percentile > 0:
        eligible = eligible & np.isfinite(scores) & (scores >= cutoffs[percentile])

    counts = eligible.sum(axis=(1, 2))
    matching = int(counts.sum())
    # Spend the display budget on matching points, proportionally across the selected views.
    budgets = counts.copy()
    if matching > MAX_POINTS:
        budgets, remainders = np.divmod(counts * MAX_POINTS, matching)
        remaining = MAX_POINTS - int(budgets.sum())
        budgets[np.argsort(-remainders, kind="stable")[:remaining]] += 1

    count = int(budgets.sum())
    positions = np.empty((count, 3), dtype="<f4")
    colors = np.empty((count, 3), dtype=np.uint8)
    confidence_ranks = np.empty(count if scores is not None else 0, dtype=np.uint8)
    rng = np.random.default_rng(0)
    offset = 0
    for index, (point_map, color_map, mask, budget) in enumerate(
        zip(prediction.points[selection], prediction.colors[selection], eligible, budgets.tolist(), strict=True)
    ):
        if not budget:
            continue
        pixels = np.flatnonzero(mask)
        if budget < len(pixels):
            pixels = rng.choice(pixels, size=budget, replace=False)
        end = offset + budget
        positions[offset:end] = point_map.reshape(-1, 3)[pixels]
        colors[offset:end] = color_map.reshape(-1, 3)[pixels]
        if scores is not None:
            sampled_scores = scores[index].reshape(-1)[pixels]
            ranks = np.searchsorted(cutoffs, sampled_scores, side="right") - 1
            ranks[~np.isfinite(sampled_scores)] = 255
            confidence_ranks[offset:end] = ranks
        offset = end
    positions[:, 1:] *= -1

    metadata = {
        "count": count,
        "matching": matching,
        "available": available,
        "has_confidence": scores is not None,
        "point_source": prediction.point_source,
        "has_direct_points": prediction.has_direct_points,
        "views": prediction.views[selection],
    }
    return metadata, positions, colors, confidence_ranks


class ViewerServer(ThreadingHTTPServer):
    def __init__(self, root: Path, port: int):
        self.root = root
        super().__init__(("127.0.0.1", port), ViewerHandler)


class ViewerHandler(BaseHTTPRequestHandler):
    server: ViewerServer

    def send_parts(self, status: int, content_type: str, *parts: bytes | memoryview) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(sum(len(part) for part in parts)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        for part in parts:
            self.wfile.write(part)

    def send_json(self, status: int, payload: dict) -> None:
        self.send_parts(status, "application/json", json.dumps(payload).encode())

    def do_GET(self) -> None:
        route = urlsplit(self.path)
        try:
            port = self.server.server_port
            allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if port == 80:
                allowed_hosts.update(("127.0.0.1", "localhost"))
            if self.headers.get("Host") not in allowed_hosts:
                self.send_json(400, {"error": "Open the viewer using its localhost URL."})
                return
            if route.path in STATIC_FILES:
                name, content_type = STATIC_FILES[route.path]
                self.send_parts(200, content_type, (ASSET_DIR / name).read_bytes())
            elif route.path == "/api/runs":
                self.send_json(200, {"runs": sorted(discover_runs(self.server.root))})
            elif route.path == "/api/cloud":
                query = parse_qs(route.query)
                run_id = query.get("run", [""])[0]
                archive = discover_runs(self.server.root).get(run_id)
                if archive is None:
                    self.send_json(404, {"error": "Prediction run was not found."})
                    return
                try:
                    percentile = int(query.get("percentile", ["40"])[0])
                    view = int(query.get("view", ["-1"])[0])
                    point_source = query.get("point_source", ["projected"])[0]
                    if not 0 <= percentile <= 100 or view < -1 or point_source not in POINT_FIELDS:
                        raise ValueError
                except ValueError:
                    self.send_json(400, {"error": "Invalid percentile, source image, or point reconstruction."})
                    return
                try:
                    metadata, positions, colors, confidence_ranks = read_cloud(archive, percentile, view, point_source)
                except (BadZipFile, EOFError, KeyError, OSError, ValueError, np.linalg.LinAlgError) as error:
                    self.send_json(422, {"error": f"Could not read predictions for {run_id}: {error}"})
                    return

                # Padded JSON, float32 XYZ, uint8 RGB, and optional uint8 confidence ranks.
                header = json.dumps(metadata).encode()
                padding = b"\0" * (-len(header) % 4)
                self.send_parts(
                    200,
                    "application/octet-stream",
                    struct.pack("<I", len(header)),
                    header,
                    padding,
                    memoryview(positions.reshape(-1)).cast("B"),
                    memoryview(colors.reshape(-1)).cast("B"),
                    memoryview(confidence_ranks),
                )
            else:
                self.send_json(404, {"error": "Page not found."})
        except (BrokenPipeError, ConnectionResetError):
            # A run change can cancel a response before it finishes streaming.
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "work/dvlt", help="Search root for prediction runs")
    parser.add_argument("--port", type=int, default=8000, help="Local HTTP port (default: 8000)")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"directory does not exist: {root}")
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    try:
        with ViewerServer(root, args.port) as server:
            print(f"DVLT viewer: http://127.0.0.1:{server.server_port}  (predictions under {root})", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        return
    except OSError as error:
        parser.exit(1, f"Could not start viewer: {error}\n")


if __name__ == "__main__":
    main()
