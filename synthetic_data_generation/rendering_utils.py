from __future__ import annotations
from pathlib import Path
import os
import struct

import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import mediapy as media
import numpy as np
import torch
import tyro
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from typing_extensions import Literal, assert_never

from nerfstudio.cameras.camera_paths import get_path_from_json, get_spiral_path
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.utils.data_utils import get_depth_image_from_path
from nerfstudio.model_components.losses import normalized_depth_scale_and_shift
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps, install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import ItersPerSecColumn
from nerfstudio.viewer.server.utils import three_js_perspective_camera_focal_length

CONSOLE = Console(width=120)

def insert_spherical_metadata_into_file(
    output_filename: Path,
) -> None:
    """Inserts spherical metadata into MP4 video file in-place.
    Args:
        output_filename: Name of the (input and) output file.
    """
    # NOTE:
    # because we didn't use faststart, the moov atom will be at the end;
    # to insert our metadata, we need to find (skip atoms until we get to) moov.
    # we should have 0x00000020 ftyp, then 0x00000008 free, then variable mdat.
    spherical_uuid = b"\xff\xcc\x82\x63\xf8\x55\x4a\x93\x88\x14\x58\x7a\x02\x52\x1f\xdd"
    spherical_metadata = bytes(
        """<rdf:SphericalVideo
xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'
xmlns:GSpherical='http://ns.google.com/videos/1.0/spherical/'>
<GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>
<GSpherical:Spherical>True</GSpherical:Spherical>
<GSpherical:Stitched>True</GSpherical:Stitched>
<GSpherical:StitchingSoftware>nerfstudio</GSpherical:StitchingSoftware>
</rdf:SphericalVideo>""",
        "utf-8",
    )
    insert_size = len(spherical_metadata) + 8 + 16
    with open(output_filename, mode="r+b") as mp4file:
        try:
            # get file size
            mp4file_size = os.stat(output_filename).st_size

            # find moov container (probably after ftyp, free, mdat)
            while True:
                pos = mp4file.tell()
                size, tag = struct.unpack(">I4s", mp4file.read(8))
                if tag == b"moov":
                    break
                mp4file.seek(pos + size)
            # if moov isn't at end, bail
            if pos + size != mp4file_size:
                # TODO: to support faststart, rewrite all stco offsets
                raise Exception("moov container not at end of file")
            # go back and write inserted size
            mp4file.seek(pos)
            mp4file.write(struct.pack(">I", size + insert_size))
            # go inside moov
            mp4file.seek(pos + 8)
            # find trak container (probably after mvhd)
            while True:
                pos = mp4file.tell()
                size, tag = struct.unpack(">I4s", mp4file.read(8))
                if tag == b"trak":
                    break
                mp4file.seek(pos + size)
            # go back and write inserted size
            mp4file.seek(pos)
            mp4file.write(struct.pack(">I", size + insert_size))
            # we need to read everything from end of trak to end of file in order to insert
            # TODO: to support faststart, make more efficient (may load nearly all data)
            mp4file.seek(pos + size)
            rest_of_file = mp4file.read(mp4file_size - pos - size)
            # go to end of trak (again)
            mp4file.seek(pos + size)
            # insert our uuid atom with spherical metadata
            mp4file.write(struct.pack(">I4s16s", insert_size, b"uuid", spherical_uuid))
            mp4file.write(spherical_metadata)
            # write rest of file
            mp4file.write(rest_of_file)
        finally:
            mp4file.close()



def _render_trajectory_video(
    pipeline: Pipeline,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_names: List[str],
    render_width: float,
    render_height: float,
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video", 'generator'] = "video",
    camera_type: CameraType = CameraType.PERSPECTIVE,
):
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        render_width: Video width to render.
        render_height: Video height to render.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
        camera_type: Camera projection format type.
    """
    CONSOLE.print("[bold green]Creating trajectory " + output_format)
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)

    fps = len(cameras) / seconds

    cameras = cameras.to(pipeline.device)

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    if output_format == "images":
        output_image_dir = output_filename.parent / output_filename.stem
        output_image_dir.mkdir(parents=True, exist_ok=True)
    if output_format == "video":
        # make the folder if it doesn't exist
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        # NOTE:
        # we could use ffmpeg_args "-movflags faststart" for progressive download,
        # which would force moov atom into known position before mdat,
        # but then we would have to move all of mdat to insert metadata atom
        # (unless we reserve enough space to overwrite with our uuid tag,
        # but we don't know how big the video file will be, so it's not certain!)

    with ExitStack() as stack:
        writer = (
            stack.enter_context(
                media.VideoWriter(
                    path=output_filename,
                    shape=(
                        int(render_height * rendered_resolution_scaling_factor),
                        int(render_width * rendered_resolution_scaling_factor) * len(rendered_output_names),
                    ),
                    fps=fps,
                )
            )
            if output_format == "video"
            else None
        )
        with progress:
            for camera_idx in progress.track(range(cameras.size), description=""):
                objdata = pipeline.datamanager.train_dataset.metadata["obj_info"][camera_idx].to(
                    pipeline.model.object_meta["obj_metadata"].device
                )
                obj_metadata = pipeline.datamanager.eval_dataset.metadata["obj_metadata"].to(
                    pipeline.model.object_meta["obj_metadata"].device
                )
                camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx)
                camera_ray_bundle.metadata["object_rays_info"] = objdata


                batch_obj_dyn = camera_ray_bundle.metadata["object_rays_info"].view(
                    camera_ray_bundle.metadata["object_rays_info"].shape[0],
                    camera_ray_bundle.metadata["object_rays_info"].shape[1],
                    pipeline.model.config.max_num_obj,
                    pipeline.model.config.ray_add_input_rows * 3,
                )
                norm_sh = camera_ray_bundle.metadata["directions_norm"].shape
                camera_ray_bundle.metadata["directions_norm"] = camera_ray_bundle.metadata["directions_norm"].reshape(
                    norm_sh[0] * norm_sh[1], norm_sh[2]
                )

                camera_ray_bundle.metadata["object_rays_info"] = batch_obj_dyn.reshape(
                    batch_obj_dyn.shape[0] * batch_obj_dyn.shape[1], batch_obj_dyn.shape[2] * batch_obj_dyn.shape[3]
                )

                with torch.no_grad():
                    outputs = pipeline.model.get_outputs_for_camera_ray_bundle_render(camera_ray_bundle)
                render_image = []
                for rendered_output_name in rendered_output_names:
                    if rendered_output_name not in outputs:
                        CONSOLE.rule("Error", style="red")
                        CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                        CONSOLE.print(
                            f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center"
                        )
                        sys.exit(1)
                    if rendered_output_name == "depth":
                        depth = outputs["depth"]

                        filepath = pipeline.datamanager.train_dataparser_outputs.metadata["depth_filenames"][camera_idx]
                        scale_factor = pipeline.datamanager.train_dataparser_outputs.dataparser_scale * 0.01
                        depth_img_gt = get_depth_image_from_path(
                            filepath=filepath, height=render_height, width=render_width, scale_factor=scale_factor
                        )
                        depth_mask = torch.abs(depth_img_gt / scale_factor - 65535) > 1e-6
                        depth_gt = depth_img_gt.to(depth)
                        depth_gt = depth_gt * outputs["directions_norm"]
                        depth[~depth_mask] = 0.0
                        max_depth = depth_img_gt.max()
                        if pipeline.config.model.mono_depth_loss_mult > 1e-8:
                            scale, shift = normalized_depth_scale_and_shift(
                                outputs["depth"][None, ...], depth_gt[None, ...], depth_gt[None, ...] > 0.0
                            )
                            depth = depth * scale + shift

                        depth[depth > max_depth] = max_depth
                        outputs["depth"] = colormaps.apply_depth_colormap(depth)
                    if rendered_output_name == "semantics":
                        semantic_labels = torch.argmax(
                            torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1
                        )
                        colormap = (
                            pipeline.model.object_meta["semantics"]
                            .colors.clone()
                            .detach()
                            .to(outputs["semantics"].device)
                        )
                        semantic_colormap = colormap[semantic_labels]
                        outputs["semantics"] = semantic_colormap / 255.0
                    output_image = outputs[rendered_output_name].cpu().numpy()
                    if output_image.shape[-1] == 1:
                        output_image = np.concatenate((output_image,) * 3, axis=-1)
                    render_image.append(output_image)
                render_image = np.concatenate(render_image, axis=1)
                if output_format == "images":
                    media.write_image(output_image_dir / f"{camera_idx:05d}.png", render_image)
                if output_format == "video" and writer is not None:
                    writer.add_image(render_image)
                if output_format == 'generator':
                    yield render_image

    if output_format == "video":
        if camera_type == CameraType.EQUIRECTANGULAR:
            insert_spherical_metadata_into_file(output_filename)