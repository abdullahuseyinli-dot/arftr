"""Label-blind exact-native three-frame cache for the motion innovation."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from hac.actor_memory_base import file_sha256
from hac.body_witness_data import (
    CenterSourceRequest,
    PADDING_RGB,
    array_digest,
    validate_decoded_center,
)
from hac.rgb_witness_data import witness_geometry

OFFSETS = (-4, 0, 4)
FRAME_NAMES = ("past", "center", "future")
PHASE_NAMES = ("past_to_center", "center_to_future")
CROP_SIZE = 64
CAMERA_SIZE = (320, 180)
GEOMETRY_DIM = 6
CAMERA_FIELDS = 3


@dataclass(frozen=True)
class NativeMotionCache:
    sample_ids: np.ndarray
    crops: np.ndarray
    frame_available: np.ndarray
    phase_available: np.ndarray
    geometry: np.ndarray
    camera: np.ndarray
    source_image_sha256: np.ndarray
    shard_receipts: tuple[dict, ...]


def offset_request(center: CenterSourceRequest, offset: int) -> CenterSourceRequest:
    """Make an exact native request without changing supplied center geometry."""
    if offset not in OFFSETS:
        raise ValueError("Unprescribed native offset")
    index = center.native_index + offset
    pts = Fraction(index) / center.video.fps / center.video.time_base
    reason = None
    if pts.denominator != 1:
        reason = "nonintegral_exact_timestamp"
    elif index < center.video.minimum_index or index > center.video.maximum_index or index in center.video.missing_indices:
        reason = "exact_native_offset_absent"
    if center.native_box is None:
        reason = reason or "invalid_supplied_center_geometry"
    return replace(
        center,
        native_index=index,
        target_pts=int(pts) if pts.denominator == 1 else -1,
        nominal_seconds=(center.observation.center_frame + offset) / 30.0,
        native_seconds=float(Fraction(index) / center.video.fps),
        available=reason is None,
        unavailable_reason=reason,
    )


def decode_exact_triplet(
    requests: Sequence[CenterSourceRequest], *, verified_video: Mapping[str, object]
) -> tuple[list[np.ndarray | None], list[dict]]:
    """Decode the three exact PTS values in one bounded seek; never substitute."""
    if len(requests) != 3 or tuple(r.native_index - requests[1].native_index for r in requests) != OFFSETS:
        raise ValueError("Expected the fixed ordered native triplet")
    source = requests[1].video
    if any(r.video != source for r in requests):
        raise ValueError("Triplet crosses video sources")
    info = source.path.stat()
    if (verified_video.get("path") != str(source.path.resolve())
            or verified_video.get("sha256") != source.sha256
            or verified_video.get("size_bytes") != info.st_size
            or verified_video.get("mtime_ns") != info.st_mtime_ns):
        raise RuntimeError("Video verification receipt changed")
    images: dict[int, np.ndarray] = {}
    if any(r.available for r in requests):
        import av
        targets = {r.target_pts for r in requests if r.available}
        with av.open(str(source.path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            if Fraction(stream.time_base) != source.time_base or Fraction(stream.average_rate) != source.fps:
                raise RuntimeError("Decoder timing differs from verified source")
            container.seek(min(targets), stream=stream, backward=True)
            for count, frame in enumerate(container.decode(stream), 1):
                if frame.pts in targets:
                    images[int(frame.pts)] = frame.to_ndarray(format="rgb24")
                if targets <= images.keys() or count >= 256 or (frame.pts is not None and frame.pts > max(targets)):
                    break
    after = source.path.stat()
    if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Native video changed during triplet decode")
    output, audits = [], []
    empty = np.empty((0, 0, 3), dtype=np.uint8)
    for request in requests:
        image = images.get(request.target_pts)
        audit = validate_decoded_center(
            request, image if image is not None else empty,
            pts=request.target_pts if image is not None else None,
            time_base=source.time_base,
        )
        audit["video_verification"] = dict(verified_video)
        output.append(image if audit["decode_valid"] else None)
        audits.append(audit)
    return output, audits


def _mask_box(box: Sequence[float], source_size: tuple[int, int], extent: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    width, height = extent*(x2-x1), extent*(y2-y1)
    sx, sy = CAMERA_SIZE[0]/source_size[0], CAMERA_SIZE[1]/source_size[1]
    return (
        max(0, math.floor((cx-width/2)*sx)), max(0, math.floor((cy-height/2)*sy)),
        min(CAMERA_SIZE[0], math.ceil((cx+width/2)*sx)),
        min(CAMERA_SIZE[1], math.ceil((cy+height/2)*sy)),
    )


def estimate_camera_translation(
    earlier_rgb: np.ndarray,
    later_rgb: np.ndarray,
    actor_box: Sequence[float],
    *,
    source_size: tuple[int, int],
) -> tuple[np.ndarray, bool]:
    """Return native-pixel translation mapping earlier coordinates to later."""
    import cv2
    expected = (source_size[1], source_size[0], 3)
    if earlier_rgb.dtype != np.uint8 or later_rgb.dtype != np.uint8 or earlier_rgb.shape != expected or later_rgb.shape != expected:
        raise ValueError("Camera estimator requires exact native RGB frames")
    values = []
    for image in (earlier_rgb, later_rgb):
        small = cv2.resize(image, CAMERA_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        left, top, right, bottom = _mask_box(actor_box, source_size, 3.0)
        gray[top:bottom, left:right] = 128.0
        values.append(gray)
    window = cv2.createHanningWindow(CAMERA_SIZE, cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(values[0], values[1], window)
    valid = bool(np.isfinite([dx,dy,response]).all()
                 and abs(dx) < CAMERA_SIZE[0]*.25 and abs(dy) < CAMERA_SIZE[1]*.25)
    native = np.asarray([
        dx*source_size[0]/CAMERA_SIZE[0], dy*source_size[1]/CAMERA_SIZE[1], response
    ], dtype=np.float32)
    return (native if valid else np.zeros(3, np.float32)), valid


def aligned_actor_crop(
    rgb: np.ndarray,
    actor_box: Sequence[float],
    *,
    source_size: tuple[int, int],
    source_sampling_shift: Sequence[float] = (0.0, 0.0),
    extent: float = 1.5,
    output_size: int = CROP_SIZE,
) -> np.ndarray:
    """Sample in center coordinates with a prescribed source-coordinate shift."""
    import cv2
    if rgb.dtype != np.uint8 or rgb.shape != (source_size[1], source_size[0], 3):
        raise ValueError("Crop source differs from declared native geometry")
    x1,y1,x2,y2 = map(float,actor_box); cx,cy=(x1+x2)/2,(y1+y2)/2
    width,height=extent*(x2-x1),extent*(y2-y1)
    left,top=cx-width/2,cy-height/2
    dx,dy=map(float,source_sampling_shift)
    sx,sy=width/output_size,height/output_size
    matrix=np.asarray([[sx,0,left+.5*sx-.5+dx],[0,sy,top+.5*sy-.5+dy]],np.float32)
    crop=cv2.warpAffine(rgb,matrix,(output_size,output_size),
        flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(PADDING_RGB))
    if crop.shape!=(output_size,output_size,3) or crop.dtype!=np.uint8:
        raise RuntimeError("Malformed aligned crop")
    return crop


def phase_inputs(crops: np.ndarray, arm: str) -> np.ndarray:
    """Build matched six-channel ordered phase images."""
    pixels=np.asarray(crops,np.float32)/255.0
    if pixels.ndim!=5 or pixels.shape[1:]!=(3,CROP_SIZE,CROP_SIZE,3):
        raise ValueError("Expected N x three x64x64 RGB crops")
    earlier=pixels[:,:2]; later=pixels[:,1:]
    mean=(earlier+later)/2
    if arm=="appearance": second=mean
    elif arm=="signed_motion": second=later-earlier
    elif arm=="phase_destroyed": second=np.abs(later-earlier)
    else: raise ValueError("Unknown motion arm")
    result=np.concatenate((mean,second),axis=-1).transpose(0,1,4,2,3).astype(np.float32)
    return result


def motion_geometry(body: np.ndarray, camera: np.ndarray) -> np.ndarray:
    result=np.concatenate((np.asarray(body,np.float32),np.asarray(camera,np.float32).reshape(len(body),-1)),axis=1)
    if result.shape!=(len(body),12) or not np.isfinite(result).all():
        raise ValueError("Motion geometry must have12 finite inference fields")
    return result


def load_native_motion_cache(directory: Path, expected_ids: np.ndarray) -> NativeMotionCache:
    directory=Path(directory); lock_path=directory/"execution_lock.json"
    if not lock_path.is_file(): raise RuntimeError("Native-motion extraction lock missing")
    lock=json.loads(lock_path.read_text(encoding="utf-8")); ids=np.asarray(expected_ids).astype(str)
    if lock.get("rows")!=len(ids) or tuple(lock.get("frame_names",()))!=FRAME_NAMES:
        raise RuntimeError("Native-motion cache lock mismatch")
    crops=np.zeros((len(ids),3,CROP_SIZE,CROP_SIZE,3),np.uint8)
    frame_available=np.zeros((len(ids),3),bool);phase_available=np.zeros((len(ids),2),bool)
    geometry=np.zeros((len(ids),6),np.float32);camera=np.zeros((len(ids),2,3),np.float32)
    hashes=np.full((len(ids),3),"",dtype="U64");seen=np.zeros(len(ids),np.int8);receipts=[]
    for receipt_path in sorted(directory.glob("shard-*.json")):
        receipt=json.loads(receipt_path.read_text(encoding="utf-8"));start,stop=int(receipt["start"]),int(receipt["stop"])
        path=directory/f"shard-{start:04d}-{stop:04d}.npz"
        if receipt.get("status")!="NATIVE_MOTION_SHARD_COMPLETE" or receipt.get("npz_sha256")!=file_sha256(path):
            raise RuntimeError(f"Invalid native-motion shard: {path}")
        with np.load(path,allow_pickle=False) as saved:
            if not np.array_equal(saved["sample_ids"].astype(str),ids[start:stop]) or saved["crops"].shape!=(stop-start,3,64,64,3):
                raise RuntimeError("Native-motion shard schema mismatch")
            crops[start:stop]=saved["crops"];frame_available[start:stop]=saved["frame_available"]
            phase_available[start:stop]=saved["phase_available"];geometry[start:stop]=saved["geometry"]
            camera[start:stop]=saved["camera"];hashes[start:stop]=saved["source_image_sha256"].astype(str)
        seen[start:stop]+=1;receipts.append(receipt)
    if not np.all(seen==1) or not np.isfinite(camera).all() or not np.isfinite(geometry).all():
        raise RuntimeError("Native-motion cache incomplete/nonfinite")
    return NativeMotionCache(ids,crops,frame_available,phase_available,geometry,camera,hashes,tuple(receipts))
