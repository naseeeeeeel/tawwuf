"""Hugging Face Space for traffic-video analysis and scenario exploration."""

from __future__ import annotations

import json
import math
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from gradio import Progress

PROGRESS_TRACKER = Progress(track_tqdm=False)

PERSON_CLASS_ID = 0
CAR_CLASS_ID = 2
MODEL_REPO = "Ultralytics/YOLOv8"
MODEL_FILE = "yolov8n.pt"
LANE_SCENE_MODEL_REPO = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
ROAD_CONDITIONS = ("Normal", "Wet road", "Roadworks", "Reduced visibility")
ROAD_CONTEXT_SPEED_CAPS = {
    "Narrow/local street": 40,
    "Urban street": 60,
    "Urban arterial": 80,
    "Highway/expressway": 100,
}
DEFAULT_ROAD_CONTEXT = "Narrow/local street"
WEATHER_ADJUSTMENT_FACTORS = {"precipitation", "visibility", "high wind"}
ROAD_CONDITION_FACTORS = {
    "Wet road": "precipitation",
    "Roadworks": "roadworks",
    "Reduced visibility": "visibility",
}
SAMPLE_VIDEO_REPO = "parsafaramarzi/Small-Crowd-and-Traffic-Video-Footages"
SAMPLE_VIDEOS = {
    "Car traffic sample 03 (13 MB)": "cartraffic03.mp4",
    "Car traffic sample 04 (9 MB)": "cartraffic04.mp4",
    "Crowd sample 03 (2.5 MB)": "crowd03.mp4",
    "Crowd sample 05 (4.1 MB)": "crowd05.mp4",
}


def recommended_speed_kmh(
    vehicle_count: int,
    lane_count: int = 1,
    road_condition: str = "Normal",
    weather_factors: tuple[str, ...] = (),
    road_context: str = DEFAULT_ROAD_CONTEXT,
) -> int:
    """Return a demo speed capped by road context and adjusted for conditions."""
    lane_count = _validate_lane_count(lane_count)
    if road_condition not in ROAD_CONDITIONS:
        raise ValueError("Choose one of the listed road conditions.")
    if road_context not in ROAD_CONTEXT_SPEED_CAPS:
        raise ValueError("Choose one of the listed road contexts.")
    unknown_factors = set(weather_factors) - WEATHER_ADJUSTMENT_FACTORS
    if unknown_factors:
        raise ValueError(f"Unknown weather adjustment factors: {unknown_factors}.")

    speed = _density_speed_kmh(vehicle_count, lane_count)

    adjustment_factors = set(weather_factors)
    if road_condition in ROAD_CONDITION_FACTORS:
        adjustment_factors.add(ROAD_CONDITION_FACTORS[road_condition])
    context_capped_speed = min(speed, ROAD_CONTEXT_SPEED_CAPS[road_context])
    return max(20, context_capped_speed - 20 * len(adjustment_factors))


def _weather_description(weather_code: int) -> str:
    if weather_code == 0:
        return "Clear"
    if weather_code in {1, 2}:
        return "Mostly clear / partly cloudy"
    if weather_code == 3:
        return "Overcast"
    if weather_code in {45, 48}:
        return "Fog"
    if weather_code in {51, 53, 55, 56, 57}:
        return "Drizzle"
    if weather_code in {61, 63, 65, 66, 67, 80, 81, 82}:
        return "Rain"
    if weather_code in {71, 73, 75, 77, 85, 86}:
        return "Snow"
    if weather_code in {95, 96, 99}:
        return "Thunderstorm"
    return "Unknown conditions"


def _weather_adjustment_factors(weather_code: int, wind_speed_kmh: float) -> set[str]:
    factors: set[str] = set()
    if weather_code in {45, 48}:
        factors.add("visibility")
    elif weather_code in {
        51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 71, 73, 75, 77,
        80, 81, 82, 85, 86, 95, 96, 99,
    }:
        factors.add("precipitation")
    if wind_speed_kmh >= 50:
        factors.add("high wind")
    return factors


def _open_meteo_json(url: str) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=10) as response:
            data = json.load(response)
    except json.JSONDecodeError as error:
        raise RuntimeError("Open-Meteo returned invalid JSON.") from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(
            "Could not retrieve current weather from Open-Meteo. "
            "Check the location and internet connection, then try again."
        ) from error
    if not isinstance(data, dict):
        raise TypeError("Open-Meteo returned an unexpected response.")
    return data


def fetch_open_meteo_weather(location: str) -> dict[str, Any]:
    """Geocode a place and retrieve its current conditions from Open-Meteo."""
    query = location.strip()
    if not query:
        raise ValueError("Enter a location for the Open-Meteo weather lookup.")

    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode(
        {"name": query, "count": 1, "language": "en", "format": "json"}
    )
    geocoding = _open_meteo_json(geocoding_url)
    results = geocoding.get("results", [])
    if not results:
        raise ValueError(f"Open-Meteo could not find a location for {query!r}.")

    place = results[0]
    if not isinstance(place, dict):
        raise TypeError("Open-Meteo returned invalid location data.")
    try:
        forecast_query = {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": (
                "temperature_2m,precipitation,rain,showers,snowfall,"
                "weather_code,wind_speed_10m"
            ),
            "wind_speed_unit": "kmh",
            "timezone": "auto",
        }
    except (KeyError, TypeError) as error:
        raise RuntimeError("Open-Meteo returned incomplete location data.") from error
    forecast_url = "https://api.open-meteo.com/v1/forecast?" + urlencode(
        forecast_query
    )
    forecast = _open_meteo_json(forecast_url)

    current = forecast.get("current")
    if not isinstance(current, dict) or "weather_code" not in current:
        raise RuntimeError("Open-Meteo returned no current weather for this location.")
    try:
        weather_code = int(current["weather_code"])
        wind_speed = float(current.get("wind_speed_10m", 0.0))
    except (TypeError, ValueError) as error:
        raise RuntimeError("Open-Meteo returned invalid current weather data.") from error

    location_parts = [str(place.get("name", query))]
    for part in (place.get("admin1"), place.get("country")):
        if part and str(part).casefold() not in {
            existing.casefold() for existing in location_parts
        }:
            location_parts.append(str(part))
    return {
        "location": ", ".join(part for part in location_parts if part),
        "timezone": str(forecast.get("timezone", place.get("timezone", ""))),
        "condition": _weather_description(weather_code),
        "weather_code": weather_code,
        "temperature_c": current.get("temperature_2m"),
        "precipitation_mm": current.get("precipitation"),
        "rain_mm": current.get("rain"),
        "snowfall_cm": current.get("snowfall"),
        "wind_speed_kmh": wind_speed,
        "adjustment_factors": _weather_adjustment_factors(
            weather_code,
            wind_speed,
        ),
    }


def _validate_lane_count(lane_count: Any) -> int:
    """Require a positive whole number of lanes."""
    try:
        normalized = int(lane_count)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Lane count must be a whole number of at least one.") from error
    if normalized < 1 or normalized != lane_count:
        raise ValueError("Lane count must be a whole number of at least one.")
    return normalized


def _density_speed_kmh(vehicle_count: int, lane_count: int) -> int:
    """Map peak vehicles per lane to an illustrative traffic speed band."""
    vehicles_per_lane = vehicle_count / lane_count
    if vehicles_per_lane < 5:
        return 100
    if vehicles_per_lane <= 15:
        return 80
    return 60


def _estimate_visible_lane_count(frame: Any, road_mask: Any = None) -> int | None:
    """Estimate lane count from road-surface line geometry in one frame."""
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    roi = np.zeros_like(edges)
    cv2.fillPoly(
        roi,
        [
            np.array(
                [
                    (0, int(height * 0.48)),
                    (width, int(height * 0.48)),
                    (width, height - 1),
                    (0, height - 1),
                ],
                dtype=np.int32,
            )
        ],
        255,
    )
    analysis_region = roi
    if road_mask is not None:
        road_mask = np.asarray(road_mask, dtype=np.uint8)
        if road_mask.shape != edges.shape:
            raise ValueError("Road segmentation mask must match the video frame.")
        road_mask = cv2.dilate(
            road_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        )
        analysis_region = cv2.bitwise_and(analysis_region, road_mask)

    lines = cv2.HoughLinesP(
        cv2.bitwise_and(edges, analysis_region),
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=max(24, int(height * 0.045)),
        maxLineGap=max(20, int(height * 0.08)),
    )
    if lines is None:
        return None

    bottom_y = height * 0.94
    candidates: list[tuple[float, float]] = []
    for line in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = (float(value) for value in line)
        delta_y = y2 - y1
        if abs(delta_y) < height * 0.04:
            continue
        slope = (x2 - x1) / delta_y
        if abs(slope) < 0.06 or abs(slope) > 6.0:
            continue
        projected_x = x1 + slope * (bottom_y - y1)
        if 0.01 * width <= projected_x <= 0.99 * width:
            candidates.append((projected_x, math.hypot(x2 - x1, delta_y)))

    if len(candidates) < 4:
        return None

    candidates.sort()
    clusters: list[list[tuple[float, float]]] = []
    cluster_tolerance = max(12.0, width * 0.025)
    for candidate in candidates:
        if (
            not clusters
            or candidate[0] - clusters[-1][-1][0] > cluster_tolerance
        ):
            clusters.append([candidate])
        else:
            clusters[-1].append(candidate)

    supported_markings = [
        cluster
        for cluster in clusters
        if sum(length for _, length in cluster) >= height * 0.08
    ]
    # Two carriageways have two more outer boundaries than lane intervals.
    estimated_lanes = len(supported_markings) - 2
    if not 1 <= estimated_lanes <= 12:
        return None
    return estimated_lanes


@lru_cache(maxsize=1)
def load_lane_scene_segmenter() -> tuple[Any, Any]:
    """Load a Cityscapes road/sidewalk segmenter from the Hugging Face Hub."""
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    processor = AutoImageProcessor.from_pretrained(LANE_SCENE_MODEL_REPO)
    model = SegformerForSemanticSegmentation.from_pretrained(
        LANE_SCENE_MODEL_REPO
    ).eval()
    return processor, model


def _segment_road_surface(frame: Any) -> Any:
    """Return a road-only mask, excluding Cityscapes sidewalk predictions."""
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from torch.nn import functional as torch_functional

    processor, model = load_lane_scene_segmenter()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=Image.fromarray(rgb_frame), return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
        labels = torch_functional.interpolate(
            logits,
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)[0]
    return (labels.cpu().numpy() == 0).astype(np.uint8) * 255


def _select_consistent_lane_estimate(estimates: list[int]) -> int | None:
    """Choose a conservative estimate or reject inconsistent multi-frame results."""
    if len(estimates) < 2:
        return None
    if len(estimates) == 2:
        return min(estimates)
    if max(estimates) - min(estimates) > 1:
        return None

    from collections import Counter

    counts = Counter(estimates)
    max_count = max(counts.values())
    return min(value for value, count in counts.items() if count == max_count)


def _detection_class_ids(model: Any) -> tuple[list[int], set[int], set[int], set[int]]:
    """Resolve target class IDs by name for COCO and fine-tuned model heads."""
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    class_groups: dict[str, set[int]] = {
        "person": set(),
        "car": set(),
        "heavy": set(),
    }
    for class_id, class_name in items:
        normalized = str(class_name).strip().lower()
        if normalized == "person":
            class_groups["person"].add(int(class_id))
        elif normalized in {"car", "cars"}:
            class_groups["car"].add(int(class_id))
        elif normalized in {"truck", "trucks", "bus", "van"}:
            class_groups["heavy"].add(int(class_id))

    if not class_groups["car"]:
        raise RuntimeError("The loaded detector has no car class.")
    target_ids = sorted(
        class_groups["person"] | class_groups["car"] | class_groups["heavy"]
    )
    return (
        target_ids,
        class_groups["person"],
        class_groups["car"],
        class_groups["heavy"],
    )


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load the pretrained YOLOv8n detector from the Hugging Face Hub."""
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    weights_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )
    return YOLO(weights_path)


def load_sample_video(sample_name: str) -> str:
    """Download a selected public sample clip from the Hugging Face Hub."""
    from huggingface_hub import hf_hub_download

    filename = SAMPLE_VIDEOS.get(sample_name)
    if filename is None:
        raise ValueError("Choose one of the listed sample videos.")

    return hf_hub_download(
        repo_id=SAMPLE_VIDEO_REPO,
        repo_type="dataset",
        filename=filename,
    )


def _context_summary(
    school_peak: bool,
    road_condition: str,
    high_risk_zone: bool,
    road_context: str = DEFAULT_ROAD_CONTEXT,
) -> str:
    """Describe user-selected scenario context without implying sensor data."""
    flags = [
        "School-peak scenario: {}".format("On" if school_peak else "Off"),
        f"Road scenario: {road_condition}",
        f"Road context: {road_context}",
        "High-risk-zone scenario: {}".format("On" if high_risk_zone else "Off"),
    ]
    return " · ".join(flags)


def _make_trend_plot(timeline: Any) -> Any:
    """Create a static detection trend chart for Gradio's Matplotlib plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 4))
    for column, label in (
        ("Cars", "Cars"),
        ("Other vehicles", "Buses/trucks/vans"),
        ("People detected", "People"),
    ):
        axis.plot(timeline["Time (seconds)"], timeline[column], label=label)

    axis.set_xlabel("Time in clip (seconds)")
    axis.set_ylabel("Detections per analyzed frame")
    axis.set_title("Vehicle and pedestrian detections")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def _resolve_video_path(video_value: Any) -> str:
    """Extract the local file path from Gradio Video input formats."""
    if isinstance(video_value, dict):
        video_value = video_value.get("video")
    if isinstance(video_value, dict):
        video_value = video_value.get("path")
    if not video_value:
        raise ValueError("Upload or load a traffic video first.")
    return str(video_value)


def _draw_cached_detections(frame: Any, detections: list[tuple]) -> Any:
    """Keep the latest sampled boxes visible between model inference frames."""
    import cv2

    annotated = frame.copy()
    for (x1, y1, x2, y2), label in detections:
        top_left = (round(x1), round(y1))
        bottom_right = (round(x2), round(y2))
        cv2.rectangle(annotated, top_left, bottom_right, (0, 220, 0), 2)
        text_y = max(18, top_left[1] - 6)
        cv2.putText(
            annotated,
            label,
            (top_left[0], text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def process_video(
    video_path: Any,
    lane_count: int = 1,
    detect_lanes_automatically: bool = False,
    frame_stride: int = 5,
    school_peak: bool = False,
    road_condition: str = "Normal",
    weather_location: str = "",
    high_risk_zone: bool = False,
    road_context: str = DEFAULT_ROAD_CONTEXT,
    progress: Progress = PROGRESS_TRACKER,
) -> tuple[str, str, str, Any, str]:
    """Annotate a video and return recommendation, trend data, and summary."""
    lane_count = _validate_lane_count(lane_count)
    if road_condition not in ROAD_CONDITIONS:
        raise ValueError("Choose one of the listed road conditions.")
    if road_context not in ROAD_CONTEXT_SPEED_CAPS:
        raise ValueError("Choose one of the listed road contexts.")
    try:
        normalized_frame_stride = int(frame_stride)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Frame interval must be a positive whole number.") from error
    if normalized_frame_stride < 1 or normalized_frame_stride != frame_stride:
        raise ValueError("Frame interval must be a positive whole number.")
    frame_stride = normalized_frame_stride
    video_path = _resolve_video_path(video_path)
    weather = (
        fetch_open_meteo_weather(weather_location) if weather_location.strip() else None
    )

    import cv2
    import pandas as pd

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        capture.release()
        raise ValueError("Could not open the uploaded video.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    lane_estimate_frame_indices = (
        {
            0,
            total_frames // 4,
            total_frames // 2,
            (3 * total_frames) // 4,
            total_frames - 1,
        }
        if total_frames > 0
        else {0}
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_file:
        output_path = Path(output_file.name)

    writer = None
    frame_number = 0
    sample_count = 0
    track_class_observations: dict[int, Counter[str]] = {}
    tracking_ids_available = True
    lane_count_source = "manual"
    lane_estimate_status = ""
    lane_estimates: list[int] = []
    records: list[dict] = []
    cached_detections: list[tuple] = []
    succeeded = False

    try:
        if detect_lanes_automatically:
            progress(
                0,
                desc="Loading road/sidewalk segmentation model from Hugging Face...",
            )
            load_lane_scene_segmenter()
        progress(0, desc="Loading YOLOv8 from Hugging Face...")
        model = load_model()
        (
            detection_class_ids,
            person_class_ids,
            car_class_ids,
            heavy_vehicle_class_ids,
        ) = _detection_class_ids(model)
        progress(0.05, desc="Analyzing video frames...")
        progress_interval = max(1, total_frames // 50) if total_frames > 0 else 30

        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break

            if (
                detect_lanes_automatically
                and frame_number in lane_estimate_frame_indices
            ):
                road_mask = _segment_road_surface(frame)
                detected_lane_count = _estimate_visible_lane_count(
                    frame,
                    road_mask=road_mask,
                )
                if detected_lane_count is not None:
                    lane_estimates.append(detected_lane_count)

            if frame_number % frame_stride == 0:
                result = model.track(
                    source=frame,
                    classes=detection_class_ids,
                    persist=True,
                    tracker="botsort.yaml",
                    verbose=False,
                )[0]
                boxes = result.boxes
                class_ids = boxes.cls.tolist() if boxes is not None else []
                people = sum(
                    int(class_id) in person_class_ids for class_id in class_ids
                )
                cars = sum(int(class_id) in car_class_ids for class_id in class_ids)
                other_vehicles = sum(
                    int(class_id) in heavy_vehicle_class_ids for class_id in class_ids
                )
                vehicles = cars + other_vehicles
                if boxes is not None and len(boxes):
                    track_ids = boxes.id
                    if track_ids is None:
                        tracking_ids_available = False
                    else:
                        for class_id, track_id in zip(
                            class_ids, track_ids.int().tolist()
                        ):
                            if int(class_id) in car_class_ids:
                                track_class_counts = track_class_observations.setdefault(
                                    track_id,
                                    Counter(),
                                )
                                track_class_counts["car"] += 1
                            elif int(class_id) in heavy_vehicle_class_ids:
                                track_class_counts = track_class_observations.setdefault(
                                    track_id,
                                    Counter(),
                                )
                                track_class_counts["other"] += 1
                    box_coords = boxes.xyxy.tolist()
                    class_names = result.names
                    cached_detections = [
                        (
                            coords,
                            str(class_names[int(class_id)]),
                        )
                        for coords, class_id in zip(box_coords, class_ids)
                    ]
                    box_area = sum(
                        max(0.0, x2 - x1) * max(0.0, y2 - y1)
                        for x1, y1, x2, y2 in box_coords
                    )
                    frame_area = float(frame.shape[0] * frame.shape[1])
                    occupancy = min(100.0, box_area / frame_area * 100.0)
                else:
                    cached_detections = []
                    occupancy = 0.0

                records.append(
                    {
                        "Time (seconds)": frame_number / fps,
                        "Vehicles": vehicles,
                        "Cars": cars,
                        "Other vehicles": other_vehicles,
                        "People detected": people,
                        "Detected box area (%)": round(occupancy, 2),
                    }
                )
                annotated_frame = result.plot()
                sample_count += 1
            else:
                annotated_frame = _draw_cached_detections(frame, cached_detections)

            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"avc1"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        "Could not create an H.264 processed video. "
                        "Check that OpenCV has an H.264 encoder available."
                    )

            writer.write(annotated_frame)
            frame_number += 1
            if frame_number % progress_interval == 0:
                if total_frames > 0:
                    fraction = min(frame_number / total_frames, 1.0)
                    progress(
                        0.05 + 0.9 * fraction,
                        desc=f"Analyzing frame {frame_number} of {total_frames}",
                    )
                else:
                    progress(
                        None,
                        desc=f"Analyzed {frame_number} frames",
                    )

        if not frame_number or not records:
            raise ValueError("The uploaded video contains no readable frames.")

        selected_lane_estimate = _select_consistent_lane_estimate(lane_estimates)
        if selected_lane_estimate is not None:
            lane_count = selected_lane_estimate
            lane_count_source = (
                f"automatic road-area estimate from {len(lane_estimates)} frame(s)"
            )
        elif detect_lanes_automatically:
            lane_estimate_status = (
                f"Automatic road-area estimates {lane_estimates or 'unavailable'} "
                "were inconclusive; the entered lane count was used."
            )

        timeline = pd.DataFrame(records)
        progress(0.97, desc="Preparing charts and results...")
        trend_plot = _make_trend_plot(timeline)
        succeeded = True

        traffic_peak = timeline.loc[timeline["Vehicles"].idxmax()]
        peak_vehicle_count = int(traffic_peak["Vehicles"])
        peak_car_count = int(traffic_peak["Cars"])
        peak_other_vehicle_count = int(traffic_peak["Other vehicles"])
        vehicles_per_lane = peak_vehicle_count / lane_count
        density_speed = _density_speed_kmh(peak_vehicle_count, lane_count)
        road_speed_cap = ROAD_CONTEXT_SPEED_CAPS[road_context]
        speed = recommended_speed_kmh(
            peak_vehicle_count,
            lane_count=lane_count,
            road_condition=road_condition,
            weather_factors=(
                tuple(weather["adjustment_factors"]) if weather is not None else ()
            ),
            road_context=road_context,
        )
        if tracking_ids_available:
            track_class_totals = Counter(
                observations.most_common(1)[0][0]
                for observations in track_class_observations.values()
            )
            total_unique_vehicles = len(track_class_observations)
            unique_vehicle_summary = (
                f"Tracked class totals: {track_class_totals['car']} cars, "
                f"{track_class_totals['other']} buses/trucks/vans"
            )
            total_vehicle_summary = (
                f"Total unique vehicles in clip: {total_unique_vehicles}"
            )
        else:
            unique_vehicle_summary = (
                "Unique vehicle count unavailable: the model did not return "
                "tracking IDs for every detected vehicle."
            )
            total_vehicle_summary = unique_vehicle_summary
        condition_text = (
            f"**Road condition scenario:** {road_condition} "
            "(manually selected)."
        )
        if weather is None:
            condition_text += (
                "\n\n**Open-Meteo weather:** not requested. Enter a location "
                "to retrieve current conditions and include weather in the "
                "demo speed adjustment."
            )
        else:
            condition_text += (
                f"\n\n**Current Open-Meteo weather for "
                f"{weather['location']}:** {weather['condition']}"
            )
            if weather["temperature_c"] is not None:
                condition_text += f", {weather['temperature_c']}°C"
            if weather["precipitation_mm"] is not None:
                condition_text += f", precipitation {weather['precipitation_mm']} mm"
            condition_text += (
                f", wind {weather['wind_speed_kmh']} km/h"
            )
            condition_text += (
                ". This is current local weather, not historical weather "
                "at the video's recording time."
            )
            if weather["adjustment_factors"]:
                condition_text += (
                    "\n\nWeather speed factors applied: "
                    + ", ".join(sorted(weather["adjustment_factors"]))
                    + "."
                )
            else:
                condition_text += "\n\nNo adverse-weather speed factor detected."
        has_manual_adjustment = road_condition in ROAD_CONDITION_FACTORS
        if has_manual_adjustment or (
            weather is not None and weather["adjustment_factors"]
        ):
            condition_text += (
                "\n\nEach distinct factor lowers the road-context-capped demo "
                "recommendation by 20 km/h, down to 20 km/h. "
                "Duplicate wet-road/rain or reduced-visibility/fog factors "
                "are counted once."
            )
        speed_text = (
            f"## {speed} km/h\n"
            "Detector: pretrained YOLOv8n (recommended)\n\n"
            f"Demo only · {total_vehicle_summary}\n\n"
            f"Peak sampled frame: {peak_vehicle_count} vehicles across "
            f"{lane_count} lane(s) "
            f"({vehicles_per_lane:.1f} vehicles per lane)\n\n"
            f"Road context: {road_context} "
            f"(demo cap {road_speed_cap} km/h) · "
            f"Traffic-density band: {density_speed} km/h\n\n"
            f"Class labels at peak: {peak_car_count} cars, "
            f"{peak_other_vehicle_count} buses/trucks/vans\n\n"
            f"Weather location: {weather['location'] if weather else 'not entered'}"
        )
        speed_text += f"\n\nLane count: {lane_count} ({lane_count_source})"
        if lane_estimate_status:
            speed_text += f"\n\n{lane_estimate_status}"
        speed_text += f"\n\n{unique_vehicle_summary}"

        traffic_lull = timeline.loc[timeline["Vehicles"].idxmin()]
        people_peak = timeline.loc[timeline["People detected"].idxmax()]
        crowd_note = (
            "People detected are a rough visual count, not a measure of "
            "pedestrian behavior or unique people."
        )
        timing_note = (
            "Peak times below are elapsed times within this uploaded clip; "
            "they do not establish daily rush-hour or school-hour patterns."
        )
        source_frames = total_frames if total_frames > 0 else frame_number
        lane_source_summary = f"- Lane count source: {lane_count_source}"
        if lane_estimates:
            lane_source_summary += f" · estimates: {lane_estimates}"
        if lane_estimate_status:
            lane_source_summary += f" · {lane_estimate_status}"
        if person_class_ids:
            people_summary = (
                f"- Peak people detections: "
                f"{int(people_peak['People detected'])} at "
                f"{float(people_peak['Time (seconds)']):.1f}s into the clip\n"
            )
        else:
            people_summary = (
                "- People detection: unavailable in this vehicle-only model\n"
            )
        summary = (
            "### Clip analysis\n"
            "- Detection model: pretrained YOLOv8n from Hugging Face\n"
            f"- Frames: {source_frames} · analyzed: {sample_count} · "
            f"detected every {frame_stride} frame(s)\n"
            f"- {total_vehicle_summary}\n"
            f"- Peak vehicles in one frame: "
            f"{peak_vehicle_count} at "
            f"{float(traffic_peak['Time (seconds)']):.1f}s into the clip\n"
            f"- Class labels at peak: {peak_car_count} cars, "
            f"{peak_other_vehicle_count} buses/trucks/vans\n"
            f"- Weather lookup: "
            f"{weather['condition'] + ' at ' + weather['location'] if weather else 'not requested'}\n"
            f"- Lowest vehicle detections: {int(traffic_lull['Vehicles'])} at "
            f"{float(traffic_lull['Time (seconds)']):.1f}s into the clip\n"
            f"{people_summary}"
            f"- Peak detected box area: "
            f"{float(timeline['Detected box area (%)'].max()):.1f}% of frame\n\n"
            f"- Lanes used: {lane_count} · peak vehicles per lane: "
            f"{vehicles_per_lane:.1f}\n\n"
            f"{lane_source_summary}\n"
            f"- {unique_vehicle_summary}\n"
            "  Tracking can still miss a vehicle or assign a new ID after "
            "occlusion, so totals are estimates rather than ground truth. "
            "Boxes are held between analyzed frames and may lag vehicle motion.\n\n"
            "**Other scenario inputs (simulated, not live feeds):** "
            f"{_context_summary(school_peak, road_condition, high_risk_zone, road_context)}\n\n"
            f"**Scope:** {crowd_note} {timing_note}\n\n"
            "**Demo rule:** thresholds apply to all detected vehicles per lane "
            "(<5: 100; 5–15: 80; >15: 60 km/h), then the selected road context "
            "caps that result (40/60/80/100 km/h). Rain, snow, fog, or wind "
            "at least 50 km/h lowers 20 km/h per distinct factor, to a 20 km/h "
            "floor; manually selected roadworks and road conditions can add factors. "
            "These values are not validated "
            "speed limits. "
            "No live camera, weather service, school calendar, road-condition "
            "feed, accident data, or traffic sign is connected."
        )
        progress(1, desc="Analysis complete")
        return str(output_path), speed_text, condition_text, trend_plot, summary
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not succeeded:
            output_path.unlink(missing_ok=True)


def search_hf_datasets(query: str) -> Any:
    """Search the Hugging Face Hub dataset catalog and report license metadata."""
    if not query.strip():
        raise ValueError("Enter a topic to search.")

    import pandas as pd
    from huggingface_hub import list_datasets

    rows = []
    for dataset in list_datasets(
        search=query.strip(),
        sort="downloads",
        direction=-1,
        limit=10,
    ):
        card = getattr(dataset, "cardData", None)
        license_name = None
        if card is not None:
            license_name = (
                card.get("license")
                if isinstance(card, dict)
                else getattr(card, "license", None)
            )
        rows.append(
            {
                "Dataset": dataset.id,
                "License in Hub card": license_name or "Not declared",
                "Downloads": getattr(dataset, "downloads", 0) or 0,
                "Hugging Face page": f"https://huggingface.co/datasets/{dataset.id}",
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Dataset",
            "License in Hub card",
            "Downloads",
            "Hugging Face page",
        ],
    )


def launch_app() -> None:
    import gradio as gr

    css = """
    :root {
        color-scheme: dark;
        --ink: #e6edf5;
        --muted: #9aaabd;
        --line: #28384a;
        --paper: #0a111b;
    }
    .gradio-container {
        max-width: 1440px !important;
        padding: 28px clamp(16px, 4vw, 56px) 48px !important;
        background:
            radial-gradient(ellipse at 9% 0%, rgba(19,173,158,.13), transparent 34rem),
            radial-gradient(ellipse at 90% 55%, rgba(48,91,157,.09), transparent 42rem),
            var(--paper);
        color: var(--ink);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .app-hero {
        position: relative; display: flex; min-height: 270px; align-items: center;
        justify-content: space-between; gap: 30px; overflow: hidden; margin: 0 0 22px;
        padding: clamp(28px, 5vw, 58px); border: 1px solid rgba(255,255,255,.12);
        border-radius: 27px;
        background: radial-gradient(circle at 82% 18%, rgba(54,211,190,.28), transparent 17rem),
            radial-gradient(circle at 100% 100%, rgba(67,132,245,.27), transparent 23rem),
            linear-gradient(118deg, #10243b 0%, #142f49 56%, #154d5d 100%);
        box-shadow: 0 20px 55px rgba(18,36,58,.15); color: white;
    }
    .app-hero::after {
        position: absolute; inset: 0;
        background-image: linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
        background-size: 38px 38px; content: ""; mask-image: linear-gradient(90deg, transparent, black);
        pointer-events: none;
    }
    .hero-copy, .hero-art { position: relative; z-index: 1; }
    .hero-copy { max-width: 700px; }
    .hero-eyebrow {
        display: inline-flex; align-items: center; gap: 9px; margin-bottom: 18px;
        color: #a7eee4; font-size: 11px; font-weight: 750; letter-spacing: .16em;
        text-transform: uppercase;
    }
    .hero-eyebrow::before {
        width: 8px; height: 8px; border-radius: 50%; background: #49e0c5;
        box-shadow: 0 0 16px rgba(73,224,197,.8); content: "";
    }
    .hero-copy h1 {
        margin: 0 0 12px; color: white; font-size: clamp(36px, 5vw, 58px);
        font-weight: 760; letter-spacing: -.055em; line-height: 1.02;
    }
    .hero-copy p { max-width: 610px; margin: 0; color: #c3d2e0; font-size: 15px; line-height: 1.7; }
    .hero-tags { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 21px; }
    .hero-tag {
        padding: 7px 11px; border: 1px solid rgba(255,255,255,.16); border-radius: 999px;
        background: rgba(255,255,255,.07); color: #e2eef5; font-size: 11px; font-weight: 600;
    }
    .hero-art { display: grid; width: 190px; height: 170px; flex: 0 0 190px; place-items: center; }
    .hero-orbit {
        display: grid; width: 140px; height: 140px; place-items: center;
        border: 1px solid rgba(157,238,222,.37); border-radius: 50%;
        background: radial-gradient(circle, rgba(61,205,185,.22), rgba(61,205,185,.025) 69%);
        box-shadow: 0 0 0 18px rgba(61,205,185,.035), 0 0 0 38px rgba(61,205,185,.025);
        transform: rotate(-12deg);
    }
    .hero-orbit-inner {
        display: grid; width: 92px; height: 92px; place-items: center;
        border: 1px solid rgba(255,255,255,.16); border-radius: 25px;
        background: linear-gradient(145deg, rgba(255,255,255,.19), rgba(255,255,255,.045));
        color: #9af0df; font-size: 38px; font-weight: 750; transform: rotate(12deg);
    }
    .safety-note {
        margin: 0 0 22px !important; padding: 13px 17px !important;
        border: 1px solid #59472c !important; border-radius: 14px !important;
        background: #241e16 !important; color: #e4c990 !important; font-size: 13px;
    }
    .input-panel, .results-panel, .research-panel {
        padding: clamp(17px, 2.2vw, 28px) !important; border: 1px solid var(--line) !important;
        border-radius: 21px !important; background: rgba(17,26,38,.96) !important;
        box-shadow: 0 14px 36px rgba(0,0,0,.20) !important;
    }
    .section-heading { margin: 2px 0 15px; }
    .section-heading .step { color: #0e978b; font-size: 10px; font-weight: 800; letter-spacing: .15em; text-transform: uppercase; }
    .section-heading .step { color: #62d7c4; }
    .section-heading h2 { margin: 5px 0 4px; color: var(--ink); font-size: 22px; font-weight: 720; letter-spacing: -.035em; }
    .section-heading p { margin: 0; color: var(--muted); font-size: 13px; line-height: 1.55; }
    .upload-video, .result-video, .trend-card {
        overflow: hidden; border: 1px solid var(--line) !important; border-radius: 15px !important;
        background: #0d1622 !important;
    }
    .input-panel label, .research-panel label { color: #d6e0ea !important; font-weight: 650 !important; }
    .input-panel input, .input-panel textarea, .input-panel select,
    .research-panel input, .input-panel [role="listbox"] {
        border-color: #344356 !important; border-radius: 11px !important;
        background-color: #111c2a !important; color: #e1eaf2 !important;
    }
    .input-panel .info, .research-panel .info { color: #91a1b3 !important; font-size: 11px !important; }
    .input-panel input::placeholder, .research-panel input::placeholder { color: #738399 !important; }
    .input-panel .wrap, .results-panel .wrap, .research-panel .wrap { color: #dce5ee; }
    .input-panel .block, .results-panel .block, .research-panel .block {
        border-color: #29394c !important;
    }
    .primary-action button {
        min-height: 49px; border: 0 !important; border-radius: 13px !important;
        background: linear-gradient(100deg, #11a998, #118f9e) !important;
        box-shadow: 0 8px 18px rgba(17,169,152,.22) !important; color: white !important;
        font-size: 14px !important; font-weight: 750 !important;
        transition: transform .18s ease, box-shadow .18s ease !important;
    }
    .primary-action button:hover { box-shadow: 0 11px 24px rgba(17,169,152,.3) !important; transform: translateY(-1px); }
    .secondary-action button {
        min-height: 42px; border: 1px solid #d6e5e9 !important; border-radius: 11px !important;
        border-color: #31534f !important; background: #142823 !important;
        color: #8ce2d2 !important; font-weight: 650 !important;
    }
    .subtle-note { color: #92a0b0; font-size: 11px; line-height: 1.6; }
    .speed-card {
        min-height: 150px; padding: 21px 24px !important;
        border: 1px solid rgba(19,173,158,.18) !important; border-radius: 18px !important;
        background: radial-gradient(circle at 100% 0%, rgba(48,204,182,.18), transparent 15rem),
            linear-gradient(120deg, #10283e, #163c50) !important;
        box-shadow: 0 12px 28px rgba(18,44,64,.12) !important; color: #dce9ef !important;
        text-align: left;
    }
    .speed-card h2 {
        margin: 2px 0 4px !important; color: #77ead6 !important;
        font-size: clamp(42px, 5vw, 58px) !important; font-weight: 780 !important;
        letter-spacing: -.06em !important; line-height: 1 !important;
    }
    .speed-card p { color: #b6cbd5 !important; font-size: 12px; }
    .condition-card {
        padding: 14px 17px !important; border: 1px solid #e1eaf0 !important;
        border-color: #2b3a4b !important; border-radius: 14px !important;
        background: #111b28 !important; color: #aab8c8 !important;
        font-size: 12px; line-height: 1.65;
    }
    .condition-card strong { color: #e1eaf3; }
    .analysis-summary { color: #aab8c8; font-size: 13px; line-height: 1.8; }
    .analysis-summary h3 { color: var(--ink); font-size: 17px; }
    .gradio-container .tabs { gap: 8px; padding: 6px; border: 1px solid #263548; border-radius: 14px; background: rgba(14,22,33,.88); }
    .gradio-container .tab-nav button { border-radius: 10px !important; color: #a3b1c2 !important; font-weight: 650 !important; }
    .gradio-container .tab-nav button.selected { background: #153b39 !important; color: #8be3d3 !important; }
    .gradio-container details { border: 1px solid #29394c !important; border-radius: 13px !important; background: #111b28 !important; color: #dce5ee; }
    .app-footer { margin-top: 25px; padding-top: 17px; border-top: 1px solid #263548; color: #8392a4; font-size: 11px; line-height: 1.7; text-align: center; }
    @media (max-width: 760px) {
        .gradio-container { padding: 14px 12px 30px !important; }
        .app-hero { min-height: auto; margin-bottom: 15px; padding: 27px 22px; border-radius: 21px; }
        .hero-art { display: none; }
        .hero-copy p { font-size: 14px; }
        .input-panel, .results-panel, .research-panel { padding: 16px !important; border-radius: 17px !important; }
    }
    """

    with gr.Blocks(
        title="Tawwuf | Traffic Intelligence Demo",
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="blue",
            neutral_hue="slate",
            radius_size="lg",
            spacing_size="md",
        ),
        css=css,
    ) as demo:
        gr.HTML(
            "<section class='app-hero'>"
            "<div class='hero-copy'>"
            "<h1>Tawwuf</h1>"
            "<p>Variable speed limits to help reduce road accidents.</p>"
            "</div>"
            "<div class='hero-art' aria-hidden='true'><div class='hero-orbit'>"
            "<div class='hero-orbit-inner'>T</div></div></div></section>"
        )
        gr.Markdown(
            "<div class='safety-note'>Research prototype · "
            "Not for operational speed control.</div>"
        )

        with gr.Column():
            with gr.Accordion("Analyze a video", open=True):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["input-panel"]):
                        gr.HTML(
                            "<div class='section-heading'>"
                            "<div class='step'>01 &nbsp; / &nbsp; Start here</div>"
                            "<h2>Bring your footage</h2>"
                            "<p>Upload a clip or load a sample to begin exploring.</p>"
                            "</div>"
                        )
                        input_video = gr.Video(
                            label="Your traffic video",
                            sources=["upload"],
                            format="mp4",
                            elem_classes=["upload-video"],
                        )
                        sample_video = gr.Dropdown(
                            choices=list(SAMPLE_VIDEOS),
                            value="Crowd sample 03 (2.5 MB)",
                            label="Or try a sample clip",
                            info="A public sample is cached after its first download.",
                        )
                        load_sample_button = gr.Button(
                            "Load selected sample",
                            elem_classes=["secondary-action"],
                        )
                        gr.Markdown(
                            "Sample source: "
                            "[Small Crowd and Traffic Video Footages]"
                            "(https://huggingface.co/datasets/"
                            "parsafaramarzi/Small-Crowd-and-Traffic-Video-Footages) "
                            "— MIT license declared in Hub metadata. Verify source "
                            "terms before reuse beyond testing.",
                            elem_classes=["subtle-note"],
                        )
                        gr.HTML(
                            "<div class='section-heading' style='margin-top:22px'>"
                            "<div class='step'>02 &nbsp; / &nbsp; Set the scene</div>"
                            "<h2>Road context</h2>"
                            "<p>Choose the visible lanes and closest road type.</p>"
                            "</div>"
                        )
                        lane_count = gr.Number(
                            value=3,
                            minimum=1,
                            maximum=12,
                            precision=0,
                            label="Visible lanes",
                            info="Used unless automatic estimation succeeds.",
                        )
                        road_context = gr.Dropdown(
                            choices=list(ROAD_CONTEXT_SPEED_CAPS),
                            value=DEFAULT_ROAD_CONTEXT,
                            label="Road type",
                            info="Sets the illustrative road-context cap.",
                        )
                        detect_lanes_automatically = gr.Checkbox(
                            label="Estimate lanes automatically",
                            value=False,
                            info=(
                                "Experimental road segmentation and line "
                                "estimation across up to five frames."
                            ),
                        )
                        frame_stride = gr.Slider(
                            minimum=1,
                            maximum=15,
                            value=5,
                            step=1,
                            label="Detection interval (every Nth frame)",
                            info="Higher intervals analyze faster; boxes may lag motion.",
                        )
                        with gr.Accordion("Fine-tune & scenario controls", open=False):
                            school_peak = gr.Checkbox(
                                label="School-peak scenario",
                                value=False,
                            )
                            road_condition = gr.Dropdown(
                                list(ROAD_CONDITIONS),
                                value="Normal",
                                label="Road condition scenario",
                                info="Manually selected; this is not detected from video.",
                            )
                            weather_location = gr.Textbox(
                                label="Location for current weather",
                                placeholder="e.g., Doha, Qatar",
                                info="Open-Meteo current weather may not match the video's recording time.",
                            )
                            high_risk_zone = gr.Checkbox(
                                label="High-risk-zone scenario",
                                value=False,
                            )
                        analyze_button = gr.Button(
                            "Analyze video",
                            variant="primary",
                            elem_classes=["primary-action"],
                        )
                        gr.Markdown(
                            "Only process video you are authorized to use. "
                            "Insights are approximate and for research only.",
                            elem_classes=["subtle-note"],
                        )
                    with gr.Column(scale=1, elem_classes=["results-panel"]):
                        gr.HTML(
                            "<div class='section-heading'>"
                            "<div class='step'>03 &nbsp; / &nbsp; Explore insights</div>"
                            "<h2>Your clip, in focus</h2>"
                            "<p>Detection summaries and visual trends from this clip.</p>"
                            "</div>"
                        )
                        speed_output = gr.Markdown(
                            "## -- km/h\nDemo estimate · run an analysis to see results",
                            elem_classes=["speed-card"],
                        )
                        condition_output = gr.Markdown(
                            "**Road conditions** &nbsp;·&nbsp; Waiting for analysis\n\n"
                            "Current weather is optional. Leave the location blank "
                            "to skip the weather lookup.",
                            elem_classes=["condition-card"],
                        )
                        output_video = gr.Video(
                            label="Processed clip",
                            elem_classes=["result-video"],
                        )
                        load_sample_button.click(
                            fn=load_sample_video,
                            inputs=sample_video,
                            outputs=input_video,
                        )
                trend_plot = gr.Plot(
                    label="Detection trends across clip time",
                    elem_classes=["trend-card"],
                )
                analysis_summary = gr.Markdown(
                    elem_classes=["analysis-summary"],
                )

                analyze_button.click(
                    fn=process_video,
                    inputs=[
                        input_video,
                        lane_count,
                        detect_lanes_automatically,
                        frame_stride,
                        school_peak,
                        road_condition,
                        weather_location,
                        high_risk_zone,
                        road_context,
                    ],
                    outputs=[
                        output_video,
                        speed_output,
                        condition_output,
                        trend_plot,
                        analysis_summary,
                    ],
                    concurrency_limit=1,
                )

            with gr.Accordion("Explore HF datasets", open=False):
                gr.HTML(
                    "<div class='section-heading'>"
                    "<div class='step'>Research tools</div>"
                    "<h2>Find data for your next experiment</h2>"
                    "<p>Search public datasets and review declared licensing.</p>"
                    "</div>"
                )
                gr.Markdown(
                    "Search the Hugging Face dataset catalog for research "
                    "candidates. Search results are not automatically used "
                    "for Qatar: review the dataset card, geographic coverage, "
                    "license, privacy terms, and data quality before use."
                )
                dataset_query = gr.Dropdown(
                    [
                        "traffic video",
                        "traffic weather",
                        "pedestrian traffic",
                        "road accident",
                    ],
                    value="traffic video",
                    label="Research topic",
                    allow_custom_value=True,
                )
                search_button = gr.Button("Search Hugging Face datasets")
                dataset_results = gr.Dataframe(
                    headers=[
                        "Dataset",
                        "License in Hub card",
                        "Downloads",
                        "Hugging Face page",
                    ],
                    datatype=["str", "str", "number", "str"],
                    interactive=False,
                    label="Dataset candidates",
                )
                search_button.click(
                    fn=search_hf_datasets,
                    inputs=dataset_query,
                    outputs=dataset_results,
                )
                gr.Markdown(
                    "**Data note:** Public datasets can help evaluate general "
                    "vehicle detection, but they do not supply authoritative "
                    "Qatar school hours, road conditions, accident-prone "
                    "locations, or citywide peak-hour patterns."
                )

            with gr.Accordion("How to interpret this demo", open=False):
                gr.HTML(
                    "<div class='section-heading'>"
                    "<div class='step'>Read the results responsibly</div>"
                    "<h2>A visual research prototype, not a control system</h2>"
                    "</div>"
                )
                gr.Markdown(
                    "### What the prototype measures\n"
                    "- YOLOv8 and BoT-SORT run on every Nth frame; boxes are "
                    "held between detections. Vehicle totals are estimated "
                    "from the analyzed frames.\n"
                    "- The trend plot shows detections during this clip only. "
                    "It does not track unique people or infer crossing intent "
                    "or other pedestrian behavior.\n"
                    "- Detected box area is a rough image-occupancy proxy, "
                    "not calibrated road density.\n"
                    "- The 100/80/60 km/h thresholds are the original demo "
                    "rule, based on vehicle detections in the busiest "
                    "frame. Wet road, roadworks, and reduced visibility "
                    "lower the demo result one band; school and high-risk-zone "
                    "controls do not alter it. Tracking totals remain "
                    "estimates because occlusion can break tracks.\n\n"
                    "### What is needed before a Qatar pilot\n"
                    "Authorized camera access; privacy review; reliable "
                    "timestamped weather, school-calendar, roadworks, and "
                    "incident feeds; locally representative labeled data; "
                    "traffic-engineering validation; human approval; and "
                    "fail-safe sign-control integration. This Space provides "
                    "none of those operational connections."
                )

        gr.HTML(
            "<footer class='app-footer'>"
            "TAWWUF &nbsp;·&nbsp; TRAFFIC INTELLIGENCE RESEARCH PROTOTYPE"
            "<br>Use only authorized video. Insights are approximate and "
            "not intended for safety-critical or operational decisions."
            "</footer>"
        )

    demo.launch()


if __name__ == "__main__":
    launch_app()
