---
title: Tawwuf Traffic Intelligence
emoji: 🚦
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: 4.44.1
python_version: 3.10
app_file: app.py
pinned: false
---

# Tawwuf Traffic Intelligence

An English-language Hugging Face Spaces demo for exploring vehicle and
pedestrian-count trends in an uploaded traffic clip.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

On Windows PowerShell, start the app from the project directory with:

```powershell
.\.venv\Scripts\python.exe app.py
```

Keep this terminal window open while using the app; the command runs the local
web server and is expected to keep running. Press `Ctrl+C` when you want to stop
the server.

Open the local URL printed by Gradio. The local app does not create a public
Gradio share link. Upload a video, enter the number of lanes
visible in the camera view, choose the road type for an illustrative speed cap,
enter a location for current weather, and select the road-condition scenario. The app
uses pretrained YOLOv8n from the Hugging Face Hub; the separately fine-tuned
model is not used because it misclassifies cars as buses, trucks, or vans. By
default, detection runs every 5th frame (about 6 detections/second for 30 FPS
video). A slider lets you trade speed for denser analysis. Boxes are kept
visible between analyzed frames using the most recent detections, so they may
lag moving vehicles by up to the selected interval. The app reports an
estimated unique-vehicle total and the peak among analyzed frames. The first
run downloads `yolov8n.pt` from
[Ultralytics/YOLOv8 on the Hugging Face Hub](https://huggingface.co/Ultralytics/YOLOv8).

When automatic lane estimation is enabled, it also downloads
[NVIDIA SegFormer B0 fine-tuned on Cityscapes](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024)
to separate road from sidewalk before estimating lane lines. These models are
public; no token is needed.

The optional **Estimate visible lane count automatically** checkbox first uses
the Hugging Face Cityscapes segmenter to restrict line analysis to predicted
road pixels, reducing sidewalk/footpath edges. OpenCV estimates lane lines at
up to five points across the video. With two estimates, it uses the lower count
to avoid a high outlier; with more estimates it requires close agreement.
Otherwise it falls back to the manual lane-count input and reports why. This is
still experimental: the segmenter is not validated for every camera angle, and
line estimates can be wrong with faded markings, occlusion, or unusual road
layouts. Verify the estimate. The first analysis can take longer while models
download and load. After that, models are cached locally. The app shows progress
for model loading and frame analysis. Longer or higher-resolution videos take
longer; raise the detection interval to reduce inference time.

Enter a place such as **Doha, Qatar** in **Location for current weather** to
look it up with Open-Meteo's free geocoding and forecast APIs. No API key is
required. Open-Meteo's current weather code maps precipitation (rain, snow,
drizzle, or thunderstorms) to a 20 km/h demo reduction, fog to a 20 km/h
reduction, and wind at or above 50 km/h to a 20 km/h reduction. Distinct
weather factors can each lower one band, down to 60 km/h. The manually selected
**Wet road** and **Reduced visibility** factors are not double-counted when
Open-Meteo reports precipitation or fog; **Roadworks** adds a separate band.
The entered location is sent to Open-Meteo for geocoding and weather lookup.
Leaving it blank skips weather lookup and leaves only the manually selected
road condition in effect.

Open-Meteo returns **current** weather, not historical weather for an uploaded
clip. It only represents the video if the clip was recorded at that place and
around the current time. Sample clips may be old or from an unknown location,
so their speed adjustment should be treated as a demonstration, not a
weather-aware measurement. The speed thresholds and weather adjustments are
illustrative, not validated traffic or safety recommendations.

## Optional Hugging Face training job

`train.py` fine-tunes YOLOv8n on the public
[UA-DETRAC-derived vehicle dataset](https://huggingface.co/datasets/thundarstrom/traffic-vehicle-detection)
and publishes the resulting weights to a model repo. The dataset Hub card
declares CC-BY-4.0; the script includes attribution in the output model card.
The dataset is not Qatar-specific. Its card reports more images than are
currently present in the pinned Hub revision, has no validation folder, and
contains orphaned training labels. The script filters to matching image/label
pairs, makes a reproducible validation split from the training images, and
leaves the test split for final evaluation.

After confirming you accept the dataset terms and possible compute charges,
authenticate with `hf auth login` and submit:

```bash
hf jobs uv run --flavor a10g-large --timeout 3h --secrets HF_TOKEN train.py \
  --epochs 20 \
  --dataset-repo thundarstrom/traffic-vehicle-detection \
  --model-repo Tawuff/tawwuf-traffic-yolov8n
```

The job installs the missing OpenCV system libraries in its container when
needed. This requires the job container to run with package-install privileges.
Jobs are billed while starting/running. At the documented A10G-large rate of
$1.50/hour, the three-hour timeout caps compute at about $4.50; actual cost
depends on runtime. The job publishes `best.pt` and test metrics to the model repo.

The completed experimental model is published at
[Tawuff/tawwuf-traffic-yolov8n](https://huggingface.co/Tawuff/tawwuf-traffic-yolov8n).
On its held-out test split it scored mAP50 0.318 and mAP50-95 0.220, so do not
assume fine-tuning improved the pretrained detector. Testing on the provided
traffic sample also showed it labeling many cars as buses; the underlying
dataset has a severe class imbalance. For that reason the app uses the
pretrained detector instead. Treat the fine-tuned model's labels and counts as
experimental; improving it requires representative, correctly labeled traffic
data and retraining.

To compare the fine-tuned checkpoint and pretrained YOLOv8n on the same public
traffic sample, run `.venv/bin/python test_model.py`. The script analyzes every
fifth frame by default, saves one annotated video per model plus
`comparison.json` under a new `runs/model-comparison-*` directory, and reports
per-frame detection summaries. Pass `--video path/to/clip.mp4` to test a local
clip or `--stride 1` to analyze every frame. This sample has no ground-truth
labels, so counts alone cannot determine which detector is more accurate.

The fine-tuned detector covers cars, buses, trucks, and vans, but not people.
When using it, pedestrian detections are unavailable. It is not validated for
Qatar roads or operational speed-control use.

The **Explore HF datasets** section searches Hugging Face's dataset catalog and
shows any license declared in each dataset card. It does not automatically
download datasets or treat search results as approved for Qatar.

For repeatable tests, select one of the sample videos in the **Analyze a
video** section and click **Load selected sample**. The first selection downloads
that MP4 from the public
[Small Crowd and Traffic Video Footages dataset](https://huggingface.co/datasets/parsafaramarzi/Small-Crowd-and-Traffic-Video-Footages);
subsequent runs use the Hugging Face cache. Its Hub metadata declares MIT, but
the dataset card has little provenance detail, so verify its terms before use
beyond testing.

## Hugging Face Space

The Space metadata is at the top of this `README.md`. The public model and
public dataset search need no token. Never commit a Hugging Face token. If
future private/gated data requires authentication, configure it as a Space
secret and review the source dataset's access terms first.

## Demo limits and safety

- Traffic and people trends describe only the uploaded clip. Peak times are
  elapsed clip time, not daily commute or school-hour patterns.
- Person detections are approximate counts, not unique-person tracking or
  pedestrian-behavior understanding. Detected box area is not calibrated road
  density.
- YOLO's BoT-SORT tracker estimates unique cars and other vehicles on sampled
  frames. The peak and speed demo use all detected car, bus, truck, and
  van classes, and show their class breakdown. A track's displayed class is
  its most frequent label across frames to reduce label flicker. The unique
  total is still an estimate, not ground truth: occlusion or a lost/reassigned
  track ID can cause missed or duplicated vehicles. A model can also
  misclassify vehicles (as the fine-tuned model does on some sample footage).
- Automatic lane estimation uses Cityscapes road/sidewalk segmentation followed
  by OpenCV line geometry, not a dedicated lane-marking model. Two frame
  estimates use the lower count; highly inconsistent estimates fall back to the
  manually entered lane count.
- The speed calculation divides vehicle detections in the busiest frame by
  the manually entered lane count or automatic estimate, then applies
  illustrative traffic bands: fewer than 5 vehicles/lane = 100 km/h, 5–15
  = 80 km/h, more than 15 = 60 km/h. A manually selected road context caps
  this result: narrow/local street 40, urban street 60, urban arterial 80,
  highway/expressway 100 km/h. Current Open-Meteo precipitation, fog, or wind
  ≥50 km/h lowers the capped result by 20 km/h per distinct factor, to a
  20 km/h floor. Manual wet
  road/reduced visibility factors are deduplicated against precipitation/fog;
  roadworks is a separate manual factor. This is only appropriate for a video
  matching the entered location and current time. School and high-risk-zone
  controls are simulated scenarios, not live feeds.
- Lane count does not establish road width, road type, posted speed, or a safe
  speed. Select the closest road context explicitly; these demo caps and
  vehicle-count bands are illustrative and are not validated traffic or legal
  speed recommendations.
- Dataset search may return data from other countries or with restrictive
  terms. Review provenance, license, privacy, and local relevance before
  research or use.
- This is not a traffic-management system. It has no government-camera
  connection, accident detector, or sign-control output. Operational use would
  require authorized data, local validation, privacy review, traffic-engineer
  approval, and fail-safe controls.

Use only video you are authorized to process.
