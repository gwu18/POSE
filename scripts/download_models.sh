#!/usr/bin/env bash
# Download MediaPipe Pose Landmarker models into models/
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
base=https://storage.googleapis.com/mediapipe-models/pose_landmarker
for v in full heavy lite; do
  out=models/pose_landmarker_${v}.task
  [ -f "$out" ] || curl -fL -o "$out" "$base/pose_landmarker_${v}/float16/latest/pose_landmarker_${v}.task"
done
ls -lh models
