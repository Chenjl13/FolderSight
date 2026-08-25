#!/usr/bin/env bash
set -euo pipefail

python pipelines/shape_prior_sam_test.py --image examples/1080p/1.jpg --output-dir outputs/demo_1080p/1 --config configs/pipeline_1080p.json --resolution 1080p --wallpaper-bg bg/bg_1080P.png
