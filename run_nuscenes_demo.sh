#!/bin/bash
# nuScenes demo: build the predict tool, run your checkpoint zero-shot on
# nuScenes pedestrians, print the results table and render videos.
#
#   bash run_nuscenes_demo.sh [checkpoint.pt] ["video label"]
#
# Env overrides: NS=<nuScenes root containing v1.0-mini/ and samples/>, VERSION=v1.0-mini
set -e
PROJ=/fs/nexus-scratch/prat03/trajectory_forecast_cpp
NS=${NS:-/fs/nexus-scratch/prat03/nuscenes_data}
VERSION=${VERSION:-v1.0-mini}
CKPT=${1:-$PROJ/build/checkpoints_pedestrian_only/latest.pt}
LABEL=${2:-"trained on ETH/UCY, zero-shot on nuScenes"}
cd "$PROJ"

[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT"; exit 1; }
[ -f "$NS/$VERSION/sample_annotation.json" ] || { echo "nuScenes tables not found in $NS/$VERSION"; exit 1; }
[ -d "$NS/samples/CAM_FRONT" ] || { echo "camera images not found in $NS/samples/CAM_FRONT"; exit 1; }

python3 -c "import matplotlib, PIL, numpy" 2>/dev/null || pip install --user matplotlib pillow numpy
command -v ffmpeg >/dev/null || python3 -c "import imageio_ffmpeg" 2>/dev/null || pip install --user imageio-ffmpeg

echo "== building predict_windows =="
grep -q "add_executable(predict_windows" CMakeLists.txt || cat >> CMakeLists.txt << 'EOF2'

add_executable(predict_windows src/graph.cpp src/encoder.cpp src/domain_adapt.cpp src/heads.cpp src/predict_windows.cpp)
target_link_libraries(predict_windows "${TORCH_LIBRARIES}")
EOF2
cmake -S . -B build -DCMAKE_PREFIX_PATH=$PROJ/libtorch -DCMAKE_CUDA_ARCHITECTURES=86 > build_cmake.log
make -C build -j"$(nproc)" predict_windows

echo "== building nuScenes windows =="
python3 ns_prepare.py "$NS" --version "$VERSION" --out ns_work

echo "== running model =="
./build/predict_windows ns_work/windows.txt "$CKPT" ns_work/predictions.txt

echo "== scoring + rendering =="
python3 ns_render.py "$NS" ns_work/meta.json ns_work/predictions.txt --out ns_videos --label "$LABEL"
echo "done: videos and metrics.json are in $PROJ/ns_videos"
