#!/usr/bin/env bash
# Reproducible build of ORB-SLAM3 (GPLv3, used as a separate executable) and
# the headless stereo-inertial driver stereo_inertial_kb.cc.
#
#   build_orbslam3.sh <step> <prefix>
#   step:   deps | pangolin | orbslam3 | driver | all
#   prefix: install tree, e.g. /opt/orbslam3 (Docker) or <worktree>/tools/orbslam3 (WSL)
#
# Used by hands/odometry/Dockerfile (one step per layer). In a Linux/WSL shell
# with sudo, `bash build_orbslam3.sh all <worktree>/tools/orbslam3` does the same.
set -euo pipefail
STEP=${1:?step}
PREFIX=${2:?prefix}
HERE="$(cd "$(dirname "$0")" && pwd)"
PANGOLIN_TAG=v0.8                                         # aff6883c83f3fd7e8268a9715e84266c42e2efe3
ORBSLAM3_COMMIT=4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4  # master, "Update Dependencies.md"
JOBS=4
SUDO=""; [ "$(id -u)" != 0 ] && SUDO=sudo
mkdir -p "$PREFIX"

deps() {
  $SUDO apt-get update
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates pkg-config libeigen3-dev libopencv-dev \
    libboost-serialization-dev libssl-dev libglew-dev libepoxy-dev libgl1-mesa-dev libpython3-dev
}

pangolin() {
  cd "$PREFIX"
  [ -d Pangolin ] || git clone --depth 1 --branch "$PANGOLIN_TAG" https://github.com/stevenlovegrove/Pangolin.git
  cmake -S Pangolin -B Pangolin/build -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_TOOLS=OFF -DBUILD_PYPANGOLIN_MODULE=OFF \
    -DBUILD_PANGOLIN_FFMPEG=OFF -DBUILD_PANGOLIN_LIBOPENNI=OFF -DBUILD_PANGOLIN_LIBOPENNI2=OFF \
    -DBUILD_PANGOLIN_REALSENSE=OFF -DBUILD_PANGOLIN_REALSENSE2=OFF -DBUILD_PANGOLIN_PLEORA=OFF \
    -DBUILD_PANGOLIN_TELICAM=OFF -DBUILD_PANGOLIN_DEPTHSENSE=OFF -DBUILD_PANGOLIN_LIBDC1394=OFF \
    -DBUILD_PANGOLIN_V4L=OFF -DBUILD_PANGOLIN_LIBREALSENSE=OFF
  cmake --build Pangolin/build -j "$JOBS"
  $SUDO cmake --install Pangolin/build
  $SUDO ldconfig
}

orbslam3() {
  cd "$PREFIX"
  if [ ! -d ORB_SLAM3 ]; then
    git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
    git -C ORB_SLAM3 checkout "$ORBSLAM3_COMMIT"
  fi
  cd ORB_SLAM3
  # Patch 1: Pangolin 0.8 headers need >= C++14 (its targets add C++17 on top).
  sed -i 's/-std=c++11/-std=c++14/g' CMakeLists.txt
  # Patch 2 (upstream bug): the KannalaBrandt8 Camera2 loader reads Camera1.k1..k4.
  sed -i '/"Camera2.cy",found);/,/vCalibration = {fx,fy,cx,cy,k0,k1,k2,k3};/ s/"Camera1\.k/"Camera2.k/' src/Settings.cc
  grep -q 'k0 = readParameter<float>(fSettings,"Camera2.k1"' src/Settings.cc
  # Patch 3: the "Not enough motion for initializing" map reset (< 2 cm over two
  # keyframes in the first 10 s after IMU init) only for monocular. Stereo gives the
  # scale; on a head-worn camera of someone standing at a task it reset the map ~50 times.
  sed -i 's/if((mTinit<10.f) \&\& (dist<0.02))/if((mTinit<10.f) \&\& (dist<0.02) \&\& mbMonocular)/' src/LocalMapping.cc
  grep -q '(dist<0.02) && mbMonocular)' src/LocalMapping.cc
  # Patch 4: SaveTrajectoryEuRoC re-anchors every pose to the first keyframe's body
  # pose; in inertial modes keep the gravity-aligned world instead (what the
  # keyframe file already uses).
  sed -i 's/Twb = vpKFs\[0\]->GetImuPose();/Twb = Sophus::SE3f();/' src/System.cc
  [ "$(grep -c 'Twb = Sophus::SE3f();' src/System.cc)" = 2 ]
  # Patch 5: no -march=native (AVX makes Eigen want 32-byte alignment; mixed with
  # g2o's allocations it crashed with "double free" in Optimizer::PoseOptimization).
  sed -i 's/ -march=native//g' CMakeLists.txt Thirdparty/DBoW2/CMakeLists.txt Thirdparty/g2o/CMakeLists.txt
  for d in DBoW2 g2o; do
    cmake -S Thirdparty/$d -B Thirdparty/$d/build -DCMAKE_BUILD_TYPE=Release
    cmake --build Thirdparty/$d/build -j "$JOBS"
  done
  [ -f Vocabulary/ORBvoc.txt ] || tar -xf Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j "$JOBS" --target ORB_SLAM3   # the library only, not the examples
}

driver() {
  cd "$PREFIX/ORB_SLAM3"
  cp "$HERE/stereo_inertial_kb.cc" Examples/Stereo-Inertial/stereo_inertial_kb.cc
  # Patch 6: one extra CMake target for the headless driver.
  grep -q stereo_inertial_kb CMakeLists.txt || cat >> CMakeLists.txt <<'CMK'

add_executable(stereo_inertial_kb Examples/Stereo-Inertial/stereo_inertial_kb.cc)
target_link_libraries(stereo_inertial_kb ${PROJECT_NAME})
set_target_properties(stereo_inertial_kb PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo-Inertial)
CMK
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j "$JOBS" --target stereo_inertial_kb
  ls -l Examples/Stereo-Inertial/stereo_inertial_kb
}

case "$STEP" in
  deps) deps ;;
  pangolin) pangolin ;;
  orbslam3) orbslam3 ;;
  driver) driver ;;
  all) deps; pangolin; orbslam3; driver ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
