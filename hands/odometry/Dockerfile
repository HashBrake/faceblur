# ORB-SLAM3 stereo-inertial runner (Ubuntu 22.04). Build from hands/odometry:
#   docker build -t faceblur-orbslam3:4452a3c "hands/odometry"
# ORB-SLAM3 is GPLv3; it runs as a separate executable that writes a trajectory file.
FROM ubuntu:22.04
# Same package list as deps() in build_orbslam3.sh, kept inline so that editing
# the script does not invalidate this (slow) layer.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates pkg-config libeigen3-dev libopencv-dev \
    libboost-serialization-dev libssl-dev libglew-dev libepoxy-dev libgl1-mesa-dev libpython3-dev \
 && rm -rf /var/lib/apt/lists/*
COPY build_orbslam3.sh /src/build_orbslam3.sh
RUN sed -i 's/\r$//' /src/build_orbslam3.sh \
 && bash /src/build_orbslam3.sh pangolin /opt/orbslam3 && rm -rf /opt/orbslam3/Pangolin
RUN bash /src/build_orbslam3.sh orbslam3 /opt/orbslam3 \
 && rm -rf /opt/orbslam3/ORB_SLAM3/.git /opt/orbslam3/ORB_SLAM3/Vocabulary/ORBvoc.txt.tar.gz
COPY stereo_inertial_kb.cc /src/stereo_inertial_kb.cc
RUN bash /src/build_orbslam3.sh driver /opt/orbslam3
WORKDIR /out
