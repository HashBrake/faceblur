"""World coordinates for the hands: stereo-inertial odometry with ORB-SLAM3.

``export`` writes a window of a recording in TUM-VI layout plus the ORB-SLAM3
settings, ``run_orbslam3`` runs the ORB-SLAM3 executable (GPLv3, built in a
Docker image by ``Dockerfile`` / ``build_orbslam3.sh``) and reads its
trajectory back.
"""
