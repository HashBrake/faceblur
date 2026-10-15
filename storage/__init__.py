"""Hot-storage tools.

Raw camera footage is kept on fast disks while it is worked on, and at the
volumes this project handles the disks fill. These tools shrink it without
losing what matters: HEVC conversion with quality and frame checks, encoder
calibration, random-access checks, CSV to Parquet for the side data, and
manifests that say what each stored file is.
"""
