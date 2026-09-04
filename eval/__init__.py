"""Evaluation harness. Measures the pipeline with no human labels.

Three automatic measurements:

- synthetic recall: real face crops pasted into real frames, with the true box
  known by construction (eval/synthetic.py)
- consensus pseudo-labels: faces that at least two of three independent
  detectors agree on, used to measure recall and off-face masking on real
  footage (eval/consensus.py, eval/measure.py)
- hands untouched: hand regions from MediaPipe Hands, used to measure how much
  of the wearer's hands the mask touches (eval/oracle_mediapipe.py)

eval/oracle_mediapipe.py runs in .venv-eval, which holds mediapipe. Everything
else runs in the main .venv. Nothing here ships in the packaged app.
"""
