# Final audit report

## Status summary

| Area | Status | Evidence |
|---|---|---|
| Legacy landmark dependency removal | PASS | No obsolete landmark-runtime references remain after audit |
| OpenCV camera and face detection | PASS | `CameraController` and Haar cascade path are implemented |
| ONNX landmark inference | PARTIAL | Implemented against a documented 68-point model contract; model binary and Pi runtime are not present locally |
| EAR/MAR calculations | PASS | Synthetic zero-width and invalid-coordinate guards are implemented |
| Drowsiness timing | PASS | Prolonged eye closure is critical; yawn alone is warning only |
| GPIO buzzer | PARTIAL | Mock mode and safe lifecycle tested; physical wiring is not tested |
| Tests | PASS | `python -m unittest discover -s tests -v`: 3 passed |
| Raspberry Pi performance | NOT TESTED | Requires the target Pi, model, and camera |

## Architecture

`main.py` coordinates `CameraController`, `UnifiedDetector`, and
`BuzzerController`. OpenCV detects the largest face; ONNX Runtime runs a
68-point landmark model once per detected face crop. Standard 68-point eye
and mouth landmarks feed EAR and MAR calculations.

## Dependencies and model

The legacy landmark runtime was removed from Python files, documentation, and
requirements. `requirements.txt` contains OpenCV, NumPy, ONNX Runtime, and Linux-only
GPIO Zero. The landmark model is intentionally not committed; see
`docs/MODEL_SETUP.md` for the verified source and model contract.

ONNX Runtime ARM64/Python 3.13 wheel availability must be confirmed on the
target Debian installation before deployment. No claim of Raspberry Pi
inference performance is made from Windows testing.

## Remaining deployment gates

1. Install the requirements in a Raspberry Pi virtual environment.
2. Download and verify `models/face_landmarks.onnx`.
3. Run the synthetic tests and a live webcam test.
4. Test the buzzer in mock mode, then validate the transistor-driven circuit.
5. Measure FPS, CPU, and memory on the actual Pi.

No commit or push was performed. Recommended commit message:

`Add ONNX Runtime Raspberry Pi drowsiness monitor`
