# Landmark model setup

The application uses `face_landmarks.onnx`, a 68-point facial landmark model
from the MIT-licensed
[yinguobing/head-pose-estimation](https://github.com/yinguobing/head-pose-estimation)
project. Its documented contract is a 128x128 RGB face crop and 136 output
values representing 68 `(x, y)` landmark coordinates. The application maps
those coordinates to the standard 68-point layout:

- eyes: points 36-47;
- mouth: points 48, 54, 62, and 66.

Download the model from the project's release page, then place it at:

```text
models/face_landmarks.onnx
```

Do not commit the model blindly. Verify its SHA-256 checksum against the
release's published checksum when available. The code fails clearly if the
file is missing or does not produce exactly 136 values.

The repository does not include model binaries, and model inference has not
been run in this Windows environment because `onnxruntime` is not installed
in the existing virtual environment.
