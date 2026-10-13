# Driver Drowsiness Detection

An **ISD Lab** computer-vision project that monitors a driver's face through a webcam and detects visual signs associated with drowsiness:

- prolonged eye closure using the **Eye Aspect Ratio (EAR)**; and
- prolonged mouth opening/yawning using the **Mouth Aspect Ratio (MAR)**.

The project uses **Python**, **OpenCV**, **MediaPipe Face Mesh**, and **NumPy**. It works in real time and displays the current measurements and detection status in a webcam window.

> **Important:** This is an academic prototype for the ISD Lab. It is not a certified driver-safety system and must not be used as the only method for deciding whether a person is fit to drive.

## Project idea

Driver fatigue can reduce attention and reaction time. This project explores a non-contact monitoring approach using a regular camera:

1. Capture frames from the webcam.
2. Detect facial landmarks with MediaPipe Face Mesh.
3. Measure the relative opening of the eyes and mouth.
4. Require the condition to persist for a short period before showing a warning. This helps avoid treating a normal blink or brief mouth movement as drowsiness.

The project proposal is included in [`Driver_Drowsiness_Proposal_ISD_Lab_Final.pdf`](./Driver_Drowsiness_Proposal_ISD_Lab_Final.pdf).

## Features

- Real-time webcam input.
- Face and facial-landmark detection.
- Eye-closure measurement using EAR.
- Mouth-opening/yawn measurement using MAR.
- Duration-based warnings instead of single-frame decisions.
- Separate scripts for testing the camera, face landmarks, eye closure, and yawning.
- A unified script for the complete demonstration.

## Requirements

### Hardware

- A working webcam. A built-in laptop camera is sufficient.
- A reasonably even-facing view of the driver's face.
- Adequate lighting. Very dark scenes, strong backlighting, and extreme side angles can reduce accuracy.

### Software

- Windows 10 or Windows 11
- Python **3.11** recommended
- Internet access while installing the Python packages
- Permission for Python/your terminal to access the camera

The application does not require a separate model download. MediaPipe's face-mesh resources are installed with the `mediapipe` package.

## Download the project

### Option 1: Clone with Git

Replace the placeholder URL with the URL of the GitHub repository after it is published:

```powershell
git clone <YOUR-REPOSITORY-URL>
cd Driver-Drowsiness-Detection
```

### Option 2: Download from GitHub

1. Open the repository page on GitHub.
2. Select **Code > Download ZIP**.
3. Extract the ZIP file.
4. Open PowerShell in the extracted `Driver-Drowsiness-Detection` folder.

## Installation on Windows

Open PowerShell in the project folder and run the following commands.

### 1. Confirm Python

```powershell
py --version
```

Python 3.11 is recommended because it is the version used while developing and checking this project. If `py` is not available, install Python from [python.org](https://www.python.org/downloads/windows/) and enable **Add python.exe to PATH** during installation.

### 2. Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

If Python 3.11 is not registered with the Python launcher, use:

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, allow scripts for the current user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then run the activation command again. Alternatively, the virtual environment can be used without activation by replacing `python` below with `.\.venv\Scripts\python.exe`.

### 4. Upgrade packaging tools

```powershell
python -m pip install --upgrade pip
```

### 5. Install dependencies

```powershell
python -m pip install opencv-python mediapipe numpy
```

The imports used by this project are:

| Package | Purpose |
|---|---|
| `opencv-python` | Webcam access, image conversion, drawing, and display |
| `mediapipe` | Face Mesh facial-landmark detection |
| `numpy` | Landmark arrays and distance calculations |

## Run the project

Run commands from the project folder with the virtual environment activated.

### Recommended: complete detection

```powershell
python unified_detection.py
```

The window is titled **Driver Drowsiness Detection System**. The screen shows EAR, MAR, and the current status.

### Run the individual demonstrations

| Command | Purpose |
|---|---|
| `python camera_test.py` | Verify that the webcam can be opened and frames can be read |
| `python face_detection.py` | Display MediaPipe face landmarks and face-detection status |
| `python eye_detection.py` | Display EAR and detect prolonged eye closure |
| `python yawn_detection.py` | Display MAR and detect a possible yawn |
| `python unified_detection.py` | Run eye-closure and mouth-opening detection together |

Press **Q** while the OpenCV window is focused to stop any script.

## How detection works

### Eye closure: EAR

The eye script uses six landmarks for each eye. EAR is calculated as:

```text
EAR = (vertical_distance_1 + vertical_distance_2)
      / (2 × horizontal_eye_width)
```

In the current implementation:

- `EAR_THRESHOLD = 0.20`
- `EYE_CLOSED_SECONDS = 1.0` in `unified_detection.py`

When the EAR remains below the threshold for at least one second, the unified detector reports prolonged eye closure.

### Mouth opening: MAR

MAR is calculated from the distance between the upper and lower lip relative to the mouth width:

```text
MAR = mouth_opening / mouth_width
```

In the current implementation:

- `MAR_THRESHOLD = 0.20`
- `YAWN_SECONDS = 1.2`

When the MAR remains above the threshold for at least 1.2 seconds, the unified detector reports a possible yawn.

These values are initial experimental settings, not universal medical or safety thresholds. Camera position, lighting, face shape, resolution, and individual behavior can affect them.

## Troubleshooting

### `Camera not found!` or `Camera not detected`

- Close applications that may already be using the webcam, such as Teams, Zoom, or another Python process.
- Check **Windows Settings > Privacy & security > Camera** and enable camera access for desktop applications.
- Try another camera index by changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in the script.
- Run `python camera_test.py` first to isolate camera problems from face-detection problems.

### `No face detected`

- Face the camera directly and move closer if the face is too small.
- Improve lighting and avoid strong light directly behind you.
- Remove anything that significantly covers the eyes or mouth.
- Keep the camera stable and within the camera's field of view.

### PowerShell will not activate `.venv`

Run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again. You can also run the scripts directly:

```powershell
.\.venv\Scripts\python.exe unified_detection.py
```

### `No module named cv2`, `mediapipe`, or `numpy`

Make sure the virtual environment is active, then run:

```powershell
python -m pip install opencv-python mediapipe numpy
```

Check which Python is being used:

```powershell
python -c "import sys; print(sys.executable)"
```

The printed path should end in `.venv\Scripts\python.exe`.

### The warning appears too early or too late

Calibrate the constants near the top of [`unified_detection.py`](./unified_detection.py):

- reduce or increase `EAR_THRESHOLD` for eye sensitivity;
- reduce or increase `EYE_CLOSED_SECONDS` for the required eye-closure duration;
- reduce or increase `MAR_THRESHOLD` for mouth-opening sensitivity; and
- reduce or increase `YAWN_SECONDS` for the required mouth-opening duration.

Change one value at a time and test under the same camera and lighting conditions. These changes are experimental and should not be interpreted as safety certification.

## Project structure

```text
Driver-Drowsiness-Detection/
├── camera_test.py
├── eye_detection.py
├── face_detection.py
├── unified_detection.py
├── yawn_detection.py
├── Driver_Drowsiness_Proposal_ISD_Lab_Final.pdf
└── README.md
```

## Limitations

- The current implementation processes one face (`max_num_faces=1`).
- It does not identify the driver or store results.
- It does not provide an audio alarm, mobile notification, database, or dashboard.
- It is sensitive to lighting, camera placement, occlusion, head pose, glasses, and landmark-tracking quality.
- EAR and MAR thresholds require calibration for a particular setup.
- A visual warning is not a substitute for rest, safe driving practices, or a validated alertness-monitoring device.

## Academic context

This repository contains the implementation for an ISD Lab project on real-time driver drowsiness detection. The proposal PDF documents the original project idea and motivation; the Python scripts provide the webcam-based proof of concept.

## License

No license has been specified for this project yet. If this repository will be shared publicly, add a license that matches the team's intended use before publishing.
