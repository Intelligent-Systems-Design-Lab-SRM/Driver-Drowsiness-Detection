# Raspberry Pi 4 setup

## Install

Use Raspberry Pi OS 64-bit and a Python virtual environment. Bookworm's
externally managed Python policy is why the virtual environment is required.

```bash
sudo apt update
sudo apt install -y python3-venv python3-opencv libatlas-base-dev
cd Driver-Drowsiness-Detection
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-pi.txt
```

Connect the USB webcam, then verify it is visible:

```bash
v4l2-ctl --list-devices
```

## Buzzer wiring

The default pin is **BCM GPIO 18**. Connect a 3.3 V-compatible active buzzer
through an appropriate transistor driver and common ground. Do not connect a
high-current or externally powered buzzer directly to a GPIO pin. Change
`DROWSINESS_BUZZER_GPIO` or `DROWSINESS_BUZZER_ACTIVE_HIGH` through the
environment when the wiring differs.

Test the application without sounding hardware:

```bash
export DROWSINESS_MOCK_BUZZER=1
python main.py
```

Press `B` to test the alarm, `R` to reset state, and `Q` to quit. Use
`DROWSINESS_HEADLESS=1` for a deployment without a desktop display.

## Run on boot

Create `/etc/systemd/system/drowsiness.service`:

```ini
[Unit]
Description=Driver drowsiness monitor
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/Driver-Drowsiness-Detection
Environment=DROWSINESS_MOCK_BUZZER=0
ExecStart=/home/pi/Driver-Drowsiness-Detection/.venv/bin/python main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable it with:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now drowsiness.service
sudo journalctl -u drowsiness.service -f
```

Stop safely with `sudo systemctl stop drowsiness.service`.

## Troubleshooting

- If the camera cannot open, check permissions and try
  `DROWSINESS_CAMERA_INDEX=1`.
- If `gpiozero` is unavailable, the application logs that it is using mock
  mode instead of failing silently.
- If ONNX Runtime cannot be installed for the Pi's Python/architecture,
  follow the documented compatibility warning in `FINAL_AUDIT_REPORT.md`
  rather than forcing a source build.
- Run `python -m unittest discover -s tests` before connecting hardware.
