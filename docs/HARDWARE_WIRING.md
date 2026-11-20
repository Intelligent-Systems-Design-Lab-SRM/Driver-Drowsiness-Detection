# Hardware wiring

The default buzzer output is **BCM GPIO 18**. The physical buzzer is not
included in this repository and its voltage/current requirements are unknown.
Use a transistor or MOSFET driver, a suitable external supply where required,
a flyback diode for inductive loads, and a shared ground. Never connect an
unknown or high-current buzzer directly to a Raspberry Pi GPIO.

Set the pin and polarity with environment variables:

```bash
export DROWSINESS_BUZZER_GPIO=18
export DROWSINESS_BUZZER_ACTIVE_HIGH=1
```

Before wiring hardware, test with:

```bash
export DROWSINESS_MOCK_BUZZER=1
python main.py
```

The buzzer is inactive until a critical prolonged eye-closure state is
detected. It is explicitly switched off during normal reset and shutdown.
