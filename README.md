# Simulink model

 `Transmission_Line_Model1.slx`
  The MATLAB script: `matlab/run_fault_sweep.m` 

![Model screenshot](model_screenshot.png)

## Block names the script looks for

| Block | Required name |
|---|---|
| Three-Phase Series RLC Branch (first line section) | `Line1` |
| Three-Phase Series RLC Branch (second line section) | `Line2` |
| Three-Phase Fault | `Fault` |
| Three-Phase RLC Load (the load at the receiving end) | `LoadBlock` |
| powergui | `powergui` |
| To Workspace (from the sending-end Vabc output) | variable `Vabc_s` (save format Array) |
| To Workspace (from the sending-end Iabc output) | variable `Iabc_s` (save format Array) |

The script finds the model by the name in its `mdl` setting (currently `Transmission_Line_Model1`)

## Parameters

| Item | Value |
|---|---|
| Frequency | 60 Hz |
| Source | Three-Phase Source, Yg, 263.1 kV line-to-line, 1 ohm, 26.526 mH (X = 10 ohm) |
| Line | R = 0.15 ohm/km, L = 1.3263 mH/km (X = 0.5 ohm/km), 40 km in total |
| Load | 381 MVA at 0.8 power factor lagging (304.8 MW, 228.6 MVAr), nominal 220 kV |
| Sampling | Discrete, 1/(60 x 360) s|
| Line1 and Line2 | set by the script each run (lengths d and 40 - d km) |
| Fault | phases, resistance and switching time set by the script; ground resistance 0.01 ohm |

The healthy model was first validated against a hand calculation: about 144.4 kV per phase at the sending end, about 127 kV and 1000 A at the receiving end, 13.6% voltage regulation and 94.4% efficiency. The source impedance was added afterwards, with the source voltage raised to 263.1 kV so that the sending-end voltage stays at about 144.3 kV.
