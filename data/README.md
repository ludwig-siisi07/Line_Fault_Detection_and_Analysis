# Data dictionary


| File | Rows | Purpose |
|---|---|---|
| `fault_dataset_train.csv` | 1500 | Training set: 300 runs for each of NF, LG, LL, LLG, LLL |
| `fault_dataset_ood.csv` | 500 | Independent test set: 100 runs per class, different seed and wider conditions |

## Label and setting columns (never used as model inputs)

| Column | Meaning |
|---|---|
| `class` | 0 = no fault (NF), 1 = LG, 2 = LL, 3 = LLG, 4 = LLL |
| `class_name` | Text version of `class` |
| `pA`, `pB`, `pC`, `pG` | 1 if phase A, B, C or ground is involved in the fault |
| `d_km` | Fault distance from the sending end (km). Empty for no-fault runs |
| `Rf_ohm` | Fault resistance per phase (ohm). Meaningless for no-fault runs |
| `tf_s` | True fault inception time (s). The pre-fault event time for no-fault runs |
| `triggered` | 1 if the current-RMS trigger fired |
| `trig_delay_ms` | Time from the true fault start to the trigger (ms). Empty for no-fault runs |

## Feature columns (37)

All computed at the sending end from one cycle (16.7 ms) of data after the trigger, compared with one cycle before it. Voltages in volts RMS, currents in amps RMS, impedances in ohm.

| Columns | Meaning |
|---|---|
| `Va Vb Vc`, `Ia Ib Ic` | Phase voltage and current magnitudes after the fault |
| `dVa dVb dVc`, `dIa dIb dIc` | Ratio of after-fault to before-fault magnitude |
| `V0 V1 V2`, `I0 I1 I2` | Zero-, positive- and negative-sequence magnitudes |
| `I0_I1`, `I2_I1` | Sequence current ratios |
| `V1_I1`, `V2_I2`, `V0_I0` | Sequence voltage-to-current magnitude ratios |
| `Rpa Rpb Rpc`, `Xpa Xpb Xpc` | Resistance and reactance of phase voltage over phase current |
| `Rab Rbc Rca`, `Xab Xbc Xca` | Resistance and reactance of phase-to-phase loops (Vab/(Ia-Ib), etc.) |
| `R1`, `X1` | Resistance and reactance of V1/I1 |

Runs where the trigger missed a fault (none in these files) would have empty feature columns.

## How the independent set differs from the training set

| Setting | Training | Independent |
|---|---|---|
| Random seed | 42 | 7 |
| Fault distance | 2 to 38 km | 1 to 39 km |
| Maximum fault resistance | 50 ohm | 100 ohm |
| Load | active and reactive power scaled together by 0.8 to 1.2 | active power 0.5 to 1.3, reactive power 0.5 to 1.5, independently |
| Measurement errors and noise | 1x | 1.5x |
