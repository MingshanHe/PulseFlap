# Fixed-base calibration data (Fig. 2)

Fixed-base load recordings of a single-wing assembly of the real wing mechanism, and the four aerodynamic models
evaluated against them.

- `fixed_base_mean_lift.csv`: mean lift at each of the six flapping frequencies 10-15 Hz (18 recordings, three per
  frequency): measurement with its SD, the four
  models uncalibrated (`be` = FlapSim's blade-element model, `flappy`, `sd` = adapted Sane-Dickinson, `ellipsoid` =
  MuJoCo's ellipsoid model), each with one output gain fitted at 10-12 Hz (`*_calibrated_mN`), and the uncalibrated
  errors in percent.
- `fixed_base_phase_example.csv`: one 15 Hz recording, periodic six-axis loads (harmonics 1-4) over one aligned
  wingbeat cycle, measurement (`actual_*`) and the four models.
- `fixed_base_nrmse.csv`: normalised RMS error of the oscillating load per model and axis.

Forces in mN, moments in mN·m. `figures/plot_figures.py calibration` draws Fig. 2 from these files.
