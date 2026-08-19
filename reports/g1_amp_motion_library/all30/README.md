# legged_lab G1-29DoF motion library analysis

Velocities are full-clip means in the root heading frame: +x is forward, +y is left, and positive yaw rate is a left/counter-clockwise turn.

The source configuration has 29 DoFs and passes exact name-based mapping to the current 23DoF policy. Removed DoFs: `waist_roll_joint, waist_pitch_joint, left_wrist_pitch_joint, right_wrist_pitch_joint, left_wrist_yaw_joint, right_wrist_yaw_joint`.

## All motions

| Motion | Frames | FPS | Duration (s) | Forward (m/s) | Lateral (m/s) | Yaw rate (rad/s) | Measured category | Source intent |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `B10_-__Walk_turn_left_45_stageii.pkl` | 179 | 29.916 | 5.950 | 0.883 | 0.004 | -0.011 | walk_straight | turn_left |
| `B11_-__Walk_turn_left_135_stageii.pkl` | 194 | 29.884 | 6.458 | 0.907 | 0.015 | 0.314 | walk_turn_left | turn_left |
| `B13_-__Walk_turn_right_90_stageii.pkl` | 168 | 29.911 | 5.583 | 0.983 | -0.056 | -0.258 | walk_turn_right | turn_right |
| `B14_-__Walk_turn_right_45_t2_stageii.pkl` | 174 | 30.000 | 5.767 | 1.031 | -0.001 | -0.167 | walk_turn_right | turn_right |
| `B15_-__Walk_turn_around_stageii.pkl` | 225 | 29.967 | 7.475 | 0.844 | -0.014 | 0.385 | walk_turn_left | mixed_direction |
| `B22_-__side_step_left_stageii.pkl` | 74 | 30.000 | 2.433 | -0.003 | 0.230 | 0.030 | walk_side_left | side_left |
| `B23_-__side_step_right_stageii.pkl` | 90 | 30.000 | 2.967 | -0.021 | -0.172 | 0.031 | walk_side_right | side_right |
| `B4_-_Stand_to_Walk_backwards_stageii.pkl` | 138 | 29.946 | 4.575 | -0.818 | 0.016 | -0.013 | walk_backward_transition | backward |
| `B9_-__Walk_turn_left_90_stageii.pkl` | 151 | 30.000 | 5.000 | 0.910 | -0.009 | 0.265 | walk_turn_left | turn_left |
| `C11_-_run_turn_left_90_stageii.pkl` | 84 | 29.822 | 2.783 | 1.925 | -0.014 | 0.520 | run_turn_left | turn_left |
| `C12_-_run_turn_left_45_stageii.pkl` | 66 | 30.000 | 2.167 | 2.685 | 0.058 | 0.402 | run_turn_left | turn_left |
| `C13_-_run_turn_left_135_stageii.pkl` | 92 | 29.838 | 3.050 | 1.891 | -0.134 | 0.690 | run_turn_left | turn_left |
| `C14_-_run_turn_right_90_stageii.pkl` | 95 | 29.921 | 3.142 | 2.033 | -0.103 | -0.429 | run_turn_right | turn_right |
| `C15_-_run_turn_right_45_stageii.pkl` | 94 | 30.000 | 3.100 | 2.199 | -0.162 | -0.196 | run_turn_right | turn_right |
| `C16_-_run_turn_right_135_stageii.pkl` | 94 | 29.763 | 3.125 | 1.901 | -0.053 | -0.786 | run_turn_right | turn_right |
| `C17_-_run_change_direction_stageii.pkl` | 91 | 29.836 | 3.016 | 2.334 | -0.262 | -0.274 | run_turn_right_transition | mixed_direction |
| `C1_-_stand_to_run_stageii.pkl` | 122 | 29.817 | 4.058 | 1.057 | -0.083 | -0.016 | run_straight_transition | straight |
| `C3_-_run_stageii.pkl` | 81 | 29.816 | 2.683 | 2.823 | -0.269 | 0.035 | run_straight | straight |
| `C4_-_run_to_walk_a_stageii.pkl` | 115 | 29.870 | 3.817 | 1.535 | -0.084 | 0.018 | run_straight_transition | straight |
| `C5_-_walk_to_run_stageii.pkl` | 107 | 30.000 | 3.533 | 2.031 | -0.005 | 0.085 | run_straight_transition | straight |
| `C6_-_stand_to_run_backwards_stageii.pkl` | 174 | 29.914 | 5.783 | -0.751 | 0.023 | -0.038 | run_backward_transition | backward |
| `C8_-_run_backwards_to_stand_stageii.pkl` | 120 | 29.938 | 3.975 | -1.241 | -0.039 | 0.187 | run_backward_transition | backward |
| `C9_-_run_backwards_turn_run_forward_stageii.pkl` | 107 | 29.860 | 3.550 | 0.241 | 0.162 | 1.017 | run_turn_left | backward |
| `Walk_B10_-_Walk_turn_left_45_stageii.pkl` | 252 | 29.911 | 8.392 | 0.734 | -0.016 | 0.102 | walk_turn_left | turn_left |
| `Walk_B13_-_Walk_turn_right_45_stageii.pkl` | 279 | 30.000 | 9.267 | 0.686 | 0.018 | -0.121 | walk_turn_right | turn_right |
| `Walk_B15_-_Walk_turn_around_stageii.pkl` | 341 | 29.934 | 11.358 | 0.596 | 0.000 | -0.250 | walk_turn_right | mixed_direction |
| `Walk_B16_-_Walk_turn_change_stageii.pkl` | 363 | 29.959 | 12.083 | 0.496 | 0.006 | -0.143 | walk_turn_right_transition | mixed_direction |
| `Walk_B22_-_Side_step_left_stageii.pkl` | 138 | 29.838 | 4.591 | 0.003 | 0.158 | 0.024 | walk_side_left | side_left |
| `Walk_B23_-_Side_step_right_stageii.pkl` | 107 | 29.860 | 3.550 | 0.016 | -0.189 | -0.017 | walk_side_right | side_right |
| `Walk_B4_-_Stand_to_Walk_Back_stageii.pkl` | 388 | 29.942 | 12.925 | -0.524 | 0.013 | -0.019 | walk_backward_transition | backward |

## Balanced pre-conversion candidates

These files still require the strict 29→23 conversion, joint-limit check, FK reconstruction, and Isaac replay before they are training-ready.

| Category | Motion | Duration (s) | Forward (m/s) | Lateral (m/s) | Yaw rate (rad/s) |
|---|---|---:|---:|---:|---:|
| walk_turn_left | `Walk_B10_-_Walk_turn_left_45_stageii.pkl` | 8.392 | 0.734 | -0.016 | 0.102 |
| walk_turn_left | `B11_-__Walk_turn_left_135_stageii.pkl` | 6.458 | 0.907 | 0.015 | 0.314 |
| walk_turn_right | `Walk_B13_-_Walk_turn_right_45_stageii.pkl` | 9.267 | 0.686 | 0.018 | -0.121 |
| walk_turn_right | `B13_-__Walk_turn_right_90_stageii.pkl` | 5.583 | 0.983 | -0.056 | -0.258 |
| walk_side_left | `B22_-__side_step_left_stageii.pkl` | 2.433 | -0.003 | 0.230 | 0.030 |
| walk_side_left | `Walk_B22_-_Side_step_left_stageii.pkl` | 4.591 | 0.003 | 0.158 | 0.024 |
| walk_side_right | `Walk_B23_-_Side_step_right_stageii.pkl` | 3.550 | 0.016 | -0.189 | -0.017 |
| walk_side_right | `B23_-__side_step_right_stageii.pkl` | 2.967 | -0.021 | -0.172 | 0.031 |
| run_straight | `C3_-_run_stageii.pkl` | 2.683 | 2.823 | -0.269 | 0.035 |
| run_turn_left | `C13_-_run_turn_left_135_stageii.pkl` | 3.050 | 1.891 | -0.134 | 0.690 |
| run_turn_left | `C11_-_run_turn_left_90_stageii.pkl` | 2.783 | 1.925 | -0.014 | 0.520 |
| run_turn_right | `C15_-_run_turn_right_45_stageii.pkl` | 3.100 | 2.199 | -0.162 | -0.196 |
| run_turn_right | `C14_-_run_turn_right_90_stageii.pkl` | 3.142 | 2.033 | -0.103 | -0.429 |

Missing desired categories: `walk_straight`.
