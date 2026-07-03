# rev13 autonomous report

_Autonomous 24h wrapper. Reset attempted via non-privileged variants only (no sudo, no reboot, no script modification)._

## Environment at start

- ts_iso: `2026-06-05T17:31:10`
- driver: `580.159.03`
- gpu_util: `17`
- gpu_temp: `53`
- git_commit: `dc478b1f88cd`

## End reason

All B-lite smoke variants failed gate. Privileged reset (sudo/reboot) needed; not attempted (user away).

## B-lite smoke variants

| variant | device | bg | sap | latency (ms) | skip% | wall (s) |
|---|---|---|---|---|---|---|
| v1 | GPU | L0 | `0.1214` | `9.4` | `0.0` | `5.4` |
| v1 | NPU | L0 | `0.0844` | `38.0` | `59.0` | `15.9` |
| v1 | GPU | L1_light | `0.1174` | `22.6` | `8.9` | `11.8` |
| v1 | NPU | L1_light | `0.0711` | `66.8` | `98.0` | `16.7` |
| v2 | GPU | L0 | `0.1214` | `8.6` | `0.0` | `5.0` |
| v2 | NPU | L0 | `0.0856` | `37.9` | `60.8` | `15.9` |
| v2 | GPU | L1_light | `0.1211` | `18.9` | `0.7` | `10.1` |
| v2 | NPU | L1_light | `0.0720` | `66.2` | `99.0` | `16.7` |
| v3 | GPU | L0 | `0.1214` | `8.5` | `0.0` | `4.9` |
| v3 | NPU | L0 | `0.0856` | `37.1` | `57.4` | `15.9` |
| v3 | GPU | L1_light | `0.1214` | `18.5` | `0.2` | `9.9` |
| v3 | NPU | L1_light | `0.0722` | `65.4` | `97.1` | `16.7` |

**winning variant**: `None`

## R1 per-detector level gates

R1 did not run.

---

_End. main_vision.tex / paper/tables/* NOT modified._
