<!-- 작업 16 — 읽기 전용 조사. 코드·데이터·원고 미수정.
     증거 수집 스크립트: collect_evidence.py → facts.json -->

# 수집 호스트 CPU 확정

## 판정

> ## **Intel Core Ultra 9 285K — 직접 증거로 확정.**
> 원고 §2의 **"Intel Core i7-13700K"는 오기**다. i7-13700K를 지지하는 증거는
> 저장소·시스템 로그 어디에도 **0건**이다.

결정적 근거: **수집 기간 전체를 포함하는 단일 부트의 커널 로그가 CPU를 직접 이름으로
기록**하고 있고, 그 구간에 재부팅이 없어 CPU 교체가 물리적으로 불가능하다.

---

## 1. 현재 호스트

| 항목 | 값 |
|---|---|
| CPU | `Intel(R) Core(TM) Ultra 9 285K` (`/proc/cpuinfo` model name) |
| 논리 CPU | **24** (`nproc`) |
| Thread(s) per core | **1** (`lscpu`) |
| hostname | `5090pc` |
| machine-id | `b7a2af2355dd46619d432e1ba3c5c8d0` |
| 커널 | `6.8.0-83-generic`, Ubuntu 22.04.5 LTS |

---

## 2. 수집 시점 증거

### 2-1. 【직접】 수집 기간을 포함하는 부트의 커널 CPU 식별 ★

`journalctl --list-boots` 로 부트 이력을 열거하면, 수집 기간(2026-07-15 ~ 07-18)과
정적 프로파일 실행(2026-07-14)이 **단 하나의 부트 안에** 들어간다:

```
-1  fc6c1e84c52e4c65be1b6aff74ffc421   2026-05-26 18:21:57 KST — 2026-07-30 16:08:41 KST
```

그 부트(`journalctl -b -1 -k`)의 커널 메시지:

```
5월 26 18:21:57 5090pc kernel: smpboot: CPU0: Intel(R) Core(TM) Ultra 9 285K
                                       (family: 0x6, model: 0xc6, stepping: 0x2)
5월 26 18:21:57 5090pc kernel: smp: Brought up 1 node, 24 CPUs
5월 26 18:21:57 5090pc kernel: smpboot: Total of 24 processors activated (176947.20 BogoMIPS)
5월 26 18:21:57 5090pc kernel: DMI: ASUS System Product Name/TUF GAMING Z890-PLUS WIFI,
                                    BIOS 1404 01/09/2025
5월 26 18:21:58 5090pc kernel: aries 0000:85:00.0: Probing a device with vendor (0x209f)...
5월 26 18:21:59 5090pc kernel: NVRM: loading NVIDIA UNIX Open Kernel Module ... 580.159.03
```

그 부트가 수집 기간에 살아 있었다는 것도 같은 저널로 확인된다 —
2026-07-15 00:00 ~ 07-18 23:59 구간에 **466,666개**의 저널 항목이 그 부트 아래에 있다.

**이 한 줄이 판정을 끝낸다.** CPU 교체에는 전원 차단·재부팅이 반드시 따르는데,
수집 기간 안에 재부팅이 **없다**(부트 경계가 05-26과 07-30). 따라서 수집 내내 CPU는
커널이 기록한 그 CPU였다.

### 2-2. 【직접】 메인보드 소켓 — i7-13700K는 물리적으로 장착 불가

같은 부트의 DMI가 보드를 **ASUS TUF GAMING Z890-PLUS WIFI**로 기록한다.
Z890은 **LGA1851** 플랫폼이고 i7-13700K는 **LGA1700**이다. 보드를 바꾸지 않는 한
두 CPU는 서로 대체될 수 없다. 그리고 그 보드는 05-26 부트와 현재 부트에서 동일하다.

### 2-3. 【직접】 CPU 토폴로지도 두 후보를 구분한다

| | 논리 CPU | Thread(s)/core | 구성 |
|---|---|---|---|
| **Core Ultra 9 285K** (관측값) | **24** | **1** | 8 P-core + 16 E-core, HT 없음 |
| i7-13700K | 24 | 2 (P-core) | 8 P-core(16T) + 8 E-core, 16코어 |

관측된 `Thread(s) per core: 1` 은 285K와 일치하고 13700K와는 **불일치**한다.
"24"라는 숫자만으로는 구분이 안 되지만(둘 다 24 논리 CPU), 스레드/코어 비가 갈라 준다.

### 2-4. 【직접】 수집 기간 로테이션 커널 로그의 호스트명

`/var/log/kern.log.4.gz` 가 **Jul 12 00:09 ~ Jul 18 23:37** 을 덮는다 — 수집 기간을
그대로 포함한다. 그 안의 모든 줄이 호스트명 **`5090pc`** 로 찍혀 있고, 커널 uptime
카운터가 `[3995250]`(Jul 12) → `[4598103]`(Jul 18)로 **연속 증가**한다
(차이 602,853초 ≈ 6.98일 = 기간과 정확히 일치). 재부팅 흔적이 없다는 독립 확인이다.

### 2-5. 【정황】 정적 프로파일 실행 기록의 호스트명

`sample_profiling_data.json` 의 `profiling_runs[0]`:

```json
{"run_id": "run_20260714_164004", "timestamp": "2026-07-14T16:40:04",
 "host": "5090pc", "python": "3.10.12", "git_commit": "521e1d5", ...}
```

수집 하루 전(07-14)에 **`host: 5090pc`** 가 기록돼 있고, 현재 호스트명과 같다.
CPU 모델명은 이 레코드에 없다.

### 2-6. 【부재】 수집 산출물 자체에는 시스템 메타데이터가 없다

전수 확인 결과, 수집이 남긴 파일에는 host/cpu/machine/uname 류 필드가 **하나도 없다**:

| 파일 | 키 |
|---|---|
| `collect_{gpu,npu}_results.jsonl` | `combo, models, n_accel, rate, set, stratum, window, y1, y2, y3` |
| `collect_{gpu,npu}_anchor.jsonl` | `at_run, t_min, y1, y2, y3` |
| `performance_*_full540.json` (창) | `combination, models, rate_factor, total, window_sec, workload` |
| `oof_{gpu,npu}.json` | 예측·라벨 배열만 |

`collect_run.py` 는 stdout 진행 로그만 찍고 파일로 남기지 않으며, 그 로그가 있던
scratchpad(`/tmp/claude-1001/.../6f0af394-.../`)는 **이미 소실**됐다.

### 2-7. 【정황】 문서

CPU 모델을 언급하는 문서는 **`PRE_COLLECTION_FINAL.md:115` 하나뿐**이다
(2026-07-15 작성 = 수집 직전, CPU 부하 A/B 실험의 스로틀링 보조 증거):

```
Intel Core Ultra 9 285K (최대 5100 MHz)
부하 중 평균 2883 MHz  ← 최대의 56%
package_throttle_count: 21,426회
```

`FULL_COLLECTION_{GPU,NPU}_REPORT.md` 에는 CPU 모델 기록이 없다.

### 2-8. 【부재】 git·병렬도 설정

- 2026-07-08 ~ 07-22 커밋 메시지·커밋 파일에 호스트 단서 없음. git config user 변화 없음.
- 코어 수가 박힌 병렬도 상수 없음 — 유일한 스레드 설정은 `model_processors.py:176`의
  `torch.set_num_threads(1)`이고 이는 코어 수와 무관하다. (지시문이 우려한 대로 이 경로로는
  두 CPU를 구분할 수 없다. 구분은 2-3의 스레드/코어 비가 해 준다.)

### 2-9. i7-13700K 지지 증거

저장소 전체(`.md/.py/.yaml/.json/.txt/.sh`, `.venv`·`.git` 제외)에서
`i7-13700`, `13700K` grep 결과 **0건**. 시스템 로그에도 없다. **지지 증거 전무.**

---

## 3. 판정 등급 정리

| 등급 | 증거 | 결론 방향 |
|---|---|---|
| **직접** | 수집 기간을 포함하는 부트의 `smpboot: CPU0: Intel(R) Core(TM) Ultra 9 285K` (2-1) | **285K** |
| **직접** | 같은 부트의 DMI = Z890(LGA1851) — 13700K(LGA1700) 장착 불가 (2-2) | **285K** |
| **직접** | Thread(s)/core = 1 — 285K와 일치, 13700K와 불일치 (2-3) | **285K** |
| **직접** | 수집 기간 커널 로그 전체가 `5090pc`, uptime 연속 → 재부팅 0 (2-4) | 교체 불가 |
| 정황 | 07-14 프로파일 실행 기록 `host: 5090pc` (2-5) | 285K 방향 |
| 정황 | 07-15 작성 문서가 285K 명시 (2-7) | 285K 방향 |
| 부재 | 수집 산출물·수집 로그·git 에 시스템 메타데이터 없음 (2-6, 2-8) | — |
| **없음** | i7-13700K 지지 증거 (2-9) | — |

> ### 결론: **"285K 확정" (직접 증거)**
>
> 작업 13에서 남긴 "호스트 교체 가능성을 배제할 직접 증거 없음"이라는 유보는
> **해소되었다.** 커널 부트 로그가 수집 기간을 통째로 덮으면서 CPU를 이름으로
> 기록하고 있고, 그 구간에 재부팅이 없다.

---

## 4. 보조 확인 — 현재 호스트 = 수집 호스트

| 항목 | 결과 |
|---|---|
| RTX 5090 | ✅ `NVIDIA GeForce RTX 5090`, 32607 MiB, driver 580.159.03 |
| Mobilint MLA100 | ✅ `/dev/aries0` 존재. 수집 부트(-1)에서도 `aries 0000:85:00.0` 프로브 확인. `qbruntime v1.2.0` |
| 수집 입력 영상 | ✅ `stockholm_1280x720.mp4` (67,130,877 B, mtime 2026-07-08) |
| machine-id | ✅ 저널 디렉터리가 `b7a2af2355dd46619d432e1ba3c5c8d0` 하나뿐 — 부트 -1과 현재 부트가 **같은 machine-id** 아래에 있다 |
| hostname | ✅ 수집 기간 로그·현재 모두 `5090pc` |

**셋 다 있고**, 게다가 수집 기간 부트에서 NPU와 GPU 드라이버 로드가 직접 확인되므로
"현재 호스트 = 수집 호스트"는 정황이 아니라 **거의 직접 확인**에 해당한다.

---

## 5. 원고 조치

`manuscript/mlforsys_main.tex:142` 의 유일한 발생 지점:

```latex
% 현재
Both platforms use the same Intel Core i7-13700K host, ...

% 교체 (최소)
Both platforms use the same Intel Core Ultra~9~285K host, ...

% 교체 (코어 수 포함 — 권장)
Both platforms use the same 24-core Intel Core Ultra~9~285K host, ...
```

작업 13에서 "저자 확인 필요"로 유보했던 부분은 이 조사로 닫혔다 — **확인 없이 교체해도
된다.** 부록 실행 설정 표에 싣는다면
`Host CPU & \multicolumn{2}{c}{Intel Core Ultra 9 285K (24 cores, no SMT)} \\`.

---

## 재현

```bash
python runs/20260809_150000_host_cpu/collect_evidence.py     # facts.json 재생성
journalctl --list-boots | tail -3                            # 부트 경계
journalctl -b -1 -k | grep -a "smpboot: CPU0\|DMI: "         # 수집 부트의 CPU·보드
zcat /var/log/kern.log.4.gz | sed -n '1p;$p'                 # 수집 기간 로그 범위
```

전 검산 11건 통과(`facts.json:checks`). 시스템 로그 조회는 읽기 전용이며 `adm` 그룹
권한으로 수행했다.
