<!-- provenance
  generated_at: 2026-07-29T18:23+09:00
  git_commit: 2157159 (pre-commit HEAD)
  action: one-time backup of the only surviving per-frame trace copy
-->

# 뷰별 프레임 트레이스 백업 기록

540조합 본수집의 뷰별 프레임 단위 원시 트레이스(수집 시 세션 scratchpad에만 존재)를
/tmp 소실 대비로 저장소 밖에 백업했다. P1~P6 실험에는 쓰이지 않지만 유일본이라 보존한다.

- 원본: `/tmp/claude-1001/-home-msyu-PycharmProjects-multimodel-scheduling-mobilint/6f0af394-5a71-4f54-bead-adacfbd8c6b9/scratchpad/collect/traces/`
  (1,140개 파일, 9.7GB — `{gpu,npu}_combination_N.jsonl` + anchor 트레이스)
- 백업: **`/home/msyu/trace_backup/full_collection_540_traces.tar.zst`**
  - 크기 1,084,578,541 B (약 1.01 GiB), 압축 전 10,392,197,120 B
  - 생성: 2026-07-29 18:23 KST, `tar -I 'zstd -T0 -5'`
  - 무결성: `zstd -t` 통과
- 복원: `tar -I zstd -xf full_collection_540_traces.tar.zst` → `traces/` 디렉터리
- 저장소에는 커밋하지 않는다(용량). 이 파일이 살아 있는 한 /tmp가 지워져도 복구 가능.
