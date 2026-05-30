# Improvement Tracker

Last updated: 2026-05-31

이 문서는 현재 코드베이스에서 개선이 필요한 항목을 우선순위와 근거 중심으로 추적합니다. 상태 값은 `Open`, `In progress`, `Done`, `Deferred` 중 하나로 갱신합니다.

## Snapshot

- 현재 브랜치: `main`
- 작업트리에는 기존 미커밋 변경과 이번 개선 변경이 함께 있음. 이번 패스 추가/수정: `.gitignore`, `doc/HEADLESS_API_KO.md`, `doc/IMPROVEMENT_TRACKER_KO.md`, `headless_server.py`, `launch.py`, `modules/inpaint/base.py`, `relay_server.py`, `tests/test_api_uploads.py`, `tests/test_inpaint_base.py`, `utils/api_uploads.py`
- `python -m pytest -q`: 현재 기본 Python 환경에는 `pytest`가 없어 실행 불가
- `python -m unittest discover -s tests -p 'test_*.py'`: 62 tests, OK

## Priority Guide

- `P1`: 릴리즈/운영/보안/데이터 관리에 직접 영향을 줄 수 있어 먼저 처리
- `P2`: 유지보수성, 안정성, 테스트 신뢰도를 높이는 일반 개선
- `P3`: 구조 개선이나 장기 정리 성격의 항목

## Backlog

| ID | Priority | Status | Area | Issue | Evidence | Next action |
| --- | --- | --- | --- | --- | --- | --- |
| IMP-001 | P1 | Open | Config | 배포용 템플릿인 `config/config.json`에 로컬 상태가 섞여 있음 | `utils/shared.py:16`에서 `config/config.json`을 템플릿으로 사용하고, `config/config.json:1`에는 `mps`, 한국어 타깃, 절대 경로 `recent_proj_list`, 사용자 폰트 경로가 포함됨 | 템플릿을 플랫폼 중립값으로 정리하고, 로컬 상태는 `config/config.local.json`에만 저장되도록 검증 추가 |
| IMP-002 | P1 | In progress | Repo hygiene | 런타임 산출물인 relay 이미지가 Git에 추적되고 있음 | `.gitignore`에 `relay_storage/` 추가 완료. 단, `git ls-files relay_storage` 결과 10개 기존 추적 파일은 아직 남아 있음 | 커밋 준비 시 `git rm --cached -r relay_storage`로 기존 산출물만 index에서 제거 |
| IMP-003 | P1 | Done | Ops docs | 공개 도메인, Cloudflare tunnel ID, 로컬 사용자 경로가 문서에 하드코딩됨 | `doc/HEADLESS_API_KO.md`를 `$BT_RELAY_PUBLIC_URL`, `<tunnel-name>`, `<tunnel-id>` placeholder 기반으로 정리 | 실제 운영 runbook이 필요하면 ignored local 문서로 분리 |
| IMP-004 | P1 | Done | Inpainting | 추상 메서드가 잘못된 예외를 raise함 | `modules/inpaint/base.py`의 `moveToDevice`가 `NotImplementedError`를 raise하도록 수정하고 `tests/test_inpaint_base.py` 추가 | 완료 |
| IMP-005 | P1 | Done | API hardening | relay/headless 업로드에 크기 제한과 실제 이미지 검증이 부족함 | `utils/api_uploads.py` 공통 헬퍼 추가, relay/headless 저장 경로에 크기 제한/이미지 검증/`413` 응답 추가, `--max-upload-mb` 지원 | 운영 기본값 50MB가 충분한지 실제 사용 이미지로 확인 |
| IMP-006 | P2 | Open | API reliability | relay job 상태가 메모리에만 있어 프로세스 재시작 시 큐/결과 메타데이터가 사라짐 | `relay_server.py:72`부터 `RelayJobStore`가 `_jobs = {}`만 사용 | SQLite 또는 job별 JSON 메타데이터로 상태 저장, stuck `running` job reclaim 정책 추가 |
| IMP-007 | P2 | Open | Observability | worker와 일부 유틸이 `print` 기반 로그를 사용함 | `local_worker.py:59`, `local_worker.py:74`, `local_worker.py:77`, `local_worker.py:148`; `utils/merger.py:218` 이후 debug print 다수 | 공통 `LOGGER`로 전환하고 job id, duration, remote status, retry count를 구조화해 기록 |
| IMP-008 | P2 | Open | Test tooling | 테스트 실행 방법이 표준화되어 있지 않고 기본 환경에 `pytest`가 없음 | `requirements.txt`에 test/dev dependency 없음, `python -m pytest -q` 실패, CI 설정 없음 | `requirements-dev.txt` 또는 `pyproject.toml`에 test deps와 명령 추가, 최소 CI에서 `unittest discover` 실행 |
| IMP-009 | P2 | Open | Dependencies | 무거운 런타임 의존성이 단일 `requirements.txt`에 넓게 선언되어 재현성이 낮음 | `requirements.txt:7`부터 `torch`, `torchvision`, `urllib3`, `requests` 등 다수 미고정 의존성 | `base/gui/api/local-model` extra 또는 constraints 파일로 분리하고 지원 Python/플랫폼 조합 문서화 |
| IMP-010 | P2 | Open | Localization | 현재 미커밋 UI 문자열이 번역 카탈로그에 아직 반영되지 않음 | `ui/canvas.py:880`의 `Add text box here`, `ui/mainwindowbars.py:782`의 tooltip이 `translate/*.ts` 검색 결과에 없음 | Qt translation update 절차를 실행하고 `.ts`/`.qm` 갱신, 한국어 README의 메뉴명도 현지화 |
| IMP-011 | P3 | Open | UI architecture | UI 핵심 파일이 커져 변경 영향 범위가 큼 | `ui/mainwindow.py` 2053 lines, `ui/canvas.py` 1145 lines, `ui/module_manager.py` 1089 lines, `ui/drawingpanel.py` 1075 lines | 새 기능 추가 시 command/service 단위로 분리하고, Canvas 상태 계산 로직은 별도 테스트 가능한 helper로 이동 |
| IMP-012 | P3 | Open | Developer experience | 코드 품질 도구가 없어 포맷/정적검사 기준이 불명확함 | `pyproject.toml`, `ruff.toml`, `.pre-commit-config.yaml`, `.github` 없음 | 우선 `ruff` check만 도입하고, 자동 포맷은 범위 합의 후 단계적으로 적용 |

## Current In-Progress Follow-Ups

현재 작업트리의 미커밋 변경은 크게 두 갈래입니다.

| Topic | Files | Follow-up |
| --- | --- | --- |
| Manual text box UX | `ui/canvas.py`, `ui/mainwindow.py`, `ui/mainwindowbars.py`, `tests/test_paint_mode.py`, docs | 번역 카탈로그 갱신, 메뉴/단축키 문구 현지화, 실제 GUI smoke test |
| PaddleX PP-OCRv5 box merge | `modules/textdetector/detector_paddlex_ppocrv5.py`, `tests/test_paddlex_ppocrv5_detector.py` | 실제 샘플 이미지에서 과병합/미병합 기준 확인, merge params 기본값 튜닝 |

## Suggested Order

1. `IMP-002`의 기존 tracked relay 산출물을 index에서 제거합니다.
2. `IMP-001`의 템플릿 config를 플랫폼 중립값으로 정리합니다.
3. `IMP-006`-`IMP-007`로 headless/relay 운영 안정성을 이어서 보강합니다.
4. `IMP-008`-`IMP-010`은 현재 기능 작업을 병합하기 전에 검증 루프를 정리합니다.
