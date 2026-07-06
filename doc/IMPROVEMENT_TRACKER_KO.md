# Improvement Tracker

Last updated: 2026-07-06

이 문서는 현재 코드베이스에서 개선이 필요해 보이는 항목을 우선순위와 근거 중심으로 추적합니다. 상태 값은 `Open`, `In progress`, `Done`, `Deferred` 중 하나로 갱신합니다.

## Snapshot

- 현재 브랜치: `main`
- 이번 패스에서 `IMP-001`, `IMP-002`를 처리함.
- `python -m unittest discover -s tests -p 'test_*.py'`: 80 tests, OK
- `python -m pytest -q`: 현재 기본 Python 환경에 `pytest`가 없어 실행 불가
- 테스트 범위는 단위 테스트 중심이며, FastAPI endpoint, 실제 GUI smoke, 실제 샘플 이미지 기반 OCR/merge 품질 검증은 아직 약함.

## Priority Guide

- `P1`: 릴리즈/운영/보안/데이터 관리에 직접 영향을 줄 수 있어 먼저 처리
- `P2`: 유지보수성, 안정성, 테스트 신뢰도를 높이는 일반 개선
- `P3`: 구조 개선이나 장기 정리 성격의 항목

## Backlog

| ID | Priority | Status | Area | Issue | Evidence | Next action |
| --- | --- | --- | --- | --- | --- | --- |
| IMP-001 | P1 | Done | Config | 배포용 템플릿인 `config/config.json`에 로컬 실행 상태가 섞여 있음 | `config/config.json`을 기본값 기반 템플릿으로 재작성함. 모듈별 device/개인 번역기 설정/최근 프로젝트 경로를 제거하고 `text_styles_path`는 `config/textstyles/default.json` 상대 경로로 변경. `tests/test_config_template.py` 추가 | 완료. 향후 템플릿에 로컬 상태가 다시 들어가면 테스트가 실패해야 함 |
| IMP-002 | P1 | Done | Repo hygiene | 런타임 산출물과 캐시가 Git에 추적되고 있음 | `.btrans_cache/cache.json` 및 `relay_storage/*` 10개 이미지를 `git rm --cached`로 index에서 제거함. 실제 파일은 working tree에 보존됨 | 완료. 커밋 시 삭제로 staged 되는 것이 정상이며, 이후에는 `.gitignore`가 재추적을 막아야 함 |
| IMP-003 | P1 | Open | Ops docs | iOS 단축어 문서에 실제 공개 도메인과 tunnel 이름이 남아 있음 | `doc/IOS_SHORTCUT_KO.md:11`의 `sekai-relay`, `doc/IOS_SHORTCUT_KO.md:19`, `:25`, `:26`, `:139`, `:152`, `:188`의 실제 URL | `$BT_RELAY_PUBLIC_URL`, `<tunnel-name>` placeholder로 바꾸고 개인 운영 runbook은 ignored local 문서로 분리 |
| IMP-004 | P1 | Open | API hardening | 공개 바인딩 시 인증 토큰이 비어 있어도 서버가 그대로 뜰 수 있음 | `launch.py:45-47`, `relay_server.py:456-460`에서 host와 token을 독립 옵션으로 받고 token 기본값은 빈 문자열. `make_auth_dependency`는 token이 없으면 인증을 생략함 | host가 loopback이 아니고 token이 비어 있으면 경고 또는 실행 거부 옵션 추가. 문서에도 공개 배포 최소 조건 명시 |
| IMP-005 | P1 | Open | API reliability | relay/headless job 상태가 메모리에만 있어 재시작 시 큐/결과 메타데이터가 사라지고, worker 사망 시 `running` job이 stuck 될 수 있음 | `relay_server.py:82-89`, `headless_server.py:223-229`에서 `_jobs = {}`와 in-memory queue만 사용. `relay_server.py:166-174`는 claim 후 lease/timeout reclaim 없음 | SQLite 또는 job별 JSON 메타데이터를 저장하고 `running` lease timeout, retry/reclaim 정책 추가 |
| IMP-006 | P2 | Open | API testing | 업로드 helper 테스트는 있으나 실제 FastAPI endpoint 흐름 테스트가 부족함 | `tests/test_api_uploads.py`는 `RelayJobStore`와 `save_upload_to_project` 중심. `/jobs`, `/translate/raw`, worker result/failure endpoint에 대한 `TestClient` 테스트 없음 | FastAPI `TestClient`로 auth, 413, invalid image, async job, worker result/failure 흐름을 추가 |
| IMP-007 | P2 | Open | Observability | worker와 일부 유틸이 `print` 기반 로그를 사용해 운영 추적성이 낮음 | `local_worker.py:59`, `:74`, `:77`, `:148`이 `print` 사용. `utils/merger.py:218-220`, `:237-255`, `:268-299`도 debug print 사용 | 공통 `LOGGER`로 전환하고 job id, duration, remote status, retry count, bytes를 구조화해 기록 |
| IMP-008 | P2 | Open | Test tooling | 테스트 실행 방법이 표준화되어 있지 않고 기본 환경에 `pytest`가 없음 | `python -m pytest -q`는 `No module named pytest`. `requirements-dev.txt`, `pyproject.toml`, `.github`가 없음 | `requirements-dev.txt` 또는 `pyproject.toml`에 test deps와 명령 추가, 최소 CI에서 `unittest discover` 실행 |
| IMP-009 | P2 | Open | Dependencies | 무거운 런타임 의존성이 단일 `requirements.txt`에 넓게 섞여 재현성과 설치 시간이 불리함 | `requirements.txt:7-70`에 GUI, API, ML, Paddle, OpenAI, platform deps가 한 파일에 혼재. `torch`, `torchvision`, `urllib3`, `requests` 등은 넓게 미고정 | `base/gui/api/local-model/paddle` extra 또는 constraints 파일로 분리하고 지원 Python/플랫폼 조합 문서화 |
| IMP-010 | P2 | Open | Localization | 새 UI/문서 문자열이 한국어 현지화에 덜 반영됨 | `ui/canvas.py:885`의 `Add text box here`가 `translate/*.ts`에 없고, `doc/README_KO.md:181`에도 영어 메뉴명이 노출됨 | Qt translation update 절차를 실행하고 `.ts`/`.qm` 갱신, 한국어 README 메뉴명 현지화 |
| IMP-011 | P2 | Open | OCR quality | PP-OCRv5 box merge 기본값은 단위 테스트가 있으나 실제 샘플 이미지 기반 품질 기준이 부족함 | `modules/textdetector/detector_paddlex_ppocrv5.py:120-136`에 merge 기본값이 있고 `tests/test_paddlex_ppocrv5_detector.py`는 synthetic polygon 테스트 중심 | 실제 만화 샘플 fixture로 과병합/미병합 회귀 테스트를 만들고 기본값 튜닝 |
| IMP-012 | P3 | Open | UI architecture | UI 핵심 파일이 커져 변경 영향 범위가 큼 | `ui/mainwindow.py` 2053 lines, `ui/canvas.py` 1150 lines, `ui/module_manager.py` 1120 lines, `ui/drawingpanel.py` 1215 lines | 새 기능 추가 시 command/service/helper 단위로 분리하고 Canvas 상태 계산 로직은 테스트 가능한 helper로 이동 |
| IMP-013 | P3 | Open | Developer experience | 코드 품질 도구가 없어 포맷/정적검사 기준이 불명확함 | `pyproject.toml`, `ruff.toml`, `.pre-commit-config.yaml`, `.github` 없음 | 우선 `ruff check`만 도입하고, 자동 포맷은 범위 합의 후 단계적으로 적용 |

## Verified Done

| ID | Area | Evidence |
| --- | --- | --- |
| DONE-001 | Inpainting base contract | `modules/inpaint/base.py`의 base method 예외 동작은 `tests/test_inpaint_base.py`에서 검증됨 |
| DONE-002 | Upload validation | `utils/api_uploads.py` 기반 크기 제한/이미지 검증은 `tests/test_api_uploads.py`에서 일부 검증됨 |
| DONE-003 | Headless API main runbook | `doc/HEADLESS_API_KO.md`는 `$BT_RELAY_PUBLIC_URL`, `<tunnel-name>`, `<tunnel-id>` placeholder 중심으로 정리되어 있음 |

## Suggested Order

1. `IMP-003`으로 iOS 단축어 문서의 실제 공개 도메인과 tunnel 이름을 placeholder로 바꿉니다.
2. `IMP-004`와 `IMP-005`로 공개 API 운영 안전성을 보강합니다.
3. `IMP-006`-`IMP-008`로 테스트와 검증 루프를 먼저 안정화합니다.
4. `IMP-009`-`IMP-013`은 기능 작업과 병행 가능한 구조/품질 개선으로 진행합니다.
