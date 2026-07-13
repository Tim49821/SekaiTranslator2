# Improvement Tracker

Last updated: 2026-07-10

최근 대규모 패치 이후의 코드 기준으로 다시 작성하고, 1차 개선 구현 결과까지 반영한 목록입니다. 단순 아이디어가 아니라 현재 코드, 테스트, Git 상태에서 확인한 근거를 우선합니다.

## Audit Snapshot

- 점검 범위: `800bf328^..ef3d9a1d`, 11 commits, 99 files changed, `+11,876 / -1,260`
- 현재 브랜치: `main` (`origin/main`보다 1 commit ahead), 아래 개선 변경은 아직 commit하지 않은 working tree 상태
- 검증 환경: Python 3.11.5, 프로젝트 `.venv`
- 최초 기준선: `python -m unittest discover -s tests -p 'test_*.py'` 92 tests, OK
- 1차 개선 후: 120 tests, OK (`lazy registry`, updater, auth, relay persistence/lease/ownership/quota, headless/relay endpoint 포함)
- `python -m compileall -q ...`: OK
- `python -m pip check`: No broken requirements found
- 개발 의존성은 `requirements-dev.txt`, 자동 검증은 `.github/workflows/core-tests.yml`에 추가
- lazy registry 진단: 36 specs, metadata warning 0건
- API `TestClient` 기반 relay client/worker/result 흐름과 headless sync/async 흐름 검증 완료
- 실제 GUI smoke, GitHub Actions 원격 실행, Windows/Linux 실기 실행, 실제 모델 다운로드·추론은 아직 검증하지 못함

## Priority Guide

- `P0`: 다음 릴리즈 전에 처리해야 할 회귀 또는 배포 차단 가능성이 높은 항목
- `P1`: 운영 안정성, 보안, 자동 설치처럼 사용자 환경에 직접 영향을 주는 항목
- `P2`: 테스트 신뢰도, 품질, 유지보수성을 높이는 항목
- `P3`: 장기 구조 개선 또는 문서 정리 항목

## Improvement Backlog

| ID | Priority | Status | Area | Issue | Evidence | Next action |
| --- | --- | --- | --- | --- | --- | --- |
| IMP-014 | P0 | Done | Lazy registry | lazy metadata 손실로 일부 모듈 설정 UI와 번역 언어 목록이 불완전해질 수 있음 | 안전한 class attribute 평가, `.copy()`, f-string, base translator 언어 상속, `lazy_params` fallback을 추가함(`modules/lazy_registry.py`). 36 specs의 warning이 17건에서 0건으로 감소했고 5개 문제 모듈의 params와 Gemma/LLM 언어 목록이 복원됨 | `tests/test_lazy_runtime.py`의 warning 0·params·언어·지연 import 회귀 테스트를 CI에서 유지 |
| IMP-015 | P0 | Done | Model preparation | lazy registry가 Hugging Face 준비 metadata를 보존하지 않아 자동 모델 준비 경로가 끊김 | `ModuleSpec`에 repo/save dir/required files/pattern/revision/prepare 필드를 추가하고(`utils/registry.py:17-32`) `ensure_module_files()`가 lazy spec에서도 snapshot 준비를 수행함(`modules/prepare_local_files.py:61`) | 실제 모델 smoke는 별도 선택 job에서 수행하고 revision pin은 `IMP-009`로 추적 |
| IMP-016 | P0 | Done | Update / release | 소스 업데이트 채널이 이 fork의 실제 원격 브랜치와 맞지 않음 | 기본 소스를 `origin/main`으로 정렬하고 환경 변수 override, ahead/behind 판정, `--ff-only` pull을 추가함(`launch.py:11-12`, `:168-241`). upstream release 알림과 fork source update UI 문구도 분리함 | 원격 CI에서 ahead/behind 시나리오 테스트를 계속 유지 |
| IMP-017 | P0 | Done | Compatibility | Python 3.8 지원 계약과 현재 핵심 코드가 충돌함 | PEP 585/604 annotation 사용 파일에 postponed annotation을 적용하고 전체 Python 파일을 3.8 grammar로 파싱하는 `tests/test_python_compat.py` 및 Python 3.8 CI matrix를 추가함 | 실제 Python 3.8 CI 결과를 첫 workflow 실행에서 확인 |
| IMP-018 | P1 | Done | Package installation | 새 설치에서 누락 패키지 자동 설치가 기본값이라 사용자 확인 없이 Python 환경을 변경할 수 있음 | 기본값을 false로 바꾸고(`utils/config.py:179`) headless는 `--allow-package-install`이 없으면 저장 설정과 무관하게 설치를 차단함(`launch.py:57`, `:176-181`) | GUI 선택과 headless opt-in 회귀 테스트 유지 |
| IMP-004 | P1 | Done | API security | 비loopback 공개 바인딩과 빈 인증 토큰 조합을 코드가 차단하지 않음 | 공통 public-bind guard를 추가해(`utils/api_security.py`) headless와 relay가 필요한 토큰 없이 공개 주소에 뜨지 못하게 했고, 명시적 `--allow-unauthenticated-public`만 예외로 허용함 | 배포 문서의 token 예시와 `tests/test_api_security.py` 유지 |
| IMP-005 | P1 | Done | API reliability | relay job이 재시작·worker 장애·오용에 취약하고 저장량 제한이 없음 | SQLite job metadata, 재시작 복구, claim lease/heartbeat/requeue, worker 소유권과 상태 전이 검사, job/storage quota를 `RelayJobStore`에 구현함(`relay_server.py:95-429`). 만료 worker와 중복 결과도 409로 거부함 | 장시간 실서비스 부하·파일시스템 장애 검증은 별도 운영 테스트로 수행 |
| IMP-008 | P1 | Implemented | CI / test tooling | 1.2만 줄 규모 변경을 자동 검증할 표준 CI와 개발 테스트 의존성이 없음 | `requirements-dev.txt`와 `.github/workflows/core-tests.yml`을 추가해 Ubuntu/Windows/macOS 및 Python 3.8/3.11에서 core tests, compileall, pip check를 실행하도록 구성함 | 첫 GitHub Actions 실행 결과를 확인한 뒤 `Done`으로 변경. 이후 ruff 도입은 `IMP-013`으로 추적 |
| IMP-006 | P1 | Done | API testing | 새 headless/relay endpoint 계약 테스트가 없음 | `tests/test_headless_api.py`, `tests/test_relay_api.py`, `tests/test_local_worker.py`, `tests/test_relay_jobs.py`에서 인증, sync/async, 실패, worker claim/heartbeat/result, ownership, 중복 완료, 재시작, quota를 검증함 | FastAPI/Starlette의 현재 TestClient deprecation warning은 의존성 정리 때 제거 |
| IMP-009 | P2 | In progress | Reproducibility | core requirements 경량화는 완료됐지만 런타임 패키지·모델 재현성이 아직 부족함 | 무거운 패키지를 module `dependencies`로 옮긴 점은 개선됨. 다만 core `requirements.txt` 대부분은 upper bound/lock이 없고 module dependency가 여러 파일에 흩어져 있음. HF `snapshot_download()`는 revision을 고정하지 않음(`modules/prepare_local_files.py:119-126`) | 지원 플랫폼별 constraints/lock 생성, module dependency catalog 자동 검증, HF revision/commit pin과 모델 manifest 기록. clean environment 설치 smoke 추가 |
| IMP-010 | P2 | Open | Localization | 대규모 패치의 새 UI 문자열이 번역 카탈로그에 반영되지 않음 | 변경 범위에 `translate/*.ts`/`.qm` 수정이 없지만 `Auto install missing packages`, `Prepare Selected Modules`, update UI, Torch install helper, restore tool 등 수십 개 `tr()` 문자열이 추가됨 | `pylupdate`/`lrelease` 절차를 문서화·실행하고 `.ts`/`.qm` 갱신. 최소 ko_KR과 zh_CN에서 새 문자열 누락 검사 |
| IMP-011 | P2 | Open | Model quality | 새 OCR/detector/inpaint 기능은 synthetic 단위 테스트 중심이고 실제 이미지 품질 기준이 없음 | PP-OCRv5 merge와 paint mode 테스트는 늘었지만 실제 만화 fixture의 과병합/미병합, OCR 정확도, SDXL 경계·색상 품질 기준은 없음 | 소형 라이선스 허용 이미지 fixture와 golden metadata를 추가. detector IoU/merge, OCR text, inpaint seam 지표를 분리하고 느린 model smoke는 별도 job으로 운영 |
| IMP-019 | P2 | Open | Platform availability | registry가 플랫폼 전용 모듈의 사용 가능 여부를 일관되게 표현하지 못함 | `ModuleSpec.available` 필드는 있지만 scanner가 값을 채우지 않음. macOS에서도 `one_ocr`이 목록에 나타나며, 선택하면 Windows-only 경고 후 비어 있는 OCR 객체가 생성됨(`modules/ocr/ocr_oneocr.py:243-298`) | 모듈에 `supported_platforms`/availability probe metadata를 두고 UI에서 비활성화 사유를 표시. Windows/macOS/Linux별 registry snapshot 테스트 추가 |
| IMP-012 | P2 | Open | Architecture | 핵심 UI 및 orchestration 파일의 책임과 크기가 패치 후 더 커짐 | `ui/mainwindow.py` 2,200 lines, `ui/module_manager.py` 1,845 lines, `ui/drawingpanel.py` 1,215 lines, `ui/canvas.py` 1,150 lines. 특히 `module_manager.py`는 pipeline, module lifecycle, package install UI, retry 상태를 함께 관리 | package preparation coordinator, pipeline controller, UI adapter로 분리. 상태 전이는 Qt와 분리한 순수 Python 객체로 옮겨 단위 테스트 가능하게 구성 |
| IMP-020 | P2 | In progress | Duplication | headless와 relay의 upload/job response 로직이 중복되어 수정 누락 위험이 큼 | public-bind 검사는 `utils/api_security.py`, upload 검증은 `utils/api_uploads.py`로 공통화했지만 upload extension/media type/auth dependency/result response/size middleware는 두 서버에 일부 중복됨 | 나머지 공통 API protocol/helper를 이동하고 서버별 차이는 store/translation adapter로 제한 |
| IMP-007 | P2 | Open | Observability | API/worker/model 준비 흐름을 job 단위로 추적하기 어려움 | `local_worker.py`는 `print` 중심이며 relay/headless 로그에 일관된 request/job/worker correlation field가 없음. 설치·다운로드 출력도 문자열 중심 | 표준 logger와 structured context를 사용해 `job_id`, `worker_id`, module, stage, duration을 남기고 token/경로는 redaction |
| IMP-013 | P3 | In progress | Developer experience | 정적 검사·포맷 기준이 아직 자동화되지 않음 | `requirements-dev.txt`에 ruff를 선언했고 현재 변경분은 `git diff --check`를 통과하지만, ruff 설정과 CI 실행, pre-commit은 없음 | 기존 코드에 적용 가능한 ruff 오류 탐지 규칙부터 CI에 추가. 자동 포맷은 별도 합의 후 점진 적용 |
| IMP-003 | P3 | Done | Ops docs / privacy | iOS 문서가 placeholder 문서와 달리 실제 tunnel 이름과 공개 도메인을 포함함 | `doc/IOS_SHORTCUT_KO.md`의 tunnel 이름과 공개 도메인을 `<tunnel-name>` 및 `https://your-domain.example` placeholder로 교체했고 실제 도메인 잔존 검색 결과가 없음 | 개인 운영값은 Git 비추적 환경 파일이나 별도 runbook에만 유지 |

## Verified Improvements Since Previous Audit

| Area | Result | Evidence |
| --- | --- | --- |
| Config hygiene | 배포 템플릿과 사용자 로컬 설정 분리 | 기본 저장 경로가 ignored `config/config.local.json`, 템플릿은 빈 module params와 상대 text style 경로 사용 |
| Repo hygiene | runtime cache/model/relay 산출물 비추적 | `.gitignore`에 `.btrans_cache/`, `relay_storage/`, `data/models/`, `data/libs/`, `.env*` 반영 |
| Upload safety | 크기 제한과 이미지 검증 helper 도입 | `utils/api_uploads.py`, 관련 unittest 통과 |
| Download safety | TLS 검증 해제는 명시적 환경 변수 opt-in | `BALLOONTRANS_ALLOW_INSECURE_DOWNLOADS`가 있을 때만 unverified context 사용 |
| Command execution | 주요 installer/launcher 명령이 argument list와 `shell=False` 사용 | `launch.py`, `utils/package_installer.py`, `utils/torch_install_helper.py` |
| Upstream overwrite guard | fork에서 upstream source zip 자동 덮어쓰기는 기본 차단 | `SEKAI_TRANSLATOR_ALLOW_UPSTREAM_SELF_UPDATE` opt-in 및 updater test |
| Core dependency size | torch/Paddle/transformers/onnxruntime 등 무거운 runtime을 module 선택 시 준비하도록 이동 | `requirements.txt` core 경량화와 module `dependencies` metadata |
| Paint/inpaint regression coverage | restore, empty mask, undo/redo 관련 단위 테스트 확대 | `tests/test_paint_mode.py`, `tests/test_inpaint_base.py`, `tests/test_imgproc_utils.py` |

## Recommended Execution Order

1. `IMP-008`의 첫 GitHub Actions 실행을 확인해 OS/Python matrix가 실제로 녹색인지 검증합니다.
2. `IMP-019`로 플랫폼 전용 모듈 availability를 registry와 UI에 일관되게 반영합니다.
3. `IMP-009`에서 constraints와 HF revision pin을 도입해 설치·모델 재현성을 높입니다.
4. `IMP-010`, `IMP-011`로 번역 카탈로그와 실제 이미지 품질 fixture를 보강합니다.
5. `IMP-020`, `IMP-007`, `IMP-012` 순으로 API 중복, 관측성, 대형 UI/orchestration 구조를 줄입니다.
6. `IMP-013`의 ruff 기준은 기존 코드에 대량 churn을 만들지 않는 범위부터 점진 적용합니다.

P0 4건, lazy metadata warning 0건, headless/relay 핵심 endpoint 테스트는 로컬에서 충족했습니다. 다음 릴리즈의 남은 최소 통과 조건은 GitHub Actions matrix 성공과 공식 지원 플랫폼의 GUI/import smoke입니다.
