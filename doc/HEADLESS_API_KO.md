# Headless API / Relay 실행 가이드

이 문서는 Headless API, Relay 서버, Cloudflare Tunnel, local worker를 실행하는 방법을 정리합니다.
iOS 단축어 설정은 [IOS_SHORTCUT_KO.md](IOS_SHORTCUT_KO.md)를 참고하세요.

## 구성

전체 파이프라인은 4개 프로세스로 동작합니다.

```text
외부 클라이언트
  -> $BT_RELAY_PUBLIC_URL/translate/raw 또는 /jobs
  -> relay_server.py
  -> local_worker.py
  -> launch.py --headless-server
  -> 번역 결과를 relay_server.py에 업로드
  -> 외부 클라이언트가 이미지 응답 또는 /jobs/{job_id}/result로 다운로드
```

각 역할은 다음과 같습니다.

| 역할 | 명령 | 설명 |
| --- | --- | --- |
| Relay 서버 | `relay_server.py` | 외부 요청, job 큐, 결과 보관 |
| Cloudflare Tunnel | `cloudflared tunnel run <tunnel-name>` | 외부 HTTPS 도메인을 relay 서버로 연결 |
| Headless 번역기 | `launch.py --headless-server` | 실제 OCR/번역/인페인트 처리 |
| Local worker | `local_worker.py` | Relay job을 가져와 로컬 번역기로 처리 |

## Cloudflare Tunnel 설정

공개 주소는 환경 변수로 둡니다.

```bash
export BT_RELAY_PUBLIC_URL="https://your-domain.example"
```

터널 정보 예시:

```text
Tunnel name: <tunnel-name>
Tunnel ID: <tunnel-id>
Config: ~/.cloudflared/config.yml
Ingress: your-domain.example -> http://127.0.0.1:9000
```

상태 확인:

```bash
curl -sS "$BT_RELAY_PUBLIC_URL/health"
```

정상 응답 예:

```json
{"ok":true,"jobs":{"queued":0,"running":0,"done":0,"failed":0,"total":0},"max_upload_bytes":52428800}
```

## 1. 토큰 준비

각 서버 프로세스는 서로 다른 터미널에서 계속 실행되므로, 토큰을 파일에 저장해두고 각 터미널에서 불러오는 방식이 가장 편합니다.

프로젝트 루트에 `.env.headless`를 만듭니다.

```bash
python - <<'PY'
from pathlib import Path
import secrets

Path(".env.headless").write_text("\n".join([
    "# Headless API / Relay tokens. Do not commit this file.",
    f'export BT_CLIENT_TOKEN="client-{secrets.token_urlsafe(32)}"',
    f'export BT_WORKER_TOKEN="worker-{secrets.token_urlsafe(32)}"',
    f'export BT_LOCAL_TOKEN="local-{secrets.token_urlsafe(32)}"',
    'export BT_RELAY_PUBLIC_URL="https://your-domain.example"',
    "",
]), encoding="utf-8")
PY
```

각 터미널에서 명령 실행 전에 토큰을 불러옵니다.

```bash
source .env.headless
```

토큰이 들어 있는 파일은 Git에 올리면 안 됩니다. `.gitignore`에 이미 추가되어 있어야 합니다.

```bash
grep -n '^.env.headless$' .gitignore
```

토큰 역할:

| 토큰 | 쓰는 곳 | 설명 |
| --- | --- | --- |
| `BT_CLIENT_TOKEN` | 클라이언트 -> Relay | 외부에서 job 생성/조회/다운로드 |
| `BT_WORKER_TOKEN` | Local worker -> Relay | worker가 job을 가져오고 결과 업로드 |
| `BT_LOCAL_TOKEN` | Local worker -> Headless 번역기 | 로컬 번역 API 접근 |

## 2. Relay 서버 실행

Relay 서버는 외부 클라이언트가 접근하는 API 서버입니다. 실제 번역 모델은 로드하지 않습니다.

```bash
source .env.headless
python relay_server.py \
  --host 127.0.0.1 \
  --port 9000 \
  --storage-dir relay_storage \
  --api-token "$BT_CLIENT_TOKEN" \
  --worker-token "$BT_WORKER_TOKEN" \
  --max-upload-mb 50 \
  --claim-lease-seconds 1800 \
  --max-jobs 1000 \
  --max-storage-mb 2048
```

로컬 확인:

```bash
curl -sS http://127.0.0.1:9000/health
```

## 3. Cloudflare Tunnel 실행

Relay 서버를 외부 HTTPS 주소로 노출합니다.

```bash
cloudflared tunnel run <tunnel-name>
```

외부 확인:

```bash
curl -sS "$BT_RELAY_PUBLIC_URL/health"
```

`~/.cloudflared/config.yml`은 다음 형태여야 합니다.

```yaml
tunnel: <tunnel-id>
credentials-file: /path/to/<tunnel-id>.json

ingress:
  - hostname: your-domain.example
    service: http://127.0.0.1:9000
  - service: http_status:404
```

설정 검사:

```bash
cloudflared tunnel ingress validate
```

## 4. Headless 번역기 실행

실제 번역 파이프라인입니다. GUI 없이 로컬 API 서버로 실행됩니다.

```bash
source .env.headless
python launch.py \
  --headless-server \
  --host 127.0.0.1 \
  --port 8000 \
  --api-token "$BT_LOCAL_TOKEN" \
  --max-upload-mb 50
```

로컬 확인:

```bash
curl -sS http://127.0.0.1:8000/health
```

## 5. Local worker 실행

Worker는 relay 서버에서 job을 가져와 로컬 headless 번역기로 처리하고 결과를 relay에 업로드합니다.

```bash
source .env.headless
python local_worker.py \
  --relay-url "$BT_RELAY_PUBLIC_URL" \
  --local-url http://127.0.0.1:8000 \
  --worker-token "$BT_WORKER_TOKEN" \
  --local-token "$BT_LOCAL_TOKEN" \
  --heartbeat-interval 60
```

한 번만 처리하고 종료하려면:

```bash
source .env.headless
python local_worker.py \
  --relay-url "$BT_RELAY_PUBLIC_URL" \
  --local-url http://127.0.0.1:8000 \
  --worker-token "$BT_WORKER_TOKEN" \
  --local-token "$BT_LOCAL_TOKEN" \
  --once
```

## 6. 클라이언트 사용법

### iOS 단축어 / 동기 호출

iOS 단축어처럼 한 번 요청하고 결과 이미지를 바로 받고 싶으면 `/translate/raw`를 사용합니다. Cloudflare를 거치는 요청은 길게 열린 응답이 끊길 수 있으므로, 외부 HTTPS에서는 `timeout`을 90초 정도로 두고 오래 걸리는 작업은 `/jobs/raw` 또는 `/jobs`를 사용하세요.

```bash
source .env.headless
curl -sS -X POST "$BT_RELAY_PUBLIC_URL/translate/raw?filename=input.png&timeout=90" \
  -H "Authorization: Bearer $BT_CLIENT_TOKEN" \
  -H "Content-Type: image/png" \
  --data-binary "@input.png" \
  -o result.png
```

multipart form을 쓰는 클라이언트는 `/translate`도 사용할 수 있습니다.

```bash
source .env.headless
curl -sS -X POST "$BT_RELAY_PUBLIC_URL/translate?timeout=90" \
  -H "Authorization: Bearer $BT_CLIENT_TOKEN" \
  -F "file=@input.png" \
  -o result.png
```

두 동기 엔드포인트는 내부적으로 Relay job을 만들고 worker가 완료할 때까지 기다린 뒤 결과 이미지를 반환합니다. `timeout` 안에 끝나지 않으면 `504`가 반환됩니다.

### iOS 단축어 / raw job 생성

단축어에서 raw 파일 본문으로 job만 만들고 싶으면 `/jobs/raw`를 사용합니다.

```bash
source .env.headless
curl -sS -X POST "$BT_RELAY_PUBLIC_URL/jobs/raw?filename=input.png" \
  -H "Authorization: Bearer $BT_CLIENT_TOKEN" \
  -H "Content-Type: image/png" \
  --data-binary "@input.png"
```

응답 형식은 `/jobs`와 같습니다. 이후 `/jobs/{job_id}`로 상태를 조회하고 `/jobs/{job_id}/result`로 결과를 받으면 됩니다.

```json
{
  "job_id": "abc123",
  "status": "queued",
  "status_url": "/jobs/abc123",
  "result_url": "/jobs/abc123/result"
}
```

### 작업 생성

```bash
source .env.headless
curl -sS -X POST "$BT_RELAY_PUBLIC_URL/jobs" \
  -H "Authorization: Bearer $BT_CLIENT_TOKEN" \
  -F "file=@input.png"
```

응답 예:

```json
{
  "job_id": "abc123",
  "status": "queued",
  "status_url": "/jobs/abc123",
  "result_url": "/jobs/abc123/result"
}
```

### 상태 조회

```bash
source .env.headless
curl -sS "$BT_RELAY_PUBLIC_URL/jobs/abc123" \
  -H "Authorization: Bearer $BT_CLIENT_TOKEN"
```

상태 값:

```text
queued   대기 중
running  worker가 처리 중
done     완료
failed   실패
```

### 결과 다운로드

```bash
source .env.headless
curl -sS "$BT_RELAY_PUBLIC_URL/jobs/abc123/result" \
  -H "Authorization: Bearer $BT_CLIENT_TOKEN" \
  -o result.png
```

아직 처리 중이면 `202`가 반환됩니다. 실패하면 에러 메시지가 반환됩니다.

## 7. 빠른 전체 실행 순서

터미널 1:

```bash
source .env.headless
python relay_server.py --host 127.0.0.1 --port 9000 --storage-dir relay_storage --api-token "$BT_CLIENT_TOKEN" --worker-token "$BT_WORKER_TOKEN" --max-upload-mb 50 --claim-lease-seconds 1800 --max-jobs 1000 --max-storage-mb 2048
```

터미널 2:

```bash
cloudflared tunnel run <tunnel-name>
```

터미널 3:

```bash
source .env.headless
python launch.py --headless-server --host 127.0.0.1 --port 8000 --api-token "$BT_LOCAL_TOKEN" --max-upload-mb 50
```

터미널 4:

```bash
source .env.headless
python local_worker.py --relay-url "$BT_RELAY_PUBLIC_URL" --local-url http://127.0.0.1:8000 --worker-token "$BT_WORKER_TOKEN" --local-token "$BT_LOCAL_TOKEN" --heartbeat-interval 60
```

클라이언트:

```bash
source .env.headless
curl -sS -X POST "$BT_RELAY_PUBLIC_URL/translate/raw?filename=input.png&timeout=90" -H "Authorization: Bearer $BT_CLIENT_TOKEN" -H "Content-Type: image/png" --data-binary "@input.png" -o result.png
```

## 8. 운영 메모

- `relay_server.py`와 `cloudflared tunnel run`은 현재 터미널 세션이 종료되면 같이 종료됩니다.
- 상시 운영하려면 macOS `launchd`, `tmux`, `screen`, 또는 별도 서버 프로세스 매니저에 등록하세요.
- `cloudflared service install`로 시스템 서비스 등록도 가능하지만, 로컬 네트워크/계정 상태에 따라 권한 설정이 필요할 수 있습니다.
- 외부 공개 주소를 사용할 때는 반드시 `--api-token`과 `--worker-token`을 설정하세요. loopback이 아닌 host에 토큰 없이 바인딩하면 서버가 시작을 거부합니다. `--allow-unauthenticated-public`은 격리된 테스트 환경 외에는 사용하지 마세요.
- Relay 서버와 Headless 번역기는 `--max-upload-mb` 또는 `BALLOONTRANS_MAX_UPLOAD_BYTES`로 업로드 크기 제한을 설정할 수 있습니다.
- Relay 서버에는 원본 이미지와 결과 이미지가 `relay_storage` 아래에 TTL 동안 저장되고, job metadata는 `relay_storage/relay_jobs.sqlite3`에 보존됩니다.
- 완료된 job은 `--result-ttl` 초 이후 정리됩니다. 기본값은 3600초입니다.
- Relay는 `--max-jobs`와 `--max-storage-mb`를 넘는 새 job을 `503`으로 거부합니다.
- worker claim은 lease 방식입니다. `local_worker.py`는 `--heartbeat-interval`마다 lease를 갱신하며, `--claim-lease-seconds` 동안 heartbeat가 없으면 job이 다시 `queued`로 돌아갑니다.
- headless 모드에서 누락된 Python 패키지 설치를 허용하려면 `launch.py`에 `--allow-package-install`을 명시해야 합니다. 기본값은 설치하지 않음입니다.

## 9. 문제 해결

### 외부 `/health`는 되는데 번역이 진행되지 않음

대부분 worker 또는 headless 번역기가 꺼져 있습니다.

```bash
curl -sS http://127.0.0.1:8000/health
```

이 요청이 실패하면 `launch.py --headless-server`를 먼저 실행하세요.

### job이 계속 `queued`

`local_worker.py`가 실행 중인지 확인하세요. worker가 relay에서 job을 가져오면 상태가 `running`으로 바뀝니다.

### job이 `running`에서 멈춤

번역 모델 로딩 중이거나 OCR/번역 처리 시간이 오래 걸리는 상태일 수 있습니다. `local_worker.py`와 `launch.py --headless-server` 터미널 로그를 확인하세요. worker가 종료되거나 heartbeat가 끊기면 claim lease 만료 후 job은 자동으로 `queued`로 복구됩니다.

### `401 Unauthorized`

클라이언트, worker, 로컬 번역기 토큰 중 하나가 맞지 않습니다.

- 클라이언트 -> relay: `BT_CLIENT_TOKEN`
- worker -> relay: `BT_WORKER_TOKEN`
- worker -> local translator: `BT_LOCAL_TOKEN`

### `502`, `1033`, 또는 Cloudflare 오류

`cloudflared tunnel run <tunnel-name>`이 꺼졌거나 relay 서버가 꺼진 상태일 가능성이 큽니다.

```bash
cloudflared tunnel ingress validate
curl -sS http://127.0.0.1:9000/health
```
