# iOS 단축어 실행 가이드

이 문서는 iPhone/iPad의 단축어 앱에서 사진을 보내고 번역된 이미지를 받는 설정 방법입니다.

## 전제 조건

아래 4개 프로세스가 실행 중이어야 합니다.

```text
relay_server.py
cloudflared tunnel run sekai-relay
launch.py --headless-server
local_worker.py
```

공개 주소는 현재 다음 값으로 맞춰져 있습니다.

```text
https://translator.allen-lee.blog
```

단축어에서는 두 가지 API를 사용할 수 있습니다.

```text
빠른 번역용: POST https://translator.allen-lee.blog/translate/raw?filename=input.png&timeout=90
긴 번역용:   POST https://translator.allen-lee.blog/jobs/raw?filename=input.png
```

`/translate/raw`는 요청 본문으로 이미지 파일 자체를 받고, 완료되면 번역된 이미지 파일을 그대로 반환합니다.
다만 Cloudflare를 거치는 긴 요청은 약 120초 전후에서 끊길 수 있으므로, 번역이 오래 걸리는 이미지는 `/jobs/raw`로 job을 만든 뒤 상태를 폴링하는 방식이 더 안정적입니다.

## 1. 빠른 번역용 단축어 만들기

단축어 앱에서 새 단축어를 만들고 이름을 예를 들어 `Sekai 이미지 번역`으로 지정합니다.

### 공유 시트에서 실행하는 방식

사진 앱에서 이미지를 열고 공유 버튼으로 실행하려면 이 구성이 가장 편합니다.

1. 단축어 세부사항을 열고 `공유 시트에서 보기`를 켭니다.
2. 입력 유형은 `이미지`만 남깁니다.
3. 액션 `URL`을 추가합니다.
4. URL 값을 아래처럼 입력합니다.

```text
https://translator.allen-lee.blog/translate/raw?filename=input.png&timeout=90
```

5. 액션 `URL 내용 가져오기` 또는 `Get Contents of URL`을 추가합니다.
6. `방법`을 `POST`로 바꿉니다.
7. `헤더`를 추가합니다.

```text
Authorization: Bearer <BT_CLIENT_TOKEN 값>
```

8. `요청 본문`을 `파일`로 바꿉니다.
9. 파일 값은 `단축어 입력`을 선택합니다.
10. 마지막 액션으로 원하는 출력 방식을 추가합니다.

추천 출력 액션:

```text
훑어보기 / Quick Look
사진 앨범에 저장 / Save to Photo Album
공유 / Share
```

처음 테스트할 때는 `훑어보기`가 가장 확인하기 쉽습니다. 잘 되면 그 뒤에 `사진 앨범에 저장`을 붙이면 됩니다.

### 단축어 앱에서 직접 사진을 고르는 방식

공유 시트가 아니라 단축어 앱 안에서 실행하려면 맨 앞에 액션 하나만 추가합니다.

1. 액션 `사진 선택` 또는 `Select Photos`를 첫 번째로 추가합니다.
2. `여러 항목 선택`은 끕니다.
3. `URL 내용 가져오기`의 요청 본문 파일 값을 `선택한 사진`으로 지정합니다.

나머지는 공유 시트 방식과 같습니다.

## 2. 긴 번역용 단축어 구조

번역이 2분 이상 걸릴 가능성이 있으면 이 방식으로 만드세요. 액션 수는 조금 많지만 Cloudflare 응답 타임아웃에 덜 취약합니다.

1. 사진 입력은 위와 동일하게 `단축어 입력` 또는 `사진 선택`을 사용합니다.
2. `URL` 액션에 아래 값을 넣습니다.

```text
https://translator.allen-lee.blog/jobs/raw?filename=input.png
```

3. `URL 내용 가져오기`를 `POST`, `요청 본문: 파일`로 설정하고 파일 값에 사진을 넣습니다.
4. 헤더는 동일하게 넣습니다.

```text
Authorization: Bearer <BT_CLIENT_TOKEN 값>
```

5. 응답에서 `사전 값 가져오기 / Get Dictionary Value`로 `job_id`를 꺼냅니다.
6. `반복 / Repeat`을 60회 정도 추가합니다.
7. 반복 안에서 아래 URL을 만들어 `GET` 요청합니다.

```text
https://translator.allen-lee.blog/jobs/<job_id>
```

8. 상태 응답에서 `status` 값을 꺼냅니다.
9. `status`가 `done`이면 아래 URL을 `GET`으로 호출하고 결과 이미지를 `훑어보기` 또는 `사진 앨범에 저장`으로 보냅니다.

```text
https://translator.allen-lee.blog/jobs/<job_id>/result
```

10. `status`가 `failed`이면 `error` 값을 알림으로 보여주고 중단합니다.
11. 아직 `queued` 또는 `running`이면 `대기 / Wait` 5초 후 다음 반복으로 넘어갑니다.

## 3. 토큰 넣는 방법

Mac에서 `.env.headless`에 들어 있는 `BT_CLIENT_TOKEN` 값을 확인한 뒤, 단축어의 Authorization 헤더에 붙입니다.

형식은 반드시 아래처럼 `Bearer ` 뒤에 토큰이 와야 합니다.

```text
Bearer client-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

주의할 점:

- 실제 토큰이 들어간 단축어를 공개 공유하지 마세요.
- 다른 사람에게 단축어를 공유하려면 Authorization 헤더를 `Bearer PASTE_CLIENT_TOKEN_HERE` 같은 자리표시자로 바꾼 뒤 공유하세요.
- 토큰이 노출됐다고 생각되면 `.env.headless`를 새로 만들고 모든 프로세스를 재시작하세요.

## 4. Mac에서 먼저 API 테스트

단축어가 안 될 때는 먼저 Mac에서 raw API가 되는지 확인합니다.

```bash
source .env.headless
curl -sS -X POST "https://translator.allen-lee.blog/translate/raw?filename=input.png&timeout=90" \
  -H "Authorization: Bearer $BT_CLIENT_TOKEN" \
  -H "Content-Type: image/png" \
  --data-binary "@input.png" \
  -o result.png
```

`result.png`가 생성되면 iOS 단축어도 같은 Relay 경로를 사용하면 됩니다.

긴 번역용 job 생성만 먼저 확인하려면:

```bash
source .env.headless
curl -sS -X POST "https://translator.allen-lee.blog/jobs/raw?filename=input.png" \
  -H "Authorization: Bearer $BT_CLIENT_TOKEN" \
  -H "Content-Type: image/png" \
  --data-binary "@input.png"
```

## 5. 공유 링크 만들기

단축어 파일 또는 iCloud 공유 링크는 iPhone의 단축어 앱에서 생성해야 합니다.

1. 단축어 앱에서 `Sekai 이미지 번역`의 `...` 버튼을 누릅니다.
2. 공유 버튼을 누릅니다.
3. `iCloud 링크 복사` 또는 파일 공유를 선택합니다.
4. 공개 공유 전에는 Authorization 헤더에 실제 토큰이 들어 있는지 꼭 확인합니다.

macOS의 `shortcuts` CLI는 실행, 목록, 보기, 서명 기능만 제공하고 새 단축어 생성/공유 링크 생성을 자동화하지 못합니다. 그래서 최종 iCloud 공유 링크 생성은 단축어 앱에서 직접 해야 합니다.

## 6. 문제 해결

### `401 Unauthorized`

Authorization 헤더가 틀린 상태입니다.

```text
Authorization: Bearer <BT_CLIENT_TOKEN>
```

Relay용 클라이언트 토큰은 `BT_CLIENT_TOKEN`입니다. `BT_WORKER_TOKEN`이나 `BT_LOCAL_TOKEN`을 넣으면 안 됩니다.

### 요청이 오래 걸리다가 실패하거나 `524`가 표시됨

Cloudflare가 긴 HTTP 응답을 기다리다가 끊은 상태일 수 있습니다. 빠른 단축어의 `timeout`은 90초 정도로 두고, 오래 걸리는 이미지는 `/jobs/raw` 폴링 방식 단축어를 사용하세요.

worker/headless 번역기가 멈춘 상태일 수도 있습니다.

```bash
curl -sS https://translator.allen-lee.blog/health
curl -sS http://127.0.0.1:8000/health
```

`local_worker.py`와 `launch.py --headless-server` 터미널 로그도 같이 확인하세요.

### Cloudflare 502 또는 1033

Relay 서버나 Cloudflare Tunnel이 꺼진 상태일 가능성이 큽니다.

```bash
curl -sS http://127.0.0.1:9000/health
cloudflared tunnel ingress validate
```
