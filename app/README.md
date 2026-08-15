# Covicare — Flutter client

Chat UI for the COVID-19 retrieval API. The app holds **no model logic**: it posts a
question to `POST /predict` and renders what comes back, including the source, trust tier
and disclaimer (PRD §10.2, §12).

## Running

Start the backend first, from the repository root:

```bash
uvicorn src.api:app          # serves on :8000
```

Then run the app, pointing it at the right host for your target:

```bash
flutter run                                                      # uses the platform default
flutter run --dart-define=API_BASE_URL=http://10.0.2.2:8000      # Android emulator
flutter run --dart-define=API_BASE_URL=http://192.168.1.42:8000  # physical device (host LAN IP)
flutter run --dart-define=API_BASE_URL=https://api.example.com   # deployed
```

### Which address, and why it matters

| Target | Address | Note |
|---|---|---|
| Android emulator | `http://10.0.2.2:8000` | **default on Android.** `127.0.0.1` is the *emulator itself* |
| Physical device | `http://<host LAN IP>:8000` | must be passed explicitly; the device shares no loopback |
| Desktop / web | `http://127.0.0.1:8000` | default elsewhere |
| Deployed | `https://…` | cleartext is blocked for non-local hosts |

This is the defect that made the previous build unusable on a device: it hardcoded
`127.0.0.1`, and `INTERNET` was declared only in the debug manifest, so release builds had
no network access at all.

## Structure

```
lib/
├── main.dart                    app + theme wiring
├── theme.dart                   Material 3 colour scheme, light and dark
├── models/
│   ├── predict_response.dart    the §10.2 contract
│   └── chat_message.dart        user / bot / error turns
├── services/
│   └── api_service.dart         HTTP, configurable base URL, typed failures
├── screens/
│   ├── welcome_screen.dart
│   └── chat_screen.dart         conversation, loading, auto-scroll, errors
└── widgets/
    ├── message_bubble.dart      answer + source chip + disclaimer
    └── typing_indicator.dart
```

## What the UI guarantees

- **Attribution is visible.** Every answer shows its source, badged `official` (WHO, CDC, a
  government health department) or `community` (WikiHow and similar). The corpus is ~44%
  community content, so presenting the two identically would misrepresent authority.
- **The disclaimer is on every answer**, including abstentions.
- **Abstention looks different from an answer.** Below the confidence threshold the bot says
  "No confident match" and shows no source — a safe fallback must never appear to carry a
  WHO badge.
- **Failures are never presented as answers.** Network errors render as an error bubble with
  a plain-language message. The previous app printed raw exception text into the transcript.

## Networking config

- **Android:** `INTERNET` is declared in the *main* manifest, and
  `res/xml/network_security_config.xml` permits cleartext only for `10.0.2.2`, `127.0.0.1`
  and `localhost`. Everything else must be HTTPS.
- **iOS:** `NSAllowsLocalNetworking` in `Info.plist`, which is the ATS exception scoped to
  local addresses.

## Tests

```bash
flutter test                                    # 32 tests, no server needed
flutter test test/integration/live_api_test.dart  # against a running API
```

`test/widget_test.dart` drives the real screens against a mocked HTTP client.
`test/integration/live_api_test.dart` is the only test that exercises the Dart client and
the Python service together — it **skips automatically** when the API isn't running, so the
default suite stays green offline.
