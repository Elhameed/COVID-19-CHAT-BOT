# Covicare — Flutter client

Chat UI for the COVID-19 retrieval API. The app holds **no model logic**: it sends a
question to `POST /predict` and renders the returned answer together with its source
and disclaimer (PRD §10.2, §12).

## Running

Start the backend first, from the repository root:

```bash
uvicorn src.api:app --reload      # serves on :8000
```

Then run the app, pointing it at the right host for your target:

```bash
flutter run --dart-define=API_BASE_URL=http://10.0.2.2:8000     # Android emulator
flutter run --dart-define=API_BASE_URL=http://127.0.0.1:8000    # desktop / web
flutter run --dart-define=API_BASE_URL=http://<LAN-IP>:8000     # physical device
```

`10.0.2.2` is the Android emulator's alias for the host machine — `127.0.0.1` there
resolves to the emulator itself, which is why the previous build could never reach the
API from a device.

## Status

Phase 0 (rebrand and hygiene) is done. The API wiring, response schema, and UX fixes
listed in PRD §11 land in **Phase 7**; until then this app still targets the old
single-field response and hardcodes its base URL.
