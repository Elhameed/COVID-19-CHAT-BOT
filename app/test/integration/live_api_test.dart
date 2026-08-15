// Integration check against a REAL running API (PRD §12, §19).
//
// Skipped automatically when the server isn't up, so `flutter test` stays green
// offline. To run it for real:
//
//     uvicorn src.api:app          # from the repository root
//     flutter test test/integration/live_api_test.dart
//
// This is the only test that proves the Dart client and the Python service
// actually agree on the wire format. Everything else mocks one side or the
// other, and a contract that both halves mock is a contract nobody checks.

import 'dart:io';

import 'package:covicare/services/api_service.dart';
import 'package:flutter_test/flutter_test.dart';

const _baseUrl = String.fromEnvironment('API_BASE_URL', defaultValue: 'http://127.0.0.1:8000');

void main() {
  setUpAll(() {
    // flutter_test installs an HttpOverrides that fakes every request; real
    // network calls need it removed.
    HttpOverrides.global = null;
  });

  late ApiService api;
  var serverUp = false;

  setUp(() async {
    api = ApiService(baseUrl: _baseUrl, timeout: const Duration(seconds: 20));
    serverUp = await api.isHealthy();
  });

  tearDown(() => api.dispose());

  test('health endpoint responds', () async {
    if (!serverUp) return markTestSkipped('API not running at $_baseUrl');
    expect(await api.isHealthy(), isTrue);
  });

  test('an in-scope question returns an attributed answer', () async {
    if (!serverUp) return markTestSkipped('API not running at $_baseUrl');

    final r = await api.predict('How long should I isolate after testing positive?');
    expect(r.abstained, isFalse);
    expect(r.answer, isNotEmpty);
    expect(r.hasSource, isTrue, reason: 'an answer must carry attribution');
    expect(r.trust, anyOf('official', 'community'));
    expect(r.disclaimer, contains('not medical advice'));
    expect(r.score, greaterThan(0));
  });

  test('an off-topic question abstains without attribution', () async {
    if (!serverUp) return markTestSkipped('API not running at $_baseUrl');

    final r = await api.predict('What is the capital of France?');
    expect(r.abstained, isTrue);
    expect(r.source, isNull);
    expect(r.trust, isNull);
    expect(r.matchedQuestion, isNull);
    expect(r.disclaimer, isNotEmpty);
  });

  test('over-long input is rejected as a client error', () async {
    if (!serverUp) return markTestSkipped('API not running at $_baseUrl');

    await expectLater(
      api.predict('x' * 501),
      throwsA(isA<ApiException>()
          .having((e) => e.failure, 'failure', ApiFailure.invalidRequest)),
    );
  });

  test('non-ASCII text survives the round trip', () async {
    if (!serverUp) return markTestSkipped('API not running at $_baseUrl');

    // Curly quotes and accents are common in the corpus; a latin-1 fallback
    // anywhere in the chain would corrupt them.
    final r = await api.predict('Do I need a mask — even outdoors?');
    expect(r.answer, isNotEmpty);
    expect(r.answer.contains('â€'), isFalse, reason: 'mojibake in the answer');
  });

  test('an unreachable server surfaces a friendly retryable error', () async {
    final dead = ApiService(
      baseUrl: 'http://127.0.0.1:9',
      timeout: const Duration(seconds: 3),
    );
    addTearDown(dead.dispose);

    await expectLater(
      dead.predict('anything'),
      throwsA(isA<ApiException>()
          .having((e) => e.isRetryable, 'isRetryable', isTrue)
          .having((e) => e.message, 'message', contains("Can't reach"))),
    );
  });
}
