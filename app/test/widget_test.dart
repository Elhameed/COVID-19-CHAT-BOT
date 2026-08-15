// Widget tests for the Covicare chat UI.
//
// These drive the real screens against a mocked HTTP client, so they exercise
// the actual request/response path rather than a stubbed-out view model. The
// guarantees under test are the ones a user would notice breaking: attribution
// and the disclaimer being visible, abstention being distinguishable from an
// answer, and failures never being presented as medical information.

import 'dart:convert';

import 'package:covicare/models/chat_message.dart';
import 'package:covicare/models/predict_response.dart';
import 'package:covicare/screens/chat_screen.dart';
import 'package:covicare/screens/welcome_screen.dart';
import 'package:covicare/services/api_service.dart';
import 'package:covicare/widgets/message_bubble.dart';
import 'package:covicare/widgets/typing_indicator.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

/// A confident, attributed answer.
const _answerJson = {
  'answer': 'Mainly person to person, within 6 feet.',
  'matched_question': 'How is COVID-19 transmitted?',
  'source': 'World Health Organization',
  'trust': 'official',
  'url': 'https://who.int/faq',
  'score': 0.78,
  'abstained': false,
  'disclaimer': 'This is general information, not medical advice.',
};

/// The abstention path: no attribution at all.
const _abstentionJson = {
  'answer': "I don't have a vetted answer for that. See who.int or cdc.gov.",
  'matched_question': null,
  'source': null,
  'trust': null,
  'url': null,
  'score': 0.29,
  'abstained': true,
  'disclaimer': 'This is general information, not medical advice.',
};

ApiService _apiReturning(
  Map<String, dynamic> body, {
  int status = 200,
  Duration delay = Duration.zero,
}) {
  return ApiService(
    baseUrl: 'http://test.local',
    client: MockClient((request) async {
      // A zero-delay mock resolves on the same microtask as the tap, so the
      // in-flight UI never renders. Tests that assert on it pass a delay.
      if (delay > Duration.zero) await Future<void>.delayed(delay);
      return http.Response(
        jsonEncode(body),
        status,
        headers: {'content-type': 'application/json; charset=utf-8'},
      );
    }),
  );
}

ApiService _apiFailing(Exception error) {
  return ApiService(
    baseUrl: 'http://test.local',
    client: MockClient((_) async => throw error),
  );
}

Future<void> _ask(WidgetTester tester, String question) async {
  await tester.enterText(find.byType(TextField), question);
  await tester.tap(find.byKey(sendButtonKey));
  await tester.pump(); // register the user message + spinner
}

Future<void> _pumpChat(
  WidgetTester tester,
  ApiService api, {
  Duration minimumResponseDelay = Duration.zero,
}) async {
  // Zero by default: the pacing floor is presentation, and paying it in every
  // test would add a second per interaction. It has dedicated tests below.
  await tester.pumpWidget(
    MaterialApp(
      home: ChatScreen(apiService: api, minimumResponseDelay: minimumResponseDelay),
    ),
  );
}

void main() {
  group('PredictResponse parsing', () {
    test('reads a full attributed answer', () {
      final r = PredictResponse.fromJson(Map<String, dynamic>.from(_answerJson));
      expect(r.answer, contains('6 feet'));
      expect(r.source, 'World Health Organization');
      expect(r.isOfficial, isTrue);
      expect(r.hasSource, isTrue);
      expect(r.abstained, isFalse);
    });

    test('an abstention exposes no source', () {
      final r = PredictResponse.fromJson(Map<String, dynamic>.from(_abstentionJson));
      expect(r.abstained, isTrue);
      expect(r.source, isNull);
      expect(r.hasSource, isFalse);
    });

    test('an integer score is accepted', () {
      // JSON has no int/double distinction, so a score of 1 arrives as `1`.
      final json = Map<String, dynamic>.from(_answerJson)..['score'] = 1;
      expect(PredictResponse.fromJson(json).score, 1.0);
    });

    test('an empty url is treated as absent', () {
      final json = Map<String, dynamic>.from(_answerJson)..['url'] = '';
      expect(PredictResponse.fromJson(json).url, isNull);
    });

    test('a community source is not marked official', () {
      final json = Map<String, dynamic>.from(_answerJson)..['trust'] = 'community';
      expect(PredictResponse.fromJson(json).isOfficial, isFalse);
    });

    test('a response without an answer is rejected', () {
      expect(
        () => PredictResponse.fromJson({'score': 0.5, 'abstained': false}),
        throwsFormatException,
      );
    });
  });

  group('ApiService', () {
    test('maps 429 to a rate-limit failure', () async {
      final api = _apiReturning(const {}, status: 429);
      await expectLater(
        api.predict('hello'),
        throwsA(isA<ApiException>()
            .having((e) => e.failure, 'failure', ApiFailure.rateLimited)),
      );
    });

    test('maps 422 to an invalid-request failure', () async {
      final api = _apiReturning(const {}, status: 422);
      await expectLater(
        api.predict('hello'),
        throwsA(isA<ApiException>()
            .having((e) => e.failure, 'failure', ApiFailure.invalidRequest)),
      );
    });

    test('reports a connection failure as retryable', () async {
      final api = _apiFailing(http.ClientException('connection refused'));
      await expectLater(
        api.predict('hello'),
        throwsA(isA<ApiException>().having((e) => e.isRetryable, 'isRetryable', isTrue)),
      );
    });

    test('rejects a blank question without calling the server', () async {
      var called = false;
      final api = ApiService(
        baseUrl: 'http://test.local',
        client: MockClient((_) async {
          called = true;
          return http.Response('{}', 200);
        }),
      );
      await expectLater(api.predict('   '), throwsA(isA<ApiException>()));
      expect(called, isFalse);
    });

    test('malformed JSON surfaces as a bad-response failure', () async {
      final api = ApiService(
        baseUrl: 'http://test.local',
        client: MockClient((_) async => http.Response('not json', 200)),
      );
      await expectLater(
        api.predict('hello'),
        throwsA(isA<ApiException>()
            .having((e) => e.failure, 'failure', ApiFailure.badResponse)),
      );
    });

    test('decodes UTF-8 rather than falling back to latin-1', () async {
      // FAQ answers contain curly quotes; a latin-1 fallback mangles them.
      final api = ApiService(
        baseUrl: 'http://test.local',
        client: MockClient((_) async => http.Response.bytes(
              utf8.encode(jsonEncode({..._answerJson, 'answer': 'It’s 20 seconds.'})),
              200,
              headers: {'content-type': 'application/json'},
            )),
      );
      final r = await api.predict('how long');
      expect(r.answer, 'It’s 20 seconds.');
    });
  });

  group('Chat screen', () {
    testWidgets('shows the scope banner and example prompts before any question',
        (tester) async {
      await _pumpChat(tester, _apiReturning(_answerJson));
      expect(find.textContaining('Not medical advice'), findsOneWidget);
      expect(find.text('Ask a COVID-19 question'), findsOneWidget);
    });

    testWidgets('renders the answer with its source and disclaimer', (tester) async {
      await _pumpChat(tester, _apiReturning(_answerJson));
      await _ask(tester, 'how does covid spread?');
      await tester.pumpAndSettle();

      expect(find.textContaining('Mainly person to person'), findsOneWidget);
      // Attribution and disclaimer must be visible, not just present in JSON.
      expect(find.textContaining('World Health Organization'), findsOneWidget);
      expect(find.textContaining('not medical advice'), findsWidgets);
      expect(find.textContaining('How is COVID-19 transmitted?'), findsOneWidget);
    });

    testWidgets('marks a community source as such', (tester) async {
      final json = Map<String, dynamic>.from(_answerJson)
        ..['trust'] = 'community'
        ..['source'] = 'WikiHow';
      await _pumpChat(tester, _apiReturning(json));
      await _ask(tester, 'anything');
      await tester.pumpAndSettle();

      expect(find.textContaining('community source'), findsOneWidget);
    });

    testWidgets('shows an abstention as a non-answer with no source', (tester) async {
      await _pumpChat(tester, _apiReturning(_abstentionJson));
      await _ask(tester, "what's the weather?");
      await tester.pumpAndSettle();

      expect(find.text('No confident match'), findsOneWidget);
      expect(find.textContaining('who.int'), findsOneWidget);
      // The dangerous bug would be a safe message wearing an authority badge.
      expect(find.textContaining('World Health Organization'), findsNothing);
    });

    testWidgets('shows a typing indicator while waiting', (tester) async {
      await _pumpChat(
        tester,
        _apiReturning(_answerJson, delay: const Duration(milliseconds: 200)),
      );
      await _ask(tester, 'how does covid spread?');

      expect(find.byType(TypingIndicator), findsOneWidget);
      await tester.pumpAndSettle();
      expect(find.byType(TypingIndicator), findsNothing);
    });

    testWidgets('disables send while a request is in flight', (tester) async {
      await _pumpChat(
        tester,
        _apiReturning(_answerJson, delay: const Duration(milliseconds: 200)),
      );
      await _ask(tester, 'first question');

      final button = tester.widget<IconButton>(find.byKey(sendButtonKey));
      expect(button.onPressed, isNull, reason: 'send must be disabled mid-request');
      await tester.pumpAndSettle();
    });

    testWidgets('an empty question sends nothing', (tester) async {
      await _pumpChat(tester, _apiReturning(_answerJson));
      await tester.tap(find.byKey(sendButtonKey));
      await tester.pumpAndSettle();
      expect(find.byType(MessageBubble), findsNothing);
    });

    testWidgets('a network failure appears as an error, never as an answer',
        (tester) async {
      await _pumpChat(tester, _apiFailing(http.ClientException('refused')));
      await _ask(tester, 'how does covid spread?');
      await tester.pumpAndSettle();

      expect(find.byIcon(Icons.cloud_off), findsOneWidget);
      expect(find.textContaining("Can't reach the Covicare server"), findsOneWidget);
      // The old app rendered raw exception text into the transcript.
      expect(find.textContaining('ClientException'), findsNothing);
      expect(find.textContaining('#0 '), findsNothing);
    });

    testWidgets('the input clears after sending', (tester) async {
      await _pumpChat(tester, _apiReturning(_answerJson));
      await _ask(tester, 'how does covid spread?');
      await tester.pumpAndSettle();

      expect(tester.widget<TextField>(find.byType(TextField)).controller!.text, isEmpty);
    });

    testWidgets('keeps the whole conversation', (tester) async {
      // ListView.builder only builds what fits; on the default 800x600 surface
      // the earliest bubble is scrolled out and never constructed.
      tester.view.physicalSize = const Size(1200, 2600);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);

      await _pumpChat(tester, _apiReturning(_answerJson));
      await _ask(tester, 'first');
      await tester.pumpAndSettle();
      await _ask(tester, 'second');
      await tester.pumpAndSettle();

      // Two questions and two answers.
      expect(find.byType(MessageBubble), findsNWidgets(4));
    });

    testWidgets('the About dialog explains the bot has limits', (tester) async {
      await _pumpChat(tester, _apiReturning(_answerJson));
      await tester.tap(find.byTooltip('About'));
      await tester.pumpAndSettle();

      expect(find.textContaining('never writes its own'), findsOneWidget);
      expect(find.textContaining('not medical advice'), findsWidgets);
    });
  });

  group('Welcome screen', () {
    testWidgets('leads to the chat screen', (tester) async {
      await tester.pumpWidget(const MaterialApp(home: WelcomeScreen()));
      expect(find.text('Welcome to Covicare'), findsOneWidget);

      await tester.tap(find.text('Ask'));
      await tester.pumpAndSettle();
      expect(find.byType(ChatScreen), findsOneWidget);
    });

    testWidgets('states the disclaimer before the user starts', (tester) async {
      await tester.pumpWidget(const MaterialApp(home: WelcomeScreen()));
      expect(find.textContaining('not medical advice'), findsOneWidget);
    });
  });

  group('Response pacing', () {
    // Retrieval answers in ~25 ms, so the typing indicator would otherwise
    // appear and vanish within a frame. The floor is presentation only.

    testWidgets('a fast answer is held back until the floor elapses',
        (tester) async {
      await _pumpChat(
        tester,
        _apiReturning(_answerJson), // responds instantly
        minimumResponseDelay: const Duration(milliseconds: 900),
      );
      await _ask(tester, 'how does covid spread?');

      // Well after the response arrived, but before the floor.
      await tester.pump(const Duration(milliseconds: 300));
      expect(find.byType(TypingIndicator), findsOneWidget);
      expect(find.textContaining('Mainly person to person'), findsNothing);

      await tester.pump(const Duration(milliseconds: 700));
      expect(find.byType(TypingIndicator), findsNothing);
      expect(find.textContaining('Mainly person to person'), findsOneWidget);
    });

    testWidgets('a slow answer is NOT delayed any further', (tester) async {
      // The response takes longer than the floor, so no padding may be added:
      // the point is to stop the indicator flashing, never to add latency.
      await _pumpChat(
        tester,
        _apiReturning(_answerJson, delay: const Duration(milliseconds: 1500)),
        minimumResponseDelay: const Duration(milliseconds: 900),
      );
      await _ask(tester, 'how does covid spread?');

      await tester.pump(const Duration(milliseconds: 1490));
      expect(find.textContaining('Mainly person to person'), findsNothing);

      // Arrives on its own schedule, with nothing added on top.
      await tester.pump(const Duration(milliseconds: 20));
      await tester.pumpAndSettle();
      expect(find.textContaining('Mainly person to person'), findsOneWidget);
    });

    testWidgets('errors are paced too, so failures do not flash', (tester) async {
      await _pumpChat(
        tester,
        _apiFailing(http.ClientException('refused')),
        minimumResponseDelay: const Duration(milliseconds: 900),
      );
      await _ask(tester, 'how does covid spread?');

      await tester.pump(const Duration(milliseconds: 300));
      expect(find.byIcon(Icons.cloud_off), findsNothing);

      await tester.pump(const Duration(milliseconds: 700));
      expect(find.byIcon(Icons.cloud_off), findsOneWidget);
    });

    testWidgets('send stays disabled for the whole hold', (tester) async {
      await _pumpChat(
        tester,
        _apiReturning(_answerJson),
        minimumResponseDelay: const Duration(milliseconds: 900),
      );
      await _ask(tester, 'first');

      await tester.pump(const Duration(milliseconds: 400));
      expect(
        tester.widget<IconButton>(find.byKey(sendButtonKey)).onPressed,
        isNull,
        reason: 'a second question must not be sendable mid-hold',
      );
      await tester.pumpAndSettle();
      expect(tester.widget<IconButton>(find.byKey(sendButtonKey)).onPressed, isNotNull);
    });

    test('the shipped default is about a second', () {
      expect(kMinimumResponseDelay.inMilliseconds, inInclusiveRange(600, 1200));
    });
  });

  group('ChatMessage', () {
    test('an error message is never mistaken for an answer', () {
      final message = ChatMessage.error('offline', detail: 'SocketException: ...');
      expect(message.isError, isTrue);
      expect(message.response, isNull);
      expect(message.isAbstention, isFalse);
    });
  });
}
