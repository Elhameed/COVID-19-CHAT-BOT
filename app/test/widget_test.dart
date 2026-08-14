// Smoke test for the Covicare app shell.
//
// This replaces the default Flutter counter template, which referenced a
// counter widget this app never had and failed to compile.
//
// Scope is deliberately narrow: it covers the widget tree as it exists after
// Phase 0. The real behavioural tests — sending a question against a mocked
// API and rendering answer/source/disclaimer/abstention — arrive with the
// Phase 7 integration work (PRD §11, §19).

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:covicare/main.dart';

void main() {
  testWidgets('welcome screen renders and navigates to the chat screen',
      (WidgetTester tester) async {
    await tester.pumpWidget(MyApp());

    expect(find.text('Welcome to Covicare'), findsOneWidget);
    expect(find.text('Ask'), findsOneWidget);

    await tester.tap(find.text('Ask'));
    await tester.pumpAndSettle();

    expect(find.byType(ChatScreen), findsOneWidget);
    expect(find.text('Type your message...'), findsOneWidget);
  });

  testWidgets('empty input does not create a message bubble',
      (WidgetTester tester) async {
    await tester.pumpWidget(MaterialApp(home: ChatScreen()));

    await tester.tap(find.byIcon(Icons.send));
    await tester.pump();

    expect(find.byType(ChatBubble), findsNothing);
  });
}
