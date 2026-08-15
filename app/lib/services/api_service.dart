import 'dart:async';
import 'dart:convert';
import 'dart:io' show SocketException;

import 'package:flutter/foundation.dart' show defaultTargetPlatform, kIsWeb, TargetPlatform;
import 'package:http/http.dart' as http;

import '../models/predict_response.dart';

/// Talks to the Covicare retrieval API (PRD §10.2, §12).
///
/// The app holds no model logic: it posts a question and renders what comes
/// back. Everything about *which* answer is chosen lives server-side.
class ApiService {
  ApiService({http.Client? client, String? baseUrl, this.timeout = const Duration(seconds: 15)})
      : _client = client ?? http.Client(),
        baseUrl = baseUrl ?? defaultBaseUrl();

  final http.Client _client;
  final String baseUrl;
  final Duration timeout;

  /// Supplied at build time, e.g.
  /// `flutter run --dart-define=API_BASE_URL=http://10.0.2.2:8000`.
  static const String _configuredBaseUrl = String.fromEnvironment('API_BASE_URL');

  /// Where to reach the API when nothing was configured.
  ///
  /// The default has to differ per platform, and getting this wrong is why the
  /// previous build could never talk to the backend from a device: on the
  /// Android emulator `127.0.0.1` is the *emulator itself*, and the host machine
  /// is reachable only at `10.0.2.2`.
  ///
  /// A physical device shares neither, so it needs the host's LAN address passed
  /// explicitly via `--dart-define`.
  static String defaultBaseUrl() {
    if (_configuredBaseUrl.isNotEmpty) return _configuredBaseUrl;
    if (!kIsWeb && defaultTargetPlatform == TargetPlatform.android) {
      return 'http://10.0.2.2:8000';
    }
    return 'http://127.0.0.1:8000';
  }

  Uri _endpoint(String path) => Uri.parse('$baseUrl$path');

  /// Ask a question. Throws [ApiException] on any failure.
  Future<PredictResponse> predict(String question) async {
    final trimmed = question.trim();
    if (trimmed.isEmpty) {
      throw const ApiException(ApiFailure.invalidRequest, 'Please type a question first.');
    }

    late final http.Response response;
    try {
      response = await _client
          .post(
            _endpoint('/predict'),
            headers: const {'Content-Type': 'application/json; charset=utf-8'},
            body: jsonEncode({'question': trimmed}),
          )
          .timeout(timeout);
    } on TimeoutException {
      throw ApiException(
        ApiFailure.timeout,
        'The server took too long to respond.',
        detail: 'No reply within ${timeout.inSeconds}s from $baseUrl',
      );
    } on SocketException catch (e) {
      throw ApiException(ApiFailure.network, _unreachableMessage(), detail: e.message);
    } on http.ClientException catch (e) {
      throw ApiException(ApiFailure.network, _unreachableMessage(), detail: e.message);
    }

    switch (response.statusCode) {
      case 200:
        break;
      case 422:
        throw const ApiException(
          ApiFailure.invalidRequest,
          "That question couldn't be sent — try rephrasing it, or keep it under 500 characters.",
        );
      case 429:
        throw const ApiException(
          ApiFailure.rateLimited,
          "You're sending questions faster than the server allows. Wait a moment and try again.",
        );
      default:
        throw ApiException(
          ApiFailure.server,
          'The server had a problem answering that.',
          detail: 'HTTP ${response.statusCode}',
        );
    }

    try {
      // utf8.decode, not response.body: the latter falls back to latin-1 when
      // the server omits a charset, which mangles the curly quotes and dashes
      // that are common in the FAQ answers.
      final decoded = jsonDecode(utf8.decode(response.bodyBytes));
      if (decoded is! Map<String, dynamic>) {
        throw const FormatException('expected a JSON object');
      }
      return PredictResponse.fromJson(decoded);
    } on FormatException catch (e) {
      throw ApiException(
        ApiFailure.badResponse,
        "The server's reply couldn't be read.",
        detail: e.message,
      );
    }
  }

  /// Readiness check, used by the connection banner.
  Future<bool> isHealthy() async {
    try {
      final response = await _client.get(_endpoint('/health')).timeout(timeout);
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  String _unreachableMessage() =>
      "Can't reach the Covicare server at $baseUrl. Make sure it's running "
      '(`uvicorn src.api:app`), and that this device can see that address.';

  void dispose() => _client.close();
}

enum ApiFailure { network, timeout, server, rateLimited, invalidRequest, badResponse }

/// A user-presentable failure.
///
/// [message] is safe to show; [detail] is for diagnostics and is deliberately
/// kept out of the chat transcript — the previous app rendered raw exception
/// text, stack traces and all, straight into the conversation.
class ApiException implements Exception {
  const ApiException(this.failure, this.message, {this.detail});

  final ApiFailure failure;
  final String message;
  final String? detail;

  bool get isRetryable =>
      failure == ApiFailure.network ||
      failure == ApiFailure.timeout ||
      failure == ApiFailure.server ||
      failure == ApiFailure.rateLimited;

  @override
  String toString() => 'ApiException(${failure.name}): $message';
}
