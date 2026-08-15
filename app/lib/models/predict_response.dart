/// The `POST /predict` response.
///
/// The API returns either a stored FAQ answer with its attribution, or an
/// abstention. On abstention `matchedQuestion`, `source`, `trust` and `url` are
/// all null — deliberately, so the UI cannot present the safe fallback message
/// as though a real public-health body had said it.
class PredictResponse {
  const PredictResponse({
    required this.answer,
    required this.score,
    required this.abstained,
    required this.disclaimer,
    this.matchedQuestion,
    this.source,
    this.trust,
    this.url,
  });

  final String answer;
  final String? matchedQuestion;
  final String? source;
  final String? trust;
  final String? url;
  final double score;
  final bool abstained;
  final String disclaimer;

  /// True when the answer came from an official body (WHO, CDC, a government
  /// health department) rather than a community source such as WikiHow.
  bool get isOfficial => trust == 'official';

  /// Whether there is any attribution worth showing.
  bool get hasSource => !abstained && source != null && source!.isNotEmpty;

  factory PredictResponse.fromJson(Map<String, dynamic> json) {
    final answer = json['answer'];
    if (answer is! String || answer.isEmpty) {
      throw const FormatException('response is missing "answer"');
    }
    return PredictResponse(
      answer: answer,
      matchedQuestion: _asNullableString(json['matched_question']),
      source: _asNullableString(json['source']),
      trust: _asNullableString(json['trust']),
      url: _asNullableString(json['url']),
      // Tolerate an int: JSON has no float/int distinction and a score of
      // exactly 1 can arrive as `1` rather than `1.0`.
      score: (json['score'] as num?)?.toDouble() ?? 0.0,
      abstained: json['abstained'] as bool? ?? false,
      disclaimer: _asNullableString(json['disclaimer']) ?? '',
    );
  }

  /// Treats an empty string as absent, so the UI never renders a blank source
  /// chip for an entry whose `url` was empty in the corpus.
  static String? _asNullableString(Object? value) {
    if (value == null) return null;
    final text = value.toString().trim();
    return text.isEmpty ? null : text;
  }
}
