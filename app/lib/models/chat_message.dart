import 'predict_response.dart';

/// One entry in the conversation.
///
/// A bot turn carries the whole [PredictResponse] rather than just its text, so
/// the UI can render the source, trust tier and disclaimer alongside the answer
/// (PRD §12: those must be visible, not merely present in the payload).
class ChatMessage {
  const ChatMessage._({
    required this.kind,
    required this.text,
    this.response,
    this.errorDetail,
  });

  factory ChatMessage.user(String text) =>
      ChatMessage._(kind: ChatMessageKind.user, text: text);

  factory ChatMessage.bot(PredictResponse response) =>
      ChatMessage._(kind: ChatMessageKind.bot, text: response.answer, response: response);

  /// A failure to reach or parse the API — never presented as an answer.
  factory ChatMessage.error(String text, {String? detail}) =>
      ChatMessage._(kind: ChatMessageKind.error, text: text, errorDetail: detail);

  final ChatMessageKind kind;
  final String text;
  final PredictResponse? response;
  final String? errorDetail;

  bool get isUser => kind == ChatMessageKind.user;
  bool get isError => kind == ChatMessageKind.error;
  bool get isAbstention => response?.abstained ?? false;
}

enum ChatMessageKind { user, bot, error }
