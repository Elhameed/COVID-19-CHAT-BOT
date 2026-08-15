import 'package:flutter/material.dart';

import '../models/chat_message.dart';
import '../services/api_service.dart';
import '../widgets/message_bubble.dart';
import '../widgets/typing_indicator.dart';

/// The conversation.
///
/// Holds no model logic: it posts to `/predict` and renders what returns,
/// including the abstention path.
/// Identifies the send button for tests.
///
/// Finding it by tooltip returns the [Tooltip] wrapper rather than the button,
/// so its `onPressed` — the thing that proves send is disabled mid-request —
/// isn't reachable that way.
const Key sendButtonKey = Key('chat-send-button');

/// How long the "thinking" state stays on screen at minimum.
///
/// Not a delay added to the request — see [ChatScreenState._holdForMinimumDuration].
/// Retrieval is fast enough (~25 ms) that without a floor the typing indicator
/// renders for roughly one frame.
const Duration kMinimumResponseDelay = Duration(milliseconds: 900);

class ChatScreen extends StatefulWidget {
  const ChatScreen({
    this.apiService,
    this.minimumResponseDelay = kMinimumResponseDelay,
    super.key,
  });

  /// Injected in tests; a real [ApiService] is built when omitted.
  final ApiService? apiService;

  /// Minimum time the typing indicator is shown. Tests set this to zero so the
  /// suite doesn't pay a second per interaction.
  final Duration minimumResponseDelay;

  @override
  State<ChatScreen> createState() => ChatScreenState();
}

class ChatScreenState extends State<ChatScreen> {
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final FocusNode _inputFocus = FocusNode();
  final List<ChatMessage> _messages = [];

  late final ApiService _api = widget.apiService ?? ApiService();
  late final bool _ownsApi = widget.apiService == null;

  bool _awaitingReply = false;

  @override
  void dispose() {
    _controller.dispose();
    _scrollController.dispose();
    _inputFocus.dispose();
    if (_ownsApi) _api.dispose();
    super.dispose();
  }

  Future<void> _send() async {
    final question = _controller.text.trim();
    // Guards both an empty box and a double tap while a request is in flight.
    if (question.isEmpty || _awaitingReply) return;

    setState(() {
      _messages.add(ChatMessage.user(question));
      _controller.clear();
      _awaitingReply = true;
    });
    _scrollToBottom();

    // Timed from the moment the request is *sent*, so the pacing below can
    // subtract real network time rather than adding to it.
    final elapsed = Stopwatch()..start();

    try {
      final response = await _api.predict(question);
      await _holdForMinimumDuration(elapsed);
      if (!mounted) return;
      setState(() => _messages.add(ChatMessage.bot(response)));
    } on ApiException catch (e) {
      await _holdForMinimumDuration(elapsed);
      if (!mounted) return;
      // The message is written for a user; `detail` stays out of the transcript.
      setState(() => _messages.add(ChatMessage.error(e.message, detail: e.detail)));
    } finally {
      if (mounted) {
        setState(() => _awaitingReply = false);
        _scrollToBottom();
        _inputFocus.requestFocus();
      }
    }
  }

  /// Keeps the "thinking" state on screen for a minimum length of time.
  ///
  /// Retrieval answers in about 25 ms, so without this the typing indicator
  /// appears and vanishes inside a single frame. The result reads as a glitch
  /// rather than as a reply, and gives no sense that anything was looked up.
  ///
  /// This pads *only the remainder*: it waits `minimum - elapsed`, so a request
  /// that already took longer than the floor waits not at all. Real latency is
  /// therefore never increased, and a slow or failing network is never made to
  /// feel slower than it is. The API call itself is untouched — this is purely
  /// presentation, applied after the response is already in hand.
  Future<void> _holdForMinimumDuration(Stopwatch elapsed) async {
    final remaining = widget.minimumResponseDelay - elapsed.elapsed;
    if (remaining > Duration.zero) {
      await Future<void>.delayed(remaining);
    }
  }

  /// Keeps the newest message in view.
  ///
  /// Deferred a frame so the list has been laid out with the new item before we
  /// measure its extent — otherwise this scrolls to the previous maximum.
  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) return;
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 280),
        curve: Curves.easeOut,
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    // One extra row holds the typing indicator while a reply is pending.
    final itemCount = _messages.length + (_awaitingReply ? 1 : 0);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Covicare'),
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            tooltip: 'About',
            onPressed: () => _showAbout(context),
          ),
        ],
      ),
      body: Column(
        children: [
          _ScopeBanner(),
          Expanded(
            child: _messages.isEmpty && !_awaitingReply
                ? const _EmptyState()
                : ListView.builder(
                    controller: _scrollController,
                    padding: const EdgeInsets.symmetric(vertical: 12),
                    itemCount: itemCount,
                    itemBuilder: (context, index) {
                      if (index >= _messages.length) return const TypingIndicator();
                      return MessageBubble(message: _messages[index]);
                    },
                  ),
          ),
          SafeArea(
            top: false,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(12, 8, 12, 12),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Expanded(
                    child: TextField(
                      controller: _controller,
                      focusNode: _inputFocus,
                      enabled: !_awaitingReply,
                      maxLength: 500,
                      maxLines: 4,
                      minLines: 1,
                      textInputAction: TextInputAction.send,
                      onSubmitted: (_) => _send(),
                      decoration: const InputDecoration(
                        hintText: 'Ask about COVID-19…',
                        counterText: '',
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  IconButton.filled(
                    key: sendButtonKey,
                    onPressed: _awaitingReply ? null : _send,
                    icon: _awaitingReply
                        ? SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: scheme.onPrimary,
                            ),
                          )
                        : const Icon(Icons.send),
                    tooltip: 'Send',
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  void _showAbout(BuildContext context) {
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('About Covicare'),
        content: const SingleChildScrollView(
          child: Text(
            'Covicare answers COVID-19 questions by finding the closest match in a '
            'fixed set of vetted public-health FAQs. It never writes its own '
            'medical text — every answer is stored content shown with its source.\n\n'
            'When nothing matches confidently it says so rather than guessing.\n\n'
            'The information dates from 2020–2021 and may be out of date. It is not '
            'medical advice.',
          ),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Close')),
        ],
      ),
    );
  }
}

/// States the bot's scope up front, so a user learns its limits before hitting
/// them.
class _ScopeBanner extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      color: scheme.surfaceContainerHighest,
      child: Text(
        'Answers come from vetted COVID-19 FAQs (2020–2021). Not medical advice.',
        textAlign: TextAlign.center,
        style: TextStyle(fontSize: 11.5, color: scheme.onSurfaceVariant),
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState();

  static const _examples = [
    'How does COVID-19 spread?',
    'How long should I isolate?',
    'Do face coverings work?',
  ];

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.forum_outlined, size: 56, color: scheme.primary.withValues(alpha: 0.5)),
            const SizedBox(height: 16),
            Text(
              'Ask a COVID-19 question',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w600,
                color: scheme.onSurface,
              ),
            ),
            const SizedBox(height: 20),
            for (final example in _examples)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Text(
                  '“$example”',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 14, color: scheme.onSurfaceVariant),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
