import 'package:flutter/material.dart';

import '../models/chat_message.dart';
import '../theme.dart';

/// Renders one conversation turn.
///
/// For a bot answer this shows the stored text **plus its source, trust tier and
/// the medical disclaimer**. Those must be visible in the UI, not
/// merely present in the JSON — an attributed answer whose attribution the user
/// never sees is functionally unattributed.
class MessageBubble extends StatelessWidget {
  const MessageBubble({required this.message, super.key});

  final ChatMessage message;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;

    if (message.isUser) {
      return _Aligned(
        alignment: Alignment.centerRight,
        child: _Container(
          color: scheme.primary,
          child: Text(message.text, style: TextStyle(color: scheme.onPrimary, fontSize: 16)),
        ),
      );
    }

    if (message.isError) {
      return _Aligned(
        alignment: Alignment.centerLeft,
        child: _Container(
          color: scheme.errorContainer,
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(Icons.cloud_off, size: 20, color: scheme.onErrorContainer),
              const SizedBox(width: 10),
              Flexible(
                child: Text(
                  message.text,
                  style: TextStyle(color: scheme.onErrorContainer, fontSize: 15),
                ),
              ),
            ],
          ),
        ),
      );
    }

    final response = message.response!;
    return _Aligned(
      alignment: Alignment.centerLeft,
      child: _Container(
        color: message.isAbstention
            ? scheme.surfaceContainerHighest
            : scheme.secondaryContainer,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (message.isAbstention) ...[
              Row(
                children: [
                  Icon(Icons.help_outline, size: 18, color: scheme.onSurfaceVariant),
                  const SizedBox(width: 6),
                  Text(
                    'No confident match',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: scheme.onSurfaceVariant,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
            ],
            SelectableText(
              response.answer,
              style: TextStyle(
                color: message.isAbstention ? scheme.onSurfaceVariant : scheme.onSecondaryContainer,
                fontSize: 16,
                height: 1.4,
              ),
            ),
            if (response.hasSource) ...[
              const SizedBox(height: 12),
              _SourceChip(source: response.source!, isOfficial: response.isOfficial),
              if (response.matchedQuestion != null) ...[
                const SizedBox(height: 6),
                Text(
                  'Matched: "${response.matchedQuestion}"',
                  style: TextStyle(
                    fontSize: 11.5,
                    fontStyle: FontStyle.italic,
                    color: scheme.onSecondaryContainer.withValues(alpha: 0.75),
                  ),
                ),
              ],
            ],
            if (response.disclaimer.isNotEmpty) ...[
              const SizedBox(height: 10),
              Divider(height: 1, color: scheme.outlineVariant),
              const SizedBox(height: 8),
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.info_outline, size: 14, color: scheme.onSurfaceVariant),
                  const SizedBox(width: 6),
                  Expanded(
                    child: Text(
                      response.disclaimer,
                      style: TextStyle(
                        fontSize: 11.5,
                        height: 1.35,
                        color: scheme.onSurfaceVariant,
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}

/// Names the source and how far to trust it.
///
/// The corpus is ~44% community content (WikiHow and similar), so presenting a
/// WikiHow answer with the same authority as a WHO one would be misleading.
class _SourceChip extends StatelessWidget {
  const _SourceChip({required this.source, required this.isOfficial});

  final String source;
  final bool isOfficial;

  @override
  Widget build(BuildContext context) {
    final color = isOfficial ? CovicareTheme.official : CovicareTheme.community;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.13),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withValues(alpha: 0.45)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(isOfficial ? Icons.verified : Icons.groups_outlined, size: 14, color: color),
          const SizedBox(width: 6),
          Flexible(
            child: Text(
              isOfficial ? source : '$source · community source',
              style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: color),
            ),
          ),
        ],
      ),
    );
  }
}

class _Aligned extends StatelessWidget {
  const _Aligned({required this.alignment, required this.child});

  final Alignment alignment;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: alignment,
      child: ConstrainedBox(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.85,
        ),
        child: child,
      ),
    );
  }
}

class _Container extends StatelessWidget {
  const _Container({required this.color, required this.child});

  final Color color;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 5, horizontal: 10),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(16)),
      child: child,
    );
  }
}
