import 'package:flutter/material.dart';

/// Shown while a request is in flight.
///
/// The previous build gave no feedback at all between tapping send and the
/// answer appearing, which reads as a frozen app on a cold start.
class TypingIndicator extends StatefulWidget {
  const TypingIndicator({super.key});

  @override
  State<TypingIndicator> createState() => _TypingIndicatorState();
}

class _TypingIndicatorState extends State<TypingIndicator>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1100),
  )..repeat();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 5, horizontal: 10),
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 15),
        decoration: BoxDecoration(
          color: scheme.secondaryContainer,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Semantics(
          label: 'Searching for an answer',
          child: AnimatedBuilder(
            animation: _controller,
            builder: (context, _) => Row(
              mainAxisSize: MainAxisSize.min,
              children: List.generate(3, (i) {
                // Stagger the three dots a third of a cycle apart.
                final t = (_controller.value + i / 3) % 1.0;
                final lift = (t < 0.5 ? t : 1 - t) * 2;
                return Padding(
                  padding: EdgeInsets.only(right: i == 2 ? 0 : 5),
                  child: Transform.translate(
                    offset: Offset(0, -3 * lift),
                    child: CircleAvatar(
                      radius: 4,
                      backgroundColor: scheme.onSecondaryContainer
                          .withValues(alpha: 0.35 + 0.45 * lift),
                    ),
                  ),
                );
              }),
            ),
          ),
        ),
      ),
    );
  }
}
