import 'package:flutter/material.dart';

/// Covicare's visual identity, in one place.
///
/// The previous build repeated `Color(0xFF6A5AE0)` at eight call sites with no
/// theme, so nothing could be restyled without hunting for literals.
abstract final class CovicareTheme {
  static const Color seed = Color(0xFF6A5AE0);

  /// Colours carrying meaning rather than decoration. Trust tier is a safety
  /// signal, so it gets a deliberate, consistent treatment.
  static const Color official = Color(0xFF2E7D5B);
  static const Color community = Color(0xFF9A6A00);

  static ThemeData light() => _base(Brightness.light);
  static ThemeData dark() => _base(Brightness.dark);

  static ThemeData _base(Brightness brightness) {
    final scheme = ColorScheme.fromSeed(seedColor: seed, brightness: brightness);
    return ThemeData(
      useMaterial3: true,
      colorScheme: scheme,
      scaffoldBackgroundColor: scheme.surface,
      appBarTheme: AppBarTheme(
        backgroundColor: scheme.primary,
        foregroundColor: scheme.onPrimary,
        centerTitle: true,
        elevation: 0,
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
          shape: const StadiumBorder(),
          textStyle: const TextStyle(fontSize: 17, fontWeight: FontWeight.w600),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: scheme.surfaceContainerHighest,
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(24),
          borderSide: BorderSide.none,
        ),
      ),
    );
  }
}
