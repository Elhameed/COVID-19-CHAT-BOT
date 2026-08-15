import 'package:flutter/material.dart';

import 'screens/welcome_screen.dart';
import 'theme.dart';

void main() => runApp(const CovicareApp());

class CovicareApp extends StatelessWidget {
  const CovicareApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Covicare',
      debugShowCheckedModeBanner: false,
      theme: CovicareTheme.light(),
      darkTheme: CovicareTheme.dark(),
      home: const WelcomeScreen(),
    );
  }
}
