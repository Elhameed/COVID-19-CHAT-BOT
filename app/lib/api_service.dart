import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = "http://127.0.0.1:8000";

  // Method to send a question to the API
  static Future<String> getResponse(String question) async {
    final url = Uri.parse('$baseUrl/predict/');
    final headers = {'Content-Type': 'application/json'};
    final body = jsonEncode({'question': question});

    try {
      final response = await http.post(url, headers: headers, body: body);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['response']; // Extract the response from the JSON
      } else {
        throw Exception('Failed to load response: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Failed to connect to the server: $e');
    }
  }
}
