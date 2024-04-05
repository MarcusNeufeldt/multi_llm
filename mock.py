import unittest
import os
from unittest.mock import patch, mock_open
from io import StringIO
from pathlib import Path
from azure.cognitiveservices.speech import ResultReason

# Import the functions to be tested
from llms import read_prompt_from_file, query_llms, save_responses, combine_responses, summarize_responses

class TestMain(unittest.TestCase):

    def test_read_prompt_from_file_valid(self):
        # Arrange
        mock_file_content = "This is a test prompt."
        mock_open_object = mock_open(read_data=mock_file_content)
        with patch('builtins.open', mock_open_object):

            # Act
            prompt = read_prompt_from_file("valid_prompt.txt")

            # Assert
            self.assertEqual(prompt, mock_file_content)

    def test_read_prompt_from_file_nonexistent(self):
        # Arrange
        with patch('builtins.open', side_effect=FileNotFoundError):

            # Act & Assert
            with self.assertRaises(FileNotFoundError):
                read_prompt_from_file("nonexistent_file.txt")

    @patch('requests.post')
    def test_query_llms_valid(self, mock_post):
        # Arrange
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Response 1"}}]
        }
        prompt = "Hello, world!"
        llm_models = ["valid_model_1", "valid_model_2"]

        # Act
        responses = query_llms(prompt, llm_models)

        # Assert
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses, ["Response 1", "Response 1"])

    @patch('requests.post')
    def test_query_llms_empty_models(self, mock_post):
        # Arrange
        prompt = "Hello, world!"
        llm_models = []

        # Act
        responses = query_llms(prompt, llm_models)

        # Assert
        self.assertEqual(responses, [])

    def test_save_responses_valid(self):
        # Arrange
        responses = ["Response 1", "Response 2"]
        output_dir = "test_output_dir"
        os.makedirs(output_dir, exist_ok=True)

        # Act
        save_responses(responses, output_dir)

        # Assert
        self.assertTrue(os.path.isfile(os.path.join(output_dir, "response_1.md")))
        self.assertTrue(os.path.isfile(os.path.join(output_dir, "response_2.md")))

        # Clean up
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        os.rmdir(output_dir)

    def test_save_responses_invalid_dir(self):
        # Arrange
        responses = ["Response 1", "Response 2"]
        invalid_dir = "/invalid/path"

        # Act & Assert
        with self.assertRaises((PermissionError, OSError)):
            save_responses(responses, invalid_dir)

    def test_combine_responses_valid(self):
        # Arrange
        output_dir = "test_output_dir"
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "response_1.md"), "w") as f:
            f.write("# Response 1\n\nThis is response 1.")
        with open(os.path.join(output_dir, "response_2.md"), "w") as f:
            f.write("# Response 2\n\nThis is response 2.")

        expected_combined_text = "# Combined Responses\n\n## response_1\n\n# Response 1\n\nThis is response 1.\n\n\n\n## response_2\n\n# Response 2\n\nThis is response 2.\n\n"

        # Act
        combined_text = combine_responses(os.path.join(output_dir, "responses"))

        # Assert
        self.assertEqual(combined_text, expected_combined_text)

        # Clean up
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        os.rmdir(output_dir)

    def test_combine_responses_empty_dir(self):
        # Arrange
        output_dir = "test_output_dir"
        os.makedirs(output_dir, exist_ok=True)

        # Act
        combined_text = combine_responses(os.path.join(output_dir, "responses"))

        # Assert
        self.assertEqual(combined_text, "# Combined Responses\n\n")

        # Clean up
        os.rmdir(output_dir)

    @patch('requests.post')
    def test_summarize_responses_valid(self, mock_post):
        # Arrange
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "This is a summary."}}]
        }
        combined_text = "Combined responses..."
        summarizer_model = "valid_model"
        original_prompt = "Hello, world!"

        # Act
        summary = summarize_responses(combined_text, summarizer_model, original_prompt)

        # Assert
        self.assertEqual(summary, "This is a summary.")

    @patch('requests.post')
    def test_summarize_responses_empty(self, mock_post):
        # Arrange
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": ""}}]
        }
        combined_text = ""
        summarizer_model = "valid_model"
        original_prompt = "Hello, world!"

        # Act
        summary = summarize_responses(combined_text, summarizer_model, original_prompt)

        # Assert
        self.assertEqual(summary, "")

    def test_main_execution_flow(self):
        # Arrange
        prompt_file_path = "test_prompt.txt"
        with open(prompt_file_path, "w") as f:
            f.write("This is a test prompt.")

        llm_models = ["mock_model_1", "mock_model_2"]
        summarizer_model = "mock_summarizer_model"

        with patch('main.query_llms') as mock_query_llms, \
             patch('main.save_responses') as mock_save_responses, \
             patch('main.combine_responses') as mock_combine_responses, \
             patch('main.summarize_responses') as mock_summarize_responses:

            mock_query_llms.return_value = ["Response 1", "Response 2"]
            mock_combine_responses.return_value = "Combined responses..."
            mock_summarize_responses.return_value = "This is a summary."

            # Act
            from main import __name__, main
            if __name__ == "__main__":
                main()

            # Assert
            mock_query_llms.assert_called_once()
            mock_save_responses.assert_called_once()
            mock_combine_responses.assert_called_once()
            mock_summarize_responses.assert_called_once()

            # Clean up
            os.remove(prompt_file_path)
            output_dir = "output_*"
            for dir_path in Path(".").glob(output_dir):
                for file_path in dir_path.glob("*"):
                    file_path.unlink()
                dir_path.rmdir()

    @patch('azure.cognitiveservices.speech.SpeechSynthesizer')
    def test_azure_tts_valid(self, mock_speech_synthesizer):
        # Arrange
        mock_speech_synthesis_result = mock_speech_synthesizer.return_value.speak_text_async.return_value.get.return_value
        mock_speech_synthesis_result.reason = ResultReason.SynthesizingAudioCompleted
        summary = "This is a valid summary."

        # Act
        from main import read_aloud
        read_aloud = True
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            from main import __name__, main
            if __name__ == "__main__":
                main()

        # Assert
        self.assertIn("Speech synthesized successfully.", mock_stdout.getvalue())

if __name__ == '__main__':
    unittest.main()