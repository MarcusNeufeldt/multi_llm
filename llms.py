import os
import requests
import json
from typing import List
from datetime import datetime
from tqdm import tqdm

# Configure OpenRouter API
OPENROUTER_API_KEY = "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"

def read_prompt_from_file(file_path: str) -> str:
    """
    Read the prompt from a text file.
    Returns the prompt as a string.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    return prompt

def query_llms(prompt: str, llm_models: List[str]) -> List[str]:
    llm_responses = []
    for llm in tqdm(llm_models, desc="Querying LLMs", unit="model"):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                },
                data=json.dumps({
                    "model": llm,
                    "messages": [{"role": "user", "content": prompt}]
                })
            )
            response.raise_for_status()
            llm_responses.append(response.json()["choices"][0]["message"]["content"])
        except requests.exceptions.RequestException as e:
            print(f"Error querying LLM '{llm}': {e}")
            llm_responses.append("Error: Unable to get response from LLM.")
    return llm_responses

def save_responses(responses: List[str], output_dir: str) -> None:
    """
    Save each LLM's response into a separate text file in the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, response in enumerate(tqdm(responses, desc="Saving responses", unit="response"), start=1):
        filename = os.path.join(output_dir, f"response_{i}.md")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# Response {i}\n\n{response}\n")

def combine_responses(output_dir: str) -> str:
    """
    Combine the individual text files into a single Markdown file.
    Returns the combined Markdown text.
    """
    combined_text = "# Combined Responses\n\n"
    for filename in tqdm(sorted(os.listdir(output_dir)), desc="Combining responses", unit="file"):
        if filename.endswith(".md"):
            with open(os.path.join(output_dir, filename), "r", encoding="utf-8") as f:
                combined_text += f"## {filename[:-3]}\n\n{f.read()}\n\n"
    return combined_text

def summarize_responses(combined_text: str, summarizer_model: str, original_prompt: str) -> str:
    """
    Summarize the combined text using another LLM.
    Returns the summary in Markdown format.
    """
    # Define the summary_prompt outside of the if statement
    summary_prompt = (
            "The following text contains code snippets from multiple sources, generated in response to the original prompt:\n"
            f"Original Prompt: {original_prompt}\n\n"
            "Please analyze the reponses, select the best parts of each and provide a combined summary of them in Markdown format.\n"
            "If you are faced with code, then pick the best provided solution and expand it with your own knowledge.\n"
            "Start your response with 'Summary' and then provide your summary. \n\n"
        + combined_text
    )

    if "```" in combined_text:
        # If the combined text contains code snippets
        summary_prompt = (
            "The following text contains code snippets from multiple sources, generated in response to the original prompt:\n"
            f"Original Prompt: {original_prompt}\n\n"
            "Please analyze the reponses, select the best parts of each and provide a combined summary of them in Markdown format.\n"
            "If you are faced with code, then pick the best provided solution and expand it with your own knowledge.\n\n"
            + combined_text
        )

    print("Summarizing responses...")
    try:
        summary_response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            },
            data=json.dumps({
                "model": summarizer_model,
                "messages": [{"role": "user", "content": summary_prompt}]
            })
        )
        response_json = summary_response.json()
        if "choices" in response_json and isinstance(response_json["choices"], list) and len(response_json["choices"]) > 0:
            return response_json["choices"][0]["message"]["content"]
        else:
            raise ValueError("Unexpected JSON response structure from OpenRouter API")
    except requests.exceptions.RequestException as e:
        print(f"Error summarizing responses: {e}")
        return "Error: Unable to summarize responses."
    except ValueError as e:
        print(f"Error summarizing responses: {e}")
        return "Error: Unexpected response from OpenRouter API."


if __name__ == "__main__":
    prompt_file_path = "prompt.txt"
    prompt = read_prompt_from_file(prompt_file_path)

    llm_models = ["perplexity/pplx-7b-online", "cognitivecomputations/dolphin-mixtral-8x7b", "google/gemini-pro","perplexity/sonar-small-online"]
    summarizer_model = "anthropic/claude-3-haiku"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    llm_responses = query_llms(prompt, llm_models)
    save_responses(llm_responses, os.path.join(output_dir, "responses"))
    combined_text = combine_responses(os.path.join(output_dir, "responses"))

    with open(os.path.join(output_dir, "combined_responses.md"), "w", encoding="utf-8") as f:
        f.write(combined_text)

    summary = summarize_responses(combined_text, summarizer_model, prompt)

    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Summary\n\n" + summary)

    print(f"All files saved in the '{output_dir}' folder.")

    # Ask the user if they want the summary read out loud
    read_aloud = input("Do you want the summary to be read out loud? (y/n): ")

    if read_aloud.lower() == "y":
        from azure.cognitiveservices.speech import AudioDataStream, SpeechConfig, SpeechSynthesizer, \
            SpeechSynthesisOutputFormat
        from azure.cognitiveservices.speech.audio import AudioOutputConfig
        from azure.cognitiveservices.speech import ResultReason, CancellationReason
        from markdown import Markdown
        from io import StringIO


        def unmark_element(element, stream=None):
            if stream is None:
                stream = StringIO()
            if element.text:
                stream.write(element.text)
            for sub in element:
                unmark_element(sub, stream)
            if element.tail:
                stream.write(element.tail)
            return stream.getvalue()


        # Patching Markdown
        Markdown.output_formats["plain"] = unmark_element
        __md = Markdown(output_format="plain")
        __md.stripTopLevelTags = False


        def unmark(text):
            return __md.convert(text)


        # Convert the Markdown summary to plain text
        plain_summary = unmark(summary)

        # Set up the Azure TTS configuration
        speech_config = SpeechConfig(subscription="e56bfb375d6c4f45bccb429fc077d103", region="germanywestcentral")
        speech_config.speech_synthesis_language = "en-US"  # Set the language to German
        speech_config.speech_synthesis_voice_name = "en-US-JennyMultilingualNeural"  # Use a German neural voice
        audio_config = AudioOutputConfig(use_default_speaker=True)

        # Create a speech synthesizer
        speech_synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

        # Synthesize speech from the plain text summary
        speech_synthesis_result = speech_synthesizer.speak_text_async(plain_summary).get()

        # Check the synthesis result
        if speech_synthesis_result.reason == ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesized successfully.")
        elif speech_synthesis_result.reason == ResultReason.Canceled:
            cancellation_details = speech_synthesis_result.cancellation_details
            print(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")