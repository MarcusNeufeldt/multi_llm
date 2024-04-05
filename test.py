import streamlit as st
import requests
import asyncio
import json
from llms import query_llms, save_responses, combine_responses, summarize_responses
from azure.cognitiveservices.speech import AudioDataStream, SpeechConfig, SpeechSynthesizer, SpeechSynthesisOutputFormat
from azure.cognitiveservices.speech.audio import AudioOutputConfig
from azure.cognitiveservices.speech import ResultReason, CancellationReason
from markdown import Markdown
from io import StringIO
import time
import shutil

st.set_page_config(page_title="Multi-LLM Chatbot", page_icon=":robot_face:", layout="wide", initial_sidebar_state="expanded", theme="dark")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_API_URL = "https://openrouter.ai/api/v1/models"
CREDITS_API_URL = "https://openrouter.ai/api/v1/auth/key"

# Add a text input for the API key with a clear label and placeholder
api_key = st.text_input("Enter your OpenRouter API key:", type="password", placeholder="Your API key")

# Provide instructions on how to obtain an API key
st.markdown("Don't have an API key? Sign up at [OpenRouter](https://openrouter.ai) to get one.")


if not api_key:
    st.warning("Please enter your OpenRouter API key to continue.")
    st.stop()

# Check if the API key is valid
headers = {
    "Authorization": f"Bearer {api_key}"
}
try:
    response = requests.get(CREDITS_API_URL, headers=headers)
    if response.status_code != 200:
        st.error("Invalid API key. Please enter a valid OpenRouter API key.")
        st.stop()
except Exception as e:
    st.error(f"An error occurred while validating the API key: {str(e)}")
    st.stop()

async def query_model(model_name, prompt, expander):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = await asyncio.to_thread(requests.post, API_URL, headers=headers, json=data, stream=True)
        response_data = ""
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                response_data += chunk.decode()

        try:
            data = json.loads(response_data)
            if "error" in data:
                error_message = data["error"]["message"]
                expander.error(f"Error: {error_message}")
                return None, error_message
            else:
                content = data["choices"][0]["message"]["content"]
                expander.markdown(content)

                # Display token information if available
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", "N/A")
                completion_tokens = usage.get("completion_tokens", "N/A")
                total_tokens = usage.get("total_tokens", "N/A")
                token_cost = usage.get("total_cost", "N/A")

                # Format token cost as currency
                if token_cost != "N/A":
                    token_cost = f"${token_cost:.5f}"

                expander.write(f"Prompt Tokens: {prompt_tokens}")
                expander.write(f"Completion Tokens: {completion_tokens}")
                expander.write(f"Total Tokens: {total_tokens}")
                expander.write(f"Token Cost: {token_cost}")

                return content, None
        except (json.JSONDecodeError, KeyError):
            expander.error("Error: Unable to parse the API response.")
            return None, "Error: Unable to parse the API response."
    except Exception as e:
        expander.error(f"An error occurred: {str(e)}")
        return None, f"An error occurred: {str(e)}"
def get_models():
    try:
        response = requests.get(MODELS_API_URL)
        if response.status_code == 200:
            models_data = response.json()
            if isinstance(models_data, dict) and "data" in models_data:
                free_models = []
                moderate_models = []
                expensive_models = []

                for model in models_data["data"]:
                    if "id" in model and "pricing" in model:
                        prompt_cost = float(model["pricing"]["prompt"])
                        completion_cost = float(model["pricing"]["completion"])

                        if prompt_cost == 0 and completion_cost == 0:
                            free_models.append(model["id"])
                        elif prompt_cost < 0.000002 and completion_cost < 0.000002:
                            moderate_models.append(model["id"])
                        else:
                            expensive_models.append(model["id"])

                return free_models, moderate_models, expensive_models
            else:
                st.error("Unexpected format of models data.")
        else:
            st.error(f"Failed to fetch models. Status code: {response.status_code}")
    except Exception as e:
        st.error(f"An error occurred while fetching models: {str(e)}")
    return [], [], []

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


def get_credits():
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    try:
        response = requests.get(CREDITS_API_URL, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "data" in data:
                credits_data = data["data"]
                usage = credits_data.get("usage", 0)
                limit = credits_data.get("limit", None)
                is_free_tier = credits_data.get("is_free_tier", False)
                rate_limit = credits_data.get("rate_limit", {})
                requests_limit = rate_limit.get("requests", "N/A")
                interval = rate_limit.get("interval", "N/A")

                # Convert usage to dollar amount
                dollar_amount = usage * 0.0001  # Assuming $0.0001 per credit

                return {
                    "dollar_amount": dollar_amount,
                    "usage": usage,
                    "limit": limit,
                    "is_free_tier": is_free_tier,
                    "rate_limit": {
                        "requests": requests_limit,
                        "interval": interval
                    }
                }
            else:
                st.error("Unexpected format of credits data.")
        else:
            st.error(f"Failed to fetch credits. Status code: {response.status_code}")
    except Exception as e:
        st.error(f"An error occurred while fetching credits: {str(e)}")
    return None


st.title("Multi-LLM Chatbot")

st.markdown("""
## Instructions
1. Select the desired models from the dropdown menus below.
2. Enter your prompt in the text input field.
3. Click the "Send" button to generate responses from the selected models.
""")

# Display available credits
credits_info = get_credits()
if credits_info is not None:
    st.write(f"Usage: {credits_info['usage']} credits")
else:
    st.write("Failed to fetch available credits.")

# Add a toggle switch for automatic summary
auto_summary = st.checkbox("Automatic Summary", value=True)

# Add the "Start Speech" checkbox below the "Automatic Summary" checkbox
start_speech = st.checkbox("Start Speech")

try:
    free_models, moderate_models, expensive_models = get_models()

    # Sort the models alphabetically
    free_models.sort()
    moderate_models.sort()
    expensive_models.sort()
except Exception as e:
    st.error(f"Failed to fetch models: {str(e)}")
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    selected_free_models = st.multiselect("Select free models", free_models)

with col2:
    selected_moderate_models = st.multiselect("Select cheap models", moderate_models)

with col3:
    selected_expensive_models = st.multiselect("Select expensive models", expensive_models)

prompt_container = st.container()
with prompt_container:
    prompt = st.text_input("Enter your prompt:", key="prompt_input", placeholder="Type your question or request here")
    send_button = st.button("Generate Responses", disabled=False, key="send_button")


if send_button:
    selected_models = selected_free_models + selected_moderate_models + selected_expensive_models
    expanders = [st.expander(model, expanded=False) for model in selected_models]

    if "responses" not in st.session_state:
        st.session_state.responses = None
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "speech_synthesizer" not in st.session_state:
        st.session_state.speech_synthesizer = None

    # Clear the contents of the "responses" folder
    shutil.rmtree("responses", ignore_errors=True)


    async def main():
        try:
            with st.spinner("Generating responses..."):
                # Add progress bar
                progress_bar = st.progress(0, text="Processing model: N/A")
                total_models = len(selected_models)

                start_time = time.time()
                responses = []
                for i, model in enumerate(selected_models):
                    with st.expander(model, expanded=False):
                        response = await query_model(model, prompt, st)
                        responses.append(response)

                    # Update progress bar
                    progress = (i + 1) / total_models
                    elapsed_time = time.time() - start_time
                    estimated_time_remaining = (elapsed_time / (i + 1)) * (total_models - i - 1)
                    progress_bar.progress(progress,
                                         text=f"Processing model: {model} ({i + 1}/{total_models}, {estimated_time_remaining:.2f}s remaining)")

                st.session_state.responses = responses

                if auto_summary:
                    with st.spinner("Generating summary..."):
                        save_responses(responses, "responses")
                        combined_text = combine_responses("responses")
                        summary = summarize_responses(combined_text, "anthropic/claude-3-haiku", prompt)
                        st.session_state.summary = summary

                    with st.expander("Summary", expanded=True):
                        st.markdown(summary)

                        if start_speech:
                            if st.session_state.summary:
                                # Convert the Markdown summary to plain text
                                plain_summary = unmark(st.session_state.summary)

                                # Set up the Azure TTS configuration
                                speech_config = SpeechConfig(subscription="e56bfb375d6c4f45bccb429fc077d103",
                                                             region="germanywestcentral")
                                speech_config.speech_synthesis_language = "en-US"  # Set the language to German
                                speech_config.speech_synthesis_voice_name = "en-US-JennyMultilingualNeural"  # Use a German neural voice
                                audio_config = AudioOutputConfig(use_default_speaker=True)

                                # Create a speech synthesizer
                                speech_synthesizer = SpeechSynthesizer(speech_config=speech_config,
                                                                       audio_config=audio_config)
                                st.session_state.speech_synthesizer = speech_synthesizer

                                # Add a "Stop Speech" button below the summary
                                if st.button("Stop Speech"):
                                    if st.session_state.speech_synthesizer:
                                        st.session_state.speech_synthesizer.stop_speaking()
                                        st.warning("Speech stopped.")

                                # Synthesize speech from the plain text summary asynchronously
                                speech_synthesis_result = speech_synthesizer.speak_text_async(plain_summary)

                                # Check the synthesis result
                                while not speech_synthesis_result.done():
                                    # Wait for the speech synthesis to complete
                                    await asyncio.sleep(0.1)

                                if speech_synthesis_result.result().reason == ResultReason.SynthesizingAudioCompleted:
                                    st.success("Speech synthesized successfully.")
                                elif speech_synthesis_result.result().reason == ResultReason.Canceled:
                                    cancellation_details = speech_synthesis_result.result().cancellation_details
                                    st.error(f"Speech synthesis canceled: {cancellation_details.reason}")
                                    if cancellation_details.reason == CancellationReason.Error:
                                        st.error(f"Error details: {cancellation_details.error_details}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            # Hide the spinner after the task is completed
            st.stop()

    asyncio.run(main())
