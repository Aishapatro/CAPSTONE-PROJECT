import base64
from functools import lru_cache
from pathlib import Path

import pandas as pd
import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Multiple Language Translator",
    page_icon="🌐",
    layout="centered"
)

st.title("🌐 Multiple Language Translator")
st.write("Translate text into different languages and listen to the translated audio.")


# ---------------------------------------------------------
# Load language dataset
# Expected CSV columns: name, iso
# Example: Odia, OW  (the code below corrects OW to or)
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
LANGUAGE_FILE = BASE_DIR / "language.csv"

# Also check the project's original Windows path if needed.
if not LANGUAGE_FILE.exists():
    LANGUAGE_FILE = Path(
        r"C:\Users\aisha\AVSCODE\CAPSTONE PROJECT\MULTPLE LANGUAGE TRANSLATION\language.csv"
    )


@st.cache_data
def load_languages(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()

    required_columns = {"name", "iso"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "language.csv must contain these columns: name and iso"
        )

    df = df[["name", "iso"]].dropna()
    df["name"] = df["name"].astype(str).str.strip()
    df["iso"] = df["iso"].astype(str).str.strip()
    df = df[(df["name"] != "") & (df["iso"] != "")]
    df = df.drop_duplicates(subset=["name"], keep="first")

    return df


# ---------------------------------------------------------
# Normalize language codes
# ---------------------------------------------------------
def normalize_language_code(code):
    """
    Convert dataset language codes into codes accepted by
    GoogleTranslator and gTTS.

    Odia's ISO 639-1 code is 'or'. Some datasets may contain
    'OW', 'OD', 'ORIYA', or 'ODIA'; these are corrected here.
    """
    code = str(code).strip().lower()

    code_map = {
        "ow": "or",
        "od": "or",
        "odia": "or",
        "oriya": "or",
        "or": "or",
    }

    return code_map.get(code, code)


# ---------------------------------------------------------
# Translation function
# ---------------------------------------------------------
@lru_cache(maxsize=256)
def translate_text(text, target_language):
    translator = GoogleTranslator(
        source="auto",
        target=target_language
    )
    return translator.translate(text)


# ---------------------------------------------------------
# Audio generation
# ---------------------------------------------------------
def create_audio(text, language_code):
    speech = gTTS(text=text, lang=language_code)
    audio_path = BASE_DIR / "translated_audio.mp3"
    speech.save(str(audio_path))
    return audio_path


# ---------------------------------------------------------
# Load the language data
# ---------------------------------------------------------
try:
    languages_df = load_languages(str(LANGUAGE_FILE))
except FileNotFoundError:
    st.error(
        "Could not find language.csv. Place language.csv in the same "
        "folder as frontend.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Unable to read language.csv: {error}")
    st.stop()


# Build the language dropdown.
language_names = languages_df["name"].tolist()

if not language_names:
    st.error("No languages were found in language.csv.")
    st.stop()


# ---------------------------------------------------------
# Streamlit interface
# ---------------------------------------------------------
st.subheader("Enter text")

input_text = st.text_area(
    "Text to translate",
    placeholder="Type or paste your text here...",
    height=150
)

selected_language = st.selectbox(
    "Select target language",
    language_names
)

if st.button("Translate", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("Please enter some text before translating.")
    else:
        selected_row = languages_df[
            languages_df["name"] == selected_language
        ].iloc[0]

        original_code = selected_row["iso"]
        target_code = normalize_language_code(original_code)

        try:
            with st.spinner(f"Translating into {selected_language}..."):
                translated_text = translate_text(
                    input_text.strip(),
                    target_code
                )

            st.session_state["translated_text"] = translated_text
            st.session_state["translated_language"] = selected_language
            st.session_state["translated_code"] = target_code

            st.success("Translation completed successfully!")

        except Exception as error:
            st.error(
                f"Translation failed: {error}\n\n"
                f"Selected language: {selected_language}\n"
                f"Dataset code: {original_code}\n"
                f"Code sent to translator: {target_code}"
            )


# ---------------------------------------------------------
# Display translated text and audio option
# ---------------------------------------------------------
if st.session_state.get("translated_text"):
    st.subheader("Translated text")

    st.text_area(
        "Translation",
        value=st.session_state["translated_text"],
        height=150,
        disabled=True
    )

    st.download_button(
        label="Download translated text",
        data=st.session_state["translated_text"],
        file_name="translated_text.txt",
        mime="text/plain",
        use_container_width=True
    )

    st.subheader("Listen to translation")

    if st.button("Generate audio", use_container_width=True):
        try:
            with st.spinner("Generating audio..."):
                audio_file = create_audio(
                    st.session_state["translated_text"],
                    st.session_state["translated_code"]
                )

            audio_bytes = audio_file.read_bytes()
            st.audio(audio_bytes, format="audio/mp3")

            st.download_button(
                label="Download audio",
                data=audio_bytes,
                file_name="translated_audio.mp3",
                mime="audio/mpeg",
                use_container_width=True
            )

        except Exception as error:
            st.warning(
                "The text was translated, but audio generation failed. "
                f"This language may not be supported by gTTS. Details: {error}"
            )


st.caption(
    "Odia language-code correction: OW / OD / Oriya / Odia → or."
)
