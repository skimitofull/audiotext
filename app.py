import os
import math
import tempfile
import subprocess
from pathlib import Path
import streamlit as st
import whisper

st.set_page_config(
    page_title="Transcriptor Whisper - 8h",
    layout="wide"
)

st.title("🎙️ Transcriptor de Audio con Whisper")
st.write("Sube un archivo de audio (hasta 8 h). La aplicación lo dividirá en fragmentos, los transcribirá y unirá el resultado.")

def get_audio_duration(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path
        ],
        capture_output=True,
        text=True
    )
    return float(result.stdout.strip())

def transcribe_large_file(model, audio_path: str, chunk_minutes: int = 30, language: str = None):
    total_sec = get_audio_duration(audio_path)
    chunk_sec = chunk_minutes * 60
    num_chunks = math.ceil(total_sec / chunk_sec)

    st.write(f"⏱️ Duración total: **{total_sec/3600:.2f} h** → {num_chunks} fragmentos de {chunk_minutes} min")
    progress = st.progress(0)

    full_text = ""
    for i in range(num_chunks):
        start = i * chunk_sec
        tmp_chunk = Path(tempfile.gettempdir()) / f"chunk_{i}.mp3"

        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-i", audio_path,
                "-ss", str(start), "-t", str(chunk_sec),
                "-acodec", "copy", str(tmp_chunk)
            ],
            capture_output=True
        )

        result = model.transcribe(
            str(tmp_chunk),
            language=language,
            verbose=False
        )
        full_text += result["text"].strip() + "\n"

        tmp_chunk.unlink(missing_ok=True)
        progress.progress((i + 1) / num_chunks)

    progress.empty()
    return full_text

@st.cache_resource(show_spinner="Descargando modelo...")
def load_model(size: str = "base"):
    return whisper.load_model(size)

with st.sidebar:
    st.header("⚙️ Configuración")
    model_size = st.selectbox(
        "Tamaño del modelo Whisper",
        ["tiny", "base", "small", "medium", "large"],
        index=1
    )
    language_opt = st.selectbox(
        "Idioma",
        ["auto", "es", "en", "fr", "de", "it", "pt"]
    )
    chunk_mins = st.slider(
        "Duración de fragmento (minutos)",
        5, 60, 30
    )

uploaded = st.file_uploader(
    "📤 Sube tu archivo de audio",
    type=["mp3", "wav", "m4a", "mp4", "mov", "avi"]
)

if uploaded:
    st.success(f"Archivo: **{uploaded.name}** ({uploaded.size/1024/1024:.1f} MB)")

    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        temp_path = tmp.name

    if st.button("🚀 Iniciar transcripción", type="primary"):
        model = load_model(model_size)
        lang_param = None if language_opt == "auto" else language_opt

        try:
            with st.spinner("Transcribiendo..."):
                texto = transcribe_large_file(
                    model,
                    temp_path,
                    chunk_minutes=chunk_mins,
                    language=lang_param
                )
            
            st.success("✅ Transcripción completada")
            st.subheader("📝 Transcripción completa")
            st.text_area("Resultado", texto, height=400)

            st.download_button(
                "💾 Descargar TXT",
                data=texto,
                file_name=f"transcripcion_{Path(uploaded.name).stem}.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            Path(temp_path).unlink(missing_ok=True)