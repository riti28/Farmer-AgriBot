"""
Step 7: Streamlit web app with 2 tabs.
  Tab 1 "Pest Risk Predictor": crop + stage + weather (typed in, or the latest real
        weather for any place from NASA POWER) -> risk, confidence, IPM tips
  Tab 2 "Ask IPM": ask a question, get an answer ONLY from our documents, with citations

Run: streamlit run app.py
"""
import importlib
import os
import pandas as pd
import streamlit as st
import rag
from fetch_data import SITES
from make_dataset import CROP_CALENDAR, STAGES
from predict import predict_risk, ipm_tips, latest_week_weather

DISCLAIMER = "Not an agronomist. No yield guarantee. Consult your local agricultural extension officer (e.g. KVK in India)."
RISK_COLOUR = {"Low": "green", "Medium": "orange", "High": "red"}


def pretty(value):
    """'finger_millet' -> 'Finger millet'"""
    return value.replace("_", " ").capitalize()


@st.cache_resource(show_spinner="Loading the search model (first visit only, ~30 s)...")
def load_search_model():
    """Load the embedding model ONCE for all users of the app."""
    return rag.get_embedder()


def notes_version():
    """Newest change time of the note files: it changes when the notes are edited or pushed."""
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
    return max(os.path.getmtime(os.path.join(folder, f)) for f in os.listdir(folder))


@st.cache_resource(show_spinner="Reading the IPM notes...")
def load_notes(version):
    """Re-read the notes (and their embeddings) whenever a note file changes.
    Without this, the running app would keep the old notes after a git push."""
    importlib.reload(rag)
    rag.EMBEDDER = load_search_model()
    rag.get_chunk_vectors()
    return version


st.set_page_config(page_title="AgriBot IPM", page_icon="🌾")
load_notes(notes_version())
rag.EMBEDDER = load_search_model()
st.title("Farmer AgriBot - Integrated Pest Management (IPM)")
st.warning(DISCLAIMER)
tab1, tab2 = st.tabs(["Pest Risk Predictor", "Ask IPM"])

# ---------------- Tab 1: Pest Risk Predictor ----------------
with tab1:
    col1, col2 = st.columns(2)
    crop = col1.selectbox("Crop", list(CROP_CALENDAR), format_func=pretty)
    stage = col2.selectbox("Crop stage", STAGES, index=1, format_func=pretty)

    # Weather: start with typical values, or load the latest real week for a location
    if "weather" not in st.session_state:
        st.session_state.weather = {"temp": 27.0, "humidity": 75.0, "rainfall": 20.0}

    with st.expander("Get the latest real weather for a location (NASA POWER, works worldwide)"):
        place = st.selectbox("Location", ["Type latitude/longitude"] + list(SITES))
        if place == "Type latitude/longitude":
            c1, c2 = st.columns(2)
            lat = c1.number_input("Latitude", -90.0, 90.0, 12.97)
            lon = c2.number_input("Longitude", -180.0, 180.0, 77.59)
        else:
            _, lat, lon = SITES[place]
        if st.button("Load last 7 days of weather"):
            try:
                st.session_state.weather = latest_week_weather(lat, lon)
                w = st.session_state.weather
                st.success(f"Loaded weather for {w['from']} to {w['to']} at ({lat:.2f}, {lon:.2f}).")
            except Exception as error:  # e.g. no internet
                st.error(f"Could not reach NASA POWER: {error}")

    w = st.session_state.weather
    c1, c2, c3 = st.columns(3)
    temp = c1.number_input("Average temperature (°C)", -10.0, 50.0, float(w["temp"]))
    humidity = c2.number_input("Relative humidity (%)", 0.0, 100.0, float(w["humidity"]))
    rainfall = c3.number_input("Rainfall in the week (mm)", 0.0, 1000.0, float(w["rainfall"]))

    if st.button("Predict risk", type="primary"):
        risk, confidence, probs = predict_risk(crop, stage, temp, humidity, rainfall)
        st.subheader(f"Pest risk: :{RISK_COLOUR[risk]}[{risk}]")
        st.write(f"Model confidence: **{confidence:.0%}**")
        st.bar_chart(pd.Series(probs).reindex(["Low", "Medium", "High"]))

        st.subheader("IPM tips from our documents")
        for tip in ipm_tips(crop):
            st.markdown(f"**{tip['pest']}**: {tip['text']}  \n"
                        f"Source: [{tip['title']} - {tip['org']}]({tip['url']})")

# ---------------- Tab 2: Ask IPM (chat) ----------------
with tab2:
    if "chat" not in st.session_state:
        st.session_state.chat = []  # list of (role, text) pairs

    st.caption("Ask in everyday words, e.g. *worms eating my maize* or *my rice leaves are turning yellow*.")
    for role, text in st.session_state.chat:
        st.chat_message(role).markdown(text)

    # "Not in my documents" -> show 3 closest topics as buttons; a click asks that topic
    clicked = None
    for i, topic in enumerate(st.session_state.get("suggestions", [])):
        if st.button(f"🔎 {topic}", key=f"suggestion_{len(st.session_state.chat)}_{i}"):
            clicked = topic

    question = st.chat_input("Ask a pest management question") or clicked
    if question:
        result = rag.answer(question)
        text = result["text"]
        if result["sources"]:
            text += "\n\n**Sources:**  \n" + "  \n".join(  # "  \n" = line break in Markdown
                f"[{n}] [{s['title']}]({s['url']}) - {s['org']} (score {s['score']})"
                for n, s in enumerate(result["sources"], start=1))
        if result["suggestions"]:
            text += "\n\n**Topics we cover that may help** (click one below):"
        st.session_state.suggestions = result["suggestions"]
        st.session_state.chat += [("user", question), ("assistant", text)]
        st.rerun()
