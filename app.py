import streamlit as st
import requests
from PIL import Image
import pandas as pd
import os

API_URL = os.getenv(
    "API_URL",
    "https://ai-image-classification-system-j3uz.onrender.com/predict"
)

CLASS_LABELS = {
    "Attire": "👕 Clothing",
    "Decorationandsignage": "🎈 Decoration & Signage",
    "Food": "🍔 Food",
    "misc": "📦 Miscellaneous"
}

st.set_page_config(
    page_title="AI Image Classification",
    page_icon="🖼️",
    layout="wide"
)

# Sidebar
with st.sidebar:

    st.title("🤖 AI Model Information")

    st.success("MobileNetV2")

    st.markdown("""
    ### Dataset Statistics

    📷 Images : 5,983

    🏷 Classes : 4

    ### Categories

    👕 Clothing

    🎈 Decoration & Signage

    🍔 Food

    📦 Miscellaneous
    """)

# Header
st.markdown("""
<h1 style='text-align:center;'>
🖼️ AI Image Classification System
</h1>

<p style='text-align:center; font-size:18px;'>
Upload an image and let AI classify it instantly
</p>
""", unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    col1, col2 = st.columns([1, 1])

    with col1:

        img = Image.open(uploaded_file)

        st.image(
            img,
            caption="Uploaded Image",
            use_container_width=True
        )

        st.markdown("### 📋 Image Details")

        st.write(f"📏 Width : {img.width}px")
        st.write(f"📐 Height : {img.height}px")
        st.write(f"🖼 Format : {img.format}")

    with col2:

        with st.spinner("🔍 Analyzing Image..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            try:

                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=60
                )

                if response.status_code == 200:

                    result = response.json()

                    prediction = result["prediction"]

                    pred_label = CLASS_LABELS.get(
                        prediction,
                        prediction
                    )

                    confidence = result["confidence"]

                    st.success("✅ Prediction Complete")

                    st.markdown("## 🎯 Prediction Result")

                    st.metric(
                        "Prediction",
                        pred_label
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence}%"
                    )

                    st.progress(
                        min(confidence / 100, 1.0)
                    )

                    if confidence >= 90:
                        st.success(
                            "Very High Confidence"
                        )
                    elif confidence >= 70:
                        st.info(
                            "Good Confidence"
                        )
                    else:
                        st.warning(
                            "Low Confidence"
                        )

                    st.markdown(
                        "### 📊 Class Probabilities"
                    )

                    df = pd.DataFrame(
                        result["all_predictions"].items(),
                        columns=[
                            "Class",
                            "Probability"
                        ]
                    )

                    df["Class"] = df["Class"].map(
                        CLASS_LABELS
                    )

                    df = df.sort_values(
                        by="Probability",
                        ascending=False
                    )

                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.bar_chart(
                        df.set_index("Class")
                    )

                    # Prediction History
                    if "history" not in st.session_state:
                        st.session_state.history = []

                    st.session_state.history.append(
                        {
                            "Image":
                            uploaded_file.name,

                            "Prediction":
                            pred_label,

                            "Confidence":
                            confidence
                        }
                    )

                else:

                    st.error(
                        f"Backend Error ({response.status_code})"
                    )

                    st.write(
                        response.text
                    )

            except Exception as e:

                st.error(
                    f"Connection Error: {e}"
                )

# History Section
if (
    "history" in st.session_state
    and len(st.session_state.history) > 0
):

    st.markdown("---")

    st.subheader(
        "📜 Prediction History"
    )

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

st.markdown("---")

st.markdown(
"""
<center>
🚀 Built with MobileNetV2, FastAPI,
Streamlit and Render
</center>
""",
unsafe_allow_html=True
)
