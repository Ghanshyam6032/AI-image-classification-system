import streamlit as st
import requests
from PIL import Image
import pandas as pd

API_URL = "https://ai-image-classification-system-j3uz.onrender.com/predict"

st.set_page_config(
    page_title="AI Image Classification",
    page_icon="🖼️",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("🤖 Model Info")
    st.info("""
    Model: MobileNetV2
    
    Classes:
    - Attire
    - Decoration & Signage
    - Food
    - Misc
    
    Accuracy: 73.18%
    Dataset Size: 5,983 Images
    """)

# Header
st.markdown("""
<h1 style='text-align:center;'>
🖼️ AI Image Classification System
</h1>
<p style='text-align:center;'>
Upload an image and let AI classify it instantly
</p>
""", unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    col1, col2 = st.columns([1,1])

    with col1:

        img = Image.open(uploaded_file)

        st.image(
            img,
            caption="Uploaded Image",
            use_container_width=True
        )

        st.markdown("### 📋 Image Details")

        st.write(f"Width : {img.width}px")
        st.write(f"Height : {img.height}px")
        st.write(f"Format : {img.format}")

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
                    files=files
                )

                if response.status_code == 200:

                    result = response.json()

                    st.success("✅ Prediction Complete")

                    st.markdown("## 🎯 Prediction")

                    st.metric(
                        label="Class",
                        value=result["prediction"]
                    )

                    st.metric(
                        label="Confidence",
                        value=f"{result['confidence']}%"
                    )

                    st.progress(
                        min(result["confidence"]/100,1.0)
                    )

                    st.markdown("### 📊 Class Probabilities")

                    df = pd.DataFrame(
                        result["all_predictions"].items(),
                        columns=[
                            "Class",
                            "Probability"
                        ]
                    )

                    df = df.sort_values(
                        by="Probability",
                        ascending=False
                    )

                    st.dataframe(
                        df,
                        use_container_width=True
                    )

                    st.bar_chart(
                        df.set_index("Class")
                    )

                    # History
                    if "history" not in st.session_state:
                        st.session_state.history = []

                    st.session_state.history.append({
                        "Image": uploaded_file.name,
                        "Prediction": result["prediction"],
                        "Confidence": result["confidence"]
                    })

                else:

                    st.error(
                        f"Backend Error ({response.status_code})"
                    )

            except Exception as e:

                st.error(
                    f"Connection Error: {e}"
                )

# History Section
if "history" in st.session_state and len(st.session_state.history) > 0:

    st.markdown("---")

    st.subheader("📜 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

st.markdown("---")

st.caption(
    "🚀 Powered by MobileNetV2 + FastAPI + Streamlit"
)
