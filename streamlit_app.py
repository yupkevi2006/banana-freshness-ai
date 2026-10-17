
import os
import tempfile
import streamlit as st
from inference import run_inference

st.set_page_config(
    page_title="Banana Freshness AI",
    page_icon="🍌",
    layout="wide"
)

st.title("🍌 Banana Freshness AI")
st.write(
    "Upload a banana image and enter the sensor readings "
    "to estimate freshness and spoilage timelines."
)

# Sidebar: sensor inputs
st.sidebar.header("Sensor Readings")

#day_num = st.sidebar.number_input(
   # "Day number",
   # min_value=1,
    #max_value=7,
    #value=1,
    ##step=1
#)

methane_left = st.sidebar.number_input(
    "Methane — Left (ppm)",
    min_value=0.0,
    value=15.0,
    step=0.5
)

methane_right = st.sidebar.number_input(
    "Methane — Right (ppm)",
    min_value=0.0,
    value=16.0,
    step=0.5
)

temp1 = st.sidebar.number_input(
    "Temperature 1 (°C)",
    value=29.5,
    step=0.5
)

temp2 = st.sidebar.number_input(
    "Temperature 2 (°C)",
    value=31.5,
    step=0.5
)

st.subheader("Upload Banana Image")

uploaded_file = st.file_uploader(
    "Choose a banana image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded banana image",
        width=350
    )

    if st.button("Predict Freshness", type="primary"):

        temp_path = None

        try:
            # Save the uploaded image temporarily
            suffix = os.path.splitext(uploaded_file.name)[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            with st.spinner("Analyzing banana freshness..."):
                results = run_inference(
                    image_path=temp_path,
                    methane_ppm_left=methane_left,
                    methane_ppm_right=methane_right,
                    temp1=temp1,
                    temp2=temp2,
                    day_num=day_num
                )

            st.success("Prediction completed!")

            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Freshness Score",
                    f"{results['freshness_score']:.1f}%"
                )

                st.progress(
                    int(results["freshness_score"])
                )

            with col2:
                st.metric(
                    "Estimated 95% Spoilage Timeline",
                    f"{results['spoil_95_days']:.2f} days"
                )

            st.subheader("Estimated Spoilage Timelines")

            a, b, c = st.columns(3)

            a.metric(
                "50% Spoilage",
                f"{results['spoil_50_days']:.2f} days"
            )

            b.metric(
                "75% Spoilage",
                f"{results['spoil_75_days']:.2f} days"
            )

            c.metric(
                "90% Spoilage",
                f"{results['spoil_90_days']:.2f} days"
            )

            st.caption(
                "These are model-generated estimates, not "
                "validated guarantees of actual shelf life."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info(
                "Check that model_bundle.pkl, "
                "model_architecture.py, and all required "
                "dependencies are available."
            )

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

else:
    st.info("Upload an image to begin.")
