import streamlit as st
from PIL import Image
import numpy as np
import os
from io import BytesIO
from api.transfer_style import transfer_style

# Set Streamlit page config
st.set_page_config(page_title="Style Transfer App", layout="centered")

# Title
st.title("🎨 Neural Style Transfer - Web App")
st.markdown("Upload a **content image** and a **style image**, and generate a stylized result using a pre-trained neural network.")

# Upload section
content_file = st.file_uploader("Upload your content image (.jpg only)", type=["jpg"])
style_file = st.file_uploader("Upload your style image (.jpg only)", type=["jpg"])

if content_file and style_file:
    # Save uploaded files to disk (temporary .jpgs)
    content_path = os.path.join("assets", "uploaded_content.jpg")
    style_path = os.path.join("assets", "uploaded_style.jpg")

    with open(content_path, "wb") as f:
        f.write(content_file.read())

    with open(style_path, "wb") as f:
        f.write(style_file.read())

    # Show input images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(content_path, caption="Content Image", use_column_width=True)
    with col2:
        st.image(style_path, caption="Style Image", use_column_width=True)

    st.markdown("---")
    st.write("✨ Click below to apply style transfer:")

    if st.button("Stylize Image"):
        with st.spinner("Processing... please wait ⏳"):
            model_path = "model/magenta_arbitrary-image-stylization-v1-256_2"
            stylized_img = transfer_style(content_path, style_path, model_path)

            # Convert and display
            output_pil = Image.fromarray((stylized_img * 255).astype(np.uint8))
            st.image(output_pil, caption="Stylized Image", use_column_width=True)

            # Download button
            buf = BytesIO()
            output_pil.save(buf, format="JPEG")
            st.download_button("📥 Download Stylized Image", buf.getvalue(), file_name="stylized_output.jpg", mime="image/jpeg")
