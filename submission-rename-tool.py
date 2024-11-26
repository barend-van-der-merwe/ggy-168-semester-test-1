import streamlit as st
import cv2 as cv
import pandas as pd
import numpy as np
from PIL import Image
import io

gc = st.file_uploader("Select the grades CSV file", type = "csv")
image = st.file_uploader("Select the student submission", type = ["png", "jpg"])
if image and gc is not None:
    img_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv.imdecode(img_bytes, 1)
    snumber = st.text_input("Student Number", autocomplete=None, help=None)
    st.image(img)
    if st.button("Load Details"):
        df = pd.read_csv(gc)
        row_index = df.index[df["Username"] == snumber].tolist()
        surname = df.iloc[row_index, 0].values[0]
        first = df.iloc[row_index, 1].values[0]
        sname = st.text_input("Surname", value=surname)
        fname = st.text_input("Initials", value=first)
        filename = f"{surname}-{first}-{snumber}.png"
        final_img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        final_img_pil = Image.fromarray(final_img_rgb)
        buffer = io.BytesIO()
        final_img_pil.save(buffer, format="PNG")
        st.download_button(label=f"Download {filename}", data=buffer, file_name=filename, mime="image/png")
