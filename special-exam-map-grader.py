import streamlit as st
import cv2 as cv
import pandas as pd
import numpy as np
from PIL import Image
import io
import re

st.title("Special - Mapwork")

st.subheader("Select the grades files")
gc = st.file_uploader("Select the grades CSV file", type = "csv")


st.subheader("Select Student Submission")
image = st.file_uploader("Select the student submission", type = ["png", "jpg"])

if image is not None:
    img_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv.imdecode(img_bytes, 1)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
    parameters =  cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(img)
    ids = np.concatenate(ids, axis=0).tolist()
    WIDTH = 712
    HEIGHT = 972
    aruco_top_left = corners[ids.index(0)]
    aruco_top_right = corners[ids.index(1)]
    aruco_bottom_right = corners[ids.index(2)]
    aruco_bottom_left = corners[ids.index(3)]
    point1 = aruco_top_left[0][0]
    point2 = aruco_top_right[0][1]
    point3 = aruco_bottom_right[0][2]
    point4 = aruco_bottom_left[0][3]
    working_image = np.float32([[point1[0], point1[1]],
                                [point2[0], point2[1]],
                                [point3[0], point3[1]],
                                [point4[0], point4[1]]])
    working_target = np.float32([[0, 0],
                                 [WIDTH, 0],
                                 [WIDTH, HEIGHT],
                                 [0, HEIGHT]])
    transformation_matrix = cv.getPerspectiveTransform(working_image, working_target)
    warped_img = cv.warpPerspective(img_gray, transformation_matrix, (WIDTH, HEIGHT))
    warped_img = cv.rectangle(warped_img, (282,758), (309,790), (0, 0, 255), 1, 8, 0)
    details = warped_img[0:250, 0:712]
    q15 = warped_img[288:318, 0:712]
    q16 = warped_img[327:355, 0:715]
    q17 = warped_img[366:472, 0:712]
    q18 = warped_img[482:972, 0:715]

    global df
    df = pd.read_csv(gc)
    snumber_from_filename = re.findall(r"-u[0-9]*", image.name)
    snumber_from_filename = re.sub("-", '', snumber_from_filename[0])
    st.write(f'Student number: {snumber_from_filename}')

    global row_index
    row_index = df.index[df["Username"] == snumber_from_filename].tolist()
    surname = df.iloc[row_index, 0].values[0]
    first = df.iloc[row_index, 1].values[0]
    st.write(f'Surname: {surname}')
    st.write(f'First name: {first}')

    st.image(details)
    st.image(q15)
    st.checkbox("Bearing corect (137° - 141°)", key="chk1")

    st.image(q16)
    st.checkbox("Distance correct (646.78m - 670.78m)", key="chk2")

    st.image(q17)
    st.slider("Magnetic declination correct", min_value=0.0, max_value=3.0, step=0.5, key="slider1")

    st.image(q18)
    st.slider("Profile correct", min_value=0.0, max_value=5.0, step=0.5, key="slider2")


    global grade
    grade = 0

    

    if st.button("Grade"):
        grade += st.session_state.chk1
        grade += st.session_state.chk2
        grade += st.session_state.slider1
        grade += st.session_state.slider2

        final_img = cv.putText(img=warped_img, text=f'{grade}', org=(583,140),
                               fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=1, color=(0, 0, 255), thickness=2)
        final_img = cv.putText(img=final_img, text=f'{int(st.session_state.chk1)}', org=(33,312),
                               fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=0.5, color=(0, 0, 255), thickness=1)
        final_img = cv.putText(img=final_img, text=f'{int(st.session_state.chk2)}', org=(33,350),
                               fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=0.5, color=(0, 0, 255), thickness=1)
        final_img = cv.putText(img=final_img, text=f'{st.session_state.slider1}', org=(33,430),
                               fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=0.5, color=(0, 0, 255), thickness=1)
        final_img = cv.putText(img=final_img, text=f'{st.session_state.slider2}', org=(33,504),
                               fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=0.5, color=(0, 0, 255), thickness=1)
        

        st.image(final_img)
        filename = f"{surname}-{first}-{snumber_from_filename}.png"
        final_img_rgb = cv.cvtColor(final_img, cv.COLOR_BGR2RGB)
        final_img_pil = Image.fromarray(final_img)
        buffer = io.BytesIO()
        final_img_pil.save(buffer, format="PNG")
        st.download_button(label=f"Download {filename}", data=buffer, file_name=filename, mime="image/jpeg")
