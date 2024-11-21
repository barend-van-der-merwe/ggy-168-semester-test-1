import streamlit as st
import cv2 as cv
import pandas as pd
import numpy as np
from PIL import Image
import io
import re

st.title("Normal Exam - Mapwork")


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
    warped_img = cv.rectangle(warped_img, (479,717), (514,752), (0, 0, 0), 1, 8, 0)
    details = warped_img[0:245, 0:712]
    q15 = warped_img[290:317, 0:712]
    table = warped_img[317:445, 0:712]
    mapwork = warped_img[445:972, 0:712]

    snumber_from_filename = re.findall(r"-u[0-9]*", image.name)
    snumber_from_filename = re.sub("-", '', snumber_from_filename[0])
    st.write(f'Student number: {snumber_from_filename}')
    global df
    df = pd.read_csv(gc)
    row_index = df.index[df["Username"] == snumber_from_filename].tolist()
    surname = df.iloc[row_index, 0].values[0]
    first = df.iloc[row_index, 1].values[0]
    
    st.image(details)
    st.write(f'Surname: {surname}')
    st.write(f'First name: {first}')

    st.image(q15)
    st.checkbox("Q15 Bearing correct (136°)", key = "chk1")

    st.image(table)

    col1, col2, col3 = st.columns(3)

    with col1:
        strip1_length = st.number_input("strip 1 length", format="%0.3f")
        strip2_length = st.number_input("strip 2 length", format="%0.3f")
        strip3_length = st.number_input("strip 3 length", format="%0.3f")
        strip4_length = st.number_input("strip 4 length", format="%0.3f")

    with col2:
        strip1_width = st.number_input("strip 1 width", format="%0.3f")
        strip2_width = st.number_input("strip 2 width", format="%0.3f")
        strip3_width = st.number_input("strip 3 width", format="%0.3f")
        strip4_width = st.number_input("strip 4 width", format="%0.3f")

    with col3:
        strip1_area = st.number_input("strip 1 area", format="%0.3f")
        strip2_area = st.number_input("strip 2 area", format="%0.3f")
        strip3_area = st.number_input("strip 3 area", format="%0.3f")
        strip4_area = st.number_input("strip 4 area", format="%0.3f")

    area = st.number_input("Calculated Area", format="%0.3f")
    st.slider("Units incorrect", min_value=-1, max_value=0, value=0, step=1, key="slider3")
    st.image(mapwork)
    st.slider("Q14) Point in the correct position", min_value=0.0, max_value=1.0, step=0.5, key="slider1")    
    st.slider("Working shown", min_value=0.0, max_value=1.0, step=0.5, key = "slider2")


    global grade
    grade = 0

    if st.button("Grade"):
        # q14
        grade += float(st.session_state.slider1)
        warped_img = cv.putText(img=warped_img, text=f"{float(st.session_state.slider1)}",
                                   org=(522, 742),
                                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                   fontScale=0.5, color=(0, 0, 255), thickness=1)
        # q15
        grade += st.session_state.chk1
        warped_img = cv.putText(img=warped_img, text=f"{int(st.session_state.chk1)}",
                                   org=(268,310),
                                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                   fontScale=0.5, color=(0, 0, 255), thickness=1)

        WIDTH_UPPER = 1.34
        WIDTH_LOWER = 0.94

        LST1_UPPER = 1.04
        LST1_LOWER = 0.64

        LST2_UPPER = 3.32
        LST2_LOWER = 2.92

        LST3_UPPER = 3.86
        LST3_LOWER = 3.46

        LST4_UPPER = 3.2
        LST4_LOWER = 2.8

        # check if width falls within range
        if strip1_width >= WIDTH_LOWER and strip1_width <= WIDTH_UPPER:
            grade += 0.5
            warped_img = cv.putText(img=warped_img, text='0.5',
                                   org=(471,357),
                                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                   fontScale=0.5, color=(0, 0, 255), thickness=1)
        else:
            warped_img = cv.putText(img=warped_img, text='0',
                                   org=(471,357),
                                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                   fontScale=0.5, color=(0, 0, 255), thickness=1)
        if strip2_width >= WIDTH_LOWER and strip2_width <= WIDTH_UPPER:
            grade += 0.5
            warped_img = cv.putText(img=warped_img, text='0.5',
                                    org=(471,376),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)
        else:
            warped_img = cv.putText(img=warped_img, text='0',
                                    org=(471,376),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)
        if strip3_width >= WIDTH_LOWER and strip3_width <= WIDTH_UPPER:
            grade += 0.5
            warped_img = cv.putText(img=warped_img, text='0.5',
                                    org=(471,394),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)
        else:
            warped_img = cv.putText(img=warped_img, text='0',
                                    org=(471,394),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)
        if strip4_width >= WIDTH_LOWER and strip4_width <= WIDTH_UPPER:
            grade += 0.5
            warped_img = cv.putText(img=warped_img, text='0.5',
                                    org=(471,415),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)
        else:
            warped_img = cv.putText(img=warped_img, text='0',
                                    org=(471,415),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)

        # check if length falls in range
        if strip1_length >= LST1_LOWER and strip1_length <= LST1_UPPER:
            grade += 0.5
            warped_img = cv.putText(img=warped_img, text='0.5',
                                    org=(330,357),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)
        else:
            warped_img = cv.putText(img=warped_img, text='0',
                                    org=(330,357),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)
        if strip2_length >= LST2_LOWER and strip2_length <= LST2_UPPER:
            grade += 0.5
            warped_img = cv.putText(img=warped_img, text='0.5',
                                    org=(330,376),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)
        else:
            warped_img = cv.putText(img=warped_img, text='0',
                                    org=(330,376),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)
        if strip3_length >= LST3_LOWER and strip3_length <= LST3_UPPER:
            grade += 0.5
            warped_img = cv.putText(img=warped_img, text='0.5',
                                    org=(330,394),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)
        else:
            warped_img = cv.putText(img=warped_img, text='0',
                                    org=(330,394),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)
        if strip4_length >= LST4_LOWER and strip4_length <= LST4_UPPER:
            grade += 0.5
            warped_img = cv.putText(img=warped_img, text='0.5',
                                    org=(330,415),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)
        else:
            warped_img = cv.putText(img=warped_img, text='0',
                                    org=(330,415),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)


        # check whether areas were calculated accurately
        if round(strip1_area,1) == round((strip1_length * strip1_width),1):
            grade += 0.5
            warped_img = cv.putText(img=warped_img, text='0.5',
                                    org=(665,357),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)
        else:
            warped_img = cv.putText(img=warped_img, text='0',
                                    org=(665,357),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)
        if round(strip2_area,1) == round((strip2_length * strip2_width),1):
            grade += 0.5
            warped_img = cv.putText(img=warped_img, text='0.5',
                                    org=(665,376),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)
        else:
            warped_img = cv.putText(img=warped_img, text='0',
                                    org=(665,376),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)
        if round(strip3_area,1) == round((strip3_length * strip3_width),1):
            grade += 0.5
            warped_img = cv.putText(img=warped_img, text='0.5',
                                    org=(665,394),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)
        else:
            warped_img = cv.putText(img=warped_img, text='0',
                                    org=(665,394),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)
        if round(strip4_area,1) == round((strip4_length * strip4_width),1):
            grade += 0.5
            warped_img = cv.putText(img=warped_img, text='0.5',
                                    org=(665,415),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)
        else:
            warped_img = cv.putText(img=warped_img, text='0',
                                    org=(665,415),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)

        # check if final area calculation is correct
        calc_area = strip1_area + strip2_area + strip3_area + strip4_area
        if round(calc_area, 1) == round(area, 1):
            grade += 1
            warped_img = cv.putText(img=warped_img, text='1',
                                    org=(670, 442),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)
        else:
            warped_img = cv.putText(img=warped_img, text='0',
                                    org=(670, 442),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0, 0, 255), thickness=1)

        # check if work is shown
        grade += float(st.session_state.slider2)
        warped_img = cv.putText(img=warped_img, text=f'{float(st.session_state.slider1)}',
                                org=(670, 642),
                                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, color=(0, 0, 255), thickness=1)
        
        #check if there is a penalty for incorrect units
        grade += int(st.session_state.slider3)
        if int(st.session_state.slider3) != 0:
            warped_img = cv.putText(img=warped_img, text=f'Units incorrect: {float(st.session_state.slider3)}',
                                org=(255,233),
                                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, color=(0, 0, 255), thickness=1)

        final_img = cv.putText(img=warped_img, text=f'{grade}', org=(583,140),
                               fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=1, color=(0, 0, 255), thickness=2)


        
            

        st.image(final_img)
        filename = f"{surname}-{first}-{snumber_from_filename}.png"
        final_img_rgb = cv.cvtColor(final_img, cv.COLOR_BGR2RGB)
        final_img_pil = Image.fromarray(final_img)
        buffer = io.BytesIO()
        final_img_pil.save(buffer, format="PNG")
        st.download_button(label=f"Download {filename}", data=buffer, file_name=filename, mime="image/jpeg")
