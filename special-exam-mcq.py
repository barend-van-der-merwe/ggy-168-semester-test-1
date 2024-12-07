import streamlit as st
import cv2 as cv
import pandas as pd
import numpy as np
from PIL import Image
import io
import re

st.title("Special Exam - Multiple Choice Questions")

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
    blur_img = cv.blur(warped_img, (5, 5))
    thresh = cv.threshold(blur_img, 200, 255, cv.THRESH_BINARY_INV)[1]
    # get responses and adjust grade
    global grade
    grade = 0

    # get answer for question 1
    mcq1 = []
    mcq1.append(thresh[319:328, 306:315].mean())
    mcq1.append(thresh[319:328, 344:353].mean())
    mcq1.append(thresh[319:328, 382:391].mean())
    mcq1.append(thresh[319:328, 420:429].mean())
    mcq1_max = max(mcq1)
    mcq1_index = mcq1.index(mcq1_max)
    if mcq1_index == 1:
        grade += 1

    # get answer for question 2
    mcq2 = []
    mcq2.append(thresh[357:366, 306:315].mean())
    mcq2.append(thresh[357:366, 344:353].mean())
    mcq2.append(thresh[357:366, 382:391].mean())
    mcq2.append(thresh[357:366, 420:429].mean())
    mcq2_max = max(mcq2)
    mcq2_index = mcq2.index(mcq2_max)
    if mcq2_index == 0:
        grade += 1

    # get answer for question 3
    mcq3 = []
    mcq3.append(thresh[396:405, 306:315].mean())
    mcq3.append(thresh[396:405, 344:353].mean())
    mcq3.append(thresh[396:405, 382:391].mean())
    mcq3.append(thresh[396:405, 420:429].mean())
    mcq3_max = max(mcq3)
    mcq3_index = mcq3.index(mcq3_max)
    if mcq3_index == 2:
        grade += 1

    # get answer for question 4
    mcq4 = []
    mcq4.append(thresh[435:444, 306:315].mean())
    mcq4.append(thresh[435:444, 344:353].mean())
    mcq4.append(thresh[435:444, 382:391].mean())
    mcq4.append(thresh[435:444, 420:429].mean())
    mcq4_max = max(mcq4)
    mcq4_index = mcq4.index(mcq4_max)
    if mcq4_index == 1:
        grade += 1

    # get answer for question 5
    mcq5 = []
    mcq5.append(thresh[474:483, 306:315].mean())
    mcq5.append(thresh[474:483, 344:353].mean())
    mcq5.append(thresh[474:483, 382:391].mean())
    mcq5.append(thresh[474:483, 420:429].mean())
    mcq5_max = max(mcq5)
    mcq5_index = mcq5.index(mcq5_max)
    if mcq5_index == 2:
        grade += 1

    # get answer for question 6
    mcq6 = []
    mcq6.append(thresh[513:522, 306:315].mean())
    mcq6.append(thresh[513:522, 344:353].mean())
    mcq6.append(thresh[513:522, 382:391].mean())
    mcq6.append(thresh[513:522, 420:429].mean())
    mcq6_max = max(mcq6)
    mcq6_index = mcq6.index(mcq6_max)
    if mcq6_index == 0:
        grade += 1

    # get answer for question 7
    mcq7 = []
    mcq7.append(thresh[552:561, 306:315].mean())
    mcq7.append(thresh[552:561, 344:353].mean())
    mcq7.append(thresh[552:561, 382:391].mean())
    mcq7.append(thresh[552:561, 420:429].mean())
    mcq7_max = max(mcq7)
    mcq7_index = mcq7.index(mcq7_max)
    if mcq7_index == 0:
        grade += 1

    # get answer for question 8
    mcq8 = []
    mcq8.append(thresh[590:599, 306:315].mean())
    mcq8.append(thresh[590:599, 344:353].mean())
    mcq8.append(thresh[590:599, 382:391].mean())
    mcq8.append(thresh[590:599, 420:429].mean())
    mcq8_max = max(mcq8)
    mcq8_index = mcq8.index(mcq8_max)
    if mcq8_index == 2:
        grade += 1

    # get answer for question 9
    mcq9 = []
    mcq9.append(thresh[629:638, 306:315].mean())
    mcq9.append(thresh[629:638, 344:353].mean())
    mcq9.append(thresh[629:638, 382:391].mean())
    mcq9.append(thresh[629:638, 420:429].mean())
    mcq9_max = max(mcq9)
    mcq9_index = mcq9.index(mcq9_max)
    if mcq9_index == 2:
        grade += 1

    # get answer for question 10
    mcq10 = []
    mcq10.append(thresh[668:677, 306:315].mean())
    mcq10.append(thresh[668:677, 344:353].mean())
    mcq10.append(thresh[668:677, 382:391].mean())
    mcq10.append(thresh[668:677, 420:429].mean())
    mcq10_max = max(mcq10)
    mcq10_index = mcq10.index(mcq10_max)
    if mcq10_index == 2:
        grade += 1

    details = warped_img[0:250, 0:712]
    #mcq = warped_img[249:760, 0:712]
    mcq = warped_img[306:689,295:443]
    #memo = cv.rectangle(warped_img, (310, 320), (319, 329), (0, 0, 0), 1, 8, 0) #1a
    #memo = cv.rectangle(warped_img, (348, 320), (357, 329), (0, 0, 0), 1, 8, 0) #1b
    #memo = cv.rectangle(warped_img, (386, 320), (395, 329), (0, 0, 0), 1, 8, 0) #1c
    #memo = cv.rectangle(warped_img, (424, 320), (433, 329), (0, 0, 0), 1, 8, 0) #1d
    #memo = cv.circle(warped_img, (311, 324), 10, (0,0,0), 2) #1a
    memo = cv.circle(warped_img, (349, 324), 10, (0,0,0), 2) #1b
    #memo = cv.circle(warped_img, (387, 324), 10, (0,0,0), 2) #1c
    #memo = cv.circle(warped_img, (425, 324), 10, (0,0,0), 2) #1d

    #memo = cv.rectangle(warped_img, (310,359), (319, 368), (0, 0, 0), 1, 8, 0)  #2a
    #memo = cv.rectangle(warped_img, (348, 359), (357, 368), (0, 0, 0), 1, 8, 0)  # 2b
    #memo = cv.rectangle(warped_img, (386, 359), (395, 368), (0, 0, 0), 1, 8, 0)  # 2c
    #memo = cv.rectangle(warped_img, (424, 359), (433, 368), (0, 0, 0), 1, 8, 0)  # 2d
    memo = cv.circle(warped_img, (311, 362), 10, (0,0,0), 2) #2a
    #memo = cv.circle(warped_img, (349, 362), 10, (0,0,0), 2) #2b
    #memo = cv.circle(warped_img, (387, 362), 10, (0,0,0), 2) #2c
    #memo = cv.circle(warped_img, (425, 363), 10, (0,0,0), 2) #2d

    #memo = cv.rectangle(warped_img, (310, 398), (319, 407), (0, 0, 0), 1, 8, 0)  # 3a
    #memo = cv.rectangle(warped_img, (348, 398), (357, 407), (0, 0, 0), 1, 8, 0)  # 3b
    #memo = cv.rectangle(warped_img, (386, 398), (395, 407), (0, 0, 0), 1, 8, 0)  # 3c
    #memo = cv.rectangle(warped_img, (424, 398), (433, 407), (0, 0, 0), 1, 8, 0)  # 3d
    #memo = cv.circle(warped_img, (311, 401), 10, (0,0,0), 2) #3a
    #memo = cv.circle(warped_img, (349, 401), 10, (0,0,0), 2) #3b
    memo = cv.circle(warped_img, (387, 401), 10, (0,0,0), 2) #3c
    #memo = cv.circle(warped_img, (425, 401), 10, (0,0,0), 2) #3d


    #memo = cv.rectangle(warped_img, (310, 437), (319, 446), (0, 0, 0), 1, 8, 0)  # 4a
    #memo = cv.rectangle(warped_img, (348, 437), (357, 446), (0, 0, 0), 1, 8, 0)  # 4b
    #memo = cv.rectangle(warped_img, (386, 437), (395, 446), (0, 0, 0), 1, 8, 0)  # 4c
    #memo = cv.rectangle(warped_img, (424, 437), (433, 446), (0, 0, 0), 1, 8, 0)  # 4d
    #memo = cv.circle(warped_img, (311, 440), 10, (0,0,0), 2) #4a
    memo = cv.circle(warped_img, (349, 440), 10, (0,0,0), 2) #4b
    #memo = cv.circle(warped_img, (387, 440), 10, (0,0,0), 2) #4c
    #memo = cv.circle(warped_img, (425, 440), 10, (0,0,0), 2) #4d

    #memo = cv.rectangle(warped_img, (310, 476), (319, 485), (0, 0, 0), 1, 8, 0)  # 5a
    #memo = cv.rectangle(warped_img, (348, 476), (357, 485), (0, 0, 0), 1, 8, 0)  # 5b
    #memo = cv.rectangle(warped_img, (386, 476), (395, 485), (0, 0, 0), 1, 8, 0)  # 5c
    #memo = cv.rectangle(warped_img, (424, 476), (433, 485), (0, 0, 0), 1, 8, 0)  # 5d
    #memo = cv.circle(warped_img, (311, 478), 10, (0,0,0), 2) #5a
    #memo = cv.circle(warped_img, (349, 478), 10, (0,0,0), 2) #5b
    memo = cv.circle(warped_img, (387, 478), 10, (0,0,0), 2) #5c
    #memo = cv.circle(warped_img, (425, 478), 10, (0,0,0), 2) #5d

    #memo = cv.rectangle(warped_img, (310, 515), (319, 524), (0, 0, 0), 1, 8, 0)  # 6a
    #memo = cv.rectangle(warped_img, (348, 515), (357, 524), (0, 0, 0), 1, 8, 0)  # 6b
    #memo = cv.rectangle(warped_img, (386, 515), (395, 524), (0, 0, 0), 1, 8, 0)  # 6c
    #memo = cv.rectangle(warped_img, (424, 515), (433, 525), (0, 0, 0), 1, 8, 0)  # 6d
    memo = cv.circle(warped_img, (311, 517), 10, (0,0,0), 2) #6a
    #memo = cv.circle(warped_img, (349, 517), 10, (0,0,0), 2) #6b
    #memo = cv.circle(warped_img, (387, 517), 10, (0,0,0), 2) #6c
    #memo = cv.circle(warped_img, (425, 517), 10, (0,0,0), 2) #6d

    #memo = cv.rectangle(warped_img, (310, 554), (319, 563), (0, 0, 0), 1, 8, 0)  # 7a
    #memo = cv.rectangle(warped_img, (348, 554), (357, 563), (0, 0, 0), 1, 8, 0)  # 7b
    #memo = cv.rectangle(warped_img, (386, 554), (395, 563), (0, 0, 0), 1, 8, 0)  # 7c
    #memo = cv.rectangle(warped_img, (424, 554), (433, 563), (0, 0, 0), 1, 8, 0)  # 7d
    memo = cv.circle(warped_img, (311, 556), 10, (0,0,0), 2) #7a
    #memo = cv.circle(warped_img, (349, 556), 10, (0,0,0), 2) #7b
    #memo = cv.circle(warped_img, (387, 556), 10, (0,0,0), 2) #7c
    #memo = cv.circle(warped_img, (425, 556), 10, (0,0,0), 2) #7d

    #memo = cv.rectangle(warped_img, (310, 592), (319, 601), (0, 0, 0), 1, 8, 0)  # 8a
    #memo = cv.rectangle(warped_img, (348, 592), (357, 601), (0, 0, 0), 1, 8, 0)  # 8b
    #memo = cv.rectangle(warped_img, (386, 592), (395, 601), (0, 0, 0), 1, 8, 0)  # 8c
    #memo = cv.rectangle(warped_img, (424, 592), (433, 601), (0, 0, 0), 1, 8, 0)  # 8d
    #memo = cv.circle(warped_img, (311, 595), 10, (0,0,0), 2) #8a
    #memo = cv.circle(warped_img, (349, 595), 10, (0,0,0), 2) #8b
    memo = cv.circle(warped_img, (387, 595), 10, (0,0,0), 2) #8c
    #memo = cv.circle(warped_img, (425, 595), 10, (0,0,0), 2) #8d

    #memo = cv.rectangle(warped_img, (310, 631), (319, 640), (0, 0, 0), 1, 8, 0)  # 9a
    #memo = cv.rectangle(warped_img, (348, 631), (357, 640), (0, 0, 0), 1, 8, 0)  # 9b
    #memo = cv.rectangle(warped_img, (386, 631), (395, 640), (0, 0, 0), 1, 8, 0)  # 9c
    #memo = cv.rectangle(warped_img, (424, 631), (433, 640), (0, 0, 0), 1, 8, 0)  # 9d
    #memo = cv.circle(warped_img, (311, 633), 10, (0,0,0), 2) #9a
    #memo = cv.circle(warped_img, (349, 633), 10, (0,0,0), 2) #9b
    memo = cv.circle(warped_img, (387, 633), 10, (0,0,0), 2) #9c
    #memo = cv.circle(warped_img, (425, 633), 10, (0,0,0), 2) #9d

    #memo = cv.rectangle(warped_img, (310, 670), (319, 679), (0, 0, 0), 1, 8, 0)  # 10a
    #memo = cv.rectangle(warped_img, (348, 670), (357, 679), (0, 0, 0), 1, 8, 0)  # 10b
    #memo = cv.rectangle(warped_img, (386, 670), (395, 679), (0, 0, 0), 1, 8, 0)  # 10c
    #memo = cv.rectangle(warped_img, (424, 670), (433, 679), (0, 0, 0), 1, 8, 0)  # 10d
    #memo = cv.circle(warped_img, (311, 672), 10, (0,0,0), 2) #10a
    #memo = cv.circle(warped_img, (349, 672), 10, (0,0,0), 2) #10b
    memo = cv.circle(warped_img, (387, 672), 10, (0,0,0), 2) #10c
    #memo = cv.circle(warped_img, (425, 672), 10, (0,0,0), 2) #10d

    st.image(details)
    snumber_from_filename = re.findall(r"-u[0-9]*", image.name)
    snumber_from_filename = re.sub("-", '', snumber_from_filename[0])
    st.write(f'Student number: {snumber_from_filename}')

    global df
    df = pd.read_csv(gc)
    global row_index
    row_index = df.index[df["Username"] == snumber_from_filename].tolist()
    surname = df.iloc[row_index, 0].values[0]
    first = df.iloc[row_index, 1].values[0]
    st.write(f'Surname: {surname}')
    st.write(f'First name: {first}')

    st.image(memo)
    st.write(f'GRADE: {grade}')
    st.slider("Mark Adjustment", min_value = -10, max_value=10, step=1, value = 0, key="mcq_adjust")

    if st.button("Grade"):
        grade += st.session_state.mcq_adjust
        df.iloc[row_index, 6] = grade
        grades = df.to_csv(index=False)


        final_img = cv.putText(img=memo, text=f'{grade}', org=(583,140),
                               fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=1, color=(0, 0, 255), thickness=2)
        st.image(final_img)

        filename = f"{surname}-{first}-{snumber_from_filename}.png"
        final_img_rgb = cv.cvtColor(final_img, cv.COLOR_BGR2RGB)
        final_img_pil = Image.fromarray(final_img)
        buffer = io.BytesIO()
        final_img_pil.save(buffer, format="PNG")
        st.download_button(label=f"Download {filename}", data=buffer, file_name=filename, mime="image/jpeg")
        st.download_button(label=f"Grade Center", data=grades, mime="text/csv")