import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import random
import tempfile

model = YOLO("yolov8x-worldv2.pt")

st.title("Zero-Shot Object Detection using Yolo-World")

# User input for classes
custom_classes = st.text_input("Enter classes (comma-separated)", "smartphone, laptop, tablet, monitor, TV, speaker")
classes = [cls.strip() for cls in custom_classes.split(",")]
model.set_classes(classes)

# Generating Distinct Colors for each class
def generate_colors(num_colors):
    return [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(num_colors)]
    # return [(random.randint(0,255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_colors)]
class_colors = {cls: color for cls, color in zip(classes, generate_colors(len(classes)))}

# Annotating the Image
def annotate_image(image, results):
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls_id = int(box.cls[0])
        label = f"{results[0].names[cls_id]} {conf:.2f}"
        color = class_colors[results[0].names[cls_id]]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return image
# Image Upload and Processing

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image.", use_container_width=True)

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = model(image_cv)

    annotated_image = annotate_image(image_cv, results)

    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Detected Objects.", use_column_width=True)

    annotated_image_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

    st.download_button(
        label="Download Annotated Image",
        data=cv2.imencode('.jpg', cv2.cvtColor(np.array(annotated_image_pil), cv2.COLOR_RGB2BGR))[1].tobytes(),
        file_name="annotated_image.jpg",
        mime="image/jpeg"
    )

# Video Upload and Processing

uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    video_cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = annotate_image(frame, results)
        out.write(annotated_frame)
        stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    video_cap.release()
    out.release()

    with open(output_video_path, "rb") as video_file:
        st.download_button(
            label="Download Annotated Video",
            data=video_file,
            file_name="annotated_video.mp4",
            mime="video/mp4"
        )

