import streamlit as st
from model import model_yolo

st.title("Horse Project ")

st.write("Projeto de Machine Learning utilizando a IA YOLO para identificação e classificação de cavalos.")

file = st.file_uploader(
    label="Faça o upload da imagem do cavalo",
    type=["jpg", "jpeg", "png", "webp"],
    on_change=None
)

if file is not None:
    image_result = model_yolo(file)
    
    if isinstance(image_result, str):
        st.error(image_result)
    else:
        caption: str = image_result["label"]
        caption = caption.replace("_", " ").title()

        percentual = image_result["conf"] * 100


        st.image(image_result["imagem"], caption=f"{caption} - {percentual:.2f}%")

        if caption.lower() in "appaloosa":
            path = r"horses_infos/appaloosa.md"
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
            st.markdown(text, unsafe_allow_html=True)

        if caption.lower() in "cavalo belga":
            path = r"horses_infos/cavalo_belga.md"
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
            st.markdown(text, unsafe_allow_html=True)

        if caption.lower() in "puro sangue ingles":
            path = r"horses_infos/puro_sangue_ingles.md"
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
            st.markdown(text, unsafe_allow_html=True)

else:
    st.write(f"O arquivo não é uma imagem válida. {file}")