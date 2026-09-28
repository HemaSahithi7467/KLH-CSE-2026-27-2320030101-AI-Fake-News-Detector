import numpy as np
import cv2
import streamlit as st
import joblib
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


def preprocess_image(image):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Upscale the image 2x - helps OCR read smaller/stylized text more clearly
    height, width = gray.shape
    gray = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

    # Remove noise while preserving edges
    denoised = cv2.fastNlMeansDenoising(gray, h=30)

    # Increase contrast using thresholding
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_text_tesseract(image):
    processed = preprocess_image(image)
    text = pytesseract.image_to_string(processed)
    return text.strip()


def explain_prediction(text, top_n=6):
    feature_names = vectorizer.get_feature_names_out()
    tfidf_vector = vectorizer.transform([text])
    coefs = model.coef_[0]

    nonzero_indices = tfidf_vector.nonzero()[1]
    contributions = []
    for idx in nonzero_indices:
        word = feature_names[idx]
        weight = coefs[idx] * tfidf_vector[0, idx]
        contributions.append((word, weight))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions[:top_n]


st.title("📰 AI Fake News Detector")
st.write("Paste a news article below and find out if it's Real or Fake.")

user_input = st.text_area("Enter news article text:", height=200)
uploaded_image = st.file_uploader("Or upload an image of a news article:", type=["png", "jpg", "jpeg"])

if st.button("Analyze"):
    final_text = user_input

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Extracting text using Tesseract..."):
            final_text = extract_text_tesseract(image)

        st.write("**Extracted text:**")
        st.write(final_text)

    if final_text.strip() == "":
        st.warning("Please paste some text or upload an image first.")
    else:
        input_tfidf = vectorizer.transform([final_text])
        prediction = model.predict(input_tfidf)[0]

        probability = model.predict_proba(input_tfidf)[0]
        confidence = max(probability) * 100

        if prediction == 1:
            st.success(f"✅ Prediction: REAL News")
        else:
            st.error(f"❌ Prediction: FAKE News")

        st.write(f"Confidence: {confidence:.2f}%")