import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown
import cv2
import tempfile
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

st.markdown("""
<style>
header {visibility: hidden;}      /* hides the top-right Deploy/menu bar */
footer {visibility: hidden;}      /* hides the “Made with Streamlit” footer */
</style>
""", unsafe_allow_html=True)
# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="PulmoScan AI", layout="wide")

# -------------------------------
# Session State
# -------------------------------
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_trained_model():
    url = "https://drive.google.com/uc?id=11WzeDiH8Hnzl7uwEI5AsPG9KSQWkCNf1"
    gdown.download(url, "model.h5", quiet=False)
    return load_model("model.h5", compile=False)

model = load_trained_model()
class_names = ['COVID', 'Normal', 'Pneumonia']
IMG_SIZE = (224, 224)
DISPLAY_SIZE = (300, 300)

# -------------------------------
# Preprocess + Prediction + Grad-CAM + PDF
# -------------------------------
def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100
    return predicted_class, confidence, img_array, predictions[0]

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if isinstance(predictions, list):
            predictions = predictions[0]
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs*pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap,0)
    if tf.reduce_max(heatmap) != 0:
        heatmap /= tf.reduce_max(heatmap)
    return heatmap.numpy()
def analyze_gradcam(heatmap, threshold, predicted_class, confidence):
    total_pixels = heatmap.size
    affected_pixels = np.sum(heatmap > threshold)
    affected_percentage = (affected_pixels / total_pixels) * 100

    # -------------------------------
    # Severity
    # -------------------------------
    if affected_percentage < 10:
        severity = "minimal"
    elif affected_percentage < 30:
        severity = "moderate"
    else:
        severity = "extensive"

    # -------------------------------
    # Spread
    # -------------------------------
    if affected_percentage < 15:
        spread = "localized"
    else:
        spread = "diffuse"

    # -------------------------------
    # Confidence Interpretation
    # -------------------------------
    if confidence > 90:
        confidence_text = "high confidence"
    elif confidence > 70:
        confidence_text = "moderate confidence"
    else:
        confidence_text = "low confidence"

    # -------------------------------
    # Class-specific Explanation
    # -------------------------------
    if predicted_class == "COVID":
        condition_text = "patterns consistent with viral infection (COVID-19), often showing peripheral or bilateral opacities"
    elif predicted_class == "Pneumonia":
        condition_text = "findings suggest lung inflammation consistent with pneumonia, typically presenting as localized consolidations"
    else:
        condition_text = "no significant abnormal patterns detected, consistent with a normal chest X-ray"

    h, w = heatmap.shape

    # -------------------------------
    # 🫁 Left vs Right Lung
    # -------------------------------
    left_region = heatmap[:, :w//2]
    right_region = heatmap[:, w//2:]

    left_activation = np.sum(left_region > threshold)
    right_activation = np.sum(right_region > threshold)

    if predicted_class == "Normal":
        lung_text = "No significant abnormal lung involvement is observed."
    else:
        if abs(left_activation - right_activation) < 0.1 * (left_activation + right_activation):
            lung_text = "The abnormalities are bilaterally distributed across both lungs."
        elif left_activation > right_activation:
            lung_text = "The abnormalities are more prominent in the left lung region."
        else:
            lung_text = "The abnormalities are more prominent in the right lung region."

    # -------------------------------
    # 🫁 Upper vs Lower Lung
    # -------------------------------
    upper_region = heatmap[:h//2, :]
    lower_region = heatmap[h//2:, :]

    upper_activation = np.sum(upper_region > threshold)
    lower_activation = np.sum(lower_region > threshold)

    if predicted_class == "Normal":
        vertical_text = ""
    else:
        if abs(upper_activation - lower_activation) < 0.1 * (upper_activation + lower_activation):
            vertical_text = "The involvement is evenly distributed between upper and lower lung zones."
        elif upper_activation > lower_activation:
            vertical_text = "Greater involvement is observed in the upper lung regions."
        else:
            vertical_text = "Greater involvement is observed in the lower lung regions."

    # -------------------------------
    # 🔍 Pattern Detection
    # -------------------------------
    active_coords = np.argwhere(heatmap > threshold)

    if len(active_coords) > 0:
        spread_x = np.std(active_coords[:, 1])
        spread_y = np.std(active_coords[:, 0])

        if spread_x + spread_y < 40:
            pattern = "focal (localized cluster)"
        elif spread_x + spread_y < 80:
            pattern = "patchy distribution"
        else:
            pattern = "diffuse spread across lung fields"
    else:
        pattern = "no significant activation detected"

    # -------------------------------
    # 💡 Intensity Analysis
    # -------------------------------
    active_values = heatmap[heatmap > threshold]

    if len(active_values) > 0:
        avg_intensity = np.mean(active_values)

        if avg_intensity < 0.4:
            intensity_text = "The detected abnormalities show mild intensity."
        elif avg_intensity < 0.7:
            intensity_text = "The detected abnormalities show moderate intensity."
        else:
            intensity_text = "The detected abnormalities show high intensity, indicating strong model attention."
    else:
        intensity_text = ""

    # -------------------------------
    # 🧠 Final Combined Analysis
    # -------------------------------
    analysis_text = f"""
    The AI model identified {affected_percentage:.2f}% of the lung region as significant, indicating {severity} involvement with a {spread} pattern.

    The prediction corresponds to {predicted_class} with {confidence_text} ({confidence:.2f}%).

    The highlighted regions suggest {condition_text}.

    {lung_text}.{vertical_text}


    The activation pattern appears {pattern}.

    {intensity_text}.These regions represent the key areas influencing the model’s decision.
    """

    return analysis_text

def generate_pdf(img, overlay, filtered_overlay, predicted_class, confidence, analysis_text):
    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("AI Chest X-ray Diagnosis Report", styles['Title']))
    elements.append(Spacer(1,10))
    elements.append(Paragraph(f"Prediction: {predicted_class}", styles['Normal']))
    elements.append(Paragraph(f"Confidence: {confidence:.2f}%", styles['Normal']))
    elements.append(Spacer(1,20))

    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    temp_heatmap = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    temp_filtered = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name

    Image.fromarray(img).save(temp_img)
    Image.fromarray(overlay).save(temp_heatmap)
    Image.fromarray(filtered_overlay).save(temp_filtered)

    elements.append(Paragraph("Original Image", styles['Heading2']))
    elements.append(RLImage(temp_img, width=300,height=300))
    elements.append(Paragraph("Grad-CAM Heatmap", styles['Heading2']))
    elements.append(RLImage(temp_heatmap, width=300,height=300))
    elements.append(Paragraph("Affected Regions", styles['Heading2']))
    elements.append(RLImage(temp_filtered, width=300,height=300))
    elements.append(Spacer(1,10))
    elements.append(Paragraph("Analysis Summary", styles['Heading2']))
    elements.append(Paragraph(analysis_text, styles['Normal']))
    elements.append(Spacer(1,10))
    # elements.append(Paragraph("Note: AI-assisted diagnosis. Not a substitute for medical advice.", styles['Italic']))

    doc.build(elements)
    return pdf_path

# -------------------------------
# Animated & Professional Welcome Page
# -------------------------------
# -------------------------------
# Animated & Professional Welcome Page (Button Centered Fix)
# -------------------------------
# -------------------------------
# Animated & Professional Welcome Page (Perfectly Centered Button)
# -------------------------------
def show_welcome_page():
    st.markdown("""
    <style>
    body {
        margin:0;
        padding:0;
        height:100vh;
        background-image: url('https://media.giphy.com/media/3oKIPwoeGErMmaI43C/giphy.gif');
        background-size: cover;
        background-position: center;
        font-family: 'Arial', sans-serif;
    }
    .center-box {
        text-align:center;
        color:white;
        animation: fadeIn 2s ease-in-out;
        margin-top:50px;
    }
    h1 {
        font-size:70px;
        color:#00FF9C;
        animation: slideDown 1.5s ease-in-out;
    }
    p {
        font-size:30 px;
        animation: fadeIn 3s ease-in-out;
    }
    .features {
        font-size:18px;
        margin-top:20px;
        animation: fadeIn 4s ease-in-out;
    }
    .side-boxes {
        display:flex;
        justify-content: space-around;
        margin-top:50px;
    }
    .box {
        background-color: rgba(0,0,0,0.65);
        padding:30px;
        border-radius:20px;
        width:45%;
        color:white;
        animation: fadeIn 2s ease-in-out;
    }

    .box p {
        font-size: 19px;   /* increase from 18px to 20px or 22px */
        line-height: 1.6;  /* better spacing between lines */
        margin: 10px 0;
    }
    .step-box {
        background-color: rgba(255,255,255,0.1);
        padding:20px;
        border-radius:15px;
        margin:10px 0;
        font-size:18px;
    }
    /* Perfectly centered button */
    .center-button-container {
        display: flex;
        justify-content: center;  /* horizontal center */
        margin-top: 40px;         /* spacing from boxes */
        
    }
    div.stButton > button:first-child {
        background-color: #00FF9C;
        color:black;
        font-size:24px;
        padding:20px 60px;
        border-radius:15px;
        font-weight:bold;
        margin-left:388%;
    }
    div.stButton > button:hover {
        background-color:#00cc7a;
    }
    
    @keyframes fadeIn {0% {opacity:0;} 100% {opacity:1;}}
    @keyframes slideDown {0% {opacity:0; transform:translateY(-50px);} 100% {opacity:1; transform:translateY(0);} }
    </style>
    
    <div class="center-box">
        <h1> 🩺 PulmoScan AI</h1>
        <p style="font-size:23px; font-weight:300;">Your AI Assistant for COVID, Pneumonia & Normal X-rays</p>
        
    </div>

    <div class="side-boxes">
        <div class="box">
            <h2 style="color:#00FF9C;">How It Works</h2>
            <div class="step-box">1️⃣ Upload Chest X-ray Image</div>
            <div class="step-box">2️⃣ Analyze & Predict Results</div>
            <div class="step-box">3️⃣ Download PDF Report with Heatmap</div>
        </div>
        <div class="box">
            <h2 style="color:#00FF9C;">Why Choose Us?</h2>
            <p>✅ Explainable Predictions &nbsp;&nbsp; ✅Free & Accessible &nbsp;&nbsp;✅ Web-based & Easy to Use</p>
            <p>✅ Secure & Private &nbsp;&nbsp; ✅ Comprehensive PDF Reports</p>
            <p>✅ Quick & Accurate Results &nbsp;&nbsp; ✅ User-Friendly Interface</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------
    # Perfectly Centered Streamlit Button
    # -------------------------------
    st.markdown('<div class="center-button-container">', unsafe_allow_html=True)
    if st.button("Get Started"):
        st.session_state.page = "analyzer"
    st.markdown('</div>', unsafe_allow_html=True)


# -------------------------------
# Analyzer Page
# -------------------------------
def show_analyzer_page():
    with st.sidebar:
        st.header("⚙️ Settings")
        threshold = st.slider("Grad-CAM Sensitivity", 100, 255, 180)
        st.sidebar.markdown("""
        <p style="font-size:14px; color:#cccccc; font-style:italic; line-height:1.5;">
        Slide to adjust Grad-CAM sensitivity:<br>
        Higher values = focused regions<br>
        Lower values = broader coverage
        </p>
        """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>🩺 PulmoScan AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>COVID-19 | Pneumonia | Normal Detection with Explainable AI</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg","png","jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img.resize((300,300)), caption="Uploaded Image")
        if st.button("Analyze Image"):
            with st.spinner("Analyzing the X-ray..."):
                predicted_class, confidence, img_array, raw_preds = predict_image(img)
                color = "#00FF9C" if predicted_class=="Normal" else "#FF4B4B"
                st.markdown(f"<div style='background-color:#1e1e1e;padding:20px;border-radius:10px;text-align:center;'><h2 style='color:{color};'>{predicted_class}</h2><h4>Confidence: {confidence:.2f}%</h4></div>", unsafe_allow_html=True)
                st.progress(int(confidence))

                tab1, tab2 = st.tabs(["📊 Results","Visual Analysis"])
                with tab1: st.bar_chart({"COVID":raw_preds[0],"Normal":raw_preds[1],"Pneumonia":raw_preds[2]})

                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="block14_sepconv2_act")
                img_np = np.array(img.resize((224,224)))
                heatmap_resized = cv2.resize(heatmap,(224,224),interpolation=cv2.INTER_CUBIC)
                heatmap_resized = np.uint8(255*heatmap_resized)
                heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img_np,0.6,heatmap_colored,0.4,0)
                mask = np.zeros_like(heatmap_resized)
                mask[heatmap_resized>threshold]=255
                red_mask = np.zeros((224,224,3),dtype=np.uint8)
                red_mask[:,:,2]=mask
                filtered_overlay = cv2.addWeighted(img_np,1.0,red_mask,0.7,0)
                overlay = cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB)
                filtered_overlay = cv2.cvtColor(filtered_overlay,cv2.COLOR_BGR2RGB)
                analysis_text = analyze_gradcam(heatmap_resized, threshold, predicted_class, confidence)

                with tab2:
                    col1,col2,col3 = st.columns(3)
                    with col1: st.image(img.resize((224,224)), caption="Original", width="stretch")
                    with col2: st.image(overlay, caption="Full Heatmap", width="stretch")
                    with col3: st.image(filtered_overlay, caption="Affected Regions", width="stretch")
                    st.markdown("### Heatmap Legend\n- 🔵 Blue → Low importance  \n- 🟢 Green → Medium importance  \n- 🔴 Red → High importance  ")
                    st.markdown("### Analysis Summarys")
                    st.write(analysis_text)
                pdf_file = generate_pdf(img_np, overlay, filtered_overlay, predicted_class, confidence, analysis_text)
                with open(pdf_file,"rb") as f:
                    st.download_button("📄 Download Full Report (PDF)", f, file_name="AI_Xray_Report.pdf", mime="application/pdf")

# -------------------------------
# Main
# -------------------------------
if st.session_state.page=="welcome":
    show_welcome_page()
else:
    show_analyzer_page()