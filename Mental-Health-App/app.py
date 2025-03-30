import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pyttsx3
from io import BytesIO
from fpdf import FPDF
from sklearn.preprocessing import StandardScaler
from gtts import gTTS
import os
import base64
import tempfile

# Load the trained model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_bytes = mp3_fp.getvalue()
        base64_audio = base64.b64encode(mp3_bytes).decode()
        audio_html = f"""
            <audio controls>
                <source src="data:audio/mpeg;base64,{base64_audio}" type="audio/mpeg">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")

def preprocess_input(data):
    binary_mappings = {
        'Gender': {'Male': 0, 'Female': 1},
        'self_employed': {'No': 0, 'Yes': 1},
        'family_history': {'No': 0, 'Yes': 1},
        'remote_work': {'No': 0, 'Yes': 1},
        'tech_company': {'No': 0, 'Yes': 1},
        'obs_consequence': {'No': 0, 'Yes': 1}
    }
    ordinal_mappings = {
        'work_interfere': {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3},
        'leave': {'Very easy': 0, 'Somewhat easy': 1, "Don't know": 2, 'Somewhat difficult': 3, 'Very difficult': 4},
        'mental_health_consequence': {'No': 0, 'Maybe': 1, 'Yes': 2},
        'phys_health_consequence': {'No': 0, 'Maybe': 1, 'Yes': 2},
        'mental_health_interview': {'No': 0, 'Maybe': 1, 'Yes': 2},
        'phys_health_interview': {'No': 0, 'Maybe': 1, 'Yes': 2}
    }
    one_hot_columns = ['no_employees', 'benefits', 'care_options', 'wellness_program',
                       'seek_help', 'anonymity', 'coworkers', 'supervisor', 'mental_vs_physical']
    
    # Apply Binary & Ordinal Encoding
    for col, mapping in binary_mappings.items():
        data[col] = mapping.get(data.get(col, "Unknown"), 0)
    for col, mapping in ordinal_mappings.items():
        data[col] = mapping.get(data.get(col, "Unknown"), 0)
    
    df_input = pd.DataFrame([data])

    # Ensure all one-hot encoded columns exist before encoding
    for col in one_hot_columns:
        if col not in df_input:
            df_input[col] = "Unknown"  # Default value if not provided

    # One-Hot Encoding
    df_input = pd.get_dummies(df_input, columns=one_hot_columns, drop_first=True)

    # Ensure feature alignment with the scaler
    expected_columns = scaler.feature_names_in_
    for col in expected_columns:
        if col not in df_input:
            df_input[col] = 0  # Add missing columns with default value

    # Reorder columns to match the trained model
    df_input = df_input[expected_columns]

    # Standardize input
    input_scaled = scaler.transform(df_input)
    return input_scaled


# Function to generate a downloadable PDF report
def generate_report(name, email, prediction, recommendations):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, "Mental Health Assessment Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Name: {name}", ln=True)
    pdf.cell(200, 10, f"Email: {email}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(200, 10, "Prediction Result:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, prediction)
    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(200, 10, "Recommendations:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, recommendations)
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output



# Streamlit UI
st.set_page_config(page_title="Mental Health Risk Prediction", layout="wide")

st.markdown(
    """
    <style>
        /* Background Styling */
        body {
            background: linear-gradient(to right, #a8e6cf, #dcedc1);
        }
        
        /* Title and Subheading Styling */
        .stTitle, .stSubheader {
            text-align: center;
        }

        /* Center all Text Inputs, Radio Buttons & Buttons */
        div[data-testid="stTextInput"] {
            margin: auto;
            display: flex;
            justify-content: center;
        }
        
        div[data-testid="stRadio"] {
            display: flex;
            justify-content: center;
        }

        /* Button Styling */
        div.stButton > button {
            background-color: #007BFF;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            padding: 10px;
            width: 200px;
            display: flex;
            margin: auto;
            justify-content: center;
        }

        div.stButton > button:hover {
            background-color: #0056b3;
        }

        /* Align Download Button */
        div[data-testid="stDownloadButton"] {
            display: flex;
            justify-content: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Mental Health Risk Prediction & Support System")
st.markdown("---")

if "user_name" not in st.session_state or "user_email" not in st.session_state:
    st.subheader("👤 Enter Your Information")
    name = st.text_input("Enter your name:")
    email = st.text_input("Enter your email:")
    if st.button("Proceed"):
        if name and email:
            st.session_state["user_name"] = name
            st.session_state["user_email"] = email
            st.session_state["question_index"] = 0
            st.session_state["responses"] = {}
            st.rerun()
        else:
            st.warning("Please enter both your name and email to continue.")
else:
    st.subheader("📋 Assessment")
    questions = [
    ("What is your gender?", ["Male", "Female"]),
    ("Are you self-employed?", ["No", "Yes"]),
    ("Do you have a family history of mental illness?", ["No", "Yes"]),
    ("How often does work interfere with your mental health?", ["Never", "Rarely", "Sometimes", "Often"]),
    ("Company Size (in terms of employee count) ?", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]),
    ("Do you work remotely?", ["No", "Yes"]),
    ("Do you work in a Tech company?", ["No", "Yes"]),
    ("Does your employer provide mental health benefits ?", ["Yes", "No", "Don't know"]),
    ("Does your employer provide mental health care assistance?", ["Yes", "No", "Not sure"]),
    ("Does your company conduct wellness programs ?", ["Yes", "No", "Don't know"]),
    ("Does your company offer external resources to seek help?", ["Yes", "No", "Don't know"]),
    ("If Yes, is anonymity protected when seeking mental health care?", ["Yes", "No", "Don't know"]),
    ("How easy is it to take work leave for mental health?", ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"]),
    ("Do you think discussing mental health could have negative consequences?", ["No", "Maybe", "Yes"]),
    ("Do you think discussing physical health could have negative consequences?", ["No", "Maybe", "Yes"]),
    ("Would you discuss your mental health with coworkers?", ["Yes", "No", "Some of them"]),
    ("Would you discuss your mental health with a supervisor at work?", ["Yes", "No", "Some of them"]),
    ("Would you bring up mental health in an interview?", ["No", "Maybe", "Yes"]),
    ("Would you bring up physical health in an interview?", ["No", "Maybe", "Yes"]),
    ("According to you, is mental health as important as physical health?", ["Yes", "No", "Don't know"]),
    ("Have you observed negative consequences for mental health issues?", ["No", "Yes"])
    ]
    
    q_index = st.session_state["question_index"]
    if q_index < len(questions):
        question, options = questions[q_index]
        st.session_state["responses"][question] = st.radio(question, options)
        if st.button("Next"):
            st.session_state["question_index"] += 1
            st.rerun()
           
    else:
        if st.button("Submit & Predict"):
            input_data = preprocess_input(st.session_state["responses"])
            prediction = model.predict(input_data)[0]

            if prediction == 1:
                st.session_state["prediction"] = "High Risk: Prioritize Your Mental Well being\n\n" \
                                         "Prediction:\n" \
                                         "Your responses indicate that you may be facing mental health challenges. This isn't something you have to go through alone support is available, and taking action now can help you feel better.\n\n"

                st.session_state["recommendations"] = "Recommendation:\n" \
                                              "What Steps to Take Next\n" \
                                              "Seek Professional Guidance : Connecting with a therapist or counselor can provide clarity and coping strategies tailored to your needs. Even an initial consultation can be a great start!\n" \
                                              "Talk to Someone You Trust : Sharing your feelings with a close friend, family member, or support group can ease the burden. Opening up is a sign of strength.\n" \
                                              "Learn About Mental Health : Educate yourself on what you're experiencing. Understanding mental health can help reduce fear and uncertainty.\n\n" \
                                              " Daily Self Care & Coping Strategies\n" \
                                              "Practice Stress Relief : Deep breathing, journaling, or even listening to calming music can help reduce tension.\n" \
                                              "Stay Physically Active : Movement, whether it's a simple walk or stretching, releases endorphins that improve mood.\n" \
                                              "Set Small, Achievable Goals : Feeling overwhelmed? Start with small, manageable steps to regain a sense of control.\n\n" \
                                              "Remember, taking action today is the first step toward feeling better. Support is available, and you deserve it!"
            else:
                st.session_state["prediction"] = "Low Risk: Keep Strengthening Your Mental Health\n\n" \
                                         "Prediction:\n" \
                                         "Your responses suggest that you're currently in a stable mental state great job! But mental wellness is an ongoing journey, and maintaining good habits will keep you feeling your best.\n\n"

                st.session_state["recommendations"] = "Recommendation:\n" \
                                              "How to Maintain a Healthy Mind\n" \
                                              "Practice Mindfulness Daily : Take a few minutes to slow down, breathe deeply, and be present. Apps like Calm or Headspace can guide you.\n" \
                                              "Stay Socially Connected : Regular chats with loved ones can provide emotional support and boost your happiness.\n" \
                                              "Keep a Balanced Routine : Having a structured day with time for work, rest, and hobbies helps maintain mental clarity.\n\n" \
                                              "Simple Habits for Long Term Well being\n" \
                                              "Move Your Body : Whether it's yoga, jogging, or dancing, movement keeps your mind and body in sync.\n" \
                                              "Fuel Your Mind with Rest & Nutrition : Aim for 7 to 9 hours of sleep and eat foods rich in nutrients to keep your brain sharp.\n" \
                                              "Take Breaks & Avoid Burnout : Make time for activities you love, whether it's reading, painting, or simply relaxing.\n\n" \
                                              "Even when things feel fine, taking care of your mental well being ensures you stay resilient. Keep prioritizing yourself!"

            st.rerun()

    
    if "prediction" in st.session_state:
        st.subheader("📊 Prediction Results")
        st.write(st.session_state["prediction"])
        st.write("### Recommendations:")
        st.write(st.session_state["recommendations"])
        if st.button("🔊 Read Out Results"):
            if "prediction" in st.session_state:
                speak(st.session_state["prediction"])
            else:
                st.write("No prediction available.")
        if st.button("📥 Download Report"):
            pdf_file = generate_report(st.session_state["user_name"], st.session_state["user_email"], st.session_state["prediction"], st.session_state["recommendations"])
            st.download_button("Download Report", data=pdf_file, file_name=f"{st.session_state['user_name']}_Mental_Health_Report.pdf", mime="application/pdf")