import streamlit as st
import subprocess
import os
import tempfile
import pandas as pd
import time
import sys

# Page Config
st.set_page_config(page_title="Student Engagement Detector", layout="wide")

# Title & Description
st.title("🎓 Student Engagement Detection System")
st.markdown("""
This system analyzes classroom videos to detect student engagement levels using **SlowFast Deep Learning Network**.
**Classes:** `Low`, `Mid`, `High`
""")

# Sidebar
st.sidebar.header("Configuration")
analysis_mode = st.sidebar.selectbox(
    "Analysis Mode", 
    ["Global Classification (Whole Class)", "Individual Student Tracker (BBox)"]
)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# File Uploader
uploaded_file = st.file_uploader("Upload a Classroom Video", type=["mp4", "avi", "mov"])

def run_prediction_global(video_path):
    cmd = [
        r"c:\Users\hoescodes\Documents\OUC-CGE\.venv\Scripts\python.exe", "tools/predict_video.py",
        video_path,
        "--cfg", "configs/Kinetics/SLOWFAST_8x8_R50.yaml",
        "--ckpt", "checkpoints/exp2/checkpoints/checkpoint_epoch_00050.pyth"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout, result.stderr
    except Exception as e:
        return None, str(e)

def run_prediction_bbox(video_path, output_path):
    cmd = [
        r"c:\Users\hoescodes\Documents\OUC-CGE\.venv\Scripts\python.exe", "tools/demo_bbox.py",
        "--video_path", video_path,
        "--output", output_path,
        "--cfg", "configs/Kinetics/SLOWFAST_8x8_R50.yaml",
        "--ckpt", "checkpoints/exp2/checkpoints/checkpoint_epoch_00050.pyth"
    ]
    try:
        # Check=True will raise error if script fails
        subprocess.run(cmd, check=True) 
        return True, "Success"
    except subprocess.CalledProcessError as e:
        return False, f"Script failed with code {e.returncode}"
    except Exception as e:
        return False, str(e)

if uploaded_file is not None:
    # Save uploaded file to temp
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
    tfile.write(uploaded_file.read())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Video Preview")
        st.video(tfile.name)
    
    with col2:
        st.info(f"Analysis: {analysis_mode}")
        if st.button("Start Analysis"):
            
            # --- GLOBAL MODE ---
            if analysis_mode == "Global Classification (Whole Class)":
                with st.spinner('Running SlowFast Classification...'):
                    stdout, stderr = run_prediction_global(tfile.name)
                    
                    if stdout and "=== Result ===" in stdout:
                        st.success("Analysis Complete!")
                        # Parsing logic
                        lines = stdout.split('\n')
                        result_data = {}
                        for line in lines:
                            if "1. Low:" in line or "1. Mid:" in line or "1. High:" in line:
                                parts = line.split(":")
                                label = parts[0].split(".")[1].strip()
                                score = float(parts[1].strip().replace("%", ""))
                                result_data[label] = score
                            elif "2. Low:" in line or "2. Mid:" in line or "2. High:" in line:
                                parts = line.split(":")
                                label = parts[0].split(".")[1].strip()
                                score = float(parts[1].strip().replace("%", ""))
                                result_data[label] = score
                            elif "3. Low:" in line or "3. Mid:" in line or "3. High:" in line:
                                parts = line.split(":")
                                label = parts[0].split(".")[1].strip()
                                score = float(parts[1].strip().replace("%", ""))
                                result_data[label] = score
                                
                        if result_data:
                            max_label = max(result_data, key=result_data.get)
                            max_score = result_data[max_label]
                            st.metric(label="Predominant Engagement Level", value=f"{max_label}", delta=f"{max_score}% Confidence")
                            df = pd.DataFrame(list(result_data.items()), columns=["Level", "Confidence"])
                            st.bar_chart(df.set_index("Level"))
                        else:
                            st.warning("Could not parse content.")
                            st.text(stdout)
                    else:
                        st.error("Prediction Failed.")
                        st.text(stdout)
                        st.text(stderr)

            # --- TRACKER BBOX MODE ---
            else:
                with st.spinner('Running AI Tracker (Object Detection + Action Recognition)... This is a heavy process...'):
                    # Output file
                    output_vid = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                    
                    success, msg = run_prediction_bbox(tfile.name, output_vid)
                    
                    if success:
                        st.success("Tracking Complete! Video Generated below:")
                        st.video(output_vid)
                    else:
                        st.error(f"Tracking Failed: {msg}")

st.markdown("---")
st.caption("Developed for OUC-CGE Project | Powered by PyTorchVideo & SlowFast")
