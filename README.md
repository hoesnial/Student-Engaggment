# Student Engagement Recognition System 🎓📊

A Deep Learning project designed to automatically classify student engagement levels in classroom videos using the **SlowFast** action recognition network.

## 🌟 Project Overview
This system analyzes video footage to determine the engagement level of students. It utilizes a state-of-the-art **SlowFast (ResNet50)** architecture to capture both spatial context (scene details) and temporal dynamics (motion/behavior).

### Classes
The model classifies engagement into three levels:
1.  **Low Engagement** 🔴 (Distracted, sleeping, looking away)
2.  **Mid Engagement** 🟡 (Passive listening, neutral posture)
3.  **High Engagement** 🟢 (Active participation, taking notes, raising hands)

---

## 🚀 Getting Started

### Prerequisites
-   **OS**: Windows 10/11 (Recommended) or Linux.
-   **Python**: 3.8 - 3.9.
-   **GPU**: NVIDIA GPU with CUDA support (Recommended for training).

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/hoesnial/Student-Engaggment.git
    cd Student-Engaggment
    ```

2.  Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # Windows
    # source .venv/bin/activate  # Linux/Mac
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## 📂 Project Structure

-   `student_engagement_FINAL_v11.ipynb`: **Main Report**. Contains the full analysis, training logs, validation results, and confusion matrices.
-   `dashboard.py`: **Interactive Demo**. A Streamlit app to test the model on new videos.
-   `configs/`: Configuration files for the SlowFast model.
-   `tools/`: Core scripts for training and inference.
-   `videos/`: Dataset directory (Gitignored).

---

---

## 📂 Dataset Source & Preprocessing
The dataset used in this project was sourced from **[OSF: Student Engagement Dataset](https://osf.io/brd2c/overview)**.

**Processing Steps:**
1.  **Download:** Raw videos were retrieved from OSF.
2.  **Conversion:** Videos were processed into frame-based formats (JPG/NPZ features) to enable efficient loading.
    -   **Script used:** `tools/convert_video_to_frames.py`
    -   **Function:** Extracts individual frames from `.mp4` files and organizes them into class-specific folders.

## 💻 How to Run

### 1. View the Analysis Report
Open the Jupyter Notebook to see the complete project walkthrough and performance metrics (Accuracy ~91%):
```bash
jupyter notebook student_engagement.ipynb
```

### 2. Run the Interactive Dashboard (Demo)
Launch the Streamlit app to upload your own videos and see the engagement prediction in real-time:
```bash
streamlit run dashboard.py
```

### 3. Run Inference via Command Line
To test the model on a specific video file without the GUI:
```bash
python tools/demo_net.py --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml --input_video your_video.mp4
```

---

## 📊 Dataset & Model Details
-   **Backbone**: ResNet-50 (SlowFast variants).
-   **Input**: MP4 Video clips.
-   **Validation Accuracy**: 84.73% (Full Manual Verification).
-   **Test Set Accuracy**: 91.67% (On Unseen Data).
-   **Pre-trained Model**: [Download Checkpoint Here (Google Drive)] https://drive.google.com/drive/folders/14ym_VKw0xfOGKdirNYaraI_xvv6cbXDI?usp=sharing
