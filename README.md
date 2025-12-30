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
**⚠️ PREREQUISITE: Run Validation & Testing First!**
The notebook requires the results from the model evaluation. Before opening the notebook, please run the following commands to generate the data:

#### A. Run Validation (On `val.csv`)
Since the test script defaults to `test.csv`, we temporarily use `val.csv`:
```bash
# 1. Backup actual test.csv
cp test.csv test_backup.csv

# 2. Use validation data as test input
cp val.csv test.csv

# 3. Run Inference (Saves to exp2_val_mp4_full)
python tools/run_net.py --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml DATA.PATH_TO_DATA_DIR ./ TEST.CHECKPOINT_FILE_PATH checkpoints/exp2/checkpoints/checkpoint_epoch_00050.pyth OUTPUT_DIR checkpoints/exp2_val_mp4_full

# 4. Restore test.csv
mv test_backup.csv test.csv
```

#### B. Run Testing (On `test.csv`)
```bash
# Run Inference (Saves to exp2_test_mp4)
python tools/run_net.py --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml DATA.PATH_TO_DATA_DIR ./ TEST.CHECKPOINT_FILE_PATH checkpoints/exp2/checkpoints/checkpoint_epoch_00050.pyth OUTPUT_DIR checkpoints/exp2_test_mp4
```

#### C. Open the Notebook
Once the results are generated, open the notebook to view the report:
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
python tools/demo_net.py --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml --input_video your_video.mp4
```

### 4. Run Full Validation & Testing
To reproduce the validation and test results (84.73% / 91.67%):

1.  **Download Dataset:**
    *   Get the pre-split dataset (Val & Test) from **[Google Drive](https://drive.google.com/drive/folders/14ym_VKw0xfOGKdirNYaraI_xvv6cbXDI?usp=sharing)**.
    *   Extract them into the `videos/` directory so you have `videos/val/` and `videos/test/`.

2.  **Run Evaluation:**
    *   Open `student_engagement.ipynb` and run **Section 4b** (Validation) and **Section 5** (Test).
    *   *Alternative (CMD):* You can run the testing script directly:
        ```bash
        python tools/run_net.py --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml DATA.PATH_TO_DATA_DIR ./ TEST.CHECKPOINT_FILE_PATH checkpoints/exp2/checkpoints/checkpoint_epoch_00050.pyth
        ```

---

## 📊 Dataset & Model Details
-   **Backbone**: ResNet-50 (SlowFast variants).
-   **Input**: MP4 Video clips.
-   **Validation Accuracy**: 84.73% (Full Manual Verification).
-   **Test Set Accuracy**: 91.67% (On Unseen Data).
-   **Pre-trained Model**: [Download Checkpoint Here (Google Drive)](https://drive.google.com/drive/folders/14ym_VKw0xfOGKdirNYaraI_xvv6cbXDI?usp=sharing) (jika tidak pakai git lfs).
