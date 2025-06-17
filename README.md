# 🧠 Schizophrenia EEG Classification Project

This project contains three sub-models designed to classify EEG data for schizophrenia diagnosis. Each model follows a different strategy of feature extraction and classification using Machine Learning and Deep Learning techniques.

---

## 📁 Project Structure

Schizophrenia-EEG-Classification/  
├── models/  
│   ├── model_1_ml_manual.py  
│   ├── model_2_dl_manual.py  
│   ├──  model_3_dl_from_ml.py  
│   └── scz-for-eeg (2).ipynb
├── data/  
│   ├── demographic.csv  
│   ├── 18.csv  
│   └── README.md  
├── requirements.txt  
├── .gitignore  
└── README.md

---

### 📄 Descriptions

- `models/`  
  - `model_1_ml_manual.py`: Manual feature extraction + ML classification  
  - `model_2_dl_manual.py`: Manual feature extraction + DL classification  
  - `model_3_dl_from_ml.py`: ML-based feature extraction + DL classification  

- `data/`  
  - `demographic.csv`: EEG metadata file (user must provide)  
  - `18.csv`: EEG signal sample file (user must provide)  
  - `README.md`: Instructions on what data files are needed and where to place them  

- `requirements.txt`: List of required Python packages  
- `.gitignore`: Specifies files and folders to be ignored by Git  
- `README.md`: Main project documentation (this file)

---

## 🚀 Getting Started

1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/Schizophrenia-EEG-Classification.git
   cd Schizophrenia-EEG-Classification
