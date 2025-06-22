# 🧠 EEG Dataset for Schizophrenia Classification

This folder is meant to store the raw EEG dataset used in this project. Due to large size constraints (~20 GB), the data is not included directly in this repository.

---

## 📦 Dataset Source

- **Kaggle Dataset**: [Button Tone Schizophrenia EEG Dataset](https://www.kaggle.com/datasets/broach/button-tone-sz)

You must download the dataset manually or use Kaggle CLI and place the files under the `data/button-tone-sz/` directory.

---

## 📁 Directory Structure (After Extraction)

Expected file structure inside the `data/` folder:

data/
└── button-tone-sz  
├── columnLabels.csv  
├── ERPdata.csv  
├── demographic.csv  
├── time.csv  
├── mergedTrialData.csv  
├── 1.csv  
├── 2.csv  
├── ...  
├── 81.csv


Each subject has their own folder (e.g., `1.csv/1.csv`, `2.csv/2.csv`, etc.) containing EEG trial data.

---

## 🧾 Description of Key Files

| File Name              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `columnLabels.csv`     | Labels for EEG signal columns (electrodes, time markers, etc.)              |
| `ERPdata.csv`          | Event-related potential data summary                                        |
| `demographic.csv`      | Age, gender, and diagnosis (control or schizophrenia) for each subject      |
| `time.csv`             | Time indices for signal trials                                              |
| `mergedTrialData.csv`  | Trial-wise EEG data merged across sessions                                  |
| `XX.csv/XX.csv`        | EEG data for subject number XX (e.g., 1–81), stored in individual folders   |

---

## 📥 How to Use in This Project

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/broach/button-tone-sz).
2. Unzip the files.
3. Place the unzipped folder inside this repo at:  
   `data/button-tone-sz/`
4. Your feature extraction and model scripts will access files like:
   - `data/button-tone-sz/1.csv/1.csv`
   - `data/button-tone-sz/columnLabels.csv`

---

## 💡 Tip: Automate with Kaggle CLI (Optional)

If you’ve configured your `kaggle.json`, you can use:

```bash
kaggle datasets download -d broach/button-tone-sz -p data/ --unzip
