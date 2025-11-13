# 🩺 Telemedicine Triage Classification

This project focuses on building a **machine learning-based triage system** to assist in rural healthcare.  
It analyzes patient symptoms, demographics, and vital signs to automatically classify the case as:  
- **Emergency** 🆘  
- **Urgent** ⚠️  
- **Routine** ✅  

By leveraging **Natural Language Processing (NLP)** and **Machine Learning (ML)**, this project aims to support healthcare workers in prioritizing patients efficiently in low-resource environments.

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/telemedicine-triage.git
```

### 2️⃣ Install dependencies
```bash

pip install -r requirements.txt
```
### 3️⃣ Download required NLTK data
```bash

python -m nltk.downloader stopwords punkt
```
### 4️⃣ Run the pipeline
```bash

python telemedicine_pipeline.py
```