# Classified_doucment
# 🛡️ Classified Document Detector

A Streamlit-based machine learning app that **classifies textual documents** into:
- 🔓 Public  
- 🔒 Secret  

This project simulates document classification for sensitive environments such as defense, legal, or corporate settings. It includes NLP-based preprocessing, spell correction, and a logistic regression model.

## 🚀 Features

- 📁 Upload your own CSV dataset (`text`, `label`)
- 📦 Use built-in dataset (`classified_documents_1000.csv`)
- ⚙️ Train a logistic regression model on the data
- ✍️ Classify new user-inputted text instantly
- 🔤 Built-in spell correction using `TextBlob`
- 💾 Saves model and label encoder (`model.pkl`, `label_encoder.pkl`)

## 🧠 Technologies Used

- Python
- Streamlit
- scikit-learn
- Pandas
- TextBlob
- NLTK

## 📁 Folder Structure

Classified_document/
│
├── data/
│ └── classified_documents_1000.csv
│
├── app.py
├── model.pkl
├── label_encoder.pkl
└── README.md

bash
Copy code

## 📦 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/classified-document-detector.git
cd classified-document-detector
2. Create and activate a virtual environment
bash
Copy code
python -m venv .venv
.venv\Scripts\activate  # On Windows
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
python -m textblob.download_corpora
4. Run the Streamlit app
bash
Copy code
streamlit run app.py
📘 Dataset Format
The dataset should be a CSV file with the following columns:

Column	Description
text	The document text to be analyzed
label	Public or Secret

Example:

csv
Copy code
text,label
"This is a public announcement",Public
"Blueprint of missile system",Secret
🧪 Sample Predictions
Input Text	Predicted Label
Testing a screate missle is sucess	Secret ✅
Government launched a new traffic app	Public ✅

💡 Future Improvements
Add BERT or transformer-based models for deeper understanding

Use SymSpell for faster spell correction

Add model evaluation and accuracy display

Improve UI for upload feedback and prediction history

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change or improve.

📜 License
This project is open source and available under the MIT License.

yaml
Copy code

---

✅ Paste this into your `README.md` file on GitHub, and it will render cleanly with headings, code blocks, tables, and emoji icons.

Let me know if you also want a **project thumbnail/banner** or want to deploy it live!







Ask ChatGPT
