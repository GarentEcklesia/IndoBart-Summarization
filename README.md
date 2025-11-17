# ✍️ IndoBart - Text Summarization

This is a Text Summarization application that uses **IndoBART**, a transformer model specifically for the Indonesian language. The project utilizes the `gaduhhartawan/indobart-base` (a Hugging Face Pretrained Model) which was fine-tuned on the BBC Indonesia article dataset available on Kaggle.

The objectives of this project are:
* To create an automatic summarization system for Indonesian text.
* To help users quickly grasp the core ideas of long articles.
* To demonstrate the application of transformer-based NLP models for a local language.

## 🚀 Live Demo

This application has been deployed using Streamlit Cloud and can be accessed here:

[**➡️ Click here to launch the Streamlit App**](https://indobart-summarization-garentecklesia.streamlit.app/)

## 💡 Application Features

* **Text Input:** An interactive form to type or paste any Indonesian article. (Also includes a button to use sample text).
* **Parameter Control:** Adjust summarization parameters like:
    * Max Length (tokens)
    * Min Length (tokens)
    * Beam Search Size
* **Instant Results:** Displays the generated summary immediately.
* **User Tips:** Provides tips for achieving the best results (e.g., using text longer than 200 words).

## ⚙️ Tech Stack

* **NLP Model:** Hugging Face `transformers` (IndoBART)
* **Core Library:** PyTorch
* **Web Framework:** Streamlit
* **Data Handling:** Pandas, NumPy
* **Deployment Platform:** Streamlit Cloud

## 🧠 Model Details

* **Base Model:** **IndoBART** (`gaduhhartawan/indobart-base` from Hugging Face).
* **Task:** Abstractive Text Summarization.
* **Fine-tuning Dataset:** BBC Indonesia articles dataset from Kaggle.

## 🛠️ How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/GarentEcklesia/Credit-Card-Fraud-Detection
    cd Credit-Card-Fraud-Detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## 📬 Contact

Garent Ecklesia - [garentecklesia45678@gmail.com](mailto:garentecklesia45678@gmail.com)

## 📝 License
This project is open-source and free to use for educational and research purposes.
