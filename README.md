# IndoBart-Summarization
Proyek ini adalah aplikasi Text Summarization yang menggunakan IndoBART — sebuah model transformer untuk Bahasa Indonesia.
Model yang digunakan adalah gaduhhartawan/indobart-base (HuggingFace Pretrained Model), kemudian dilakukan fine-tuning menggunakan dataset artikel BBC Indonesia yang tersedia di Kaggle.

Tujuan proyek ini adalah untuk:
1. Membuat sistem ringkasan otomatis teks berbahasa Indonesia.
2. Memudahkan pengguna dalam membaca inti artikel panjang.
3. Menunjukkan penerapan transformer-based NLP model dalam bahasa lokal.

Fitur-fitur pada Aplikasi ini:
1. Input teks artikel bahasa Indonesia melalui form interaktif.(Type/Paste teks & Gunakan teks contoh)
2. Kontrol parameter summarization (Panjang maksimal ringkasan (tokens), panjang minimal ringkasan (tokens), dan ukuran beam search)
3. Menampilkan hasil ringkasan secara instan.
4. Tips untuk hasil terbaik (misalnya teks lebih dari 200 kata).

Aplikasi ini sudah dideploy menggunakan Streamlit Cloud dan dapat diakses di sini: https://indobart-summarization-garentecklesia.streamlit.app/

Preview Aplikasi:
<img width="1919" height="753" alt="image" src="https://github.com/user-attachments/assets/b2e1521e-9dbb-4f95-b1ff-c717d5f2ec6a" />
<img width="1917" height="905" alt="image" src="https://github.com/user-attachments/assets/7655dfea-f5b5-482e-9cd4-d887582e59d4" />
