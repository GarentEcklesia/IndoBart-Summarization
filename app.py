import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import time
from datetime import datetime
import re

st.set_page_config(
    page_title="IndoBART Text Summarization",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .summary-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .input-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 1rem 0;
    }
    .metrics-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the IndoBART model and tokenizer with caching."""
    model_path = "GarentEcklesia/IndoBart_Summarization"
    
    try:
        if not os.path.exists(model_path):
            st.error(f"Direktori model '{model_path}' tidak ditemukan!")
            st.info("Pastikan folder model Anda berada di direktori yang sama dengan script ini.")
            return None, None
        
        with st.spinner("Memuat model IndoBART... Harap tunggu sebentar."):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval() 
        
        return tokenizer, model, device
    
    except Exception as e:
        st.error(f"Error saat memuat model: {str(e)}")
        return None, None, None

def preprocess_text(text):
    ### Clean and preprocess input text.
    text = re.sub(r'\s+', ' ', text.strip())
    
    text = re.sub(r'[^\w\s\.,!?;:-]', '', text)
    
    return text

def generate_summary(text, tokenizer, model, device, max_length=150, min_length=30, num_beams=4):
    ### Generate summary using the IndoBART model.
    try:
        text = preprocess_text(text)
        
        inputs = tokenizer(
            text,
            max_length=1024, 
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                do_sample=False
            )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    except Exception as e:
        st.error(f"Error saat membuat ringkasan: {str(e)}")
        return None

def calculate_metrics(original_text, summary):
    """Calculate basic text metrics."""
    original_words = len(original_text.split())
    summary_words = len(summary.split())
    compression_ratio = (1 - summary_words / original_words) * 100 if original_words > 0 else 0
    
    return {
        "original_words": original_words,
        "summary_words": summary_words,
        "compression_ratio": compression_ratio
    }

def main():
    st.markdown('<div class="main-header">📰 Summarization Teks IndoBART</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Summarization Teks Bahasa Indonesia untuk Artikel</div>', unsafe_allow_html=True)
    
    tokenizer, model, device = load_model()
    
    if tokenizer is None or model is None:
        st.stop()
    
    st.success(f"✅ Model berhasil dimuat! Berjalan di: {device}")
    
    st.sidebar.header("⚙️ Pengaturan Summarization")
    
    max_length = st.sidebar.slider(
        "Panjang Maksimal Ringkasan (Tokens)",
        min_value=50,
        max_value=300,
        value=150,
        step=10,
        help="Panjang maksimal ringkasan yang akan dihasilkan"
    )
    
    min_length = st.sidebar.slider(
        "Panjang Minimal Ringkasan (Tokens)",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        help="Panjang minimal ringkasan yang akan dihasilkan"
    )
    
    num_beams = st.sidebar.selectbox(
        "Ukuran Beam Search",
        [2, 4, 6, 8],
        index=1,
        help="Nilai lebih tinggi mungkin menghasilkan kualitas lebih baik tapi lebih lambat"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Tips untuk Hasil Terbaik:")
    st.sidebar.markdown("• Gunakan teks bahasa Indonesia")
    st.sidebar.markdown("• Artikel bekerja paling baik (200+ kata)")
    st.sidebar.markdown("• Periksa format teks")
    st.sidebar.markdown("• Teks lebih panjang membutuhkan waktu lebih lama")
    
    st.markdown("### 📝 Masukkan Teks Artikel Indonesia")
    
    sample_text = """Jakarta - Presiden Joko Widodo mengumumkan kebijakan baru mengenai pengembangan infrastruktur digital di Indonesia. 
    Kebijakan ini bertujuan untuk mempercepat transformasi digital dan meningkatkan konektivitas internet di seluruh nusantara. 
    Menurut Presiden, investasi dalam infrastruktur digital akan menciptakan lapangan kerja baru dan mendorong pertumbuhan ekonomi. 
    Program ini akan dilaksanakan secara bertahap selama lima tahun ke depan dengan anggaran yang mencapai triliunan rupiah. 
    Pemerintah juga akan bekerja sama dengan sektor swasta untuk memastikan implementasi yang efektif. 
    Diharapkan program ini dapat meningkatkan daya saing Indonesia di era digital global."""
    
    input_method = st.radio(
        "Pilih metode input:",
        ["Type/Paste Teks", "Gunakan Teks Contoh"],
        horizontal=True
    )
    
    if input_method == "Gunakan Teks Contoh":
        input_text = st.text_area(
            "Artikel Indonesia contoh (Anda bisa mengeditnya):",
            value=sample_text,
            height=200,
            help="Ini adalah artikel Indonesia contoh untuk demonstrasi"
        )
    else:
        input_text = st.text_area(
            "Masukkan artikel Indonesia Anda di sini:",
            height=200,
            placeholder="Masukkan teks artikel bahasa Indonesia di sini...",
            help="Masukkan teks artikel Indonesia Anda di sini untuk diringkas"
        )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        summarize_button = st.button(
            "🚀 Buat Ringkasan",
            type="primary",
            use_container_width=True
        )

    if summarize_button and input_text.strip():
        if len(input_text.split()) < 20:
            st.warning("⚠️ Teks tampak terlalu pendek. Harap berikan setidaknya 20 kata untuk peringkasan yang lebih baik.")
        else:
            start_time = time.time()
            
            with st.spinner("Membuat ringkasan... Harap tunggu."):
                summary = generate_summary(
                    input_text, 
                    tokenizer, 
                    model, 
                    device,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams
                )
            
            end_time = time.time()
            
            if summary:
                st.markdown("---")
                st.markdown("### 📊 Hasil Peringkasan")
                
                metrics = calculate_metrics(input_text, summary)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Kata Asli", metrics["original_words"])
                
                with col2:
                    st.metric("Kata Ringkasan", metrics["summary_words"])
                
                with col3:
                    st.metric("Kompresi", f"{metrics['compression_ratio']:.1f}%")
                
                with col4:
                    st.metric("Waktu Proses", f"{end_time - start_time:.2f}s")
                
                st.markdown("### 📝 Ringkasan yang Dihasilkan")
                st.markdown(
                    f'<div class="summary-box"><strong>Ringkasan:</strong><br>{summary}</div>',
                    unsafe_allow_html=True
                )
                
                st.markdown("### 📋 Salin Ringkasan")
                st.code(summary, language=None)
                
                st.download_button(
                    label="💾 Download Ringkasan sebagai File Teks",
                    data=f"Teks Asli ({metrics['original_words']} kata):\n{input_text}\n\n---\n\nRingkasan ({metrics['summary_words']} kata):\n{summary}\n\nDibuat pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    file_name=f"ringkasan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    elif summarize_button and not input_text.strip():
        st.warning("⚠️ Harap masukkan teks untuk diringkas.")
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9rem;'>
            <p>🤖 Didukung oleh IndoBART | Dibuat dengan Streamlit</p>
            <p>Untuk Summarization teks Indonesia | Hasil terbaik didapatkan dengan artikel berita</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
