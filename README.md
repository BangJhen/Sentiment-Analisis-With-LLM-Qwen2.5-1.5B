# 🐦 Analisis Sentimen Twitter Indonesia

Aplikasi web untuk analisis sentimen teks berbahasa Indonesia menggunakan IndoBERT dan evaluasi dengan OLLAMA/Qwen.

## ✨ Fitur Utama

- **Analisis Sentimen Real-time**: Menggunakan model IndoBERT yang sudah fine-tuned
- **Multiple Input Methods**: Text manual, upload CSV, atau sample data
- **Visualisasi Interaktif**: Charts dan graphs menggunakan Plotly
- **AI Evaluation**: Evaluasi mendalam menggunakan OLLAMA/Qwen
- **Export Results**: Download hasil dalam format CSV
- **Fallback Model**: TextBlob sebagai backup jika IndoBERT gagal

## ➡ Streamlit Web App
https://sentiment-analysis-with-qwen.streamlit.app


## 🚀 Quick Start

### 1. Instalasi Otomatis
```bash
./run_app.sh
```

### 2. Instalasi Manual
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run pengajar.py
```

## 📋 Requirements

### Python Dependencies
- Python 3.9+
- Streamlit 1.28+
- PyTorch 2.0+
- Transformers 4.21+
- Plotly 5.15+

### Optional (untuk evaluasi OLLAMA)
- OLLAMA server
- Qwen2.5 model

## 🔧 Setup OLLAMA (Opsional)

```bash
# Install OLLAMA
curl -fsSL https://ollama.ai/install.sh | sh

# Start server
ollama serve

# Download Qwen model
ollama pull qwen2.5:0.5b
```

## 📊 Cara Penggunaan

### 1. Input Data
- **Manual**: Masukkan teks langsung di text area
- **CSV Upload**: Upload file CSV dengan kolom 'text'
- **Sample Data**: Gunakan 10 contoh teks untuk testing

### 2. Analisis
- Klik tombol "🔍 Analisis Sentimen"
- Tunggu proses analisis selesai
- Lihat hasil di tab "Results & Visualizations"

### 3. Visualisasi
- Pie chart distribusi sentimen
- Bar chart jumlah sentimen
- Histogram confidence score
- Tabel detail hasil

### 4. Evaluasi AI (OLLAMA)
- Pilih model Qwen yang tersedia
- Klik "🤖 Evaluasi Dataset dengan OLLAMA"
- Dapatkan insight mendalam dari AI

## 🎯 Model yang Digunakan

### IndoBERT Models (Prioritas)
1. `mdhugol/indonesia-bert-sentiment-classification`
2. `ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa`
3. `w11wo/indonesian-roberta-base-sentiment-classifier`

### Fallback Model
- TextBlob + Google Translate (jika IndoBERT gagal)

### OLLAMA Models
- Qwen2.5:0.5b (Recommended - Cepat)
- Qwen2.5:1.5b (Seimbang)
- Qwen2.5:3b (Akurat)

## 📁 Struktur File

```
├── pengajar.py          # Main application
├── requirements.txt     # Python dependencies
├── run_app.sh          # Auto-run script
└── README.md           # Documentation
```

## 🔍 Output Format

### Hasil Analisis
- `text`: Teks input
- `sentiment`: positif/negatif/netral
- `confidence`: Skor kepercayaan (0-1)
- `prob_positif`: Probabilitas sentimen positif
- `prob_netral`: Probabilitas sentimen netral
- `prob_negatif`: Probabilitas sentimen negatif

### Evaluasi OLLAMA
- Ringkasan distribusi sentimen
- Kualitas prediksi
- Insight bisnis
- Rekomendasi tindakan

## 🛠️ Troubleshooting

### Model Loading Error
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/

# Reinstall transformers
pip uninstall transformers
pip install transformers
```

### OLLAMA Connection Error
```bash
# Check if OLLAMA is running
curl http://localhost:11434/api/tags

# Restart OLLAMA
ollama serve
```

### Memory Issues
- Gunakan model yang lebih kecil (qwen2.5:0.5b)
- Kurangi batch size untuk dataset besar
- Tutup aplikasi lain yang menggunakan GPU/RAM

## 📈 Performance Tips

1. **GPU**: Gunakan CUDA jika tersedia untuk performa terbaik
2. **Model Size**: Pilih model sesuai kapasitas hardware
3. **Batch Processing**: Untuk dataset besar, proses secara bertahap
4. **Confidence Threshold**: Filter hasil dengan confidence >0.7

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Create pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **IndoBERT**: Hugging Face Indonesian BERT models
- **OLLAMA**: Local LLM runtime
- **Qwen**: Alibaba Cloud's language model
- **Streamlit**: Web app framework
- **Plotly**: Interactive visualizations

---

**Made with ❤️ for Indonesian NLP community**
