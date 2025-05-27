# 🚀 Demo Aplikasi Analisis Sentimen

## Quick Demo

Jalankan aplikasi:
```bash
./start.sh
```

Atau manual:
```bash
streamlit run pengajar.py
```

## Demo Steps

### 1. 📝 Input Data
- Pilih "Sample Data" untuk demo cepat
- Atau upload file `sample_data.csv` 
- Atau masukkan teks manual

### 2. 🔍 Analisis
- Klik tombol "🔍 Analisis Sentimen"
- Tunggu proses loading model (pertama kali ~30 detik)
- Lihat progress bar analisis

### 3. 📊 Results
- Lihat distribusi sentimen (pie chart)
- Cek confidence scores (histogram)
- Filter hasil berdasarkan sentimen
- Download CSV hasil

### 4. 🤖 OLLAMA Evaluation (Optional)
- Install OLLAMA jika belum ada
- Download model Qwen
- Evaluasi mendalam dengan AI

## Sample Outputs

**Contoh Hasil Positif:**
```
Text: "Produk ini sangat bagus dan berkualitas tinggi!"
Sentiment: positif
Confidence: 0.956
Prob Positif: 0.956
Prob Netral: 0.032
Prob Negatif: 0.012
```

**Contoh Hasil Negatif:**
```
Text: "Pelayanan customer service sangat mengecewakan"
Sentiment: negatif  
Confidence: 0.891
Prob Positif: 0.024
Prob Netral: 0.085
Prob Negatif: 0.891
```

## Screenshots URLs

Ketika aplikasi berjalan, kunjungi:
- Main App: http://localhost:8501
- Health Check: http://localhost:8501/healthz

## Troubleshooting

### Model Loading Issues
```bash
# Clear cache
rm -rf ~/.cache/huggingface/

# Restart app
./start.sh
```

### Port Already in Use
```bash
# Kill existing streamlit
pkill -f streamlit

# Or use different port
streamlit run pengajar.py --server.port 8502
```
