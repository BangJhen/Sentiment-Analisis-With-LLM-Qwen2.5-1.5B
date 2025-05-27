import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from config import MODEL_CONFIG, OLLAMA_CONFIG, UI_CONFIG, SAMPLE_TEXTS

# Konfigurasi Streamlit
st.set_page_config(page_title=UI_CONFIG["page_title"], layout=UI_CONFIG["layout"])
st.title("🐦 Analisis Sentimen Twitter Indonesia")
st.write("Analisis sentimen berbasis tweet menggunakan IndoBERT dan evaluasi dengan OLLAMA")

# Inisialisasi IndoBERT
@st.cache_resource
def load_indobert():
    """
    Load model IndoBERT yang sudah fine-tuned untuk analisis sentimen Indonesia
    dengan fallback ke model alternatif jika gagal
    """    # Daftar model yang bisa digunakan (urutan prioritas)
    model_options = MODEL_CONFIG["primary_models"]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i, model_info in enumerate(model_options):
        try:
            model_name = model_info["name"]
            st.info(f"🔄 Mencoba model {i+1}/{len(model_options)}: {model_info['description']}")
            
            # Load tokenizer dan model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Pindahkan model ke device yang tepat
            model.to(device)
            model.eval()
            
            # Validasi model dengan test input
            test_success = validate_model(tokenizer, model, device)
            
            if test_success:
                st.success(f"✅ Model berhasil dimuat: {model_info['description']}")
                st.info(f"📱 Device: {device}")
                
                # Update global label mapping
                global LABEL_MAPPING
                LABEL_MAPPING = model_info["labels"]
                
                return tokenizer, model, device, model_info
            else:
                st.warning(f"⚠️ Model {model_name} tidak dapat divalidasi")
                continue
                
        except Exception as e:
            st.warning(f"❌ Gagal memuat model {model_info['name']}: {str(e)}")
            continue
    
    # Jika semua model gagal, coba fallback
    st.error("❌ Semua model IndoBERT gagal dimuat")
    return load_fallback_model()

def validate_model(tokenizer, model, device):
    """
    Validasi model dengan test input untuk memastikan model bekerja dengan benar
    """
    try:
        # Test dengan beberapa contoh teks Indonesia
        test_texts = [
            "Saya sangat senang dengan produk ini",
            "Produk ini biasa saja",
            "Sangat kecewa dengan pelayanannya"
        ]
        
        for text in test_texts:
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=128
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                
                # Pastikan output memiliki format yang benar
                if not hasattr(outputs, 'logits'):
                    return False
                
                # Pastikan output memiliki 3 kelas (negatif, netral, positif)
                if outputs.logits.shape[-1] != 3:
                    return False
                
                # Test softmax
                probs = F.softmax(outputs.logits, dim=-1)
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    return False
        
        return True
        
    except Exception as e:
        st.error(f"Validasi model gagal: {str(e)}")
        return False

def load_fallback_model():
    """
    Fallback ke model sederhana jika semua model transformer gagal
    """
    try:
        st.warning("🔄 Menggunakan model fallback sederhana...")
        
        # Import library untuk fallback
        try:
            from textblob import TextBlob
            from googletrans import Translator
            
            # Setup translator untuk Indonesia
            translator = Translator()
            
            st.info("✅ Fallback model (TextBlob + Translation) siap")
            
            # Return format yang konsisten
            return "textblob", translator, "cpu", {
                "name": "textblob_fallback",
                "description": "TextBlob dengan Google Translate",
                "labels": {0: 'negatif', 1: 'netral', 2: 'positif'}
            }
            
        except ImportError:
            st.error("❌ Library fallback tidak tersedia. Install dengan: pip install textblob googletrans==4.0.0rc1")
            return None, None, None, None
            
    except Exception as e:
        st.error(f"❌ Fallback model juga gagal: {str(e)}")
        return None, None, None, None

def analyze_sentiment_fallback(texts, translator):
    """
    Analisis sentimen menggunakan fallback model (TextBlob)
    """
    from textblob import TextBlob
    
    results = []
    for text in texts:
        try:
            # Translate ke bahasa Inggris untuk TextBlob
            translated = translator.translate(text, src='id', dest='en').text
            
            # Analisis dengan TextBlob
            blob = TextBlob(translated)
            polarity = blob.sentiment.polarity
            
            # Convert polarity to sentiment
            if polarity > 0.1:
                sentiment = 'negatif'
                label = 2
            elif polarity < -0.1:
                sentiment = 'positif' 
                label = 0
            else:
                sentiment = 'netral'
                label = 1
            
            # Simulate confidence based on polarity strength
            confidence = min(abs(polarity) + 0.5, 1.0)
            
            # Simulate probabilities
            if sentiment == 'negatif':
                prob_positif = confidence
                prob_negatif = (1 - confidence) / 2
                prob_netral = (1 - confidence) / 2
            elif sentiment == 'positif':
                prob_negatif = confidence
                prob_positif = (1 - confidence) / 2
                prob_netral = (1 - confidence) / 2
            else:
                prob_netral = confidence
                prob_positif = (1 - confidence) / 2
                prob_negatif = (1 - confidence) / 2
            
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence,
                'prob_negatif': prob_negatif,
                'prob_netral': prob_netral,
                'prob_positif': prob_positif
            })
            
        except Exception as e:
            # Jika gagal, beri sentiment netral
            results.append({
                'text': text,
                'sentiment': 'netral',
                'confidence': 0.5,
                'prob_negatif': 0.33,
                'prob_netral': 0.34,
                'prob_positif': 0.33
            })
    
    return pd.DataFrame(results)

# Update fungsi load model dan variabel global
tokenizer, model, device, model_info = load_indobert()

# Label mapping untuk sentimen
LABEL_MAPPING = {0: 'negatif', 1: 'netral', 2: 'positif'}

# Fungsi analisis sentimen
def analyze_sentiment(texts):
    if not tokenizer or not model:
        st.error("❌ Model tidak tersedia")
        return pd.DataFrame()
    
    # Check if using fallback model
    if tokenizer == "textblob":
        st.info("🔄 Menggunakan fallback model untuk analisis...")
        return analyze_sentiment_fallback(texts, model)  # model is actually translator here
    
    if not texts or len(texts) == 0:
        st.warning("⚠️ Tidak ada teks untuk dianalisis")
        return pd.DataFrame()
    
    results = []
    errors = []
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, text in enumerate(texts):
        try:
            # Update progress
            progress = (i + 1) / len(texts)
            progress_bar.progress(progress)
            status_text.text(f"Memproses {i+1}/{len(texts)} tweet...")
            
            if not text or len(text.strip()) == 0:
                continue
                
            # Batasi panjang teks (Twitter limit)
            text = str(text)[:280]
            
            # Tokenize dengan error handling
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=128
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                predicted_label = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][predicted_label].item()
            
            if predicted_label == 0:
                sentiment = 'positif'
            elif predicted_label == 1:
                sentiment = 'netral'
            else:
                sentiment = 'negatif'
            # sentiment = LABEL_MAPPING[predicted_label]
            
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence,
                'prob_negatif': probs[0][0].item(),
                'prob_netral': probs[0][1].item(),
                'prob_positif': probs[0][2].item()
            })
            
        except Exception as e:
            errors.append(f"Error pada teks '{text[:50]}...': {str(e)}")
            continue
    
    # Clear progress
    progress_bar.empty()
    status_text.empty()
    
    # Show errors if any
    if errors:
        with st.expander(f"⚠️ {len(errors)} error ditemukan"):
            for error in errors:
                st.text(error)
    
    return pd.DataFrame(results)

# Fungsi untuk OLLAMA evaluation (keseluruhan)
def check_ollama_status():
    """Check if OLLAMA server is running and get available models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            return True, available_models
        return False, []
    except:
        return False, []

def evaluate_with_ollama(df_results, model="qwen2.5:0.5b"):
    """Evaluasi hasil analisis sentimen menggunakan OLLAMA dengan model yang dipilih"""
    
    # Check OLLAMA status first
    is_running, available_models = check_ollama_status()
    
    if not is_running:
        return """❌ OLLAMA server tidak berjalan. 
        
Untuk menjalankan OLLAMA:
1. Buka terminal
2. Jalankan: `ollama serve`
3. Di terminal lain, install model: `ollama pull qwen2.5:0.5b`
"""
    
    # Check if requested model is available
    model_available = any(model in available_model for available_model in available_models)
    if not model_available:
        return f"""❌ Model {model} tidak tersedia. 

Model yang tersedia: {available_models}

Untuk install model Qwen:
```bash
ollama pull qwen2.5:0.5b
# atau
ollama pull qwen2.5:1.5b
# atau  
ollama pull qwen2.5:3b
```
"""
    
    # Hitung statistik
    sentiment_counts = df_results['sentiment'].value_counts()
    avg_confidence = df_results['confidence'].mean()
    
    # Analisis per sentiment dengan contoh berkualitas tinggi
    sentiment_analysis = {}
    for sentiment in ['negatif', 'netral', 'positif']:
        sentiment_df = df_results[df_results['sentiment'] == sentiment]
        if len(sentiment_df) > 0:
            # Ambil contoh dengan confidence tertinggi
            high_conf_samples = sentiment_df.nlargest(3, 'confidence')[['text', 'confidence']]
            sentiment_analysis[sentiment] = {
                'count': len(sentiment_df),
                'avg_confidence': sentiment_df['confidence'].mean(),
                'examples': high_conf_samples.to_dict('records')
            }
    
    # Buat prompt yang dioptimalkan untuk Qwen
    prompt = f"""Analisis hasil klasifikasi sentimen untuk {len(df_results)} tweet Indonesia:

DISTRIBUSI SENTIMEN:
• Positif: {sentiment_counts.get('positif', 0)} tweet ({sentiment_counts.get('positif', 0)/len(df_results)*100:.1f}%)
• Netral: {sentiment_counts.get('netral', 0)} tweet ({sentiment_counts.get('netral', 0)/len(df_results)*100:.1f}%)  
• Negatif: {sentiment_counts.get('negatif', 0)} tweet ({sentiment_counts.get('negatif', 0)/len(df_results)*100:.1f}%)

CONFIDENCE RATA-RATA: {avg_confidence:.3f}

CONTOH TWEET TERBAIK PER KATEGORI:"""

    for sentiment, data in sentiment_analysis.items():
        prompt += f"\n\n{sentiment.upper()} (confidence rata-rata: {data['avg_confidence']:.3f}):"
        for i, example in enumerate(data['examples'][:2], 1):
            prompt += f"\n  {i}. \"{example['text']}\" (confidence: {example['confidence']:.3f})"

    prompt += f"""

Berikan evaluasi dalam format berikut:

## 📊 RINGKASAN DISTRIBUSI
[Analisis distribusi sentimen dan apa artinya]

## 🎯 KUALITAS PREDIKSI  
[Evaluasi confidence score dan akurasi prediksi]

## 💼 INSIGHT BISNIS
[Kesimpulan tentang produk/layanan berdasarkan sentimen]

## 🚀 REKOMENDASI
[Saran tindakan berdasarkan hasil analisis]

Catatan: Ini adalah data Twitter Indonesia dengan bahasa informal dan slang."""
    
    try:
        with st.spinner(f"🤖 Mengevaluasi dengan {model}..."):
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "max_tokens": 1000,
                        "stop": ["</s>", "<|im_end|>"]
                    }
                },
                timeout=90  # Increased timeout for smaller models
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'Error: Empty response')
            else:
                return f"❌ Error dari OLLAMA: HTTP {response.status_code}\n{response.text}"
                
    except requests.Timeout:
        return "⏱️ Timeout: Model membutuhkan waktu terlalu lama. Coba dengan model yang lebih kecil."
    except Exception as e:
        return f"❌ Error koneksi: {str(e)}"

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📝 Input Data", "📊 Results & Visualizations", "🤖 OLLAMA Evaluation"])

# Tab 1: Input Data
with tab1:
    st.header("📝 Input Data untuk Analisis Sentimen")
    
    # Method selection
    input_method = st.radio(
        "Pilih metode input:",
        ["Manual Text Input", "Upload CSV File", "Sample Data"]
    )
    
    texts_to_analyze = []
    
    if input_method == "Manual Text Input":
        st.subheader("Input Teks Manual")
        
        # Single text input
        single_text = st.text_area(
            "Masukkan teks untuk analisis:",
            placeholder="Contoh: Produk ini sangat bagus dan saya rekomendasikan!",
            height=100
        )
        
        # Multiple texts input
        multiple_texts = st.text_area(
            "Atau masukkan beberapa teks (pisahkan dengan baris baru):",
            placeholder="Teks 1\nTeks 2\nTeks 3",
            height=150
        )
        
        if single_text.strip():
            texts_to_analyze = [single_text.strip()]
        elif multiple_texts.strip():
            texts_to_analyze = [text.strip() for text in multiple_texts.split('\n') if text.strip()]
    
    elif input_method == "Upload CSV File":
        st.subheader("Upload File CSV")
        uploaded_file = st.file_uploader(
            "Pilih file CSV",
            type=['csv'],
            help="File CSV harus memiliki kolom 'text' yang berisi teks untuk dianalisis"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File berhasil diupload: {len(df)} baris")
                
                # Show preview
                st.subheader("Preview Data:")
                st.dataframe(df.head())
                
                # Select text column
                if 'text' in df.columns:
                    text_column = 'text'
                else:
                    text_column = st.selectbox(
                        "Pilih kolom yang berisi teks:",
                        df.columns.tolist()
                    )
                
                if text_column:
                    texts_to_analyze = df[text_column].dropna().astype(str).tolist()
                    st.info(f"📊 {len(texts_to_analyze)} teks siap untuk dianalisis")
                    
            except Exception as e:
                st.error(f"❌ Error membaca file: {str(e)}")
    elif input_method == "Sample Data":
        st.subheader("Data Contoh")
        sample_texts = SAMPLE_TEXTS
        
        st.info("✨ Menggunakan 10 contoh teks untuk demo")
        for i, text in enumerate(sample_texts, 1):
            st.text(f"{i}. {text}")
        
        texts_to_analyze = sample_texts
    
    # Analysis button
    if texts_to_analyze:
        st.subheader("🚀 Mulai Analisis")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.metric("Total Teks", len(texts_to_analyze))
        
        with col2:
            analyze_button = st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True)
        
        if analyze_button:
            with st.spinner("🔄 Sedang menganalisis sentimen..."):
                results = analyze_sentiment(texts_to_analyze)
                
                if not results.empty:
                    st.session_state['results'] = results
                    st.success("✅ Analisis selesai! Lihat hasil di tab 'Results & Visualizations'")
                    st.balloons()
                else:
                    st.error("❌ Gagal menganalisis teks")
    else:
        st.info("👆 Pilih metode input dan masukkan teks untuk memulai analisis")

# Tab 2: Results & Visualizations  
with tab2:
    st.header("📊 Hasil Analisis & Visualisasi")
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # Summary statistics
        st.subheader("📈 Ringkasan Statistik")
        col1, col2, col3, col4 = st.columns(4)
        
        sentiment_counts = results['sentiment'].value_counts()
        
        with col1:
            st.metric(
                "Total Teks", 
                len(results),
                help="Jumlah total teks yang dianalisis"
            )
        
        with col2:
            st.metric(
                "Sentimen Positif", 
                sentiment_counts.get('positif', 0),
                f"{sentiment_counts.get('positif', 0)/len(results)*100:.1f}%"
            )
        
        with col3:
            st.metric(
                "Sentimen Negatif", 
                sentiment_counts.get('negatif', 0),
                f"{sentiment_counts.get('negatif', 0)/len(results)*100:.1f}%"
            )
        
        with col4:
            st.metric(
                "Confidence Rata-rata", 
                f"{results['confidence'].mean():.3f}",
                help="Tingkat kepercayaan prediksi model"
            )
        
        # Visualizations
        st.subheader("📊 Visualisasi")
        
        col1, col2 = st.columns(2)
        
        with col1:            # Pie chart for sentiment distribution
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Distribusi Sentimen",
                color_discrete_map=UI_CONFIG["colors"]
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:            # Bar chart for sentiment counts
            fig_bar = px.bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                title="Jumlah Sentimen",
                color=sentiment_counts.index,
                color_discrete_map=UI_CONFIG["colors"]
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Confidence distribution
        st.subheader("📈 Distribusi Confidence Score") 
        fig_hist = px.histogram(
            results, 
            x='confidence', 
            color='sentiment',
            title="Distribusi Confidence Score per Sentimen",
            nbins=20,
            color_discrete_map=UI_CONFIG["colors"]
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Detailed results table
        st.subheader("📋 Hasil Detail")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            sentiment_filter = st.selectbox(
                "Filter berdasarkan sentimen:",
                ["Semua", "negatif", "netral", "positif"]
            )
        
        with col2:
            min_confidence = st.slider(
                "Confidence minimum:",
                0.0, 1.0, 0.0, 0.1
            )
        
        # Apply filters
        filtered_results = results.copy()
        if sentiment_filter != "Semua":
            filtered_results = filtered_results[filtered_results['sentiment'] == sentiment_filter]
        
        filtered_results = filtered_results[filtered_results['confidence'] >= min_confidence]
        
        # Sort by confidence
        filtered_results = filtered_results.sort_values('confidence', ascending=False)
        
        # Display results
        st.dataframe(
            filtered_results[['text', 'sentiment', 'confidence', 'prob_positif', 'prob_netral',  'prob_negatif']],
            use_container_width=True,
            column_config={
                "text": st.column_config.TextColumn("Teks", width="large"),
                "sentiment": st.column_config.TextColumn("Sentimen", width="small"),
                "confidence": st.column_config.NumberColumn("Confidence", format="%.3f"),
                "prob_positif": st.column_config.NumberColumn("Prob. Negatif", format="%.3f"),
                "prob_netral": st.column_config.NumberColumn("Prob. Netral", format="%.3f"),
                "prob_negatif": st.column_config.NumberColumn("Prob. Positif", format="%.3f"),
            }
        )
        
        # Download results
        st.subheader("💾 Download Hasil")
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name='sentiment_analysis_results.csv',
            mime='text/csv'
        )
        
    else:
        st.info("📝 Belum ada hasil analisis. Silakan lakukan analisis di tab 'Input Data' terlebih dahulu.")

# Tab 3: OLLAMA Evaluation
with tab3:
    st.header("Evaluasi dengan OLLAMA")
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # Check OLLAMA status
        ollama_running, available_models = check_ollama_status()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if ollama_running:
                st.success("✅ OLLAMA Connected")
                
                # Pilih model OLLAMA dengan prioritas Qwen
                qwen_models = [m for m in available_models if 'qwen' in m.lower()]
                other_models = [m for m in available_models if 'qwen' not in m.lower()]
                
                model_options = qwen_models + other_models
                
                if not model_options:
                    st.warning("Tidak ada model tersedia. Install model dengan:")
                    st.code("ollama pull qwen2.5:0.5b")
                    model_options = ["qwen2.5:0.5b"]  # Default fallback
                
                ollama_model = st.selectbox(
                    "Pilih Model OLLAMA:",
                    model_options,
                    index=0
                )
                
                if 'qwen' in ollama_model.lower():
                    st.info("🎯 Menggunakan model Qwen yang dioptimalkan untuk analisis")
                
            else:
                st.error("❌ OLLAMA Offline")
                st.warning("Jalankan OLLAMA server:")
                st.code("""ollama serve
# Di terminal lain:
ollama pull qwen2.5:0.5b""")
                ollama_model = "qwen2.5:0.5b"
        
        with col2:
            st.metric("Total Tweets", len(results))
            st.metric("Avg Confidence", f"{results['confidence'].mean():.3f}")
            
            # Model size info
            if 'qwen' in ollama_model.lower():
                if '0.5b' in ollama_model:
                    st.caption("🚀 Model kecil, cepat")
                elif '1.5b' in ollama_model:
                    st.caption("⚖️ Model sedang, seimbang")
                elif '3b' in ollama_model:
                    st.caption("🎯 Model besar, akurat")
        
        # Evaluasi keseluruhan
        st.subheader("Evaluasi Keseluruhan Dataset")
        
        if st.button("🤖 Evaluasi Dataset dengan OLLAMA", type="primary"):
            evaluation = evaluate_with_ollama(results, ollama_model)
            
            st.subheader("📋 Hasil Evaluasi OLLAMA:")
            st.markdown(evaluation)
        
        # Quick stats
        with st.expander("📊 Statistik Cepat"):
            col1, col2, col3 = st.columns(3)
            
            sentiment_counts = results['sentiment'].value_counts()
            
            with col1:
                st.metric(
                    "Sentimen Positif", 
                    sentiment_counts.get('positif', 0),
                    f"{sentiment_counts.get('positif', 0)/len(results)*100:.1f}%"
                )
            
            with col2:
                st.metric(
                    "Sentimen Netral", 
                    sentiment_counts.get('netral', 0),
                    f"{sentiment_counts.get('netral', 0)/len(results)*100:.1f}%"
                )
            
            with col3:
                st.metric(
                    "Sentimen Negatif", 
                    sentiment_counts.get('negatif', 0),
                    f"{sentiment_counts.get('negatif', 0)/len(results)*100:.1f}%"
                )
    
    # Sidebar untuk informasi dan status
    with st.sidebar:
        st.header("ℹ️ Informasi Aplikasi")
        st.markdown("""
        **Model yang digunakan:**
        - **IndoBERT**: Fine-tuned untuk sentimen Indonesia
        - **OLLAMA**: Qwen2.5 (Recommended)
        
        **Fitur Utama:**
        - Analisis sentimen real-time
        - Visualisasi interaktif
        - Export hasil CSV
        - Evaluasi dengan AI
        """)
        
        st.header("🔧 Status Sistem")
        
        # IndoBERT status
        if 'tokenizer' in globals() and 'model' in globals():
            if tokenizer and model:
                if tokenizer == "textblob":
                    st.warning("⚠️ Fallback Model Active")
                    st.caption("Menggunakan TextBlob + Google Translate")
                else:
                    st.success("✅ IndoBERT Ready")
                    if 'model_info' in globals() and model_info:
                        st.caption(f"Model: {model_info.get('description', 'Unknown')}")
            else:
                st.error("❌ IndoBERT Error")
        else:
            st.info("🔄 Loading IndoBERT...")
        
        # OLLAMA status
        try:
            ollama_running, available_models = check_ollama_status()
            if ollama_running:
                st.success("✅ OLLAMA Connected")
                
                # Show available models with special marking for Qwen
                qwen_models = [m for m in available_models if 'qwen' in m.lower()]
                if qwen_models:
                    st.info(f"🎯 Qwen models: {len(qwen_models)}")
                    for model_name in qwen_models[:3]:  # Show first 3
                        st.caption(f"  • {model_name}")
                
                st.caption(f"Total models: {len(available_models)}")
            else:
                st.error("❌ OLLAMA Offline")
                with st.expander("Setup OLLAMA"):
                    st.code("""# Terminal 1
    ollama serve
    
    # Terminal 2  
    ollama pull qwen2.5:0.5b""")
        except:
            st.warning("⚠️ OLLAMA Status Unknown")
        
        # Hardware status
        if torch.cuda.is_available():
            st.success(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            st.caption(f"Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
        else:
            st.info("💻 CPU Mode")
        
        # Performance metrics if results exist
        if 'results' in st.session_state:
            st.header("📊 Statistik Sesi")
            results = st.session_state['results']
            st.metric("Total Analisis", len(results))
            st.metric("Avg Confidence", f"{results['confidence'].mean():.3f}")
            
            # Sentiment breakdown
            sentiment_counts = results['sentiment'].value_counts()
            for sentiment, count in sentiment_counts.items():
                percentage = count / len(results) * 100
                st.metric(
                    f"Sentimen {sentiment.title()}", 
                    count, 
                    f"{percentage:.1f}%"
                )
        
        st.header("📚 Panduan Penggunaan")
        with st.expander("Cara Menggunakan"):
            st.markdown("""
            1. **Input Data**: Masukkan teks manual, upload CSV, atau gunakan sample data
            2. **Analisis**: Klik tombol "Analisis Sentimen" 
            3. **Results**: Lihat visualisasi dan hasil detail
            4. **Evaluation**: Gunakan OLLAMA untuk evaluasi mendalam
            5. **Download**: Export hasil dalam format CSV
            """)
        
        with st.expander("Tips & Tricks"):
            st.markdown("""
            - **Teks berkualitas**: Gunakan kalimat lengkap untuk hasil terbaik
            - **Bahasa Indonesia**: Model dioptimalkan untuk bahasa Indonesia
            - **Confidence score**: Skor >0.7 menunjukkan prediksi yang baik
            - **CSV format**: Pastikan ada kolom 'text' untuk upload
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🐦 Analisis Sentimen Twitter Indonesia | Built with Streamlit & IndoBERT</p>
    <p>Powered by Transformers & OLLAMA</p>
</div>
""", unsafe_allow_html=True)