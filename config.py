"""
Configuration file untuk aplikasi Analisis Sentimen
"""

# Model Configuration
MODEL_CONFIG = {
    "primary_models": [
        {
            "name": "mdhugol/indonesia-bert-sentiment-classification",
            "description": "IndoBERT fine-tuned untuk sentimen Indonesia",
            "labels": {0: 'negatif', 1: 'netral', 2: 'positif'}
        },
        {
            "name": "ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa",
            "description": "BERT Indonesia untuk analisis sentimen",
            "labels": {0: 'negatif', 1: 'netral', 2: 'positif'}
        },
        {
            "name": "w11wo/indonesian-roberta-base-sentiment-classifier",
            "description": "RoBERTa Indonesia untuk klasifikasi sentimen",
            "labels": {0: 'negatif', 1: 'netral', 2: 'positif'}
        }
    ],
    "fallback_enabled": True,
    "max_length": 128,
    "batch_size": 32
}

# OLLAMA Configuration
OLLAMA_CONFIG = {
    "host": "localhost",
    "port": 11434,
    "timeout": 90,
    "default_model": "qwen2.5:0.5b",
    "preferred_models": ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b"]
}

# UI Configuration
UI_CONFIG = {
    "page_title": "Analisis Sentimen Twitter IndoBERT & OLLAMA",
    "layout": "wide",
    "colors": {
        "positif": "#2E8B57",
        "negatif": "#DC143C", 
        "netral": "#FFD700"
    }
}

# Sample Data
SAMPLE_TEXTS = [
    "Produk ini sangat bagus dan berkualitas tinggi!",
    "Pelayanan customer service sangat mengecewakan",
    "Harga produk ini cukup standar untuk kualitasnya",
    "Saya sangat puas dengan pembelian ini, recommended!",
    "Pengiriman lambat dan produk tidak sesuai ekspektasi",
    "Aplikasi ini mudah digunakan dan fiturnya lengkap",
    "Website sering error dan susah diakses",
    "Tim support sangat responsif dan membantu",
    "Produk OK lah, tidak istimewa tapi juga tidak buruk",
    "Kecewa banget sama kualitas produknya, uang terbuang"
]
