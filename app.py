import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import tensorflow as tf
from streamlit_option_menu import option_menu
import base64

# --- Import tambahan untuk memuat model Keras 3 ---
from tensorflow.keras.layers import TFSMLayer
from tensorflow.keras.models import Sequential

# ======================== KONFIGURASI ==========================
st.set_page_config(
    page_title="🍅 Tomato Leaf Disease Classifier",
    page_icon="🍅",
    layout="wide"
)

# --- Path model diubah ke folder best_model_tf ---
MODEL_PATH = Path("models/best_model_tf")
IMG_SIZE = (256, 256)

# ====================== MEMUAT MODEL & LABEL =========================

@st.cache_resource(show_spinner="Memuat model AI...")
def load_tf_model():
    """
    Memuat model dari format TensorFlow SavedModel (folder) menggunakan TFSMLayer
    yang kompatibel dengan TensorFlow/Keras versi terbaru.
    """
    try:
        if not MODEL_PATH.exists():
            st.error(f"Folder model tidak ditemukan di: {MODEL_PATH}")
            return None
        
        # Buat layer inferensi dari folder model
        inference_layer = TFSMLayer(str(MODEL_PATH), call_endpoint='serving_default')
        
        # Bungkus layer di dalam model Sequential agar mudah digunakan
        model = Sequential([inference_layer])
        return model
        
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

def load_manual_labels():
    """
    Menyediakan daftar kelas (label) secara manual.
    Urutan ini harus sama persis dengan urutan saat training model.
    """
    return [
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___healthy',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
    ]

# Memuat model dan label saat aplikasi dimulai
model = load_tf_model()
class_labels = load_manual_labels()

# ================= FUNGSI UTILITAS (TIDAK BERUBAH) =================

def clean_label(lbl: str) -> str:
    """Membersihkan nama label mentah menjadi format yang mudah dibaca."""
    lbl = lbl.replace("Tomato___", "").replace("_", " ").strip()
    if "Two-spotted spider mite" in lbl:
        return "Spider Mites"
    return " ".join(part.capitalize() for part in lbl.split())

CLASS_IMAGES = {
    "Bacterial Spot": "img/bacterial_spot.JPG",
    "Early Blight": "img/early_blight.JPG",
    "Late Blight": "img/late_blight.JPG",
    "Leaf Mold": "img/leaf_mold.JPG",
    "Septoria Leaf Spot": "img/septoria_leaf_spot.JPG",
    "Spider Mites": "img/spider_mites.JPG",
    "Target Spot": "img/target_spot.JPG",
    "Yellow Leaf Curl Virus": "img/yellow_leaf_curl_virus.JPG",
    "Mosaic Virus": "img/mosaic_virus.JPG",
    "Healthy": "img/healthy.JPG",
}

CLASS_DESCRIPTIONS = {
    "Bacterial Spot": "Penyakit ini disebabkan oleh bakteri *Xanthomonas campestris pv. vesicatoria*. Gejalanya berupa bercak kecil berwarna cokelat kehitaman pada daun yang terkadang dikelilingi halo kuning. Infeksi juga dapat menyebar ke buah. Dampaknya adalah daun mudah rontok sehingga proses fotosintesis terganggu dan hasil panen berkurang.",
    "Early Blight": "Penyakit bercak daun awal ini disebabkan oleh jamur *Alternaria solani*. Gejalanya berupa bercak cokelat dengan pola lingkaran konsentris menyerupai cincin pada daun tua. Seiring waktu, daun menguning dan rontok. Dampaknya membuat luas daun hijau berkurang, buah lebih kecil, dan tanaman menjadi lemah.",
    "Late Blight": "Penyakit hawar daun lanjut disebabkan oleh jamur semu *Phytophthora infestans*. Gejalanya berupa bercak hijau gelap atau cokelat berair pada daun yang cepat meluas. Pada kondisi lembab sering muncul lapisan jamur putih di tepi bercak. Penyakit ini sangat merusak karena dapat membunuh tanaman hanya dalam beberapa hari.",
    "Leaf Mold": "Penyakit bercak jamur ini disebabkan oleh *Passalora fulva*. Gejalanya berupa bercak kuning di permukaan atas daun, sementara di bagian bawah daun muncul lapisan beludru berwarna hijau keabu-abu-an. Akibatnya, daun mengering dan tanaman kekurangan energi untuk tumbuh optimal.",
    "Septoria Leaf Spot": "Penyakit ini disebabkan oleh jamur *Septoria lycopersici*. Gejalanya berupa bercak kecil berbentuk bulat berwarna cokelat dengan pusat abu-abu pucat, biasanya muncul pada daun tua. Penyakit ini sering mempercepat kerontokan daun, terutama pada lingkungan dengan kelembaban tinggi.",
    "Spider Mites": "Gangguan ini disebabkan oleh serangan hama *Tetranychus urticae*. Gejalanya berupa daun yang menguning dengan bercak kecil, serta adanya jaring halus di bagian bawah daun. Dampaknya adalah penurunan fotosintesis, tanaman melemah, dan dalam kondisi parah daun bisa kering serta mati.",
    "Target Spot": "Penyakit bercak sasaran disebabkan oleh jamur *Corynespora cassiicola*. Gejalanya adalah bercak cokelat dengan lingkaran konsentris yang mirip sasaran tembak. Penyakit ini dapat menyebabkan kerontokan daun yang berat, terutama ketika kondisi lingkungan lembab.",
    "Yellow Leaf Curl Virus": "Penyakit ini disebabkan oleh virus yang ditularkan oleh kutu kebul *Bemisia tabaci*. Gejalanya berupa daun yang menguning, menggulung ke atas, serta pertumbuhan tanaman yang terhambat sehingga menjadi kerdil. Dampaknya adalah tanaman sulit berbuah atau menghasilkan buah yang kecil sehingga merugikan petani.",
    "Mosaic Virus": "Penyakit ini disebabkan oleh *Tomato mosaic virus (ToMV)* atau *Tobacco mosaic virus (TMV)*. Gejalanya berupa daun belang dengan pola hijau tua dan muda (mosaik), keriting, serta pertumbuhan yang tidak normal. Akibatnya, tanaman menjadi lemah dan hasil panen menurun.",
    "Healthy": "Tanaman tomat yang sehat memiliki daun hijau segar tanpa bercak, tidak mengalami penggulungan ataupun perubahan warna. Pertumbuhan tanaman berjalan normal sehingga mampu menghasilkan buah dengan baik."
}

def resolve_image_path(p: str) -> Path | None:
    base = Path(p)
    if base.exists():
        return base
    exts = [".JPG", ".jpg", ".jpeg", ".png", ".PNG"]
    stem = base.with_suffix("")
    for ext in exts:
        cand = stem.with_suffix(ext)
        if cand.exists():
            return cand
    return None

# ================== FUNGSI PREDIKSI =======================
def preprocess_image(image: Image.Image):
    """Mempersiapkan gambar untuk input model."""
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    # Hapus alpha channel jika ada (misal dari gambar PNG)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    return np.expand_dims(img_array, axis=0)

def predict_image(img: Image.Image):
    """Melakukan prediksi pada gambar dan mengembalikan hasilnya."""
    if model is None:
        st.error("Model tidak berhasil dimuat.")
        return None, None
        
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img, verbose=0)

    # Menangani output model dari TFSMLayer yang mungkin berbentuk dictionary
    if isinstance(predictions, dict):
        # Ambil tensor output dari dictionary
        output_key = next(iter(predictions))
        preds = predictions[output_key][0]
    else:
        preds = predictions[0]

    top_index = np.argmax(preds)
    top_label = class_labels[top_index] if class_labels else str(top_index)
    
    return preds, top_label

# ======================= UI APLIKASI ============================
# Navbar
with st.container():
    selected = option_menu(
        menu_title=None,
        options=["Beranda", "Deteksi Tanaman"],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#1e1e1e"},
            "icon": {"display": "none"},
            "nav-link": {
                "font-size": "16px", "text-align": "center", "margin": "0px",
                "--hover-color": "#333333",
            },
            "nav-link-selected": {"background-color": "#ff6f61", "color": "white"},
        },
    )

# Halaman Beranda
if selected == "Beranda":
    st.title("🍅 Tomato Leaf Disease Classifier")
    st.markdown(
        """
        <div style="padding:20px; background-color:#2c2c2c; border-radius:10px; margin-bottom:20px; color:#f1f1f1;">
        <h3>Selamat Datang di Website Deteksi Penyakit Tomat</h3>
        <p>Aplikasi ini menggunakan model <b>Convolutional Neural Network (CNN)</b> untuk mendeteksi 10 jenis kondisi daun tomat (9 penyakit dan 1 kondisi sehat) secara otomatis. Silakan jelajahi daftar penyakit di bawah ini atau pindah ke halaman "Deteksi Tanaman" untuk mencoba modelnya.</p>
        </div>
        """, unsafe_allow_html=True
    )

    if class_labels:
        st.subheader("Daftar Jenis Penyakit")
        st.markdown("<br>", unsafe_allow_html=True)

        cols = st.columns(3)
        for idx, raw_lbl in enumerate(sorted(class_labels)): # Diurutkan agar 'Healthy' tidak di tengah
            clean_lbl = clean_label(raw_lbl)
            mapped = CLASS_IMAGES.get(clean_lbl)
            img_path = resolve_image_path(mapped) if mapped else None
            desc = CLASS_DESCRIPTIONS.get(clean_lbl, "Deskripsi belum tersedia.")

            with cols[idx % 3]:
                st.markdown(f"<h4 style='text-align: center;'>{clean_lbl}</h4>", unsafe_allow_html=True)
                
                if img_path:
                    try:
                        img_bytes = base64.b64encode(open(img_path, "rb").read()).decode()
                        st.markdown(
                            f"<div style='display:flex; justify-content:center; margin-bottom:15px;'><img src='data:image/jpeg;base64,{img_bytes}' width='200'></div>",
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.warning(f"Gagal memuat gambar untuk {clean_lbl}")
                
                with st.expander("Lihat Deskripsi"):
                    st.markdown(f"<div style='text-align:justify; line-height:1.6;'>{desc}</div>", unsafe_allow_html=True)
                st.markdown("---")


# Halaman Deteksi
elif selected == "Deteksi Tanaman":
    st.title("🔎 Deteksi Penyakit Daun Tomat")
    uploaded_file = st.file_uploader("📤 Unggah Gambar Daun Tomat Di Sini", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1.2])

        with col1:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="📷 Gambar Anda", use_column_width=True)

        with col2:
            if st.button("Jalankan Prediksi", use_container_width=True, type="primary"):
                with st.spinner("🧠 Menganalisis gambar..."):
                    preds, label = predict_image(img)
                
                if preds is not None:
                    st.success(f"**Hasil Deteksi: {clean_label(label)}**")
                    
                    st.markdown("**Deskripsi Penyakit:**")
                    st.info(CLASS_DESCRIPTIONS.get(clean_label(label), "Deskripsi tidak tersedia."))
                    
                    st.markdown("**Distribusi Probabilitas:**")
                    prob_df = pd.DataFrame({
                        'Kelas': [clean_label(cls) for cls in class_labels],
                        'Probabilitas': preds * 100
                    }).sort_values(by="Probabilitas", ascending=False).reset_index(drop=True)
                    
                    st.dataframe(prob_df, use_container_width=True, hide_index=True, column_config={
                        "Probabilitas": st.column_config.ProgressColumn(
                            "Probabilitas (%)",
                            format="%.2f%%",
                            min_value=0,
                            max_value=100,
                        )
                    })
