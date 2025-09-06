import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import json
from pathlib import Path
import tensorflow as tf
from streamlit_option_menu import option_menu
import base64

# ======================== CONFIG ==========================
st.set_page_config(
    page_title="🍅 Tomato Leaf Disease Classifier",
    page_icon="🍅",
    layout="wide"
)

MODEL_PATH = Path("models/best_model.keras")   # gunakan file .keras terbaru
LABEL_PATH = Path("models/class_labels.json")
IMG_SIZE = (256, 256)

# ====================== LOAD MODEL =========================
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        # load tanpa compile → inference-only
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        # Debug: tampilkan info model
        st.write("✅ Model berhasil dimuat!")
        st.write(f"📊 Model input shape: {model.input_shape}")
        st.write(f"📈 Model output shape: {model.output_shape}")
        
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.write("💡 Kemungkinan penyebab:")
        st.write("- File model tidak ditemukan")
        st.write("- Model architecture tidak kompatibel")
        st.write("- Versi TensorFlow berbeda saat training vs inference")
        return None

model = load_model()

# ===================== LOAD LABELS =========================
def load_labels():
    if LABEL_PATH.exists():
        try:
            with open(LABEL_PATH, "r") as f:
                data = json.load(f)
            classes = data.get("classes", [])
            if classes:  # kalau json ada isinya
                return classes
        except Exception as e:
            st.warning(f"⚠️ Gagal load labels: {e}")
    
    # fallback default - sesuaikan dengan jumlah class yang umum
    default_classes = [
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight", 
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ]
    
    if model:
        num_classes = model.output_shape[-1]
        if len(default_classes) >= num_classes:
            return default_classes[:num_classes]
        else:
            return [f"Class_{i}" for i in range(num_classes)]
    
    return default_classes

class_labels = load_labels()

# ================= LABEL CLEANER ===========================
def clean_label(lbl: str) -> str:
    lbl = lbl.replace("Tomato", "").replace("___", " ").strip()
    lbl = lbl.replace("_", " ")
    return " ".join(part.capitalize() for part in lbl.split())

# ================== CLASS IMAGE MAPPING ====================
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
    "Bacterial Spot": "Penyakit ini disebabkan oleh bakteri *Xanthomonas campestris pv. vesicatoria*. "
    "Gejalanya berupa bercak kecil berwarna cokelat kehitaman pada daun yang terkadang dikelilingi halo kuning. "
    "Infeksi juga dapat menyebar ke buah. Dampaknya adalah daun mudah rontok sehingga proses fotosintesis terganggu dan hasil panen berkurang.",

    "Early Blight": "Penyakit bercak daun awal ini disebabkan oleh jamur *Alternaria solani*. "
    "Gejalanya berupa bercak cokelat dengan pola lingkaran konsentris menyerupai cincin pada daun tua. "
    "Seiring waktu, daun menguning dan rontok. Dampaknya membuat luas daun hijau berkurang, buah lebih kecil, dan tanaman menjadi lemah.",

    "Late Blight": "Penyakit hawar daun lanjut disebabkan oleh jamur semu *Phytophthora infestans*. "
    "Gejalanya berupa bercak hijau gelap atau cokelat berair pada daun yang cepat meluas. "
    "Pada kondisi lembab sering muncul lapisan jamur putih di tepi bercak. "
    "Penyakit ini sangat merusak karena dapat membunuh tanaman hanya dalam beberapa hari.",

    "Leaf Mold": "Penyakit bercak jamur ini disebabkan oleh *Passalora fulva*. "
    "Gejalanya berupa bercak kuning di permukaan atas daun, sementara di bagian bawah daun muncul lapisan beludru berwarna hijau keabu-abuan. "
    "Akibatnya, daun mengering dan tanaman kekurangan energi untuk tumbuh optimal.",

    "Septoria Leaf Spot": "Penyakit ini disebabkan oleh jamur *Septoria lycopersici*. "
    "Gejalanya berupa bercak kecil berbentuk bulat berwarna cokelat dengan pusat abu-abu pucat, biasanya muncul pada daun tua. "
    "Penyakit ini sering mempercepat kerontokan daun, terutama pada lingkungan dengan kelembaban tinggi.",

    "Spider Mites": "Gangguan ini disebabkan oleh serangan hama *Tetranychus urticae*. "
    "Gejalanya berupa daun yang menguning dengan bercak kecil, serta adanya jaring halus di bagian bawah daun. "
    "Dampaknya adalah penurunan fotosintesis, tanaman melemah, dan dalam kondisi parah daun bisa kering serta mati.",

    "Target Spot": "Penyakit bercak sasaran disebabkan oleh jamur *Corynespora cassiicola*. "
    "Gejalanya adalah bercak cokelat dengan lingkaran konsentris yang mirip sasaran tembak. "
    "Penyakit ini dapat menyebabkan kerontokan daun yang berat, terutama ketika kondisi lingkungan lembab.",

    "Yellow Leaf Curl Virus": "Penyakit ini disebabkan oleh virus yang ditularkan oleh kutu kebul *Bemisia tabaci*. "
    "Gejalanya berupa daun yang menguning, menggulung ke atas, serta pertumbuhan tanaman yang terhambat sehingga menjadi kerdil."
    " Dampaknya adalah tanaman sulit berbuah atau menghasilkan buah yang kecil sehingga merugikan petani.",

    "Mosaic Virus": "Penyakit ini disebabkan oleh *Tomato mosaic virus (ToMV)* atau *Tobacco mosaic virus (TMV)*. "
    "Gejalanya berupa daun belang dengan pola hijau tua dan muda (mosaik), keriting, serta pertumbuhan yang tidak normal. "
    "Akibatnya, tanaman menjadi lemah dan hasil panen menurun.",

    "Healthy": "Tanaman tomat yang sehat memiliki daun hijau segar tanpa bercak, tidak mengalami penggulungan ataupun perubahan warna. "
    "Pertumbuhan tanaman berjalan normal sehingga mampu menghasilkan buah dengan baik."
}

def resolve_image_path(p: str) -> Path | None:
    if not p:
        return None
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

# ================== PREDICT FUNCTION =======================
def preprocess_image(image: Image.Image):
    """
    Preprocessing gambar untuk model CNN
    """
    try:
        # Resize dan pastikan RGB
        img = image.resize(IMG_SIZE).convert("RGB")
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Pastikan shape (256, 256, 3)
        if img_array.ndim == 2:  # grayscale → ubah jadi 3 channel
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 4:  # RGBA → buang alpha
            img_array = img_array[..., :3]
        elif img_array.shape[-1] != 3:
            raise ValueError(f"Unexpected number of channels: {img_array.shape[-1]}")

        # Expand ke batch dimension (1, 256, 256, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        st.error(f"❌ Error dalam preprocessing: {e}")
        return None

def predict_image(img: Image.Image):
    """
    Melakukan prediksi pada gambar
    """
    if model is None:
        st.error("❌ Model tidak tersedia!")
        return None, None
        
    try:
        # Preprocessing
        x = preprocess_image(img)
        if x is None:
            return None, None
        
        # Debug info
        with st.expander("🔍 Debug Info"):
            st.write(f"📐 Input shape: {x.shape}")
            st.write(f"📊 Input dtype: {x.dtype}")
            st.write(f"📈 Input range: [{x.min():.3f}, {x.max():.3f}]")
        
        # Prediksi
        with st.spinner("🔄 Sedang melakukan prediksi..."):
            preds = model.predict(x, verbose=0)
            
        # Validasi output
        if preds is None or len(preds) == 0:
            st.error("❌ Model tidak menghasilkan output!")
            return None, None
            
        # Ambil hasil untuk batch pertama
        pred_probs = preds[0]
        
        # Debug output
        with st.expander("📋 Prediction Details"):
            st.write(f"📊 Raw predictions shape: {preds.shape}")
            st.write(f"🎯 Prediction probabilities: {pred_probs}")
            st.write(f"📈 Max probability: {np.max(pred_probs):.4f}")
            st.write(f"🏆 Predicted class index: {np.argmax(pred_probs)}")
        
        # Validasi jumlah class
        if len(pred_probs) != len(class_labels):
            st.warning(f"⚠️ Mismatch: Model output {len(pred_probs)} classes, labels have {len(class_labels)}")
            
        # Ambil label prediksi
        predicted_idx = np.argmax(pred_probs)
        if predicted_idx < len(class_labels):
            predicted_label = class_labels[predicted_idx]
        else:
            predicted_label = f"Unknown_Class_{predicted_idx}"
            
        return pred_probs, predicted_label
        
    except Exception as e:
        st.error(f"❌ Error dalam prediksi: {e}")
        st.write("💡 Kemungkinan penyebab:")
        st.write("- Input shape tidak sesuai dengan yang diharapkan model")
        st.write("- Model architecture bermasalah")
        st.write("- Preprocessing gambar gagal")
        return None, None

# ======================= NAVBAR ============================
with st.container():
    selected = option_menu(
        menu_title="",
        options=["Beranda", "Deteksi Tanaman"],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#1e1e1e"},
            "icon": {"display":"none"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#333333",
            },
            "nav-link-selected": {"background-color": "#ff6f61", "color": "white"},
        },
    )

st.markdown("""<style>.nav-link::before { display: none !important; }</style>""", unsafe_allow_html=True)

# ====================== MAIN PAGE ==========================
if selected == "Beranda":
    st.title("🍅 Tomato Leaf Disease Classifier")
    st.markdown(
        """
        <div style="padding:20px; background-color:#2c2c2c; border-radius:10px; margin-bottom:20px; color:#f1f1f1;">
        <h3>Selamat Datang Di Website</h3>
        <p>Aplikasi ini menggunakan model <b>Convolutional Neural Network (CNN)</b> 
        untuk mendeteksi penyakit pada daun tomat secara otomatis</p>
        <p>🔍 Pada halaman ini terdapat 9 jenis penyakit tanaman tomat beserta deskripsi penyakitnya</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if class_labels:
        st.subheader("📋 Daftar Kelas")
        st.markdown(f"📊 Total kelas: {len(class_labels)}")
        st.markdown("<br>", unsafe_allow_html=True)

        cols = st.columns(3)
        for idx, raw_lbl in enumerate(class_labels):
            clean_lbl = clean_label(raw_lbl)
            mapped = CLASS_IMAGES.get(clean_lbl)
            img_path = resolve_image_path(mapped) if mapped else None
            desc = CLASS_DESCRIPTIONS.get(clean_lbl, "Deskripsi belum tersedia.")

            with cols[idx % 3]:
                # Judul center
                st.markdown(
                    f"""
                    <div style='display:flex; justify-content:center; align-items:center;'>
                        <div style='font-size:18px; font-weight:bold; margin-bottom:8px; text-align:center;'>
                            {clean_lbl}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if img_path:
                    try:
                        # Gambar center + kasih jarak bawah
                        img_b64 = base64.b64encode(open(img_path, "rb").read()).decode()
                        st.markdown(
                            f"""
                            <div style='display:flex; justify-content:center; margin-bottom:15px;'>
                                <img src="data:image/png;base64,{img_b64}" width="200" style="border-radius:8px;">
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.markdown(
                            f"<div style='padding:15px; background:#333; border-radius:8px; text-align:center; margin-bottom:15px;'>❌ Error loading image: {str(e)[:50]}...</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        f"<div style='padding:15px; background:#333; border-radius:8px; text-align:center; margin-bottom:15px;'>❌ (Gambar tidak ditemukan)</div>",
                        unsafe_allow_html=True
                    )

                # Deskripsi justify biar rapi
                with st.expander("📖 Deskripsi Tanaman"):
                    st.markdown(f"<div style='text-align:justify; line-height:1.6;'>{desc}</div>", unsafe_allow_html=True)

elif selected == "Deteksi Tanaman":
    st.title("📸 Deteksi Penyakit Daun Tomat")
    
    # Cek apakah model tersedia
    if model is None:
        st.error("❌ Model tidak dapat dimuat. Silakan periksa file model dan coba lagi.")
        st.stop()

    # Pilih sumber input
    option = st.radio("📂 Pilih sumber gambar:", ["Upload Gambar", "Gunakan Kamera"])

    img = None

    if option == "Upload Gambar":
        uploaded_file = st.file_uploader("📤 Upload Gambar Daun Tomat", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            try:
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption="📷 Gambar yang diupload", use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error membuka gambar: {e}")

    elif option == "Gunakan Kamera":
        camera_file = st.camera_input("📷 Ambil foto dengan kamera")
        if camera_file:
            try:
                img = Image.open(camera_file).convert("RGB")
                st.image(img, caption="📷 Foto dari kamera", use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error membuka foto dari kamera: {e}")

    # Prediksi kalau ada gambar
    if img is not None:
        if st.button("🔍 Jalankan Prediksi", type="primary"):
            preds, label = predict_image(img)
            if preds is not None and label is not None:
                clean_lbl = clean_label(label)
                confidence = np.max(preds)
                
                # Hasil prediksi
                st.success(f"🎯 **Hasil Prediksi: {clean_lbl}**")
                st.info(f"📊 **Confidence: {confidence:.2%}**")
                
                # Chart prediksi
                st.subheader("📈 Distribusi Probabilitas")
                
                # Buat DataFrame untuk chart
                chart_data = pd.DataFrame({
                    'Class': [clean_label(cls) for cls in class_labels],
                    'Probability': preds
                })
                chart_data = chart_data.sort_values('Probability', ascending=True)
                
                st.bar_chart(chart_data.set_index('Class')['Probability'])
                
                # Top 3 prediksi
                st.subheader("🏆 Top 3 Prediksi")
                top_indices = np.argsort(preds)[-3:][::-1]
                
                for i, idx in enumerate(top_indices):
                    rank = ["🥇", "🥈", "🥉"][i]
                    class_name = clean_label(class_labels[idx])
                    prob = preds[idx]
                    st.write(f"{rank} **{class_name}**: {prob:.2%}")
                
                # Deskripsi hasil
                desc = CLASS_DESCRIPTIONS.get(clean_lbl, "Deskripsi belum tersedia.")
                if desc != "Deskripsi belum tersedia.":
                    st.subheader("📋 Deskripsi")
                    st.markdown(f"<div style='text-align:justify; line-height:1.6; padding:15px; background-color:#2c2c2c; border-radius:8px;'>{desc}</div>", unsafe_allow_html=True)
            else:
                st.error("❌ Gagal melakukan prediksi. Silakan coba lagi dengan gambar yang berbeda.")

    else:
        st.info("📷 Silakan upload gambar atau ambil foto untuk memulai deteksi.")
