"""
Sistem Rekomendasi Paket Wisata Kapal Phinisi – Labuan Bajo
Implementasi Sentence-BERT + Content-Based Filtering
"""

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rekomendasi Wisata Phinisi",
    page_icon="🚢",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Kartu hasil rekomendasi */
.result-card {
    background: #ffffff;
    border: 1px solid #e8e8e8;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.dark .result-card {
    background: #1e1e1e;
    border-color: #333;
}
/* Badge skor kecocokan */
.score-badge {
    display: inline-block;
    background: #EEF2FF;
    color: #3730A3;
    font-size: 12px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
}
/* Tag metadata */
.tag {
    display: inline-block;
    background: #f3f4f6;
    color: #6b7280;
    font-size: 12px;
    padding: 2px 9px;
    border-radius: 999px;
    margin-right: 4px;
}
.rank-label {
    font-size: 11px;
    color: #9ca3af;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 2px;
}
.divider { border: none; border-top: 1px solid #f0f0f0; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🚢 Sistem Rekomendasi Paket Wisata Phinisi")
st.caption("Masukkan preferensi perjalanan Anda, kami carikan kapal Phinisi terbaik di Labuan Bajo.")
st.divider()

# ── Input form ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    selected_paket = st.selectbox(
        "🧭 Pilih jenis paket wisata",
        options=[
            "Private Trip",
            "Open Trip",
            "Family Trip",
            "Honeymoon",
            "Diving Trip",
            "Luxury Trip",
        ],
    )

with col_right:
    top_n = st.slider(
        "🔢 Jumlah rekomendasi",
        min_value=1,
        max_value=10,
        value=5,
    )

user_desc = st.text_area(
    "📝 Deskripsikan kebutuhan perjalanan Anda",
    placeholder=(
        "Contoh: snorkeling, spa, chef pribadi, raja ampat, "
        "kabin privat, honeymoon, underwater photography..."
    ),
    height=100,
)

search_clicked = st.button("🔍 Cari Rekomendasi", use_container_width=True, type="primary")

# ── Hasil rekomendasi ──────────────────────────────────────────────────────────
if search_clicked:
    if not user_desc.strip():
        st.warning("⚠️ Tuliskan deskripsi kebutuhan perjalanan terlebih dahulu.")
        st.stop()

    with st.spinner("Memproses rekomendasi..."):

        # ── Gabungkan query ────────────────────────────────────────────────────
        query = f"{selected_paket} {user_desc}"

        # ── Encode & hitung similarity ─────────────────────────────────────────
        # Ganti bagian ini dengan model & embeddings asli Anda:
        #
        #   query_embedding = model.encode([query])
        #   scores = cosine_similarity(query_embedding, embeddings)[0]
        #   top_indices = scores.argsort()[::-1][:top_n]
        #
        # Contoh di bawah menggunakan data dummy untuk demo:
        import numpy as np
        scores = np.random.uniform(0.75, 0.99, len(df))
        top_indices = scores.argsort()[::-1][:top_n]

    # ── Tampilkan hasil ────────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"✨ {top_n} Paket Terbaik untuk Anda")
    st.caption(f'Query: *"{query}"*')

    for rank, idx in enumerate(top_indices, start=1):
        item  = df.iloc[idx]
        score = scores[idx]

        # ── Kartu tiap rekomendasi ─────────────────────────────────────────────
        with st.container():
            # Gambar + konten berdampingan
            img_col, info_col = st.columns([1, 3], gap="medium")

            with img_col:
                img_url = str(item.get("image_url", ""))
                if img_url.startswith("http"):
                    st.image(img_url, use_container_width=True)
                else:
                    st.markdown(
                        "<div style='height:90px;background:#f3f4f6;border-radius:8px;"
                        "display:flex;align-items:center;justify-content:center;"
                        "font-size:2rem;'>🚢</div>",
                        unsafe_allow_html=True,
                    )

            with info_col:
                # Rank + nama kapal + badge skor
                st.markdown(
                    f"<div class='rank-label'>#{rank}</div>"
                    f"<div style='display:flex;align-items:center;gap:10px;'>"
                    f"<span style='font-size:16px;font-weight:600;'>{item['nama_kapal']}</span>"
                    f"<span class='score-badge'>⚡ {round(score, 4)}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Tag metadata
                tags_html = (
                    f"<span class='tag'>📦 {item['kategori']}</span>"
                    f"<span class='tag'>💰 {item['harga']}</span>"
                )
                st.markdown(tags_html, unsafe_allow_html=True)

                # Detail dalam kolom kecil
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown(
                        f"📍 **Destinasi**  \n{item['destinasi']}  \n"
                        f"⭐ **Layanan**  \n{item['layanan']}"
                    )
                with d2:
                    st.markdown(
                        f"🏠 **Fasilitas**  \n{item['fasilitas']}"
                    )

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
