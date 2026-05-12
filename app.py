import streamlit as st
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Rekomendasi Wisata Phinisi",
    page_icon="🚢",
    layout="wide"
)


# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():

    df = pd.read_csv(
        "dataset_kapal_preprocessing.csv"
    )

    df.columns = df.columns.str.strip()

    return df


# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


# ==========================================
# CREATE EMBEDDING
# ==========================================
@st.cache_resource
def create_embeddings(
    data,
    model
):

    texts = data[
        "processed_text"
    ].fillna("").astype(str).tolist()

    embeddings = model.encode(
        texts
    )

    return embeddings


# ==========================================
# LOAD
# ==========================================
df = load_data()

model = load_model()

embeddings = create_embeddings(
    df,
    model
)


# ==========================================
# HEADER
# ==========================================
st.title(
    "🚢 Sistem Rekomendasi Paket Wisata Phinisi"
)

st.write(
    "Cari paket wisata berdasarkan jenis trip dan deskripsi perjalanan."
)


# ==========================================
# INPUT
# ==========================================
col1, col2 = st.columns(
    2
)

with col1:

    selected_paket = st.selectbox(

        "Pilih Paket Wisata",

        [

            "Private Trip",

            "Open Trip",

            "Family Trip",

            "Honeymoon",

            "Diving Trip",

            "Luxury Trip"

        ]

    )


with col2:

    top_n = st.slider(

        "Jumlah Rekomendasi",

        1,

        10,

        5

    )


user_desc = st.text_area(

    "Deskripsikan kebutuhan perjalanan",

    placeholder="Contoh: snorkeling, spa, chef pribadi, honeymoon, sunset dinner"

)


# ==========================================
# SEARCH
# ==========================================
if st.button(
    "Cari Rekomendasi"
):

    if user_desc.strip() == "":

        st.warning(
            "Masukkan deskripsi perjalanan."
        )

        st.stop()


    # query user
    query = (
        selected_paket
        + " " +
        user_desc
    )


    # ======================================
    # SBERT
    # ======================================
    query_embedding = model.encode(
        [query]
    )


    # ======================================
    # COSINE SIMILARITY
    # ======================================
    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]


    top_indices = scores.argsort()[
        ::-1
    ][:top_n]


    # ======================================
    # OUTPUT
    # ======================================
    st.subheader(
        "Paket Terbaik Untuk Anda"
    )


    for rank, idx in enumerate(
        top_indices,
        start=1
    ):

        item = df.iloc[idx]


        col_img, col_info = st.columns(
            [1, 3]
        )


        with col_img:

            img_url = str(
                item.get(
                    "image_url",
                    ""
                )
            )

            if img_url.startswith(
                "http"
            ):

                st.image(
                    img_url,
                    use_container_width=True
                )


        with col_info:

            st.markdown(
                f"""
                ### #{rank} {item['nama_kapal']}

                **Kategori:** {item.get('kategori','-')}

                **Harga:** {item.get('harga','-')}

                **Destinasi:** {item.get('destinasi','-')}

                **Layanan:** {item.get('layanan','-')}

                **Fasilitas:** {item.get('fasilitas','-')}

                **Skor Kecocokan:** {round(scores[idx],4)}
                """
            )

            st.divider()
