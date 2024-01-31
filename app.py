import streamlit as st
import datetime
from lorem_text import lorem

st.set_page_config(
    page_title = 'Mari belajar Streamlit',
    layout='wide'
)

# Menulis teks di streamlit

st.write("Hello world!")

st.markdown("## ini ditulis menggunakan _markdown_")

"Hello world!" # <- magic

"_Ini juga hello world_"

"**Ini juga**"

"# Ini adalah header"

"## ini adalah subheader"

st.title("Ini judul")
st.header("Ini juga header")

st.caption("Langit senja minum kopi")

st.code("import streamlit as st")

st.code('''
import pandas as pd
import streamlit as st # ini untuk memanggil package streamlit
''')

# Latex
st.latex("ax^2 + bx + c = 0")

# WIDGET <- Input elements

ini_tombol = st.button("Tekan tombol ini")
ini_tombol

saya_setuju = st.checkbox("Centang jika setuju")
if saya_setuju:
    st.write("anda setuju untuk belajar lebih giat")
else:
    st.write("Ayo belajar")

# radio button, memilih salah satu opsi dari sekian opsi
buah_favorit = st.radio(
    "Pilihan buah favorit kamu",
    ['Apel','Anggur','Jeruk','Mangga']
)
buah_favorit

makanan = st.selectbox(
    "Pilih makanan yang akan diorder",
    ['Paket 1','Paket Goceng','Kids meal']
)
makanan

belanjaan = st.multiselect(
    "Pilih item belanjaan kalian",
    ['Terigu','Minyak goreng','Biskuit','Minuman']
)
belanjaan
belanjaan[0]

parameter_alpha = st.slider(
    "Insert alpha value",
    min_value=0.0,
    max_value=1.0,
    step=0.1,
    value=0.5
)
parameter_alpha

ukuran_baju = st.select_slider(
    "Ukuran baju",
    ['SS','S','M','L','XL','XXL']
)
ukuran_baju

kode_pos = st.number_input(
    "Masukkan nomor hp kalian",
    min_value=0,
    max_value=999999999,
    step=1
)
kode_pos

nama = st.text_input("Masukkan nama kalian")

komentar = st.text_area("Masukkan komentar kamu")
komentar

tanggal_jadian = st.date_input(
    "Tanggal lahir",
    min_value=datetime.date(1990,1,1)

)

jam_mulai = st.time_input("masukkan jam mulai")

warna = st.color_picker("masukkan warna")
warna

# masukkan image, video, sama suara

# container and layouting

# Kolom

col1, col2, col3 = st.columns(3)

with col1:
    lahir_saya = st.date_input("Tanggal lahir kamu")

with col2:
    lahir_gebetan = st.date_input("Tanggal lahir dia")

with col3:
    jadian = st.date_input("Tangal jadian")

st.button("Hitung")

kol1, kol2 = st.columns([2,6])

with kol1:
    lahir_aku = st.date_input("Tanggal lahir aku")

with kol2:
    lahir_kamu = st.date_input("Tanggal lahir dirinya")

with st.sidebar:
    st.title("Titanic survival model explorer")
    your_name = st.text_input("Enter your name")

    with st.expander("Lorem ipsum"):
        st.write(lorem.paragraphs(1))

tab1, tab2, tab3 = st.tabs(['Tab1', 'Tab2', 'Tab3'])

with tab1:
    st.write(lorem.paragraphs(1))

with tab2:
    st.write(lorem.paragraphs(2))

with tab3:
    st.write(lorem.paragraphs(3))

with st.container(): # bagian metrics
    st.write("ini teks di dalam container")
    st.write("ini teks di dalam container")
    st.write("ini teks di dalam container")
    st.write("ini teks di dalam container")
    st.write("ini teks di dalam container")
    st.write("ini teks di dalam container")
    st.write("ini teks di dalam container")
    st.write("ini teks di dalam container")

st.write("ini text di luar container")

st.markdown(''' div.stButton > button:first-child {
background-color: #00cc00;color:white;font-size:20px;height:3em;width:30em;border-radius:10px 10px 10px 10px;
}
''', unsafe_allow_html=True)