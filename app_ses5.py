import streamlit as st

st.set_page_config(
    page_title = 'Belajar Streamlit'
    ,layout='wide'
)

# Menulis teks
# st.write("Hello world!")

st.title("Title")
st.header("Header")
st.subheader("Subheader")
st.text("Text")
st.code("import streamlit as st")
st.latex("y = 2x^2 + 3x + 5")
# Hands on activity: adjust text as class context

# markdown text
st.markdown("# Title")
st.markdown("_italic_")
st.markdown("__bold__")
st.markdown("""
    1. Number 1
    2. Number 2
    3. Number 3
""")

# Quizz
# 1. How to write bullet points?
# 2. How to write text with hyperlink?

# WIDGET <- Input elements
st.header("Widget")

ini_tombol = st.button("Tekan tombol ini")
ini_tombol

saya_setuju = st.checkbox("Centang jika setuju")
if saya_setuju:
    st.write("Anda setuju untuk belajar lebih giat")
else:
    st.write("Ayo belajar")

# radio button, memilih salah satu opsi dari sekian opsi
buah_favorit = st.radio(
    "Pilihan buah favorit kamu",
    ['Apel','Anggur','Jeruk','Mangga']
)
buah_favorit
st.write(f"Buah kesukaanku adalah {buah_favorit}")

# selectbox menampilkan menu dropdown untuk dipilih
makanan = st.selectbox(
    "Pilih makanan yang akan diorder",
    ['Paket 1','Paket Goceng','Kids meal']
)
st.write(f"Anda akan mengorder {makanan}")

# multiselect memungkinkan kita memilih multiple choices
belanjaan = st.multiselect(
    "Pilih item belanjaan kalian",
    ['Terigu','Minyak goreng','Biskuit','Minuman']
)
belanjaan
if len(belanjaan) > 0:
    st.write(f"Pilihan pertama Anda adalah {belanjaan[0]}")

# slider memungkinkan kita untuk memilih numerik input dengan range yang ditentukan
parameter_alpha = st.slider(
    "Insert alpha value",
    min_value=0.0,
    max_value=1.0,
    step=0.1,
    value=0.5
)
st.write(f"Parameter yang dipilih adalah {parameter_alpha}")

# Jika input berupa kategorikal, kita dapat menggunakan select_slider
ukuran_baju = st.select_slider(
    "Ukuran baju",
    ['SS','S','M','L','XL','XXL']
)
st.write(f"Ukuran baju yang dipilih adalah {ukuran_baju}")

# number_input untuk input number
kode_pos = st.number_input(
    "Masukkan kode pos",
    min_value=0,
    max_value=999999,
    step=1
)
kode_pos

# Text input
nama = st.text_input("Masukkan nama kalian")
nama

# Text area
komentar = st.text_area("Masukkan komentar kamu")
komentar

# Input tanggal
import datetime

tanggal_lahir = st.date_input(
    "Tanggal lahir"
)
tanggal_lahir

# Input waktu
jam_mulai = st.time_input("masukkan jam mulai", step=600)
jam_mulai

# Input warna
warna = st.color_picker("masukkan warna")
warna
# Use HTML and inline CSS to set the color of the text
styled_text = f'<span style="color:{warna};">Ini adalah text dengan warna yang dipilih.</span>'
# Display the text using st.write
st.write(styled_text, unsafe_allow_html=True)

# Menampilkan media

# Image
from PIL import Image

image = Image.open("streamlit_app\media\sunrise.jpeg")
st.image(
    image,
    caption="A beautiful sunrise"
)

# Home exercise
# Tampilkan audio
# Tampilkan video

# container and layouting

# Kolom

col1, col2, col3 = st.columns(3)

with col1:
    lahir_saya = st.date_input("Tanggal lahir kamu")

with col2:
    lahir_gebetan = st.date_input("Tanggal lahir dia")

with col3:
    jadian = st.date_input("Tangal jadian")

# kolom dengan lebar custom
kol1, kol2 = st.columns([2,6])

with kol1:
    lahir_aku = st.date_input("Tanggal lahir aku")

with kol2:
    lahir_kamu = st.date_input("Tanggal lahir dirinya")

# sidebar
with st.sidebar:
    st.title("Titanic survival model explorer")
    your_name = st.text_input("Enter your name")

    with st.expander("Lorem ipsum"):
        st.write("dolor sit amet")

# Tabbing
tab1, tab2, tab3 = st.tabs(['Tab1', 'Tab2', 'Tab3'])

with tab1:
    st.write("Tab 1: Lorem ipsum dolor sit amet")

with tab2:
    st.write("Tab 2: Lorem ipsum dolor sit amet")

with tab3:
    st.write("Tab 3: Lorem ipsum dolor sit amet")

# Container
custom_container = st.container(border=True)
with custom_container:
    st.write("ini teks di dalam container")

st.write("SEDANGKAN teks ini di luar container")

# Basic plotting
import pandas as pd

# Bar chart
df = pd.DataFrame({
    "name": ["A","B","C"],
    "value": [1,2,3]
})

st.bar_chart(data=df, x="name", y="value")

# Line plot
df = pd.DataFrame({
    "date": pd.date_range(start='2024-01-01', end='2024-01-10', freq='D'),
    "value": [1,2,3,4,3,2,1,2,3,4]
})

st.line_chart(data=df, x="date", y="value")

# scatter plot
import numpy as np
np.random.seed(42)
random_numbers = np.random.uniform(0, 10, 200)

df = pd.DataFrame({
    "value1": random_numbers[:100],
    "value2": random_numbers[100:]
})

st.scatter_chart(data=df, x="value1", y="value2")

# Home exercise: berikan centered title pada setiap plot di atas

# Selesai, tekan control+C pada terminal untuk mengakhiri app
# Lalu exit untuk turn off virtual environment
