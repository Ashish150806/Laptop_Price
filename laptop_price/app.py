import streamlit as st
import pickle
import numpy as np

# --- Page config ---
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main { background-color: #fafafa; }

    .block-container {
        max-width: 720px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1 { font-weight: 700 !important; letter-spacing: -0.5px; }

    .subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: -1rem;
        margin-bottom: 2rem;
    }

    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stSlider > div {
        border-radius: 8px !important;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(26, 26, 46, 0.3);
    }

    .price-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }

    .price-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.3rem;
    }

    .price-value {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -1px;
    }

    .divider {
        height: 1px;
        background: #e5e7eb;
        margin: 1.5rem 0;
        border: none;
    }

    .section-label {
        color: #374151;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Load model ---
pipe = pickle.load(open('models/pipe.pkl', 'rb'))
df = pickle.load(open('models/df.pkl', 'rb'))

# --- Header ---
st.title("Laptop Price Predictor")
st.markdown('<p class="subtitle">Get an estimated price based on laptop specifications.</p>', unsafe_allow_html=True)

# --- Form ---
st.markdown('<p class="section-label">Basic Info</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    company = st.selectbox('Brand', sorted(df['Company'].unique()))
with col2:
    type_name = st.selectbox('Type', sorted(df['TypeName'].unique()))

col3, col4 = st.columns(2)
with col3:
    cpu = st.selectbox('Processor', df['Cpu brand'].unique())
with col4:
    gpu = st.selectbox('Graphics', df['Gpu brand'].unique())

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">Display</p>', unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 15.6, step=0.1)
with col6:
    resolution = st.selectbox('Resolution', [
        '1920x1080', '1366x768', '1600x900', '3840x2160',
        '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
    ])

col7, col8 = st.columns(2)
with col7:
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
with col8:
    ips = st.selectbox('IPS Display', ['No', 'Yes'])

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">Hardware</p>', unsafe_allow_html=True)

col9, col10, col11 = st.columns(3)
with col9:
    ram = st.selectbox('RAM (GB)', sorted(df['Ram'].unique()))
with col10:
    weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=1.5, step=0.1)
with col11:
    os_choice = st.selectbox('OS', df['os'].unique())

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --- Predict ---
if st.button('Predict Price'):
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size

    query = np.array([company, type_name, ram, weight, touchscreen_val, ips_val, ppi, cpu, gpu, os_choice], dtype=object)
    query = query.reshape(1, -1)

    predicted_price = int(np.exp(pipe.predict(query)[0]))

    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">Estimated Price</div>
        <div class="price-value">{predicted_price:,}</div>
    </div>
    """, unsafe_allow_html=True)
