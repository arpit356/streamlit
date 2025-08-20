import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Background Theme Demo", 
    page_icon="🎨", 
    layout="centered"
)

# Theme Selection
theme_choice = st.sidebar.selectbox(
    "Choose a Background Theme:",
    [
        "Animated Gradient (Current)",
        "Soft Pastel", 
        "Dark Professional",
        "Nature Inspired",
        "Sunset Warm",
        "Ocean Breeze"
    ]
)

# Theme CSS Definitions
themes = {
    "Animated Gradient (Current)": """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe, #00f2fe);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        min-height: 100vh;
        color: #ffffff;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        color: #ffffff !important;
    }
    h1, h2, h3 { color: #ffffff !important; text-align: center; }
    </style>
    """,
    
    "Soft Pastel": """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    .stApp {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        background-attachment: fixed;
        min-height: 100vh;
        color: #2c3e50;
        font-family: 'Poppins', sans-serif;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        color: #2c3e50 !important;
    }
    h1, h2, h3 { color: #2c3e50 !important; text-align: center; }
    </style>
    """,
    
    "Dark Professional": """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        background-attachment: fixed;
        min-height: 100vh;
        color: #ffffff;
        font-family: 'Roboto', sans-serif;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
        color: #ffffff !important;
    }
    h1, h2, h3 { color: #ffffff !important; text-align: center; }
    </style>
    """,
    
    "Nature Inspired": """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;500;600;700&display=swap');
    .stApp {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 25%, #00b894 50%, #00cec9 75%, #6c5ce7 100%);
        background-size: 300% 300%;
        animation: natureFlow 20s ease infinite;
        min-height: 100vh;
        color: #2d3436;
        font-family: 'Nunito', sans-serif;
    }
    @keyframes natureFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 2rem;
        border-radius: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        color: #2d3436 !important;
    }
    h1, h2, h3 { color: #2d3436 !important; text-align: center; }
    </style>
    """,
    
    "Sunset Warm": """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;500;600;700&display=swap');
    .stApp {
        background: linear-gradient(45deg, #ff9a9e 0%, #fecfef 25%, #fecfef 50%, #fecfef 75%, #ff9a9e 100%);
        background-size: 400% 400%;
        animation: sunsetGlow 18s ease infinite;
        min-height: 100vh;
        color: #5d4e75;
        font-family: 'Open Sans', sans-serif;
    }
    @keyframes sunsetGlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        color: #5d4e75 !important;
    }
    h1, h2, h3 { color: #5d4e75 !important; text-align: center; }
    </style>
    """,
    
    "Ocean Breeze": """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;500;700&display=swap');
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #89f7fe 50%, #66a6ff 75%, #667eea 100%);
        background-size: 300% 300%;
        animation: oceanWave 25s ease infinite;
        min-height: 100vh;
        color: #ffffff;
        font-family: 'Lato', sans-serif;
    }
    @keyframes oceanWave {
        0% { background-position: 0% 50%; }
        33% { background-position: 100% 50%; }
        66% { background-position: 0% 100%; }
        100% { background-position: 0% 50%; }
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(25px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        color: #ffffff !important;
    }
    h1, h2, h3 { color: #ffffff !important; text-align: center; }
    </style>
    """
}

# Apply selected theme
st.markdown(themes[theme_choice], unsafe_allow_html=True)

# Demo Content
st.markdown(
    f"""
    <div class="glass-card">
        <h1>🎨 Background Theme Demo</h1>
        <p style="text-align: center; font-size: 18px;">
            Currently showing: <strong>{theme_choice}</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="glass-card">
        <h2>✨ Features</h2>
        <ul>
            <li>🎭 Multiple beautiful background themes</li>
            <li>🔄 Smooth animations and transitions</li>
            <li>🪟 Glass morphism design elements</li>
            <li>📱 Responsive and mobile-friendly</li>
            <li>🎨 Professional color schemes</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# Interactive elements
col1, col2 = st.columns(2)

with col1:
    st.button("🚀 Sample Button")
    st.slider("Sample Slider", 0, 100, 50)

with col2:
    st.selectbox("Sample Dropdown", ["Option 1", "Option 2", "Option 3"])
    st.text_input("Sample Input")

st.info("💡 Use the sidebar to switch between different background themes!")
