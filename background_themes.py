# Background Theme Options for Streamlit Apps
# Copy and paste any of these themes into your app's CSS section

# Theme 1: Animated Gradient (Current in app1.py)
ANIMATED_GRADIENT = """
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
</style>
"""

# Theme 2: Soft Pastel Gradient
SOFT_PASTEL = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
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
</style>
"""

# Theme 3: Dark Professional
DARK_PROFESSIONAL = """
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
</style>
"""

# Theme 4: Nature Inspired
NATURE_INSPIRED = """
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
</style>
"""

# Theme 5: Sunset Warm
SUNSET_WARM = """
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
</style>
"""

# Theme 6: Ocean Breeze
OCEAN_BREEZE = """
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
</style>
"""
