import streamlit as st
st.title("this is my first project on  Streamlit!")
st.write("this is the first project on Streamlit")
slider_value = st.slider("Select a value", 0, 100, 50)
st.write(f"You selected: {slider_value}!")
#checkbox
st.title("Checkbox Example")
checkbox_value = st.checkbox("Check me!")
if checkbox_value:
    st.write("Checkbox is checked!")        
else:
    st.write("Checkbox is not checked!")
# For text

st.title('Interactive Text Input Example')

user_text = st.text_input('Enter your text:')
st.write(f'You entered: {user_text}')