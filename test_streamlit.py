import streamlit as st

st.title('My first Streamlit app')

st.write('Hello, *World!* :sunglasses:')

st.sidebar.header('Sidebar')
st.sidebar.selectbox('Select an option:', 
                      ['A', 'B', 'C'])
                      
st.write('Slider')
st.slider('Select a value')

st.write('Checkbox')  
st.checkbox('Check me!')

st.button('Click me')
