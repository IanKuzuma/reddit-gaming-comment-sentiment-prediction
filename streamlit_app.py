import prediction, eda
import streamlit as st
import base64

# Create sidebar navigation
with st.sidebar:
    st.subheader('PAGE NAVIGATION')

    # Select page
    page = st.selectbox('Choose Page', ['Data Exploration', 'Sentiment Prediction'])

    # Sidebar info
    st.subheader('About')

    # Load and encode the gif (optional — replace jorb.gif with something project-relevant if needed)
    file_ = open("jorb.gif", "rb")
    contents = file_.read()
    data_url = "data:image/gif;base64," + base64.b64encode(contents).decode("utf-8")
    st.markdown(f'<img src="{data_url}" alt="gif" style="width:25%;" />', unsafe_allow_html=True)

    st.write(" ")
    st.markdown('''This project, developed by :blue-background[Ian Ladityarsa], aims to classify Reddit gaming comments into one of three sentiment classes — Negative, Neutral, or Positive — using a deep learning model powered by BERT and BiLSTM. The model can be used for social listening, community management, and strategy alignment.''')

# Main page dispatcher
if page == 'Data Exploration':
    eda.run()
else:
    prediction.run()