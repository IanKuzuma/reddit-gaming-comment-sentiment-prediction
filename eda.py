# Importing libraries
import streamlit as st
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

def run():
    # Title and intro
    st.title("REDDIT r/gaming COMMENT SENTIMENT ANALYSIS - EDA OVERVIEW")

    file_ = open("opus.gif", "rb")
    contents = file_.read()
    data_url = "data:image/gif;base64," + base64.b64encode(contents).decode("utf-8")
    st.markdown(f'<img src="{data_url}" alt="gif" style="width:100%;" />', unsafe_allow_html=True)

    st.header("Project Description")
    st.markdown("""
        This project aims to classify Reddit comments from the gaming subreddit into three sentiment categories:
        **positive**, **neutral**, and **negative**. The goal is to gain insights from player discourse to inform 
        better community management and content strategies.
    """)

    # Load data
    df = pd.read_csv("r_gaming_comments_sentiments_dataset.csv")
    st.header("Dataset Preview")
    st.dataframe(df.head(10))

    # === 1. Target Distribution ===
    st.subheader("1. Target Sentiment Distribution")
    st.markdown("""
        Because this project’s objective is to predict comment sentiment accurately,
        it is crucial to understand how balanced the sentiment labels are.
    """)

    dist = df['sentiment'].value_counts(normalize=True) * 100
    fig1 = plt.figure(figsize=(6, 6))
    plt.pie(dist, labels=dist.index, autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black'})
    plt.title("Sentiment Class Distribution")
    st.pyplot(fig1)

    st.markdown("""
        The dataset is **imbalanced**, with more positive and neutral examples than negative.
        We **do not apply oversampling**, as generating synthetic Reddit comments would introduce incoherent noise.
        Instead, we use **macro-averaged F1 score** during evaluation to treat all classes equally.
    """)

    # === 2. Comment Length Analysis ===
    st.subheader("2. Comment Length Distribution")
    st.markdown("""
        Analyzing comment length helps inform the optimal input length for our tokenizer
        and helps avoid excessive padding or truncation.
    """)

    df['length'] = df['Comment'].apply(lambda x: len(str(x).split()))
    fig2 = plt.figure(figsize=(10, 4))
    sns.histplot(df['length'], bins=50, kde=True)
    plt.title("Distribution of Comment Length (in words)")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    st.pyplot(fig2)

    st.markdown("""
        - Most comments are short, but some are extremely long.
        - **Median length** is ~12 words, while the **max is over 600**.
        - We choose **128 tokens** as the truncation length for BERT to balance context retention and efficiency.
    """)

    # === 3. Top Words per Class ===
    st.subheader("3. Top Words per Sentiment Class")
    st.markdown("""
        Examining the most frequent words in each sentiment class gives insight into common expressions
        and emotional tone used in Reddit gaming discussions.
    """)

    for sentiment in df['sentiment'].unique():
        words = " ".join(df[df['sentiment'] == sentiment]['Comment']).lower()
        top_words = Counter(words.split()).most_common(10)

        st.write(f"Top words in **{sentiment}** comments:")
        st.write(top_words)

        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords='english').generate(words)
        fig_wc = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"WordCloud for {sentiment} Sentiment")
        st.pyplot(fig_wc)

    st.markdown("""
        - High-frequency **stopwords** like "the", "i", and "you" are common across all classes.
        - Word clouds reveal subtle emotional expressions — words like *hate* and *love* appear contextually.
        - These patterns motivate us to apply **stopword removal and subword embeddings** in preprocessing.
    """)

    # === Summary ===
    st.header("EDA Summary and Preprocessing Strategy")
    st.markdown("""
    - **Class imbalance** is handled by using macro-averaged F1, not sampling.
    - **Max sequence length** is set to 128 tokens based on comment length analysis.
    - **Stopword filtering** and **contextual embeddings** are prioritized over word frequency alone.

    These decisions ensure that our downstream LSTM and BERT models receive clean, meaningful input without sacrificing realism or interpretability.
    """)


if __name__ == '__main__':
    run()