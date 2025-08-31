import streamlit as st 
from App.app import chain as chatbot


st.header("hi there  ,i'am here  to answer your question.")




question = st.text_input("whis your question : ")
if question :
    answer = chatbot.invoke(question)
    st.write(answer)








