import streamlit as st

st.title("Moje první Streamlit aplikace")
st.write("Ahoj světe! Tato aplikace funguje.")

jmeno = st.text_input("Jak se jmenuješ?")
if jmeno:
    st.write(f"Rád tě poznávám, {jmeno}!")
