from __future__ import annotations

import os
import io
import requests
import pandas as pd
import streamlit as st
from typing import Dict, Any, List

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# -----------------------------------------------------------------------------
# 1. KONFIGURACE STRÁNKY
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ČSÚ DataStat Assistant",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ČSÚ DataStat Assistant")
st.caption("Chytrý asistent pro otevřená data Českého statistického úřadu")

PERSIST_DIR = "./faiss_csu_index"
CSV_FILE = "output.csv"
DOTAZ_URL = "https://data.csu.gov.cz/api/dotaz/v1"

# -----------------------------------------------------------------------------
# 2. INICIALIZACE A CACHEVÁNÍ FAISS DB (BEZ SQLITE ZÁVISLOSTI)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_vector_db() -> FAISS:
    """Načte nebo vytvoří FAISS vektorovou databázi plně kompatibilní s Python 3.8."""
    embeddings = FastEmbedEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Pokud uložený FAISS index existuje na disku, načteme ho
    if os.path.exists(PERSIST_DIR):
        return FAISS.load_local(
            PERSIST_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )

    if not os.path.exists(CSV_FILE):
        st.error(f"Soubor `{CSV_FILE}` nebyl nalezen v adresáři!")
        st.stop()

    df = pd.read_csv(CSV_FILE)
    documents: List[Document] = [
        Document(
            page_content=f"Kód datové sady ČSÚ: {row['kod']} | Název: {row['nazev']} | Periody: {row['urovneTypObdobi']}  | Teritoria: {row['urovneTypUzemi']}",
            metadata={"kod": str(row["kod"]), "nazev": str(row["nazev"]), "perioda": str(row["urovneTypObdobi"]), "teritorium": str(row["urovneTypUzemi"])}
        )
        for _, row in df.iterrows()
    ]

    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local(PERSIST_DIR)
    return vector_db

# -----------------------------------------------------------------------------
# 3. TEXT-TO-PANDAS ANALÝZA
# -----------------------------------------------------------------------------
def analyze_csv_with_groq(df: pd.DataFrame, user_question: str, api_key: str, perioda: str, teritorium: str) -> Dict[str, Any]:
    print("df")
    df = df.drop(columns=["MJ_SYMBOL","POZNAMKA","POZNAMKA_TEXT"])    
    print(df)

    llm = ChatGroq(
        model_name="openai/gpt-oss-120b",
        temperature=0,
        groq_api_key=api_key
    )

    buffer = [
        f"Sloupce a datové typy:\n{df.dtypes.to_string()}",
        f"\nPerioda obdobi:\n{perioda}",
        f"\nUrovne teritoria uzemi:\n{teritorium}",
        f"\nUkázka prvních 3 řádků:\n{df.head(3).to_string()}",

    ]
    csv_metadata = "\n".join(buffer)

    code_prompt = ChatPromptTemplate.from_template("""
Jsi Python Data Analyst. Máš k dispozici Pandas DataFrame v proměnné `df`.

Struktura DataFrame `df`:
{csv_metadata}

Otázka uživatele: {question}

Napiš POUZE platný Python kód (bez vysvětlování, bez markdown obalů ```python), 
který agreguje výsledek z hodnot ve sloupci `Hodnota` a uloží ho do proměnné `result`.
Filtruj spravne zaznamy ze sloupce `Ukazatel` jako ekvivalenty ukazatele, na ktery se uzivatel pta.
Pokud otázka vyžaduje časový trend, potom urci spravne periodu obdobi, vyfiltruj příslušné periody a seřaď je.
Pokud otázka vyžaduje uzemni celek, potom urci spravne uroven teritoria a vyfiltruj příslušné urovne obvykle ze sloupce `Území`.
Pokud jsou v dataframe dalsi sloupce, ktere nevyuzivas ve filtrovani, potom na ne pridej filtr na hodnotu `Celkem` nebo null.
Nikdy ale nefitruj tyto sloupce [`MJ_TEXT`].
""")

    print(f"code_prompt1:{code_prompt}")
    code_chain = code_prompt | llm | StrOutputParser()
    raw_code = code_chain.invoke({"csv_metadata": csv_metadata, "question": user_question})
    clean_code = raw_code.replace("```python", "").replace("```", "").strip()

    local_vars = {"df": df}
    try:
        exec(clean_code, {}, local_vars)
        result = local_vars.get("result", "Kód nevytvořil proměnnou 'result'.")
    except Exception as e:
        return {
            "summary": f"Chyba při vykonávání analytického dotazu: {e}",
            "code": clean_code,
            "result_data": None
        }

    summary_prompt = ChatPromptTemplate.from_template("""
Uživatel se ptal: {question}

Zde jsou přesně spočítaná data z celého souboru CSV (proměnná result):
{result}

Vysvětli uživateli stručně, přehledně a lidsky v češtině, jaký je závěr.
""")
    print(f"summary_prompt2:{summary_prompt}")
    summary_chain = summary_prompt | llm | StrOutputParser()
    summary_text = summary_chain.invoke({"question": user_question, "result": str(result)})

    return {
        "summary": summary_text,
        "code": clean_code,
        "result_data": result
    }

# -----------------------------------------------------------------------------
# 4. APPLIKACE
# -----------------------------------------------------------------------------
groq_key_input = os.environ.get("GROQ_API_KEY", "")

# Zde se bezpečně zavolá funkce
with st.spinner("Načítání FAISS databáze katalogu..."):
    vector_db = get_vector_db()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Dobrý den! Jaká data ČSÚ vás zajímají? (např. *'Jaký je trend v rozvodovosti mezi roky 2020 a 2025?'*)"
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "dataframe" in msg:
            st.dataframe(msg["dataframe"], use_container_width=True)

# -----------------------------------------------------------------------------
# 5. CHAT PIPELINE
# -----------------------------------------------------------------------------
if user_query := st.chat_input("Zadejte váš dotaz..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.status("🔍 Vyhledávám a analyzuji data ČSÚ...", expanded=True) as status:
            
            status.write("1. Hledám datovou sadu ve FAISS indexu...")
            search_results = vector_db.similarity_search(user_query, k=3)
            context_str = "\n".join([doc.page_content for doc in search_results])

            status.write("2. Určuji kód sady přes Groq...")
            llm_selector = ChatGroq(
                model_name="openai/gpt-oss-120b",
                temperature=0,
                groq_api_key=groq_key_input
            )
            
            prompt_select = ChatPromptTemplate.from_template("""
Jsi asistent pro ČSÚ. Vyber kód sady odpovídající dotazu. Odhadni spravne periodu obdobi a uroven uzemniho teritoria.

Otázka: {question}
Katalog: {context}

Vrať POUZE JSON:
{{
    "selected_code": "kód sady",
    "dataset_name": "název sady",
    "perioda": "perioda",
    "teritorium": "teritorium"
}}
""")
            chain_select = prompt_select | llm_selector | JsonOutputParser()
            match_res = chain_select.invoke({"question": user_query, "context": context_str})
            
            sada_kod = match_res.get("selected_code")
            sada_nazev = match_res.get("dataset_name")
            perioda = match_res.get("perioda")
            teritorium = match_res.get("teritorium")
            
            api_url = f"{DOTAZ_URL}/data/sady/{sada_kod}?format=CSV"
            status.markdown(f"3. Nalezena sada  [`{sada_kod}`]({api_url}) ({sada_nazev}), perioda {perioda}, teritorium {teritorium}")

            status.write("4. Stahuji CSV z ČSÚ API...")                        
            csv_df = None
            try:
                response = requests.get(api_url, timeout=15)
                if response.status_code == 200:
                    csv_df = pd.read_csv(io.StringIO(response.text))
                    status.write(f"5. Data úspěšně stažena ({len(csv_df)} řádků).")
                else:
                    status.update(label="⚠️ Nepodařilo se stáhnout data z API", state="error")
            except Exception as e:
                status.update(label=f"⚠️ Chyba komunikace s API: {e}", state="error")

            if csv_df is not None:
                status.write("6. Spouštím Pandas analýzu přes Groq...")
                analysis_result = analyze_csv_with_groq(csv_df, user_query, groq_key_input, perioda, teritorium)
                status.update(label="✅ Analýza dokončena!", state="complete", expanded=False)

        if csv_df is not None:
            st.markdown(analysis_result["summary"])
            
            with st.expander("🛠️ Zobrazit vygenerovaný analytický Python kód"):
                st.code(analysis_result["code"], language="python")

            if isinstance(analysis_result["result_data"], pd.DataFrame): 
                st.markdown("### 📊 Data vysledku:")
                st.dataframe(analysis_result["result_data"], use_container_width=True, hide_index=True)

            st.markdown("### 📊 Ukázka datového souboru:")
            st.dataframe(csv_df.head(10), use_container_width=True, hide_index=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": analysis_result["summary"],
                "dataframe": csv_df.head(10)
            })
        else:
            err_msg = "Omlouvám se, ale nepodařilo se mi stáhnout požadovaná data z API ČSÚ."
            st.markdown(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})