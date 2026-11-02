import os
import io
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def analyze_csv_with_groq(df: pd.DataFrame, user_question: str, api_key: str) -> str:
    # 1. Inicializace ChatGroq s přímým předáním API klíče
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=api_key  # Přímé předání API klíče
    )

    # 2. Příprava struktury a ukázky dat pro LLM
    buffer = [
        f"Sloupce a datové typy:\n{df.dtypes.to_string()}",
        f"\nUkázka prvních 3 řádků:\n{df.head(3).to_string()}"
    ]
    csv_metadata = "\n".join(buffer)

    # 3. Prompt pro vygenerování ČISTÉHO Pandas Python kódu
    code_prompt = ChatPromptTemplate.from_template("""
Jsi Python Data Analyst. Máš k dispozici Pandas DataFrame v proměnné `df`.

Struktura DataFrame `df`:
{csv_metadata}

Otázka uživatele: {question}

Napiš POUZE platný Python kód (bez vysvětlování, bez markdown obalů ```python), 
který spočítá výsledek a uloží ho do proměnné `result`.
Pokud otázka vyžaduje časový trend, vyfiltruj příslušné roky a seřaď je.
""")

    # 4. Generování kódu
    code_chain = code_prompt | llm | StrOutputParser()
    raw_code = code_chain.invoke({"csv_metadata": csv_metadata, "question": user_question})
    
    # Očištění případných markdown značek
    clean_code = raw_code.replace("```python", "").replace("```", "").strip()

    # 5. Bezpečné spuštění kódu v lokálním kontextu
    local_vars = {"df": df}
    try:
        exec(clean_code, {}, local_vars)
        result = local_vars.get("result", "Kód nevytvořil proměnnou 'result'.")
    except Exception as e:
        return f"Chyba při vykonávání dotazu: {e}\nGenerovaný kód byl:\n{clean_code}"

    # 6. Shrnutí výsledků pomocí Groq
    summary_prompt = ChatPromptTemplate.from_template("""
Uživatel se ptal: {question}

Zde jsou přesně spočítaná data z celého souboru CSV:
{result}

Vysvětli uživateli stručně a přehledně v češtině, jaký je závěr (např. zda má rozvodovost stoupající nebo klesající trend).
""")
    summary_chain = summary_prompt | llm | StrOutputParser()
    
    return summary_chain.invoke({"question": user_question, "result": str(result)})