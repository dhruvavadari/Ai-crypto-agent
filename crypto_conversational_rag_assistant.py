import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np
import torch

# Load datasets
def load_data():
    df1 = pd.read_csv("mining_performance_dataset_90_days.csv")
    df2 = pd.read_csv("asic_specifications_dataset.csv")
    return df1, df2

# Convert rows to text
def dataframe_to_texts(df, label):
    return [f"{label} Row {i}: " + ", ".join(f"{col}: {val}" for col, val in row.items()) for i, row in df.iterrows()]

# Build FAISS index
def build_faiss_index(texts, model):
    embeddings = model.encode(texts, show_progress_bar=False)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings

# Retrieve chunks
def retrieve_chunks(query, texts, model, index, top_k=5):
    query_vec = model.encode([query])[0].astype("float32")
    D, I = index.search(np.array([query_vec]), top_k)
    return [texts[i] for i in I[0]]

# Detect if query is math-based
def is_math_query(question: str) -> bool:
    q = question.lower()
    keywords = [
        "total", "monthly", "sum", "average", "income", "revenue",
        "earned", "profit", "cost", "electricity", "daily", "mined",
        "produced", "who mined the most", "most mined"
    ]
    return any(word in q for word in keywords)

# Calculate mined totals per miner
def mined_by_each_miner(df: pd.DataFrame) -> str:
    if 'ASIC Model' in df.columns and 'Daily Mined' in df.columns:
        summary = df.groupby('ASIC Model')['Daily Mined'].sum().sort_values(ascending=False)
        result = "Total mined by each ASIC:\n"
        for model, total in summary.items():
            result += f"- {model}: {round(total, 2)} coins\n"
        return result
    elif 'Miner ID' in df.columns and 'Daily Mined' in df.columns:
        summary = df.groupby('Miner ID')['Daily Mined'].sum().sort_values(ascending=False)
        result = "Total mined by each miner:\n"
        for miner, total in summary.items():
            result += f"- {miner}: {round(total, 2)} coins\n"
        return result
    else:
        return "Could not calculate mined totals. Required columns missing."

# Math-based answers
def answer_math_question(df: pd.DataFrame, question: str) -> str:
    q = question.lower()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    if "who mined the most" in q or "most mined" in q:
        if 'ASIC Model' in df.columns and 'Daily Mined' in df.columns:
            summary = df.groupby('ASIC Model')['Daily Mined'].sum()
            top_miner = summary.idxmax()
            amount = summary.max()
            return f"{top_miner} mined the most with a total of {round(amount, 2)} coins."
        elif 'Miner ID' in df.columns and 'Daily Mined' in df.columns:
            summary = df.groupby('Miner ID')['Daily Mined'].sum()
            top_miner = summary.idxmax()
            amount = summary.max()
            return f"{top_miner} mined the most with a total of {round(amount, 2)} coins."
        return "Required columns to calculate mined totals are missing."

    if "mined" in q and ("each" in q or "per" in q or "by" in q):
        return mined_by_each_miner(df)

    if "monthly" in q and "profit" in q:
        last_month = df['Date'].max().month - 1
        monthly_df = df[df['Date'].dt.month == last_month]
        if 'Daily Profit (USD)' in df.columns:
            total = monthly_df['Daily Profit (USD)'].sum()
            return f"Monthly profit for last month: ${round(total, 2)}"

    if "monthly" in q and "revenue" in q:
        last_month = df['Date'].max().month - 1
        monthly_df = df[df['Date'].dt.month == last_month]
        if 'Daily Revenue (USD)' in df.columns:
            total = monthly_df['Daily Revenue (USD)'].sum()
            return f"Monthly revenue for last month: ${round(total, 2)}"

    if "daily profit" in q:
        if 'Daily Profit (USD)' in df.columns:
            avg = df['Daily Profit (USD)'].mean()
            return f"Average daily profit: ${round(avg, 2)}"

    if "daily revenue" in q:
        if 'Daily Revenue (USD)' in df.columns:
            avg = df['Daily Revenue (USD)'].mean()
            return f"Average daily revenue: ${round(avg, 2)}"

    if "total profit" in q or q.strip() == "profit":
        if 'Daily Profit (USD)' in df.columns:
            total = df['Daily Profit (USD)'].sum()
            return f"Total profit: ${round(total, 2)}"

    if "total cost" in q or "electricity" in q or q.strip() == "cost":
        if 'Daily Electricity Cost (USD)' in df.columns:
            total = df['Daily Electricity Cost (USD)'].sum()
            return f"Total electricity cost: ${round(total, 2)}"

    if "total" in q and ("earned" in q or "revenue" in q or "income" in q):
        if 'Daily Revenue (USD)' in df.columns:
            total = df['Daily Revenue (USD)'].sum()
            return f"Total revenue: ${round(total, 2)}"

    if "income" in q or "revenue" in q:
        if 'Daily Revenue (USD)' in df.columns:
            total = df['Daily Revenue (USD)'].sum()
            return f"Total revenue: ${round(total, 2)}"

    return "Couldn't compute. Please rephrase the question."

# Load small conversational model (CPU-safe)
def load_llm():
    model_name = "tiiuae/falcon-rw-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(torch.device("cpu"))  # Ensure CPU use
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

# Handle a question (math or chat)
def ask_question(query, df_perf, df_specs, all_texts, embed_model, index, llm):
    if is_math_query(query):
        return answer_math_question(df_perf, query)

    chunks = retrieve_chunks(query, all_texts, embed_model, index)
    context = "\n".join(set(chunks))
    prompt = f"""You are a helpful assistant. Here's some context:

{context}

User: {query}
Assistant:"""
    response = llm(prompt)[0]['generated_text']
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    return response

# Main chat loop
if __name__ == "__main__":
    df_perf, df_specs = load_data()
    all_texts = dataframe_to_texts(df_perf, "Performance") + dataframe_to_texts(df_specs, "Specs")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    index, _ = build_faiss_index(all_texts, embed_model)
    llm = load_llm()

    print("🤖 Crypto AI Assistant (CPU + Falcon RW + RAG + Math). Type 'exit' to quit.")
    while True:
        q = input("\nAsk your crypto question: ")
        if q.lower() == 'exit':
            break
        print("\nAnswer:", ask_question(q, df_perf, df_specs, all_texts, embed_model, index, llm))
