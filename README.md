# 🧠 Conversational Crypto Mining Assistant

An intelligent assistant that answers natural language queries about ASIC miners and crypto mining performance using a Retrieval-Augmented Generation (RAG) architecture.

## 🔧 Features
- Semantic search with FAISS and SentenceTransformers
- Conversational LLM (Falcon-RW) for natural interaction
- Pandas math engine for computing KPIs like profit, revenue, and efficiency
- Supports questions like:
  - "Who mined the most coins?"
  - "What’s the monthly profit?"
  - "Tell me about the Antminer S19 Pro."

## 📁 Dataset
Place the following CSV files under the `datasets/` folder:
- `mining_performance_dataset_90_days.csv`
- `asic_specifications_dataset.csv`

## 🛠 Requirements



## 🔜 Upcoming
Memory for multi-turn conversations

Voice assistant integration

Real-time mining API

pip install -r requirements.txt
