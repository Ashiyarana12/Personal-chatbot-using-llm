
# 🤖 Personal Chatbot Using LLM + Audio 📚🔊

This is a **Streamlit-powered intelligent chatbot** that uses **NVIDIA's LLM APIs** to:
- Perform semantic book search using embeddings
- Generate book summaries using LLaMA 3.1
- Provide voice-based narration using NVIDIA's TTS service

🔗 **Live Repo:** [github.com/Ashiyarana12/Personal-chatbot-using-llm](https://github.com/Ashiyarana12/Personal-chatbot-using-llm)

---

## 🚀 Features

- 📂 Upload book CSVs
- 🔍 Perform smart search using **NVIDIA's EmbedQAv5**
- ✍️ Auto-generate summaries via **LLaMA 3.1**
- 🔊 Convert responses to **spoken audio**
- ⚡ Uses **FAISS** for fast vector retrieval
- 🧠 Works with natural language queries

---

## 🧾 CSV Format Example

Ensure your CSV follows this format:

```csv
Author's Name,Title,Publisher / Place,Year
J.K. Rowling,Harry Potter,Bloomsbury,1997
George Orwell,1984,Secker & Warburg,1949
```

---

## 🛠️ Tech Stack

| Layer       | Tech Used                            |
|-------------|---------------------------------------|
| UI          | Streamlit                            |
| Embeddings  | NVIDIA NIM `nv-embedqa-e5-v5`         |
| LLM         | Meta LLaMA-3.1 via NIM Chat API       |
| TTS         | NVIDIA TTS via gRPC (`talk.py`)       |
| Storage     | FAISS, NumPy                          |
| Language    | Python                                |

---

## 📁 Folder Structure

```bash
.
├── test.py                 # Main Streamlit app
├── audio.wav              # Audio summary file
├── metadata.npy           # Stored metadata (book info)
├── faiss_index            # FAISS vector index
├── python-clients/        # NVIDIA TTS gRPC client
├── logo2.png              # Logo (optional)
└── README.md              # Project documentation
```

---

## 💡 How It Works

1. **Upload CSV** → App generates embedding vectors
2. **Enter a search query** → FAISS returns top-k results
3. **Summary Generation** → LLM gives clean descriptions
4. **Audio Creation** → TTS reads out the results

---

## 🧪 Running the App

### 1. Clone the repo

```bash
git clone https://github.com/Ashiyarana12/Personal-chatbot-using-llm.git
cd Personal-chatbot-using-llm
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

> ⚠️ Make sure you install FAISS (`faiss-cpu`) and NVIDIA’s `openai`-compatible client.

### 3. Add your NVIDIA API Key

Replace placeholders in `test.py`:

```python
api_key = "nvapi-..."  # Your NVIDIA NIM API key
```

Update it in the TTS `talk.py` call too.

### 4. Launch the app

```bash
streamlit run test.py
```

---

## 🔐 NVIDIA API Usage

Your NIM API key must support:

- 🧠 `nv-embedqa-e5-v5` (Embeddings)
- ✍️ `meta/llama-3.1-8b-instruct` (Chat)
- 🔊 NVIDIA TTS (gRPC using `talk.py`)

---

## 👨‍💻 Author

**Ashiya Rana**  
📎 GitHub: [Ashiyarana12](https://github.com/Ashiyarana12)

---

## 📄 License

This project is licensed under the MIT License.
