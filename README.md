# Deep Search

> AI-Powered Contextual Research Tool

**Deep Search** combines DuckDuckGo web searches with local Ollama language models to deliver structured, source-backed answers for any question.

---

## 🚀 Features

* **Auto-Planning**: Let the AI generate optimized sub-queries and keywords.
* **Multiple Models**: Choose from a default set of models or detect installed Ollama models.
* **Structured Output**: Use predefined JSON schemas (Summary Report, Research Analysis, Fact Check) or define your own.
* **Source Citations**: Inline links and snippets for every result.
* **Export Formats**: Markdown, JSON, or CSV exports with timestamps.
* **Gradio Interface**: Intuitive web UI, no coding required.

---

## 📦 Prerequisites

* Python 3.8+
* [Ollama](https://ollama.com/) running locally with at least one model installed

---

## 🔧 Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/your-org/deep-search.git
   cd deep-search
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Ollama daemon is running:

   ```bash
   ollama serve
   ```

---

## ⚙️ Configuration

* **Environment Variables**:

  * `BASE_OLLAMA` (default: [http://localhost:11434](http://localhost:11434))

* **DEFAULT\_MODELS**: List in `app.py` is merged with detected Ollama models.

* **EXAMPLE\_SCHEMAS**: Predefined in `app.py` under `EXAMPLE_SCHEMAS`.

---

## 🏁 Usage

1. Launch the UI:

   ```bash
   python app.py --host 0.0.0.0 --port 7860
   ```
2. Open your browser at `http://localhost:7860` (or your host/port).
3. Enter a research question, select options, and click **Search**.
4. View answer, plan, and sources.
5. Export results via the **Export** tab.

---

## 📂 Project Structure

```
.
├── app.py                # Main Gradio UI
├── backend/              # Core search logic
│   ├── main.py           # deep_search implementation
│   ├── ollama_client.py  # _ask_ollama wrapper
│   ├── constant.py       # BASE_OLLAMA, SearchResult
├── requirements.txt      # Python dependencies
└── README.md             # This file
```
