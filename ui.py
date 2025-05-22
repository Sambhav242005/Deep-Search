import json, traceback
from datetime import datetime
import requests
import gradio as gr

from backend.constant import BASE_OLLAMA
from backend.main import deep_search
from backend.ollama_client import _ask_ollama

# ─── Configurations ────────────────────────────────────────────────────────
DEFAULT_MODELS = [
    "llama3.2", "llama3.2:1b", "llama3.2:3b", "llama3.1", "llama3.1:8b",
    "mistral", "mixtral", "codellama", "phi3", "qwen2", "gemma2"
]
EXAMPLE_SCHEMAS = {
    "Summary Report": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "key_points": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["summary", "key_points"]
    },
    # ... other schemas ...
}

# ─── Helpers ──────────────────────────────────────────────────────────────

def get_models():
    try:
        resp = requests.get(f"{BASE_OLLAMA}/api/tags", timeout=3).json()
        names = [m["name"] for m in resp.get("models", [])]
        return names + DEFAULT_MODELS
    except:
        return DEFAULT_MODELS


def format_plan(plan):
    if not plan: return ""
    out = "**Search Plan:**\n"
    for i,(q,n,kw) in enumerate(plan,1):
        out += f"{i}. {q} ({n} results)" + (f" Keywords: {', '.join(kw)}" if kw else "") + "\n"
    return out


def format_sources(srcs):
    if not srcs: return ""
    out = "**Sources:**\n"
    for i,s in enumerate(srcs,1):
        out += f"{i}. [{s.title or 'Doc'}]({s.href})\n"
    return out


def perform_search(q, model, auto, k, schema_type, custom, progress=gr.Progress()):
    if not q.strip(): return "Enter a query.","","",""
    # parse schema
    schema = None if schema_type=='None' else (json.loads(custom) if schema_type=='Custom' else EXAMPLE_SCHEMAS.get(schema_type))
    try:
        ans, srcs, plan = deep_search(q, model, k=k, auto=auto, schema=schema)
        return (
            ans,
            format_plan(plan),
            format_sources(srcs),
            json.dumps({
                "query": q, "answer": ans,
                "sources": [{"title": s.title, "url": s.href} for s in srcs],
                "plan": plan, "time": datetime.now().isoformat()
            }, indent=2)
        )
    except Exception as e:
        tb = traceback.format_exc()
        return f"Error: {e}\n"


def test_conn(model):
    try:
        _ask_ollama(model, "Ping", max_retries=1)
        return f"Connected to {model}"  
    except Exception as e:
        return f"Conn failed: {e}"


def preview_schema(t, custom):
    if t=='None': return "No schema"
    try:
        sch = json.loads(custom) if t=='Custom' else EXAMPLE_SCHEMAS.get(t)
        return f"```json\n{json.dumps(sch, indent=2)}\n```"
    except Exception as e:
        return f"Invalid JSON: {e}"

# ─── UI ────────────────────────────────────────────────────────────────────

def create_interface():
    css = ".container{max-width:900px;} .src{background:#f0f0f0;padding:4px;}"
    with gr.Blocks(title="Deep Search", css=css) as app:
        gr.Markdown("# 🔍 Deep Search Tool")
        with gr.Tab("Search"):
            q = gr.Textbox(label="Question", lines=2)
            m = gr.Dropdown(get_models(), value="llama3.2", label="Model")
            auto = gr.Checkbox(value=True, label="Auto-plan")
            k = gr.Slider(1,15,5, label="Results")
            st = gr.Dropdown(["None"]+list(EXAMPLE_SCHEMAS)+["Custom"], value="None", label="Schema")
            cs = gr.Code(language="json", visible=False)
            gr.Button("Search").click(
                perform_search, inputs=[q,m,auto,k,st,cs],
                outputs=[gr.Markdown(), gr.Markdown(), gr.Markdown(), gr.Textbox()]
            )
        with gr.Tab("Settings"):
            gr.Button("Test Connection").click(test_conn, inputs=[m], outputs=[gr.Markdown()])
            gr.Button("Preview Schema").click(preview_schema, inputs=[st,cs], outputs=[gr.Markdown()])
        with gr.Tab("Export"):
            fmt = gr.Radio(["markdown","json","csv"], value="markdown", label="Format")
            out = gr.Code()
            dl = gr.DownloadButton("Download")
        return app

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    app = create_interface()
    print(f"Running on http://{args.host}:{args.port}")
    app.launch(server_name=args.host, server_port=args.port)
