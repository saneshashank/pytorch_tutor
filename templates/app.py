
#!/usr/bin/env python3
import os, json, uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, make_response
from dotenv import load_dotenv

try:
    import markdown
except Exception:
    markdown = None

_USE_NEW = True
try:
    from google import genai as google_genai
    from google.genai import types as new_types
except Exception:
    google_genai = None
    new_types = None
    _USE_NEW = False

try:
    import google.generativeai as legacy_genai
except Exception:
    legacy_genai = None

load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

PYTORCH_TUTOR_SYSTEM = (
    "You are PyTorch Tutor, a precise, pragmatic assistant for learning PyTorch.\n"
    "Keep examples small and CPU-runnable. Avoid hallucinating APIs.\n"
)

MODE_HINTS = {
    "explain": "Explain concept with a short runnable example and plain-English intuition."
}

def _call_new_sdk(contents: str, temperature: float = 0.2):
    if google_genai is None or new_types is None:
        raise RuntimeError("New SDK not installed")
    client = google_genai.Client(api_key=API_KEY)
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=new_types.GenerateContentConfig(
            system_instruction=PYTORCH_TUTOR_SYSTEM,
            temperature=float(temperature),
        ),
    )
    return getattr(resp, "text", "") or ""

def _call_legacy_sdk(contents: str, temperature: float = 0.2):
    if legacy_genai is None:
        raise RuntimeError("Legacy SDK not installed")
    legacy_genai.configure(api_key=API_KEY)
    model = legacy_genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=PYTORCH_TUTOR_SYSTEM,
    )
    resp = model.generate_content(contents, generation_config={"temperature": float(temperature)})
    txt = getattr(resp, "text", None)
    if not txt and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
            txt = "".join(getattr(p, "text", "") for p in parts)
        except Exception:
            txt = ""
    return txt or ""

def call_model_text(prompt: str, temperature: float = 0.2) -> str:
    try:
        if _USE_NEW:
            return _call_new_sdk(prompt, temperature=temperature)
        return _call_legacy_sdk(prompt, temperature=temperature)
    except Exception:
        if _USE_NEW and legacy_genai is not None:
            return _call_legacy_sdk(prompt, temperature=temperature)
        raise

def safe_json_from_text(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON found in model output")
    blob = text[start:end+1]
    return json.loads(blob)

def parse_tutorial_sections(md_path: Path):
    raw = md_path.read_text(encoding="utf-8")
    sections = []
    current = None
    for line in raw.splitlines():
        if line.startswith("## "):
            if current: sections.append(current)
            current = {"title": line[3:].strip(), "content": ""}
        else:
            if current is None: continue
            current["content"] += line + "\n"
    if current: sections.append(current)
    return [s for s in sections if s["content"].strip()]

def build_lesson_from_section(section):
    prompt = f'''
You are preparing a MICRO-LESSON for a PyTorch beginner based on a tutorial section.

SECTION TITLE:
{section['title']}

SECTION MATERIAL (verbatim):
{section['content']}

REQUIREMENTS:
- Keep it minimal and runnable on CPU.
- Start with a SHORT intro (<= 120 words) that frames what the learner will do.
- Provide ONE tiny example code block in PyTorch.
- Then provide ONE practice question for the learner to solve (code or short answer).
- Provide an expected_answer (concise) and an evaluation rubric (3-6 bullet checks).
- Provide 3 progressively revealing hints (short, 1-2 lines each).

Return ONLY a JSON object with these keys exactly:
{{
  "intro": "<plain text>",
  "example": "```python\n...\n```",
  "question": "<plain text problem statement>",
  "expected_answer": "<concise description and shape expectations if any>",
  "rubric": ["...", "..."],
  "hints": ["...", "...", "..."]
}}
'''
    text = call_model_text(prompt, temperature=0.3)
    try:
        data = safe_json_from_text(text)
    except Exception:
        data = {
            "intro": "Let's learn this topic.",
            "example": "```python\nimport torch\nx = torch.tensor([1,2,3])\nprint(x.shape)\n```",
            "question": "Create a 2x3 tensor and print its shape.",
            "expected_answer": "A 2x3 tensor and torch.Size([2,3]) printed.",
            "rubric": ["Creates a tensor", "Correct shape 2x3", "Prints shape"],
            "hints": ["Use torch.tensor or torch.ones.", "Aim for shape [2,3].", "Remember print(x.shape)."]
        }
    return data

def evaluate_answer(lesson, user_answer, used_hints):
    prompt = f'''
Evaluate a learner's answer to a short PyTorch exercise.

LESSON QUESTION:
{lesson['question']}

EXPECTED ANSWER (guidance, not shown to learner):
{lesson['expected_answer']}

RUBRIC (checks):
{json.dumps(lesson['rubric'])}

LEARNER ANSWER:
{user_answer}

USED_HINTS_COUNT: {used_hints}

Return ONLY a JSON object with keys:
{{
  "correct": true/false,
  "feedback": "<1-3 sentences, plain>",
  "next_hint": "<the next hint if incorrect, else empty string>",
  "award_completion": true/false
}}

Rules:
- If correct, set award_completion=true.
- If incorrect, set award_completion=false and choose the next unused hint from the list (index = USED_HINTS_COUNT).
'''
    text = call_model_text(prompt, temperature=0.2)
    try:
        out = safe_json_from_text(text)
    except Exception:
        out = {
            "correct": False,
            "feedback": "Couldn't parse evaluation. Try simplifying your answer.",
            "next_hint": lesson["hints"][min(used_hints, len(lesson["hints"])-1)] if lesson.get("hints") else "",
            "award_completion": False
        }
    out["correct"] = bool(out.get("correct"))
    out["award_completion"] = bool(out.get("award_completion"))
    out["feedback"] = str(out.get("feedback","")).strip()[:600]
    out["next_hint"] = str(out.get("next_hint","")).strip()
    return out

SESSIONS = {}

def get_or_create_sid():
    from flask import request
    sid = request.cookies.get("tutor_sid")
    if not sid:
        sid = str(uuid.uuid4())
    return sid

def get_state(sid):
    st = SESSIONS.get(sid)
    if not st:
        sections = parse_tutorial_sections(Path("content/tutorial.md"))
        st = {
            "sections": sections,
            "current_idx": 0,
            "completed": [False]*len(sections),
            "current_lesson": None,
            "used_hints": 0,
        }
        SESSIONS[sid] = st
    return st

def summarize_progress(state):
    return {
        "total": len(state["sections"]),
        "current_idx": state["current_idx"],
        "completed": state["completed"],
        "titles": [s["title"] for s in state["sections"]],
    }

@app.route("/")
def index():
    return render_template("index.html")

def _render_md(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    if markdown is None:
        safe = raw.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        return f"<pre>{safe}</pre>"
    return markdown.markdown(raw, extensions=["fenced_code","tables","toc"])

@app.get("/tutorial")
def tutorial_page():
    html = _render_md(Path("content/tutorial.md"))
    return render_template("page.html", title="Tensor Basics Tutorial", body_html=html)

@app.get("/reference/einsum")
def einsum_page():
    html = _render_md(Path("content/einsum.md"))
    return render_template("page.html", title="Einsum Reference", body_html=html)

@app.post("/api/tutor/start")
def tutor_start():
    from flask import request
    if not API_KEY:
        return jsonify({"error": "Missing GEMINI_API_KEY in environment (.env)."}), 400
    sid = get_or_create_sid()
    st = get_state(sid)
    st["current_idx"] = 0
    st["completed"] = [False]*len(st["sections"])
    st["used_hints"] = 0
    st["current_lesson"] = build_lesson_from_section(st["sections"][0])
    resp = make_response(jsonify({
        "sid": sid,
        "progress": summarize_progress(st),
        "section_title": st["sections"][0]["title"],
        "lesson": st["current_lesson"],
    }))
    resp.set_cookie("tutor_sid", sid, httponly=True, samesite="Lax")
    return resp

@app.post("/api/tutor/answer")
def tutor_answer():
    from flask import request
    if not API_KEY:
        return jsonify({"error": "Missing GEMINI_API_KEY in environment (.env)."}), 400
    sid = get_or_create_sid()
    st = get_state(sid)
    data = request.get_json(force=True) or {}
    user_answer = str(data.get("answer","")).strip()
    if not user_answer:
        return jsonify({"error": "Empty answer"}), 400
    if st["current_lesson"] is None:
        st["current_lesson"] = build_lesson_from_section(st["sections"][st["current_idx"]])

    eval_out = evaluate_answer(st["current_lesson"], user_answer, st["used_hints"])

    if eval_out.get("correct") or eval_out.get("award_completion"):
        st["completed"][st["current_idx"]] = True
        st["used_hints"] = 0
        if st["current_idx"] + 1 < len(st["sections"]):
            st["current_idx"] += 1
            st["current_lesson"] = build_lesson_from_section(st["sections"][st["current_idx"]])
            advanced = True
        else:
            advanced = False
            st["current_lesson"] = None
        return jsonify({
            "result": eval_out,
            "advanced": advanced,
            "progress": summarize_progress(st),
            "section_title": st["sections"][st["current_idx"]]["title"] if advanced else "All sections complete",
            "lesson": st["current_lesson"],
        })
    else:
        st["used_hints"] += 1
        return jsonify({
            "result": eval_out,
            "advanced": False,
            "progress": summarize_progress(st),
            "section_title": st["sections"][st["current_idx"]]["title"],
            "lesson": st["current_lesson"],
        })

@app.post("/api/ask")
def ask():
    from flask import request
    data = request.get_json(force=True) or {}
    user_msg = str(data.get("message", "")).strip()
    if not API_KEY:
        return jsonify({"error": "Missing GEMINI_API_KEY in environment (.env)."}), 400
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400
    contents = f"[MODE: explain] {MODE_HINTS['explain']}\n\nUSER INPUT:\n{user_msg}"
    try:
        text = call_model_text(contents, temperature=0.2)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
