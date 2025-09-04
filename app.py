#!/usr/bin/env python3
import os, json, uuid, re
from pathlib import Path
from flask import Flask, render_template, request, jsonify, make_response

try:
    import markdown
except Exception:
    markdown = None

# --- LLM Setup (for Gemini) ---
try:
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-2.5-flash') # Changed from 'gemini-pro'
except Exception as e:
    print(f"Warning: Could not configure Gemini API: {e}")
    genai = None
    model = None

# Resolve absolute folders so Flask always finds templates/static
BASE = Path(__file__).resolve().parent
TEMPLATES = BASE / "templates"
STATIC = BASE / "static"
CONTENT = BASE / "content"

app = Flask(
    __name__,
    template_folder=str(TEMPLATES),
    static_folder=str(STATIC),
    static_url_path="/static",
)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ------------------ Tutorial parsing ------------------
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

# ------------------ Local lessons & evaluation ------------------
def lesson_for_title(title: str, content: str):
    t = title.lower()
    lesson = {
        "intro": f"In this section, we explore: {title}. You'll get a clear explanation, a tiny runnable example, then a different practice task.",
        "example": "```python\n# Example will appear here\n```",
        "question": "Describe or write code per the instructions.",
        "expected_answer": "",
        "rubric": [],
        "hints": ["Focus on shapes.", "Use functions shown in the tutorial.", "Keep code minimal."]
    }
    if "what is a tensor" in t or "tensor?" in t:
        lesson.update({
            "intro": "A tensor is an N-D array with shape and dtype. Rank (ndim) is number of axes. Typical vision shapes are [N,C,H,W].",
            "example": "```python\nimport torch\nx = torch.tensor([[1.,2.,3.],[4.,5.,6.]])\nprint(x.shape)  # torch.Size([2, 3])\nprint(x.dtype)  # torch.float32\nprint(x.ndim)   # 2\n```",
            "question": "Create a 3x2 float tensor and print its shape, dtype, and ndim.",
            "expected_answer": "torch.Size([3, 2]) printed; dtype float; ndim 2",
            "rubric": ["creates 3x2", "prints shape", "prints dtype", "prints ndim"],
            "hints": ["Use torch.tensor([...]).", "Access x.shape, x.dtype, x.ndim.", "Matrices have ndim=2."]
        })
    elif "indexing" in t or "slicing" in t:
        lesson.update({
            "intro": "Index selects positions (t[i,j]); slices select ranges (start:stop). Negative indices work from the end; slices are often views.",
            "example": "```python\nimport torch\nT = torch.tensor([[10,11,12,13],[20,21,22,23],[30,31,32,33]])\nprint(T[-1, :])    # last row\nprint(T[:, 1:3])   # columns 1..2\n```",
            "question": "Give expressions for: (1) top-left 2x2 block, (2) last column slice.",
            "expected_answer": "T[:2, :2] and T[:, -1]",
            "rubric": ["top-left block", "last column slice"],
            "hints": ["Use :2 on rows/cols.", "Last col is -1.", "Shapes should be [2,2] then [3]."]
        })
    elif "reshape" in t and "view" in t:
        lesson.update({
            "intro": "reshape/view change shape without copy when possible; -1 infers a dimension. Flatten then reshape is common.",
            "example": "```python\nimport torch\nm = torch.arange(12).reshape(3,4)\nprint(m.reshape(-1).shape)\nprint(m.reshape(4,3).shape)\n```",
            "question": "With a = torch.arange(12).reshape(2,3,2), flatten then reshape to [3,4].",
            "expected_answer": "a.reshape(-1) then a.reshape(3,4)",
            "rubric": ["flatten once", "reshape to 3,4"],
            "hints": ["Use a.reshape(-1).", "12 elements → shape [3,4].", "Intermediate variable is fine."]
        })
    elif "unsqueeze" in t or "squeeze" in t:
        lesson.update({
            "intro": "unsqueeze(dim) inserts a size-1 dimension; squeeze(dim) removes it. Useful for batch/channel axes.",
            "example": "```python\nimport torch\nx = torch.tensor([7,8,9])\nprint(x.unsqueeze(0).shape)  # [1,3]\nprint(x.unsqueeze(1).shape)  # [3,1]\n```",
            "question": "Make shape [3,1] with unsqueeze(1), then remove that dim back to [3].",
            "expected_answer": "x.unsqueeze(1) then x.squeeze(1)",
            "rubric": ["unsqueeze(1)", "squeeze(1)"],
            "hints": ["Position 1 is after first dim.", "unsqueeze(1) → [3,1].", "squeeze(1) removes it."]
        })
    elif "permute" in t:
        lesson.update({
            "intro": "permute reorders axes. NCHW↔NHWC is common in vision. Non-contiguous tensors may need .contiguous().",
            "example": "```python\nimport torch\nimg = torch.randn(3, 32, 48)\nprint(img.permute(1,2,0).shape)  # [H,W,C]\n```",
            "question": "For b = torch.randn(8,3,32,48) in NCHW, permute to NHWC.",
            "expected_answer": "b.permute(0, 2, 3, 1)",
            "rubric": ["permute nhwc"],
            "hints": ["Order is new axis order.", "NCHW→NHWC = (0,2,3,1).", "Result [8,32,48,3]."]
        })
    elif "repeat" in t and "expand" in t:
        lesson.update({
            "intro": "expand makes a broadcasted view (no copy); repeat tiles data (copies). Prefer expand when possible.",
            "example": "```python\nimport torch\ncol = torch.tensor([[10],[20],[30]])\nprint(col.expand(3,4).shape)\n```",
            "question": "Make a [3,4] tensor from col using repeat (not expand).",
            "expected_answer": "col.repeat(1,4)",
            "rubric": ["repeat used", "target 3,4"],
            "hints": ["Use repeat(1,4).", "repeat copies memory.", "Target shape [3,4]."]
        })
    elif "broadcast" in t:
        lesson.update({
            "intro": "Broadcasting stretches size-1 dims to match. Use unsqueeze to align row/column vectors with matrices.",
            "example": "```python\nimport torch\nM = torch.zeros(4,3)\nv = torch.tensor([1.,2.,3.])\nprint(M + v)  # row-wise add\n```",
            "question": "Add v=[1,2,3,4] as a column to M=zeros(4,3) using broadcasting.",
            "expected_answer": "M + v.unsqueeze(1)",
            "rubric": ["column broadcast"],
            "hints": ["v.unsqueeze(1) → [4,1].", "Then M + that.", "[:, None] also works."]
        })
    elif "einops" in t:
        lesson.update({
            "intro": "einops.rearrange expresses reshape/transpose with readable patterns. Great for complex models.",
            "example": "```python\nimport torch\nfrom einops import rearrange\nx = torch.arange(2*3*4).reshape(2,3,4)\nprint(rearrange(x, 'a b c -> b (a c)').shape)\n```",
            "question": "From x [2,3,4], get [2,12] by flattening last two dims via einops.",
            "expected_answer": "rearrange(x, 'a b c -> a (b c)')",
            "rubric": ["einops rearrange", "a (b c)"],
            "hints": ["Keep a; merge b and c.", "Pattern uses parentheses.", "Result [2,12]."]
        })
    return lesson

def build_lesson_from_section(section):
    return lesson_for_title(section["title"], section["content"])

def evaluate_answer_local(lesson, user_answer: str, used_hints: int):
    # --- LLM Evaluation Integration ---
    correct = False
    feedback = ""
    next_hint = ""
    award_completion = False

    if genai and model: # Check if Gemini is configured
        prompt = f"""
        You are an AI tutor. Evaluate the user's answer to a programming question.

        Lesson Title: {lesson.get("title", "N/A")}
        Question: {lesson.get("question", "N/A")}
        Expected Answer Criteria: {lesson.get("expected_answer", "N/A")}
        User's Answer:
        ```
        {user_answer}
        ```

        Provide feedback, state if the answer is correct (True/False), and suggest a hint if incorrect and appropriate.
        Format your response as a JSON object with 'correct' (boolean), 'feedback' (string), and 'next_hint' (string, or empty string if no hint).Donot add any other text to your response.
        """
        try:
            response = model.generate_content(prompt)
            llm_response_str = response.text.strip()
            match = re.search(r"```json\s*(.*?)\s*```", llm_response_str, re.DOTALL)
            if match:
               json_payload = match.group(1)
            else:
                json_payload = llm_response_str
            # If no markdown block is found, assume the whole response is JSON
      
            # Attempt to parse LLM's JSON response
            llm_eval_output = json.loads(json_payload)
            correct = llm_eval_output.get("correct", False)
            feedback = llm_eval_output.get("feedback", "No feedback from LLM.")
            next_hint = llm_eval_output.get("next_hint", "")
            award_completion = correct # Award completion if LLM says it's correct

        except Exception as e:
            print(f"Error calling LLM or parsing response: {e}")
            feedback = f"Error with AI evaluation: {e}. Falling back to local evaluation."
            # Fallback to local evaluation if LLM fails
            llm_response_str = "" # Clear for local evaluation to take over

    if not (genai and model and llm_response_str): # If LLM failed or not configured, use local evaluation
        text = user_answer.strip()
        lc = text.lower()

        def has_any(substrs):
            return any(s in lc for s in substrs)

        rub = [r.lower() for r in lesson.get("rubric", [])]

        if "creates 3x2" in rub:
            ok_shape = ("torch.size([3, 2])" in lc) or ("[3,2]" in lc) or ("3x2" in lc)
            ok_make = has_any(["torch.tensor(", "torch.ones(", "torch.rand("])
            ok_prints = ("print(" in lc) and ("dtype" in lc) and ("ndim" in lc or "dim" in lc)
            correct = ok_shape and ok_make and ok_prints
            feedback = "Nice work—correct shape/dtype/ndim printed." if correct else "Expect a 3x2 float tensor and prints of shape, dtype, and ndim."
        elif "last column slice" in rub:
            top_left = ("[:2,:2]" in lc.replace(" ", ""))
            last_col = ("[:,-1]" in lc.replace(" ", "")) or ("[:, -1]" in lc)
            correct = top_left and last_col
            feedback = "Correct 2x2 block and last column." if correct else "Use T[:2, :2] and T[:, -1]."
        elif "reshape to 3,4" in rub:
            ok_flat = "reshape(-1)" in lc or "view(-1)" in lc or "flatten(" in lc
            ok_shape = "reshape(3,4)" in lc or "view(3,4)" in lc
            correct = ok_flat and ok_shape
            feedback = "Good reshape pipeline." if correct else "Flatten first, then reshape to [3,4]."
        elif "squeeze(1)" in rub:
            ok_unsq = "unsqueeze(1)" in lc
            ok_sq = "squeeze(1)" in lc
            correct = ok_unsq and ok_sq
            feedback = "Channel dim add/remove looks right." if correct else "Use x.unsqueeze(1) then x.squeeze(1)."
        elif "permute nhwc" in rub:
            correct = ("permute(0, 2, 3, 1)" in lc) or ("permute(0,2,3,1)" in lc)
            feedback = "NCHW→NHWC is correct." if correct else "Use b.permute(0, 2, 3, 1)."
        elif "repeat used" in rub:
            correct = ("repeat(1,4)" in lc)
            feedback = "repeat(1,4) gives [3,4]." if correct else "Use col.repeat(1,4)."
        elif "column broadcast" in rub:
            correct = ("unsqueeze(1)" in lc) or ("[:, none]" in lc) or ("[:,none]" in lc)
            feedback = "Column broadcast set up correctly." if correct else "Turn v into [4,1] via v.unsqueeze(1) (or v[:, None]) then add."
        elif "a (b c)" in rub:
            correct = ("rearrange(" in lc) and ("'a b c -> a (b c)'" in text or '"a b c -> a (b c)"' in text)
            feedback = "Correct einops pattern." if correct else "Use rearrange(x, 'a b c -> a (b c)')."
        else:
            tokens = [t for t in re.split(r"[\W_]+", lesson.get("expected_answer","").lower()) if t]
            matched = sum(1 for t in tokens if t in lc)
            correct = matched >= max(1, len(tokens)//3)
            feedback = "Looks consistent with the expected answer." if correct else "Doesn't match the expected answer yet."

        # Reset next_hint if LLM evaluation failed or not used
        next_hint = ""
        hints = lesson.get("hints", [])
        if not correct and used_hints < len(hints):
            next_hint = hints[used_hints]
        award_completion = correct

    return {
        "correct": bool(correct),
        "feedback": feedback,
        "next_hint": next_hint,
        "award_completion": bool(award_completion),
    }

# ------------------ Session state ------------------
SESSIONS = {}

def get_or_create_sid():
    sid = request.cookies.get("tutor_sid")
    if not sid:
        sid = str(uuid.uuid4())
    return sid

def get_state(sid):
    st = SESSIONS.get(sid)
    if not st:
        sections = parse_tutorial_sections(CONTENT / "tutorial.md")
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

# ------------------ Utility to render MD safely ------------------
def _render_md(path: Path):
    raw = path.read_text(encoding="utf-8")
    if markdown is None:
        safe = (raw.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
        return f"<pre>{safe}</pre>", "" # Return empty TOC

    md = markdown.Markdown(extensions=["fenced_code","tables","toc"])
    body_html = md.convert(raw)
    toc_html = md.toc # Access the generated Table of Contents
    return body_html, toc_html

# ------------------ Routes ------------------
@app.route("/")
def index():
    # Fail fast if template missing, to give a clearer error
    idx = TEMPLATES / "index.html"
    if not idx.exists():
        return f"Template missing at: {idx}", 500
    return render_template("index.html")

@app.get("/tutorial")
def tutorial_page():
    body_html, toc_html = _render_md(CONTENT / "tutorial.md")
    return render_template("page.html", title="Tensor Basics Tutorial (Markdown)",
                           body_html=body_html, toc_html=toc_html)

@app.get("/reference/einsum")
def einsum_page():
    body_html, toc_html = _render_md(CONTENT / "einsum.md")
    return render_template("page.html", title="Einsum Reference",
                           body_html=body_html, toc_html=toc_html)

@app.post("/api/tutor/start")
def tutor_start():
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
    sid = get_or_create_sid()
    st = get_state(sid)
    data = request.get_json(force=True) or {}
    user_answer = str(data.get("answer","")).strip()
    if not user_answer:
        return jsonify({"error": "Empty answer"}), 400
    if st["current_lesson"] is None:
        st["current_lesson"] = build_lesson_from_section(st["sections"][st["current_idx"]])

    eval_out = evaluate_answer_local(st["current_lesson"], user_answer, st["used_hints"])

    if eval_out.get("correct") or eval_out.get("award_completion"):
        st["completed"][st["current_idx"]] = True
        st["used_hints"] = 0
        if st["current_idx"] + 1 < len(st["sections"]):
            st["current_idx"] += 1
            st["current_lesson"] = build_lesson_from_section(st["sections"][st["current_idx"]])
            advanced = True
            next_title = st["sections"][st["current_idx"]]["title"]
            next_lesson = st["current_lesson"]
        else:
            advanced = False
            next_title = "All sections complete"
            next_lesson = None
            st["current_lesson"] = None
        return jsonify({
            "result": eval_out,
            "advanced": advanced,
            "progress": summarize_progress(st),
            "section_title": next_title,
            "lesson": next_lesson,
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

@app.post("/api/tutor/jump")
def tutor_jump():
    sid = get_or_create_sid()
    st = get_state(sid)
    data = request.get_json(force=True) or {}
    section_idx = data.get("section_idx")

    if section_idx is None or not isinstance(section_idx, int) or \
       not (0 <= section_idx < len(st["sections"])):
        return jsonify({"error": "Invalid section_idx provided"}), 400

    st["current_idx"] = section_idx
    st["used_hints"] = 0  # Reset hints when jumping to a new section
    st["current_lesson"] = build_lesson_from_section(st["sections"][st["current_idx"]])

    return jsonify({
        "progress": summarize_progress(st),
        "section_title": st["sections"][st["current_idx"]]["title"],
        "lesson": st["current_lesson"],
    })

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "base": str(BASE),
        "templates_exists": (TEMPLATES / "index.html").exists(),
        "templates_dir": str(TEMPLATES),
        "static_dir": str(STATIC),
        "content_dir": str(CONTENT),
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)