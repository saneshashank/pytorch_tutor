
# PyTorch Tutor · Gemini API (Flask) — v4 (simplified UX)

- Guided flow that reads `content/tutorial.md`: short intro → tiny example → practice → hints → completion per section.
- **Simplified UI:** removed mode/temperature controls; only Guided Tutor + an optional Quick Chat box.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # set GEMINI_API_KEY
python app.py
```
