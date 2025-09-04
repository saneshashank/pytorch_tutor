// Guided-only front-end with hard auto-advance
const $ = s => document.querySelector(s);
const startBtn = $("#startLesson");
const lessonBox = $("#lessonBox");
const sectionTitle = $("#sectionTitle");
const intro = $("#intro");
const example = $("#example");
const question = $("#question");
const progress = $("#progress");
const answerForm = $("#answerForm");
const answerBox = $("#answer");
const feedback = $("#feedback");

function renderProgress(p) {
  const rows = p.titles.map((t,i) => {
    const done = p.completed[i] ? "✅" : (i === p.current_idx ? "👉" : "•");
    return `<div>${done} ${i+1}. ${t}</div>`;
  }).join("");
  progress.innerHTML = `<div><strong>Progress:</strong></div>${rows}`;
}

function renderLesson(title, lesson) {
  lessonBox.classList.remove("hidden");
  sectionTitle.textContent = title;
  intro.textContent = lesson.intro || "";
  const ex = (lesson.example || "").replace(/^```python\n?|```$/g, "");
  example.textContent = ex;
  question.textContent = lesson.question || "";
  feedback.textContent = "";
  answerBox.value = "";
  answerBox.focus();
}

async function startLesson() {
  try {
    const res = await fetch("/api/tutor/start", { method: "POST" });
    const data = await res.json();
    if(!res.ok) throw new Error(data.error || "Failed to start lesson");
    renderProgress(data.progress);
    renderLesson(data.section_title, data.lesson);
  } catch (e) {
    alert(e.message || e);
  }
}

async function submitAnswer(evt) {
  evt.preventDefault();
  try {
    const answer = answerBox.value.trim();
    if(!answer) return;
    feedback.textContent = "Evaluating...";
    const res = await fetch("/api/tutor/answer", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ answer })
    });
    const data = await res.json();
    if(!res.ok) throw new Error(data.error || "Submission failed");
    renderProgress(data.progress);
    const r = data.result || {};
    feedback.textContent = r.feedback || "";
    if (data.advanced) {
      renderLesson(data.section_title, data.lesson);
    } else if (!r.correct && r.next_hint) {
      feedback.textContent += "\nHint: " + r.next_hint;
    } else if (!r.correct) {
      // stay on this section
    } else if (!data.lesson) {
      sectionTitle.textContent = "🎉 All sections complete!";
      question.textContent = "Great job — you've finished all sections.";
      example.textContent = "";
      intro.textContent = "";
    }
  } catch (e) {
    feedback.textContent = e.message || String(e);
  }
}

if (startBtn) startBtn.addEventListener("click", startLesson);
if (answerForm) answerForm.addEventListener("submit", submitAnswer);
