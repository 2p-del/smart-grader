# SmartGrader

A local prototype that helps teachers grade short written answers. Google's Gemini model
**suggests** a score and feedback for each answer using the rubric you provide. **Those scores are
suggestions, not final grades:** AI can misread an answer, be inconsistent, or be fooled by
a cleverly worded one. The teacher reviews, edits and owns every final grade, and the app is built
around that review step.

Workflow: enter a question and rubric, paste student answers, get suggestions, edit scores and
feedback, export a CSV.

## 1. Get a Gemini API key

Create a key in [Google AI Studio](https://aistudio.google.com/apikey). Keep it private: anyone with
the key can use your quota.

## 2. Run it locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Give the app your key in any of these ways. A key typed into the sidebar takes priority for that session; otherwise the environment variable takes priority over Streamlit secrets.

**A. Masked key field in the app**

When SmartGrader opens, enter your key in the **Gemini API key** field in the sidebar. The field masks the key. It stays in the active session only and is not written to a file or log. Use **Forget entered key** to clear it before closing the app.

**B. Environment variable** (lasts for that terminal window)

```bash
export GEMINI_API_KEY="your-key"   # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```

**C. Streamlit secrets file** (saved on your computer, so you set it once)

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
2. Replace `paste-your-key-here` with your key and save.
3. Run `streamlit run app.py`.

`secrets.toml` is listed in `.gitignore`, so it is not committed. Never put the key in `app.py` or any
other source file.

The app opens at http://localhost:8501. Without a key, you can still enter the question, rubric and answers, and use the review/export interface; AI grading remains switched off until a key is provided.

### Using Streamlit Community Cloud instead

Deploy the app, then open the app's **Settings → Secrets** and add `GEMINI_API_KEY = "your-key"`
(same format as `secrets.toml`). Don't commit the key to the repository. Note that student answers
will then travel through a hosted service, so read the privacy notes below first.

## 3. Optional settings

Each can be set as an environment variable or in `secrets.toml` (the environment variable wins).

| Setting | Default | What it does |
|---|---|---|
| `SMARTGRADER_MODEL` | `gemini-3.8-flash` | Which Gemini model to use. Change it if Google renames or retires the default. |
| `SMARTGRADER_TIMEOUT_SECONDS` | `30` | How long to wait for Gemini before giving up on one answer (allowed range 5–120). |

## 4. Using it

1. Type the question and fill in the rubric table (one row per criterion, with its points).
   **Load example** fills in a sample question and four answers, one of which tries to talk the grader
   into full marks, so you can see the review flag.
2. Paste the answers into the box, separated by a line containing only `---`.
3. Click **Get AI suggestions**. Answers are graded in parallel, so a class of 30 takes seconds rather
   than minutes.
4. In **Scores and feedback**, change **Final score** and **Feedback** as you see fit, then tick
   **Reviewed**. Rows marked ⚠ need a closer look. **Answer details** shows each answer with the
   AI's reason for every criterion.
5. Download the CSV. It shows the AI score next to your final score, plus review flags.

Results stay in the browser session until you close the tab or press **Clear results**.

## How the AI is asked to behave

Gemini is told to use **only your rubric** (no extra criteria), to give **partial credit in proportion**
to what each criterion asks for, to keep feedback to two short sentences, to treat the student's text
purely as data, and to **flag an answer for your review** when it is ambiguous, borderline, off-topic
or addresses the grader. Its reply must be JSON in a fixed shape. The app checks that reply
against your rubric (right criteria, scores clamped to each maximum) before showing anything.

## When something goes wrong

The app shows a plain message instead of failing silently.

| You see | Meaning and fix |
|---|---|
| "No Gemini API key found" | Add the key with the environment variable or `secrets.toml`, then restart the app. |
| "Gemini rejected the API key" | The key is wrong, expired or not allowed. Check it and restart. |
| "Gemini could not find the model ..." | Set `SMARTGRADER_MODEL` to a model your key can use. |
| "Gemini is rate-limiting requests" | Too many requests, or the free quota is used up. Wait a minute and retry. |
| "Gemini did not answer within N seconds" | Slow network or busy service. Retry; raise `SMARTGRADER_TIMEOUT_SECONDS` if it keeps happening. |
| "Could not reach Gemini" | No internet connection from this computer. |
| "Gemini's reply could not be used" | The model returned something that did not match the rubric. The answer is flagged for manual grading. |

Answers that fail are never guessed: they appear in the table marked for manual grading, with the reason.

## Layout

| File | Job |
|---|---|
| `app.py` | Streamlit page: inputs, results, export |
| `gemini_client.py` | **All Gemini access:** reads the key, model and timeout; sends the request; turns every failure into a clear message; hides the key |
| `grading.py` | Prompt, JSON schema, validation against the rubric, review flags, parallel batches |
| `models.py` | Rubric, suggestion and graded-answer models; score clamping |
| `review.py` | Converts between the on-screen tables and the models (no Streamlit) |
| `io_utils.py` | Splitting pasted answers, CSV export |
| `styles.py`, `theme.css`, `.streamlit/config.toml` | The ivory, lilac and editorial look |
| `.streamlit/secrets.toml.example` | Template for a local key and optional settings |
| `tests/` | `pip install pytest`, then `pytest`. No API key needed; Gemini is faked. |

`gemini_client.py` knows nothing about rubrics, so you can reuse it in another project.

## Safety and privacy

- **API key:** the sidebar field masks the key as you type. A key entered there is sent to the app server for the Gemini request and held only in the active session; it is not saved to a file or log. Environment-variable and Streamlit-secrets keys remain server-side. Error messages scrub the key.
- **Student data:** answers are sent to Google's Gemini API. Check Google's current data-use terms for
  your API tier, and remove names and ID numbers first if that matters for your class. Nothing is saved
  to disk by the app.
- **Prompt injection:** a student answer is treated as data, delimited in the prompt, and tag look-alikes
  are neutralised. Answers that address the grader are flagged. This helps but is not foolproof: your
  review is the real safeguard.
- **CSV:** cells starting with `= + - @` are prefixed with `'` so an answer can't run as a spreadsheet formula.

## Not included (by design, for now)

Accounts, a database, saved sessions, file import, per-criterion score editing, LMS integration.
