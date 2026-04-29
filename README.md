# Research Paper Summarizer

An interactive Streamlit app for summarizing research papers with Groq LLMs, extracting key metrics, and asking follow-up questions in text or by voice.

It is designed for fast paper review, lightweight analysis, and a smoother reading workflow when you want answers without manually digging through every section of a PDF.

## Highlights

- Upload a single research-paper PDF and analyze it in one session
- Generate 5 summary styles:
  - Abstract
  - Detailed
  - Bullet Points
  - Technical
  - Executive
- Run a fuller analysis workflow with multiple summaries plus metric extraction
- Ask follow-up questions about the uploaded paper
- Speak a question and transcribe it with Groq Whisper `whisper-large-v3`
- Download summaries and JSON analysis output
- Keep a lightweight local history in the Streamlit session
- Protect your Groq key with anonymous per-user token limits

## Built With

- `Streamlit` for the web UI
- `Groq` for LLM inference and speech-to-text
- `PyPDF` for PDF text extraction
- `python-dotenv` for local environment variable loading
- `uv` for dependency and runtime management

## What The App Does

Once a user uploads a PDF, the app extracts its text, then uses Groq to:

1. Summarize the paper in the selected style
2. Extract important metrics and numerical findings
3. Answer follow-up questions using prior summary context
4. Transcribe spoken questions before sending them into the Q&A flow

The app intentionally works on trimmed excerpts of the paper for LLM calls instead of sending the entire document, which keeps responses fast and usage more predictable.

## Features

### 1. PDF Summarization

Users can upload one PDF at a time and choose from multiple summary types depending on their goal:

- `Abstract`: quick top-level overview
- `Detailed`: broader multi-paragraph summary
- `Bullet Points`: key takeaways in scan-friendly form
- `Technical`: focuses on method, setup, and results
- `Executive`: simplified explanation for non-specialists

### 2. Full Analysis

The app can also run a broader paper pass that combines:

- abstract summary
- detailed summary
- bullet-point summary
- metrics extraction

This is useful for deeper paper review or generating structured notes.

### 3. Q&A Over The Paper

After a paper has been processed, users can ask questions such as:

- What problem does this paper solve?
- What are the main limitations?
- What dataset was used?
- What are the real-world implications?

The app keeps conversation context so follow-up questions feel more natural.

### 4. Voice Questions

Inside the Q&A tab, users can:

1. Record a spoken question
2. Transcribe it with Groq Whisper `whisper-large-v3`
3. Review or edit the transcript
4. Submit it as a normal follow-up question

This makes the app more accessible and faster to use during reading sessions.

### 5. Token Guardrails

To reduce the risk of anonymous visitors burning your Groq API credits, the app assigns each user an anonymous key and enforces a per-user token limit.

Current default:

- `7000` tokens per anonymous user

Important note:

- This is a lightweight product safeguard, not hard security.
- A determined user could still bypass it by forcing a new anonymous identity.
- For production-grade protection, use a backend service, authentication, and server-side rate limiting.

## Screens In The App

- `Summarize`: upload the PDF, choose a summary type, run one summary or full analysis
- `Detailed Analysis`: generate alternate summary views and extract metrics from the current paper
- `Q&A`: ask typed or spoken questions about the paper
- `History`: review saved summaries from the current session

## Project Structure

```text
AI summarizeer/
├── app.py            # Streamlit UI and app flow
├── agent.py          # Groq client logic, summarization, metrics, Q&A, transcription
├── pyproject.toml    # Project metadata and dependencies
├── uv.lock           # Locked dependency graph
└── README.md         # Project documentation
```

## Requirements

- Python `3.11+`
- A Groq API key
- Internet access when calling Groq APIs

## Installation

### 1. Clone or open the project folder

```powershell
cd "C:\Users\HP\Desktop\AI summarizeer"
```

### 2. Install dependencies

If you are using `uv`:

```powershell
uv sync
```

## Configuration

You can provide your Groq API key in any of these ways.

### Option 1: PowerShell environment variable

```powershell
$env:GROQ_API_KEY="your-api-key-here"
```

### Option 2: `.env` file

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your-api-key-here
```

### Option 3: In-app entry

If the key is not found, the app shows a `Connect Groq API` input so you can paste it directly into the UI.

## Run The App

Use either of these commands from the project folder:

```powershell
uv run app.py
```

or:

```powershell
uv run streamlit run app.py
```

`app.py` automatically forwards plain script execution to Streamlit's runner, so `uv run app.py` works cleanly.

## Token Limit Configuration

The anonymous token budget can be changed with an environment variable:

```powershell
$env:ANON_USER_TOKEN_LIMIT="7000"
```

If unset, the app uses:

```text
7000
```

## Typical Usage Flow

1. Launch the app
2. Connect your Groq API key
3. Upload one PDF
4. Generate a summary or run full analysis
5. Explore the `Detailed Analysis` tab if needed
6. Ask typed or spoken questions in the `Q&A` tab
7. Download results or save them to session history

## How Voice Input Works

The voice workflow uses Groq speech-to-text:

- Model: `whisper-large-v3`
- Input source: Streamlit `audio_input`
- Output: text transcript inserted into the question box

From there, the transcript is treated exactly like a typed question.

## Notes On PDF Handling

- The app is designed for one active PDF per user session
- Uploading a different PDF replaces the current paper context
- Clearing the uploader or pressing the clear action resets the current paper state
- PDF extraction quality depends on the structure of the source document

Scanned PDFs or poorly encoded PDFs may produce weaker extraction results unless OCR is added in a future version.

## Known Limitations

- Anonymous quota tracking is local and lightweight
- Session history is not a permanent database
- Some research PDFs extract text imperfectly
- Very large or complex papers may lose nuance because only excerpts are sent to the LLM
- The current app does not perform OCR for image-only PDFs

## Future Improvements

- Proper backend authentication and server-side quota enforcement
- Persistent database-backed usage tracking
- OCR support for scanned PDFs
- Better citation-aware answer generation
- Section-aware paper chunking for higher fidelity summaries
- Admin dashboard for token usage and analytics
- Multi-paper workspace support

## Troubleshooting

### Streamlit warnings about `ScriptRunContext`

Run the app with:

```powershell
uv run app.py
```

The project already auto-forwards into Streamlit's proper runner.

### API key not found

Make sure one of these is set:

- `GROQ_API_KEY` in the environment
- `GROQ_API_KEY` in `.env`
- pasted into the in-app API key field

### No answer in Q&A

The app requires a processed paper first. Upload a PDF and generate at least one summary before asking follow-up questions.

### Voice transcription fails

Check that:

- your audio recording was captured successfully
- your Groq key is valid
- the network request to Groq is available

## Developer Notes

- Main UI logic lives in [app.py](./app.py)
- Core summarization and Groq API logic lives in [agent.py](./agent.py)
- The app uses Groq chat completions for summaries and Q&A
- The app uses Groq audio transcription for speech input

## Why This Project Is Useful

Research papers are often dense, long, and repetitive when you only need a fast understanding of:

- the problem
- the method
- the findings
- the limitations

This app shortens that loop and gives you a practical reading assistant for papers, reports, and technical PDFs.
