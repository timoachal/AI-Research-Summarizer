# Research Paper Summarizer

## Run

Use either of these commands from the project folder:

```powershell
uv run app.py
```

```powershell
uv run streamlit run app.py
```

The app is a Streamlit app, so it needs to run through Streamlit's runner. `app.py`
now auto-forwards plain script execution to the correct launcher.

## API Key

You can provide your Groq API key in any of these ways:

```powershell
$env:GROQ_API_KEY="your-api-key-here"
```

Or create a `.env` file in the project folder:

```env
GROQ_API_KEY=your-api-key-here
```

You can also paste the key directly into the app's "Connect Groq API" field.
