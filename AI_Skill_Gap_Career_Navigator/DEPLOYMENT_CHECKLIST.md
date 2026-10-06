# Deployment checklist

## Render
- Runtime: Python 3
- Build: `pip install -r requirements.txt`
- Start: `gunicorn app:app --workers 1 --threads 4 --timeout 120`
- Python: 3.13.5
- Set `GEMINI_API_KEY`
- Set `DATABASE_URL` to a persistent PostgreSQL database
- Set `SECRET_KEY`
- For Google Sign-In set `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`

## Google redirect URI
Local:
`http://127.0.0.1:5000/auth/google/callback`

Production:
`https://YOUR-SERVICE.onrender.com/auth/google/callback`

The production URI must exactly match the authorized redirect URI in Google Cloud Console.

## Gemini
Primary model: `gemini-3.8-flash`
Fallbacks: 3.7 Flash, 3.6 Flash, 3.5 Flash, 3.1 Flash-Lite, 2.5 Flash, 2.5 Flash-Lite.

The app also has an offline/local fallback for skill analysis and roadmaps, so a temporary Gemini outage does not turn the whole site into a 500 page.

## Health checks
- `/health`
- `/gemini-status`
- `/gemini-status?test=1`
