AI SKILL GAP & CAREER NAVIGATOR - DEPLOYMENT READY

LOCAL WINDOWS
1. Open this folder in VS Code.
2. Create .env from .env.example.
3. Put your Gemini key in GEMINI_API_KEY.
4. Optional Google Sign-In: add GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET and GOOGLE_REDIRECT_URI.
5. Run: pip install -r requirements.txt
6. Run: python app.py
7. Open: http://127.0.0.1:5000
8. Test: http://127.0.0.1:5000/health
9. Test Gemini: http://127.0.0.1:5000/gemini-status?test=1

RENDER
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app --workers 1 --threads 4 --timeout 120
Python: 3.13.5

REQUIRED RENDER ENVIRONMENT VARIABLES
GEMINI_API_KEY = your real Gemini API key
GEMINI_MODEL = gemini-3.8-flash
GEMINI_THINKING_LEVEL = low
GEMINI_RPM = 4
SECRET_KEY = generate a secure random value
DATABASE_URL = persistent PostgreSQL connection string

GOOGLE SIGN-IN ENVIRONMENT VARIABLES
GOOGLE_CLIENT_ID = Google OAuth web client ID
GOOGLE_CLIENT_SECRET = Google OAuth web client secret
GOOGLE_REDIRECT_URI = https://YOUR-SERVICE.onrender.com/auth/google/callback

IMPORTANT
- Never put a real API key or Google client secret in GitHub.
- Google OAuth redirect URI must exactly match the URI configured in Google Cloud Console.
- The app initializes its database when imported, so Gunicorn/Render does not hit the old missing-table 500 error.
- For persistent accounts on Render, use DATABASE_URL instead of SQLite. Free web-service filesystems can be ephemeral.
