import os, re, json, time, random, hashlib, threading, requests, traceback, secrets
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from authlib.integrations.flask_client import OAuth
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import inspect, text

from seed_questions import SEED

load_dotenv()
basedir = os.path.abspath(os.path.dirname(__file__))
os.makedirs(os.path.join(basedir, "instance"), exist_ok=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY") or secrets.token_hex(32)

# Local development uses SQLite. Render/production should set DATABASE_URL to a
# persistent PostgreSQL database (for example a free hosted PostgreSQL provider).
database_url = (os.getenv("DATABASE_URL") or "").strip()
if database_url.startswith("postgres://"):
    database_url = "postgresql+psycopg://" + database_url[len("postgres://"):]
elif database_url.startswith("postgresql://"):
    database_url = "postgresql+psycopg://" + database_url[len("postgresql://"):]
if not database_url:
    database_url = 'sqlite:///' + os.path.join(basedir, 'instance', 'app.db')

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.getenv('COOKIE_SECURE', 'auto').lower() == 'true' or bool(os.getenv('RENDER_EXTERNAL_URL', '').startswith('https://'))
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to continue.'
login_manager.login_message_category = 'warning'

# Optional Google Sign-In. The button is shown only when credentials are configured.
GOOGLE_CLIENT_ID = (os.getenv('GOOGLE_CLIENT_ID') or '').strip()
GOOGLE_CLIENT_SECRET = (os.getenv('GOOGLE_CLIENT_SECRET') or '').strip()
GOOGLE_REDIRECT_URI = (os.getenv('GOOGLE_REDIRECT_URI') or '').strip()
oauth = OAuth(app)
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth_google = oauth.register(
        name='google',
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'}
    )
else:
    oauth_google = None

def google_enabled():
    return oauth_google is not None

# =====================================================================
#  REAL GOOGLE GEMINI CLIENT
#  - local rate limiter (never sends more than GEMINI_RPM calls/minute per model)
#  - reads Google's "retry in 57s" and puts that model on a cool-down
#  - instantly moves to the next model (each model has its OWN free quota)
# =====================================================================
GEMINI_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
GEMINI_MODEL = (os.getenv("GEMINI_MODEL") or "gemini-3.8-flash").strip()
GEMINI_RPM = max(1, min(30, int(os.getenv("GEMINI_RPM", "4"))))

_extra = [m.strip() for m in (os.getenv("GEMINI_FALLBACK_MODELS") or "").split(",") if m.strip()]
_default_chain = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
                  "gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
MODEL_CHAIN = []
for _m in [GEMINI_MODEL] + _extra + _default_chain:
    if _m and _m not in MODEL_CHAIN:
        MODEL_CHAIN.append(_m)

DEAD_MODELS = set()                  # models that returned 404 (wrong / retired name)
LAST_GOOD = {"model": None}
_LOCK = threading.Lock()
_CALLS = {}                          # model -> [timestamps of calls in the last 60 s]
_COOL = {}                           # model -> epoch time until which we do not call it


class GeminiError(Exception):
    """Gemini could not answer (busy, quota, network...)."""
    wait = 0                         # seconds until it is worth trying again


class GeminiKeyError(GeminiError):
    """API key missing / invalid - retrying will not help."""


class _Transient(Exception):
    pass


class _SkipModel(Exception):
    pass


def gemini_ready():
    return bool(GEMINI_KEY) and "PASTE" not in GEMINI_KEY.upper()


def _reserve(model, background=False):
    """Reserve one call slot for this model. Returns 0 if reserved, otherwise seconds to wait."""
    now = time.time()
    with _LOCK:
        if _COOL.get(model, 0) > now:
            return _COOL[model] - now
        calls = [t for t in _CALLS.get(model, []) if now - t < 60]
        limit = max(1, GEMINI_RPM - 2) if background else GEMINI_RPM    # background keeps slots free for users
        if len(calls) >= limit:
            _CALLS[model] = calls
            return max(0.5, 60 - (now - calls[0]))
        calls.append(now)
        _CALLS[model] = calls
        return 0


def _cool(model, seconds):
    with _LOCK:
        _COOL[model] = time.time() + seconds


def _call_model(model, prompt, system, as_json, timeout, max_tokens, history):
    contents = [{"role": r, "parts": [{"text": t}]} for r, t in (history or [])]
    contents.append({"role": "user", "parts": [{"text": prompt}]})
    body = {"contents": contents}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    cfg = {"temperature": 1.0 if as_json else 0.7}
    if as_json:
        cfg["responseMimeType"] = "application/json"
    if max_tokens:
        cfg["maxOutputTokens"] = max_tokens
    if model.startswith("gemini-3"):
        cfg["thinkingConfig"] = {"thinkingLevel": os.getenv("GEMINI_THINKING_LEVEL", "low")}
    elif "2.5-flash" in model:
        cfg["thinkingConfig"] = {"thinkingBudget": 0}
    body["generationConfig"] = cfg
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": GEMINI_KEY, "Content-Type": "application/json"},
            json=body, timeout=(10, timeout))
    except (requests.Timeout, requests.ConnectionError) as e:
        raise _Transient(f"network/timeout: {e.__class__.__name__}")

    if r.status_code == 200:
        try:
            data = r.json()
            parts = data["candidates"][0]["content"]["parts"]
            out = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        except Exception:
            raise _Transient("empty or blocked answer")
        if not out:
            raise _Transient("empty answer")
        return out

    raw = r.text or ""
    try:
        msg = r.json().get("error", {}).get("message", "") or raw[:200]
    except Exception:
        msg = raw[:200]
    if r.status_code in (401, 403) or (r.status_code == 400 and "API key" in msg):
        raise GeminiKeyError(f"Gemini API key problem: {msg[:200]}")
    if r.status_code == 404:
        DEAD_MODELS.add(model)
        raise _SkipModel(msg)
    if r.status_code == 400:
        raise _SkipModel(msg)
    if r.status_code == 429:                      # quota / rate limit -> cool this model down, use another
        m = re.search(r"retry in ([\d.]+)\s*s", msg, re.I) or re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', raw)
        wait = float(m.group(1)) + 1 if m else 30
        if re.search(r"PerDay|per day|daily", raw, re.I):
            wait = max(wait, 1800)
        _cool(model, wait)
        raise _Transient(f"429 quota ({int(wait)}s cool-down)")
    if r.status_code in (500, 502, 503, 504):
        _cool(model, 6)
    raise _Transient(f"{r.status_code}: {msg[:120]}")


def gemini(prompt, system=None, as_json=False, deadline=45, max_tokens=None, history=None, background=False):
    """Call the real Gemini API. Rate-limited, with automatic fallback across several models."""
    if not gemini_ready():
        raise GeminiKeyError("GEMINI_API_KEY is missing. Put your real key in the .env file.")
    start = time.time()
    last = "unknown"
    for _pass in range(3):
        models = [m for m in MODEL_CHAIN if m not in DEAD_MODELS]
        if LAST_GOOD["model"] in models:                      # try what worked last time first
            models.remove(LAST_GOOD["model"])
            models.insert(0, LAST_GOOD["model"])
        waits = []
        for model in models:
            left = deadline - (time.time() - start)
            if left <= 3:
                break
            w = _reserve(model, background)
            if w > 0:
                waits.append(w)
                continue
            try:
                out = _call_model(model, prompt, system, as_json, min(40, left), max_tokens, history)
                LAST_GOOD["model"] = model
                return out
            except GeminiKeyError:
                raise
            except (_SkipModel, _Transient) as e:
                last = f"{model}: {e}"
        left = deadline - (time.time() - start)
        if background or left <= 5:
            break
        if waits:                                             # everything is cooling down
            soonest = min(waits)
            if soonest > min(left - 4, 12):
                break
            time.sleep(soonest + 0.3)
        else:
            time.sleep(1.5)
    with _LOCK:
        now = time.time()
        pend = [max(0, t - now) for m, t in _COOL.items() if m not in DEAD_MODELS and t > now]
    err = GeminiError(f"Gemini is busy right now ({last[:150]})")
    err.wait = int(min(pend)) + 1 if pend and len(pend) >= len([m for m in MODEL_CHAIN if m not in DEAD_MODELS]) else 0
    raise err


def parse_json(text_):
    t = re.sub(r"^```(?:json)?|```$", "", text_.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    for a, b in (("{", "}"), ("[", "]")):
        i, j = t.find(a), t.rfind(b)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                continue
    raise GeminiError("Gemini returned an unreadable answer")


def why_busy(e):
    """Friendly one-line reason shown to the user."""
    if isinstance(e, GeminiKeyError):
        return str(e)
    if getattr(e, "wait", 0):
        return f"Gemini free-tier limit reached - it frees up again in about {e.wait} seconds."
    return "Gemini is busy right now."


# =====================================================================
#  DATA
# =====================================================================
SKILLS_LIST = [
    "Python", "Java", "C++", "JavaScript", "TypeScript",
    "HTML_CSS", "React", "SQL", "NoSQL", "Cloud_Computing",
    "DevOps", "DSA", "Machine_Learning", "Deep_Learning", "NLP",
    "Computer_Vision", "Data_Engineering", "Cyber_Security", "Git", "System_Design",
    "REST_APIs", "Agile", "Business_Intelligence", "Technical_Writing", "Communication"
]

JOB_ROLES = [
    {"title": "AI Engineer", "key_skills": ["Python", "NLP", "Deep_Learning", "Machine_Learning"]},
    {"title": "Machine Learning Engineer", "key_skills": ["Python", "Machine_Learning", "Deep_Learning", "DSA"]},
    {"title": "Data Scientist", "key_skills": ["Python", "Machine_Learning", "SQL", "Business_Intelligence"]},
    {"title": "Data Engineer", "key_skills": ["Data_Engineering", "SQL", "Python", "NoSQL"]},
    {"title": "Full-Stack Developer", "key_skills": ["React", "HTML_CSS", "JavaScript", "REST_APIs", "SQL"]},
    {"title": "Backend Engineer", "key_skills": ["Python", "Java", "REST_APIs", "System_Design", "SQL"]},
    {"title": "Frontend Developer", "key_skills": ["React", "TypeScript", "JavaScript", "HTML_CSS"]},
    {"title": "Cloud Architect", "key_skills": ["Cloud_Computing", "DevOps", "System_Design", "Cyber_Security"]},
    {"title": "DevOps Engineer", "key_skills": ["DevOps", "Cloud_Computing", "Git", "System_Design"]},
    {"title": "Cyber Security Analyst", "key_skills": ["Cyber_Security", "Cloud_Computing", "System_Design"]},
]

# Each user gets a RANDOM mix of these sub-topics, so two users never get the same test.
SUBTOPICS = {k: v.split("|") for k, v in {
    "Python": "data types & mutability|list/dict/set comprehensions|functions, *args/**kwargs|closures & decorators|generators & iterators|OOP, inheritance, dunder methods|exceptions & context managers|modules, packages, venv|file I/O & JSON|standard library (collections, itertools)|lambda, map, filter|async/await & threading|slicing & strings|type hints & dataclasses|memory, GIL, performance",
    "Java": "OOP principles|collections framework|generics|exceptions|streams & lambdas|multithreading & concurrency|JVM, GC & memory|interfaces vs abstract classes|string handling|Spring basics|Optional & records|immutability|equals/hashCode|IO/NIO|design patterns",
    "C++": "pointers & references|memory management & RAII|STL containers|templates|OOP & virtual functions|smart pointers|move semantics|operator overloading|lambdas|const-correctness|exceptions|iterators & algorithms|undefined behaviour|multithreading|build & linking",
    "JavaScript": "scope, hoisting, closures|this keyword|promises & async/await|event loop|prototypes & classes|ES6+ features|array methods|DOM & events|modules|error handling|equality & type coercion|fetch & JSON|currying & debounce|Map/Set/WeakMap|performance",
    "TypeScript": "basic & union types|interfaces vs types|generics|enums|type guards & narrowing|utility types (Partial, Pick...)|classes & access modifiers|tsconfig & strict mode|mapped & conditional types|declaration files|async typing|readonly & const assertions|modules|unknown vs any|decorators",
    "HTML_CSS": "semantic HTML|forms & validation|flexbox|grid|box model|positioning|specificity & cascade|responsive design & media queries|accessibility (ARIA)|animations & transitions|CSS variables|pseudo-classes/elements|SEO basics|units (rem/em/vw)|performance",
    "React": "components & props|useState|useEffect|useRef & useMemo|context API|keys & lists|controlled forms|custom hooks|rendering & reconciliation|routing|state management (Redux)|performance (memo)|error boundaries|testing|server components & Next.js",
    "SQL": "SELECT & filtering|JOIN types|GROUP BY & HAVING|subqueries & CTEs|window functions|indexes|normalization|transactions & ACID|constraints & keys|views & stored procedures|query optimisation|set operations|NULL handling|date & string functions|DDL vs DML",
    "NoSQL": "document stores (MongoDB)|key-value stores (Redis)|column-family stores|graph databases|CAP theorem|sharding & replication|consistency models|indexing in NoSQL|aggregation pipelines|data modelling|when to choose NoSQL|BASE vs ACID|TTL & caching|transactions in NoSQL|scaling",
    "Cloud_Computing": "IaaS/PaaS/SaaS|virtual machines & containers|object storage|serverless|load balancing & auto-scaling|IAM & security|VPC & networking|cloud databases|cost optimisation|high availability & regions|CDN|monitoring & logging|shared responsibility|infrastructure as code|migration strategies",
    "DevOps": "CI/CD pipelines|Docker|Kubernetes|Terraform & IaC|monitoring & observability|Git workflows|configuration management|deployment strategies (blue/green, canary)|Linux basics|secrets management|logging|SRE & SLAs|container networking|build tools|security (DevSecOps)",
    "DSA": "arrays & strings|linked lists|stacks & queues|hash tables|trees & BST|heaps|graphs & BFS/DFS|sorting algorithms|searching & binary search|recursion & backtracking|dynamic programming|greedy|time/space complexity|two pointers & sliding window|tries",
    "Machine_Learning": "supervised vs unsupervised|linear & logistic regression|decision trees & random forests|SVM|clustering|bias-variance|overfitting & regularisation|cross-validation|evaluation metrics|feature engineering|gradient descent|ensemble methods|dimensionality reduction|data leakage|model deployment",
    "Deep_Learning": "neural network basics|activation functions|backpropagation|CNNs|RNN/LSTM|transformers & attention|optimisers|regularisation (dropout, batchnorm)|loss functions|transfer learning|GANs|autoencoders|vanishing gradients|PyTorch/TensorFlow|training tricks",
    "NLP": "tokenisation|embeddings|TF-IDF|transformers & BERT|LLMs & prompting|RAG|text classification|named entity recognition|sequence models|evaluation (BLEU, ROUGE)|fine-tuning|stemming & lemmatisation|attention|summarisation & translation|hallucination & safety",
    "Computer_Vision": "image basics & filters|edge detection|CNN architectures|object detection (YOLO)|segmentation|data augmentation|OpenCV|transfer learning|image classification metrics|feature extraction (SIFT)|face recognition|optical flow|vision transformers|camera geometry|deployment",
    "Data_Engineering": "ETL vs ELT|data warehouses & lakes|Apache Spark|Kafka & streaming|Airflow orchestration|batch vs stream|data modelling (star schema)|partitioning & file formats (Parquet)|data quality|CDC|cloud data platforms|SQL performance|schema evolution|data governance|lakehouse",
    "Cyber_Security": "CIA triad|common attacks (XSS, SQLi, CSRF)|encryption & hashing|authentication & MFA|network security & firewalls|OWASP Top 10|malware types|incident response|PKI & TLS|social engineering|vulnerability scanning|zero trust|secure coding|cloud security|compliance",
    "Git": "commit, add, status|branching & merging|rebase vs merge|resolving conflicts|stash|reset vs revert|cherry-pick|remote repos & pull requests|tags & releases|gitignore|log & diff|reflog & recovery|hooks|submodules|workflows (GitFlow)",
    "System_Design": "scalability basics|load balancers|caching strategies|database sharding|CAP & consistency|message queues|microservices vs monolith|rate limiting|CDN|API gateway|replication|designing URL shortener/chat|idempotency|observability|fault tolerance",
    "REST_APIs": "HTTP methods|status codes|resource naming|statelessness|authentication (JWT, OAuth)|pagination & filtering|versioning|idempotency|HATEOAS|error handling|rate limiting|CORS|caching headers|OpenAPI|REST vs GraphQL",
    "Agile": "Scrum roles|sprints & ceremonies|user stories & acceptance criteria|Kanban|backlog refinement|velocity & estimation|retrospectives|Definition of Done|XP practices|agile metrics|scaling agile|product owner duties|continuous improvement|waterfall vs agile|stakeholder management",
    "Business_Intelligence": "KPIs & metrics|dashboards (Power BI, Tableau)|data warehousing|OLAP|ETL for BI|DAX & calculated fields|data visualisation best practices|star schema|data storytelling|self-service BI|SQL for analysts|data cleaning|forecasting basics|cohort & funnel analysis|governance",
    "Technical_Writing": "audience analysis|clear & concise style|API documentation|README & tutorials|release notes|diagrams in docs|style guides|docs-as-code|information architecture|editing & review|accessibility in docs|user manuals|error messages|versioning docs|SEO for docs",
    "Communication": "active listening|written communication & email|presenting technical ideas|giving feedback|conflict resolution|stakeholder communication|meeting effectiveness|cross-team collaboration|asking good questions|non-verbal & remote communication|storytelling|negotiation|explaining to non-technical people|interview communication|empathy",
}.items()}
QUESTION_STYLES = ["short code-output questions", "real-world scenario questions", "debugging / what-is-wrong questions",
                   "conceptual 'why' questions", "best-practice / trade-off questions", "interview-style questions"]


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(150), nullable=True)
    password_hash = db.Column(db.String(200), nullable=False)
    google_id = db.Column(db.String(255), unique=True, nullable=True)
    assessments = db.relationship('Assessment', backref='user', lazy=True)
    def set_password(self, p): self.password_hash = generate_password_hash(p)
    def check_password(self, p): return check_password_hash(self.password_hash, p)


class Assessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    predicted_career = db.Column(db.String(100), nullable=False)
    skill_gaps = db.Column(db.String(500), nullable=False)
    score_summary = db.Column(db.String(500), nullable=True)
    roadmap = db.Column(db.Text, nullable=True)


class QuizSession(db.Model):
    """The running quiz (with answers) is kept in the DB, so a server restart never breaks it."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False, index=True)
    kind = db.Column(db.String(60), nullable=False)         # skill name or '__all__'
    data = db.Column(db.Text, nullable=False)
    created = db.Column(db.DateTime, default=datetime.utcnow)


class QuestionBank(db.Model):
    """Every question Gemini creates is cached here and reused if Gemini is down."""
    id = db.Column(db.Integer, primary_key=True)
    skill = db.Column(db.String(60), nullable=False, index=True)
    qhash = db.Column(db.String(32), nullable=False)
    data = db.Column(db.Text, nullable=False)


class SeenQuestion(db.Model):
    """Questions each user has already received - so the same user never gets a repeat."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False, index=True)
    skill = db.Column(db.String(60), nullable=False, index=True)
    qhash = db.Column(db.String(32), nullable=False)
    text = db.Column(db.String(160), nullable=False)


class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False, index=True)
    role = db.Column(db.String(10), nullable=False)         # 'user' | 'model'
    content = db.Column(db.Text, nullable=False)
    created = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(uid):
    return db.session.get(User, int(uid))


def ensure_schema():
    db.create_all()
    insp = inspect(db.engine)
    acols = {c['name'] for c in insp.get_columns('assessment')}
    ucols = {c['name'] for c in insp.get_columns('user')}
    with db.engine.begin() as conn:                         # upgrade an old app.db automatically
        if 'score_summary' not in acols:
            conn.execute(text("ALTER TABLE assessment ADD COLUMN score_summary VARCHAR(500)"))
        if 'roadmap' not in acols:
            conn.execute(text("ALTER TABLE assessment ADD COLUMN roadmap TEXT"))
        if 'email' not in ucols:
            conn.execute(text('ALTER TABLE "user" ADD COLUMN email VARCHAR(150)'))
        if 'google_id' not in ucols:
            conn.execute(text('ALTER TABLE "user" ADD COLUMN google_id VARCHAR(255)'))

# IMPORTANT for Gunicorn/Render: __main__ is NOT executed by Gunicorn.
# Creating/upgrading the schema here prevents the deployed app from returning 500s
# simply because the database tables have not been initialized.
with app.app_context():
    ensure_schema()


# =====================================================================
#  QUESTION ENGINE  (different questions for every user)
# =====================================================================
BANK_MAX = 300
FORMAT = ('Return ONLY JSON of the form {"questions":[{"question":"...","options":["a","b","c","d"],'
          '"answer":"must be copied exactly from one of the 4 options","level":"easy|medium|hard","skill":"..."}]}. '
          'Exactly 4 distinct options per question, never use "all of the above".')


def _norm(s):
    return re.sub(r'[^a-z0-9+#]', '', str(s).lower())

SKILL_BY_NORM = {_norm(s): s for s in SKILLS_LIST}


def qhash(question):
    return hashlib.md5(str(question).casefold().strip().encode()).hexdigest()


def clean_questions(raw, forced_skill=None):
    out, seen = [], set()
    if isinstance(raw, dict):
        raw = raw.get("questions", [])
    if not isinstance(raw, list):
        return out
    for q in raw:
        if not isinstance(q, dict):
            continue
        qtext = str(q.get("question", "")).strip()
        opts = [str(o).strip() for o in (q.get("options") or [])]
        if not qtext or len(opts) != 4 or len(set(opts)) != 4:
            continue
        ans = str(q.get("answer", "")).strip()
        if ans not in opts:
            m = [o for o in opts if o.casefold() == ans.casefold()]
            if m:
                ans = m[0]
            elif re.fullmatch(r'[A-Da-d]', ans):
                ans = opts["abcd".index(ans.lower())]
            else:
                continue
        skill = forced_skill or SKILL_BY_NORM.get(_norm(q.get("skill", "")))
        if not skill or qtext.casefold() in seen:
            continue
        seen.add(qtext.casefold())
        random.shuffle(opts)
        out.append({"question": qtext, "options": opts, "answer": ans,
                    "level": str(q.get("level", "")).lower(), "skill": skill})
    return out


def save_to_bank(skill, qs):
    try:
        have = {r.qhash for r in QuestionBank.query.filter_by(skill=skill).all()}
        for q in qs:
            if len(have) >= BANK_MAX:
                break
            h = qhash(q["question"])
            if h in have:
                continue
            db.session.add(QuestionBank(skill=skill, qhash=h, data=json.dumps(
                {k: q[k] for k in ("question", "options", "answer", "level")})))
            have.add(h)
        db.session.commit()
    except Exception:
        db.session.rollback()


def seen_hashes(user_id, skill=None):
    q = SeenQuestion.query.filter_by(user_id=user_id)
    if skill:
        q = q.filter_by(skill=skill)
    return {r.qhash for r in q.all()}


def recent_seen_texts(user_id, skill, n=25):
    rows = SeenQuestion.query.filter_by(user_id=user_id, skill=skill).order_by(SeenQuestion.id.desc()).limit(n).all()
    return [r.text for r in rows]


def mark_seen(user_id, qs):
    try:
        for q in qs:
            db.session.add(SeenQuestion(user_id=user_id, skill=q["skill"], qhash=qhash(q["question"]),
                                        text=q["question"][:150]))
        db.session.commit()
    except Exception:
        db.session.rollback()


def offline_pool(skill, avoid=None):
    """Backup questions (database cache + built-in seed). Questions this user has NOT seen come first."""
    avoid = avoid or set()
    pool, seen = [], set()
    items = [json.loads(r.data) for r in QuestionBank.query.filter_by(skill=skill).all()]
    items += [{"question": t[0], "options": [t[1], t[2], t[3], t[4]], "answer": t[1], "level": t[5]}
              for t in SEED.get(skill, [])]
    for q in items:
        k = q["question"].casefold()
        if k not in seen:
            seen.add(k)
            opts = list(q["options"]); random.shuffle(opts)
            pool.append({"question": q["question"], "options": opts, "answer": q["answer"],
                         "level": q.get("level", ""), "skill": skill})
    random.shuffle(pool)
    pool.sort(key=lambda q: qhash(q["question"]) in avoid)      # unseen first (stable sort keeps the shuffle)
    return pool


def previous_pct(user_id, pretty):
    rows = Assessment.query.filter_by(user_id=user_id, predicted_career=f"{pretty} Skill Test") \
        .order_by(Assessment.id.desc()).all()
    m = re.search(r"\((\d+)%\)", rows[0].score_summary or "") if rows else None
    return (int(m.group(1)) if m else None), len(rows)


def skill_prompt(name, pretty, user_id):
    pct, attempts = previous_pct(user_id, pretty)
    if pct is None:
        mix = "8 easy, 9 medium, 8 hard"
    elif pct < 40:
        mix = "12 easy, 9 medium, 4 hard"
    elif pct < 75:
        mix = "6 easy, 12 medium, 7 hard"
    else:
        mix = "3 easy, 9 medium, 13 hard"
    topics = SUBTOPICS.get(name, [])
    focus = ", ".join(random.sample(topics, min(10, len(topics))))
    style = ", ".join(random.sample(QUESTION_STYLES, 3))
    avoid = recent_seen_texts(user_id, name)
    avoid_txt = ("\nThis learner has ALREADY seen these questions - do NOT repeat or rephrase them:\n- "
                 + "\n- ".join(avoid)) if avoid else ""
    return (f"Create 25 different multiple-choice questions that test {pretty}: {mix}. "
            f"Personalised test #{attempts + 1} for learner #{user_id} (random seed {random.randint(1000, 999999)}). "
            f"Make most questions cover these sub-topics: {focus}; use the remaining questions for other sub-topics you choose. "
            f"Preferred styles: {style}. Every question must cover a different idea, no repeats, practical and "
            f"technically correct, vary the position of the correct answer. "
            f"Put \"{pretty}\" in the skill field. {FORMAT}{avoid_txt}")


def fetch_skill_questions(name, pretty, user_id=0, background=False):
    raw = parse_json(gemini(skill_prompt(name, pretty, user_id),
                            system="You are an expert technical examiner. Return only valid JSON.",
                            as_json=True, deadline=45, max_tokens=16000, background=background))
    return clean_questions(raw, name)


def build_skill_quiz(name, user_id):
    pretty = name.replace('_', ' ')
    seen = seen_hashes(user_id, name)
    live, busy_msg = [], ""
    try:
        live_all = fetch_skill_questions(name, pretty, user_id)
        if live_all:
            save_to_bank(name, live_all)
        live = [q for q in live_all if qhash(q["question"]) not in seen]    # never repeat for the same user
    except GeminiError as e:
        busy_msg = why_busy(e)
    qs = live[:25]
    if len(qs) < 25:                                       # top up from the backup bank (unseen first)
        have = {q["question"].casefold() for q in qs}
        qs += [q for q in offline_pool(name, seen) if q["question"].casefold() not in have][:25 - len(qs)]
    random.shuffle(qs)
    for i, q in enumerate(qs, 1):
        q["id"] = i
    if len(live) >= 25:
        label = "25 fresh questions generated live by Google Gemini just for you"
    elif live:
        label = f"{len(live)} questions by Gemini + {len(qs) - len(live)} from the backup bank"
    else:
        label = "Backup question bank (Gemini limit reached)"
        flash(f"{busy_msg or 'Gemini is busy right now.'} Your test was loaded from the backup bank. "
              "Try again after the wait for brand-new Gemini questions.")
    return qs, label


def build_full_quiz(user_id):
    by_skill, busy_msg = {}, ""
    seen = seen_hashes(user_id)
    plan = {s: random.choice(SUBTOPICS[s]) for s in SKILLS_LIST}
    plan_txt = "; ".join(f"{s} -> {t}" for s, t in plan.items())
    try:
        raw = parse_json(gemini(
            "Create exactly 25 multiple-choice questions, exactly ONE for each skill in this list: "
            f"{json.dumps(SKILLS_LIST)}. Base each question on the sub-topic given here: {plan_txt}. "
            f"Mix of easy/medium/hard. Random seed {random.randint(1000, 999999)} for learner #{user_id}. "
            f"Put the skill name exactly as written in the 'skill' field. {FORMAT}",
            system="You are an expert technical examiner. Return only valid JSON.",
            as_json=True, deadline=45, max_tokens=16000))
        for q in clean_questions(raw):
            if qhash(q["question"]) not in seen:
                by_skill.setdefault(q["skill"], q)
        for s, q in by_skill.items():
            save_to_bank(s, [q])
    except GeminiError as e:
        busy_msg = why_busy(e)
    qs = []
    for s in SKILLS_LIST:                                   # always exactly 25 (one per skill)
        if s in by_skill:
            qs.append(by_skill[s])
        else:
            pool = offline_pool(s, seen)
            if pool:
                qs.append(pool[0])
    random.shuffle(qs)
    for i, q in enumerate(qs, 1):
        q["id"] = i
    if len(by_skill) < 25:
        flash(f"{busy_msg or 'Gemini is busy right now.'} Part of this assessment came from the backup bank.")
    return qs


def store_quiz(user_id, kind, qs):
    """Save the running quiz and return its id. Several quizzes can exist at once
    (two browser tabs, back button), so each form carries its own quiz_id."""
    row = QuizSession(user_id=user_id, kind=kind, data=json.dumps(qs))
    db.session.add(row)
    db.session.commit()
    old = (QuizSession.query.filter_by(user_id=user_id).order_by(QuizSession.id.desc()).offset(15).all())
    for o in old:
        db.session.delete(o)
    db.session.commit()
    mark_seen(user_id, qs)
    return row.id


def pop_quiz(user_id, kind, quiz_id=None):
    q = QuizSession.query.filter_by(user_id=user_id, kind=kind)
    if quiz_id and str(quiz_id).isdigit():
        row = q.filter_by(id=int(quiz_id)).first()
    else:
        row = q.order_by(QuizSession.id.desc()).first()
    if not row:
        return None
    qs = json.loads(row.data)
    db.session.delete(row)
    db.session.commit()
    return qs


# ---------- offline analysis (used only when Gemini cannot write the analysis) ----------
def local_analysis(pretty, correct, total, pct, wrong_topics, skill_key=None):
    level = "Beginner" if pct < 40 else "Intermediate" if pct < 75 else "Advanced"
    roles = [j["title"] for j in JOB_ROLES if skill_key in j["key_skills"]][:3] or ["Software Engineer"]
    gaps = "\n".join(f"   - {g}" for g in wrong_topics[:8]) or "   - No major gaps found. Great job!"
    return (f"(Gemini limit was reached, so this analysis was created by the built-in advisor.)\n\n"
            f"1) Skill level: {level} ({correct}/{total}, {pct}%)\n\n"
            f"2) Skill gaps - review these topics:\n{gaps}\n\n"
            f"3) 4-week roadmap for {pretty}:\n"
            f"   Week 1 - Revise the fundamentals and the topics you missed.\n"
            f"   Week 2 - Build 2-3 small hands-on exercises.\n"
            f"   Week 3 - Build one mini project and solve practice questions.\n"
            f"   Week 4 - Take this test again, polish the project and publish it on GitHub.\n\n"
            f"4) Fitting roles: {', '.join(roles)}. Next step: build a portfolio project using {pretty}.\n\n"
            f"5) Free resources: official documentation, freeCodeCamp, roadmap.sh, YouTube tutorials.")


# =====================================================================
#  AUTH  (register / login / logout)
# =====================================================================
@app.route('/')
def index():
    return redirect(url_for('dashboard' if current_user.is_authenticated else 'login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''
        if not re.fullmatch(r'[A-Za-z0-9_.-]{3,40}', username):
            flash('Username must be 3-40 characters and use only letters, numbers, dot, underscore or hyphen.')
            return redirect(url_for('register'))
        if len(password) < 6:
            flash('Password must contain at least 6 characters.')
            return redirect(url_for('register'))
        if User.query.filter(db.func.lower(User.username) == username.lower()).first():
            flash('Username already exists. Please choose another or log in.')
            return redirect(url_for('register'))
        if email and User.query.filter(db.func.lower(User.email) == email).first():
            flash('This email is already registered. Please log in instead.')
            return redirect(url_for('register'))
        u = User(username=username, email=email or None)
        u.set_password(password)
        db.session.add(u)
        db.session.commit()
        login_user(u, remember=True, fresh=True)
        session.permanent = True
        flash(f'Welcome, {username}! Your account is ready.')
        return redirect(url_for('dashboard'))
    return render_template('login.html', register_mode=True, google_enabled=google_enabled())


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        identifier = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        # Accept either username OR email. This prevents the common
        # "registered but cannot log in again" confusion.
        u = User.query.filter(db.func.lower(User.username) == identifier.lower()).first()
        if not u and '@' in identifier:
            u = User.query.filter(db.func.lower(User.email) == identifier.lower()).first()
        if u and u.password_hash and u.check_password(password):
            login_user(u, remember=True, fresh=True)
            session.permanent = True
            return redirect(url_for('dashboard'))
        flash('Invalid username/email or password. Please try again.')
    return render_template('login.html', register_mode=False, google_enabled=google_enabled())


@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for('login'))


@app.route('/auth/google')
def google_login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if not google_enabled():
        flash('Google Sign-In is not configured yet. Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in the environment.')
        return redirect(url_for('login'))
    redirect_uri = GOOGLE_REDIRECT_URI or url_for('google_callback', _external=True)
    return oauth_google.authorize_redirect(redirect_uri)


@app.route('/auth/google/callback')
def google_callback():
    if not google_enabled():
        flash('Google Sign-In is not configured.')
        return redirect(url_for('login'))
    try:
        token = oauth_google.authorize_access_token()
        userinfo = token.get('userinfo')
        if not userinfo:
            response = oauth_google.get('https://openidconnect.googleapis.com/v1/userinfo')
            response.raise_for_status()
            userinfo = response.json()
        google_id = str(userinfo.get('sub') or '').strip()
        email = str(userinfo.get('email') or '').strip().lower()
        name = str(userinfo.get('name') or userinfo.get('given_name') or 'Google User').strip()
        if not google_id or not email:
            flash('Google did not return a usable email address. Please try again.')
            return redirect(url_for('login'))

        u = User.query.filter_by(google_id=google_id).first()
        if not u:
            u = User.query.filter(db.func.lower(User.email) == email).first()
        if not u:
            base = re.sub(r'[^A-Za-z0-9_.-]', '', name.replace(' ', '_'))[:32] or 'google_user'
            username = base
            n = 1
            while User.query.filter(db.func.lower(User.username) == username.lower()).first():
                n += 1
                username = f'{base}_{n}'[:40]
            u = User(username=username, email=email, google_id=google_id,
                     password_hash=generate_password_hash(secrets.token_urlsafe(32)))
            db.session.add(u)
        else:
            if not u.email:
                u.email = email
            u.google_id = google_id
        db.session.commit()
        login_user(u, remember=True, fresh=True)
        session.permanent = True
        return redirect(url_for('dashboard'))
    except Exception:
        app.logger.exception('Google OAuth callback failed')
        flash('Google Sign-In could not be completed. Check the Google OAuth redirect URI and try again.')
        return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    q = Assessment.query.filter_by(user_id=current_user.id).order_by(Assessment.id.desc())
    return render_template('dashboard.html', assessment=q.first(), history=q.limit(8).all())


# =====================================================================
#  1) SINGLE-SKILL TEST  (25 questions per skill, 25 skills)
# =====================================================================
@app.route('/skills')
@login_required
def skills():
    return render_template('skills.html', skills=SKILLS_LIST)


@app.route('/skill/<name>', methods=['GET', 'POST'])
@login_required
def skill_test(name):
    if name not in SKILLS_LIST:
        return redirect(url_for('skills'))
    pretty = name.replace('_', ' ')
    if request.method == 'POST':
        qs = pop_quiz(current_user.id, name, request.form.get('quiz_id'))
        if not qs:
            flash('This test was already submitted (or reopened in another tab). Your result is saved on the Dashboard.')
            return redirect(url_for('dashboard'))
        correct, wrong, review = 0, [], []
        for q in qs:
            ans = (request.form.get(f"q_{q['id']}") or '').strip()
            ok = ans == q['answer'].strip()
            correct += ok
            if not ok:
                wrong.append(f"{q['question']} [{q.get('level', '')}]")
            review.append({**q, "your": ans, "ok": ok})
        pct = round(correct * 100 / len(qs))
        wrong_topics = [w.split(' [')[0] for w in wrong]
        try:
            analysis = gemini(
                f"A learner took a {pretty} test and scored {correct}/{len(qs)} ({pct}%).\n"
                f"Questions answered wrongly:\n" + "\n".join(wrong) + "\n\n"
                "Reply with: 1) Skill level (Beginner/Intermediate/Advanced) 2) Specific skill gaps "
                "3) A 4-week learning roadmap 4) Three career roles that fit + next step 5) Free resources.",
                system="You are a professional technical career advisor. Be clear and concise.", deadline=45)
        except GeminiError as e:
            analysis = local_analysis(pretty, correct, len(qs), pct, wrong_topics, name)
            flash(why_busy(e) + " The analysis was written by the built-in advisor this time.")
        db.session.add(Assessment(user_id=current_user.id, predicted_career=f"{pretty} Skill Test",
                                  skill_gaps=("; ".join(wrong_topics))[:490] or "None",
                                  score_summary=f"Score: {correct}/{len(qs)} Correct ({pct}%)", roadmap=analysis))
        db.session.commit()
        return render_template('skill_result.html', skill=pretty, correct=correct, total=len(qs),
                               pct=pct, analysis=analysis, review=review)
    qs, label = build_skill_quiz(name, current_user.id)
    qid = store_quiz(current_user.id, name, qs)
    return render_template('skill_quiz.html', skill=pretty, name=name, questions=qs, label=label, quiz_id=qid)


# =====================================================================
#  2) FULL ASSESSMENT  (1 question for each of the 25 skills)
# =====================================================================
@app.route('/assessment', methods=['GET', 'POST'])
@login_required
def assessment():
    if request.method == 'POST':
        questions = pop_quiz(current_user.id, '__all__', request.form.get('quiz_id'))
        if not questions:
            flash('This assessment was already submitted. Your result is saved on the Dashboard.')
            return redirect(url_for('dashboard'))
        passed, failed = [], []
        scores = {s: 0 for s in SKILLS_LIST}
        for q in questions:
            if (request.form.get(f"q_{q['id']}") or '').strip() == q['answer'].strip():
                passed.append(q['skill']); scores[q['skill']] = 10
            else:
                failed.append(q['skill'])
        best, top = "Full-Stack Developer", -1
        for job in JOB_ROLES:
            sc = sum(scores.get(s, 0) for s in job['key_skills'])
            if sc > top:
                best, top = job['title'], sc
        gaps = ", ".join(failed) if failed else "None"
        try:
            roadmap = gemini(f"Target Role: {best}.\nPassed Skills: {', '.join(passed) or 'None'}.\n"
                             f"Failed Skill Gaps: {gaps}.\nProvide a 30-day learning roadmap to bridge these skill gaps.",
                             system="You are a professional technical career advisor.", deadline=45)
        except GeminiError as e:
            flash(why_busy(e) + " The roadmap was written by the built-in advisor this time.")
            if failed:
                roadmap = ("(Gemini limit reached - built-in 30-day plan)\n"
                           "Week 1 - Fundamentals of: " + ", ".join(failed[:4]) + "\n"
                           "Week 2 - Hands-on practice for: " + ", ".join(failed[4:8] or failed[:4]) + "\n"
                           f"Week 3 - Build a mini project for a {best} portfolio\n"
                           "Week 4 - Retake the assessment and publish your project on GitHub")
            else:
                roadmap = f"Excellent! You passed every skill. Build advanced projects for a {best} role."
        db.session.add(Assessment(user_id=current_user.id, predicted_career=best, skill_gaps=gaps[:490],
                                  score_summary=f"Score: {len(passed)}/{len(questions)} Correct", roadmap=roadmap))
        db.session.commit()
        return redirect(url_for('dashboard'))
    qs = build_full_quiz(current_user.id)
    qid = store_quiz(current_user.id, '__all__', qs)
    return render_template('index.html', questions=qs, quiz_id=qid)


# =====================================================================
#  CHAT (remembers the conversation per user) + STATUS
# =====================================================================
@app.route('/chat', methods=['POST'])
@login_required
def chat():
    msg = (request.get_json(silent=True) or {}).get("message", "").strip()
    if not msg:
        return jsonify({"reply": "Please enter a message."})
    latest = Assessment.query.filter_by(user_id=current_user.id).order_by(Assessment.id.desc()).first()
    ctx = f"Latest result: {latest.predicted_career}. Gaps: {latest.skill_gaps}." if latest else "No test taken yet."
    rows = ChatMessage.query.filter_by(user_id=current_user.id).order_by(ChatMessage.id.desc()).limit(8).all()
    history = [(r.role, r.content[:1500]) for r in reversed(rows)]
    try:
        reply = gemini(msg, deadline=45, history=history, system=(
            "You are an AI Career Assistant. Help with technical careers, code debugging and interview prep. "
            f"User context: {ctx}"))
    except GeminiError as e:
        return jsonify({"reply": why_busy(e) + " Please send your message again in a few seconds."})
    db.session.add(ChatMessage(user_id=current_user.id, role="user", content=msg))
    db.session.add(ChatMessage(user_id=current_user.id, role="model", content=reply))
    db.session.commit()
    return jsonify({"reply": reply})


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'database': 'configured',
        'gemini_key_loaded': gemini_ready(),
        'google_sign_in_configured': google_enabled()
    })


@app.route('/gemini-status')
def gemini_status():
    """Open http://127.0.0.1:5000/gemini-status to see your Gemini setup.
       Add ?test=1 to make one real test call:  /gemini-status?test=1"""
    now = time.time()
    info = {"key_loaded": gemini_ready(), "models_in_order": MODEL_CHAIN, "dead_models": sorted(DEAD_MODELS),
            "cooling_down_seconds": {m: int(t - now) for m, t in _COOL.items() if t > now},
            "calls_per_minute_limit_used": GEMINI_RPM}
    if request.args.get("test"):
        try:
            info["reply"] = gemini("Reply with the single word: OK", deadline=25)
            info["working_model"] = LAST_GOOD["model"]
            info["status"] = "GEMINI IS WORKING"
        except GeminiError as e:
            info["status"] = "FAILED"
            info["error"] = str(e)
    else:
        info["status"] = "Key found. Add ?test=1 to run a live test." if gemini_ready() else "NO API KEY IN .env"
    return jsonify(info)


@app.errorhandler(500)
def internal_error(error):
    # Keep the browser from showing a blank generic Internal Server Error page.
    # Full details stay in Render logs, not in the public response.
    db.session.rollback()
    app.logger.error('Unhandled 500 error: %s\n%s', error, traceback.format_exc())
    if request.path.startswith('/chat'):
        return jsonify({'reply': 'The server had a temporary problem. Please try again.'}), 500
    return render_template('error.html'), 500


# =====================================================================
#  OPTIONAL BACKGROUND WARM-UP (off by default - it uses your free quota!)
#  It only runs when WARM_BANK=1 and always leaves free slots for real users.
# =====================================================================
def _warm_bank():
    time.sleep(30)
    while True:
        busy = False
        for s in SKILLS_LIST:
            with app.app_context():
                if QuestionBank.query.filter_by(skill=s).count() >= 50:
                    continue
                try:
                    save_to_bank(s, fetch_skill_questions(s, s.replace('_', ' '), 0, background=True))
                except GeminiKeyError:
                    return
                except GeminiError:
                    busy = True
                    break
            time.sleep(20)
        time.sleep(600 if busy else 3600)


if __name__ == '__main__':
    # Local development only. Render uses: gunicorn app:app
    if os.getenv("WARM_BANK", "0") == "1" and gemini_ready():
        threading.Thread(target=_warm_bank, daemon=True).start()
    port = int(os.getenv('PORT', '5000'))
    print('Gemini key loaded:', gemini_ready(), '| first model:', MODEL_CHAIN[0])
    print('Google Sign-In configured:', google_enabled())
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
