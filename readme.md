# CareerLens — ML-Based Career Guidance System

A machine learning system that analyzes a student's resume and questionnaire answers to provide personalized career path recommendations, skill gap analysis, and course suggestions.

Built with Python, Flask, scikit-learn, and pdfplumber. No external AI APIs used — everything runs locally.

---

## What It Does

Upload a resume PDF and answer 8 questions. The system will give you:

- **Top 3 career path recommendations** with confidence percentages
- **Skills you already have** that are relevant to your best-fit career
- **Skill gaps** — required skills you're missing
- **Good-to-have skills** for your target career
- **Improvement areas** to focus on
- **Recommended courses** (with platform links) for each skill gap

---

## Career Labels Supported (15 total)

| | | |
|---|---|---|
| Software Engineer | Frontend Developer | Backend Engineer |
| Full Stack Developer | Mobile Developer | Data Scientist |
| ML Engineer | Data Analyst | DevOps Engineer |
| Cybersecurity Analyst | UI/UX Designer | Business Analyst |
| QA Engineer | Cloud Engineer | Embedded Systems Engineer |

---

## Project Structure

```
career_guidance_ml/
│
├── app.py                        # Flask entry point (port 5000)
├── requirements.txt              # Python dependencies
├── test_pipeline.py              # End-to-end pipeline test (no server needed)
│
├── data/
│   ├── generate_dataset.py       # Generates synthetic training data (7500 rows)
│   └── student_profiles.csv      # Generated dataset (auto-created on running above)
│
├── ml/
│   ├── train.py                  # Model training script
│   ├── career_classifier.pkl     # Trained model (auto-created after training)
│   ├── model_metadata.json       # Accuracy report and label list
│   ├── skill_data.json           # Skill taxonomy for all 15 careers
│   └── course_map.json           # Skill → course/platform/URL mapping
│
├── services/
│   ├── skill_keywords.py         # Master list of 120+ tech skill keywords
│   ├── resume_parser.py          # Extracts skills, education, CGPA from resume text
│   ├── feature_builder.py        # Combines resume + Q&A into ML input
│   └── guidance_engine.py        # Runs model, builds full guidance output
│
├── routes/
│   ├── guidance.py               # API route: POST /api/generate-guidance (for app)
│   └── web.py                    # Web route: POST /api/analyze (for website)
│
└── templates/
    └── index.html                # Showcase website (multi-step form + results)
```

---

## How It Works

```
Resume PDF + Q&A answers
         │
         ▼
   pdfplumber extracts text from PDF
         │
         ▼
   Resume Parser scans text for 120+ skill keywords,
   detects degree, branch, CGPA, internship, projects
         │
         ▼
   Feature Builder merges resume data + Q&A
   into a single text string
         │
         ▼
   TF-IDF Vectorizer converts text → numbers
         │
         ▼
   Random Forest Classifier → Top 3 career predictions
   with confidence percentages
         │
         ▼
   Skill Taxonomy (skill_data.json) compares
   student skills vs career requirements
         │
         ▼
   Course Map (course_map.json) maps each
   skill gap to a real course
         │
         ▼
   Full guidance JSON → displayed on website
```

---

## Setup and Installation

### Requirements

- Python 3.9 or higher
- pip
- Virtual environment (recommended)

---

### Step 1 — Clone or download the project

```bash
# If you have git
git clone <your-repo-url>
cd career_guidance_ml

# Or unzip the downloaded folder and navigate into it
cd career_guidance_ml
```

---

### Step 2 — Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt includes:**
```
flask==3.0.0
flask-cors==4.0.0
scikit-learn==1.3.2
pandas==2.1.0
numpy==1.26.0
pdfplumber==0.10.3
requests==2.31.0
joblib==1.3.2
gunicorn==21.2.0
python-dotenv==1.0.0
```

---

### Step 4 — Generate the training dataset

```bash
python data/generate_dataset.py
```

**What this does:**
Generates 7,500 synthetic student profiles — 500 per career label — and saves them to `data/student_profiles.csv`. Each profile contains skills, interests, career goals, projects, education branch, and a career label.

**Expected output:**
```
Generating synthetic student profile dataset...
  Generating 500 samples for: Software Engineer
  Generating 500 samples for: Frontend Developer
  ...
Dataset saved: data/student_profiles.csv
Total records  : 7500
Career labels  : 15
```

---

### Step 5 — Train the ML model

```bash
python ml/train.py
```

**What this does:**
Trains 3 classifiers (Random Forest, Logistic Regression, Linear SVC) on the dataset using TF-IDF features. Compares them using 5-fold cross-validation, selects the best one, and saves it as `ml/career_classifier.pkl`.

**Expected output:**
```
Training: Random Forest...
  CV F1 (5-fold): 1.0000 ± 0.0000
  Test Accuracy : 1.0000

Training: Logistic Regression...
  ...

[BEST MODEL] Random Forest
[SAVED] ml/career_classifier.pkl
[SAVED] ml/model_metadata.json

[TEST INFERENCE] Running sample predictions...
  Profile: Data Science student
  1. Data Scientist     93.0%
  2. Data Analyst        4.7%
  3. ML Engineer         1.7%
```

> **Note:** The model file `career_classifier.pkl` (~27 MB) is created in the `ml/` folder after this step. The training takes 1–3 minutes depending on your machine.

---

### Step 6 — (Optional) Run the pipeline test

This tests the full system without starting the Flask server.

```bash
python test_pipeline.py
```

You should see career guidance output for 4 mock student profiles (Data Scientist, Mobile Developer, Cybersecurity Analyst, Frontend Developer).

---

### Step 7 — Start the Flask server

```bash
python app.py
```

**Expected output:**
```
 * Running on http://0.0.0.0:5000
 * Debug mode: on
```

---

### Step 8 — Open the website

Open your browser and go to:

```
http://127.0.0.1:5000
```

You will see the CareerLens showcase website. Fill in the 3-step form:
1. Upload your resume (PDF, text-based, not scanned)
2. Fill in your background (branch, year, skills, projects, internship)
3. Fill in your goals (interests, career goal, weaknesses, work preference)

Click **Analyze my profile** and wait ~5 seconds for results.

---

### Step 9 — Verify the API health check

```
http://127.0.0.1:5000/health
```

Should return:
```json
{ "status": "ok", "service": "Career Guidance API" }
```

---

## API Reference

### Web Endpoint (used by the website)

```
POST /api/analyze
Content-Type: multipart/form-data
```

| Field | Type | Description |
|-------|------|-------------|
| `resume` | File (PDF) | Student's resume |
| `interests` | string | What the student is interested in |
| `known_skills` | string | Comma-separated skills |
| `career_goal` | string | What career they want |
| `projects_done` | string | Projects they've built |
| `education_branch` | string | e.g. Computer Science |
| `year_of_study` | string | e.g. 3rd year |
| `has_internship` | string | "true" or "false" |
| `self_weakness` | string | Areas they feel weak in |
| `preferred_work` | string | startup / product company / remote |

---

### App Integration Endpoint (for mobile/web app)

```
POST /api/generate-guidance
Content-Type: application/json
```

```json
{
  "resume_url": "https://your-supabase-url/resume.pdf",
  "qa_responses": {
    "interests": "machine learning, data science",
    "known_skills": "python, sql, pandas",
    "career_goal": "want to become a data scientist",
    "projects_done": "movie recommendation system",
    "education_branch": "Computer Science",
    "year_of_study": "3rd year",
    "has_internship": false,
    "self_weakness": "weak in deep learning"
  }
}
```

---

### Sample Response

```json
{
  "status": "success",
  "student_profile": {
    "skills_detected": ["python", "sql", "pandas", "numpy"],
    "education_branch": "Computer Science",
    "education_degree": "B.Tech",
    "cgpa": 8.2,
    "has_internship": false
  },
  "guidance": {
    "top_career_recommendations": [
      { "career": "Data Scientist", "confidence_percent": 81.3 },
      { "career": "ML Engineer",    "confidence_percent": 9.7 },
      { "career": "Data Analyst",   "confidence_percent": 4.2 }
    ],
    "primary_career": {
      "name": "Data Scientist",
      "confidence_percent": 81.3,
      "skills_you_have": ["python", "sql", "pandas", "statistics"],
      "skill_gaps": ["machine learning", "probability"],
      "good_to_have_skills": ["deep learning", "tensorflow", "spark"],
      "improvement_areas": ["feature engineering", "model evaluation"],
      "recommended_courses": [
        {
          "skill": "machine learning",
          "course": "Machine Learning Specialization",
          "platform": "Coursera",
          "url": "https://coursera.org/..."
        }
      ]
    },
    "summary": "Based on your profile, you are well-suited for a career as a Data Scientist..."
  }
}
```

---

## ML Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest (300 trees) |
| Vectorizer | TF-IDF (bigrams, 8000 features, sublinear_tf) |
| Training samples | 6,000 (80% of 7,500) |
| Test samples | 1,500 (20% of 7,500) |
| Career labels | 15 |
| Cross-validation | 5-fold Stratified |
| Model file size | ~27 MB |

The pipeline is saved as a single `.pkl` file that contains both the vectorizer and classifier — no separate vectorizer file needed.

---

## Common Issues

**`ModuleNotFoundError: No module named 'pdfplumber'`**
```bash
pip install pdfplumber
```

**`career_classifier.pkl not found`**
You skipped Step 5. Run `python ml/train.py` first.

**`PDF appears to be empty or image-based`**
The uploaded PDF is a scanned image, not text-based. Use a PDF that was exported from Word/LaTeX/Google Docs.

**Port 5000 already in use**
```bash
# Change port in app.py
app.run(debug=True, host="0.0.0.0", port=5001)
```

**Flask ImportError on routes**
Make sure `routes/__init__.py` and `services/__init__.py` exist. Create them if missing:
```bash
# Windows
type nul > routes\__init__.py
type nul > services\__init__.py

# macOS / Linux
touch routes/__init__.py
touch services/__init__.py
```

---

## Future Improvements

- Replace synthetic training data with real resume datasets from Kaggle
- Add sentence-transformers embeddings for better semantic matching
- Store results in Supabase for the mobile app
- Add user authentication via Supabase Auth
- Deploy Flask API on Railway or Render for production

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| ML | scikit-learn (Random Forest + TF-IDF) |
| PDF Parsing | pdfplumber |
| Frontend | HTML, CSS, Vanilla JS |
| Data | Pandas, NumPy |
| Deployment | Gunicorn (production) |