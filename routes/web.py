"""
Web Route — Showcase Website
Serves the demo UI and handles PDF + form submission
"""

from flask import Blueprint, request, jsonify, render_template_string
import pdfplumber
import io
import traceback
import os

from services.resume_parser import parse_resume
from services.feature_builder import build_feature_text, merge_skills
from services.guidance_engine import get_guidance

web_bp = Blueprint("web", __name__)

# Load the HTML template once
TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "index.html")


@web_bp.route("/", methods=["GET"])
def index():
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    return render_template_string(html)


@web_bp.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Accepts multipart/form-data:
      - resume       : PDF file
      - interests, known_skills, career_goal, projects_done,
        education_branch, year_of_study, has_internship,
        self_weakness, preferred_work
    """
    try:
        # ── Get uploaded PDF ─────────────────────────────
        resume_file = request.files.get("resume")
        if not resume_file:
            return jsonify({"error": "No resume file uploaded"}), 400

        pdf_bytes = resume_file.read()
        resume_text = ""
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    resume_text += page.extract_text() or ""
        except Exception:
            return jsonify({"error": "Could not read PDF. Make sure it's a valid PDF file."}), 422

        if not resume_text.strip():
            return jsonify({"error": "PDF appears to be empty or image-based. Please use a text-based PDF."}), 422

        # ── Get Q&A fields ────────────────────────────────
        qa = {
            "interests":        request.form.get("interests", ""),
            "known_skills":     request.form.get("known_skills", ""),
            "career_goal":      request.form.get("career_goal", ""),
            "projects_done":    request.form.get("projects_done", ""),
            "education_branch": request.form.get("education_branch", ""),
            "year_of_study":    request.form.get("year_of_study", ""),
            "has_internship":   request.form.get("has_internship", "false") == "true",
            "self_weakness":    request.form.get("self_weakness", ""),
            "preferred_work":   request.form.get("preferred_work", ""),
        }

        # ── Parse resume ──────────────────────────────────
        parsed = parse_resume(resume_text)

        if qa.get("education_branch"):
            parsed["education_branch"] = qa["education_branch"]
        parsed["has_internship"] = qa["has_internship"]

        # ── Build features & run model ────────────────────
        feature_text   = build_feature_text(parsed, qa)
        student_skills = merge_skills(parsed["skills"], qa.get("known_skills", ""))
        guidance       = get_guidance(feature_text, student_skills)

        return jsonify({
            "status": "success",
            "student_profile": {
                "skills_detected":  student_skills,
                "education_branch": parsed["education_branch"],
                "education_degree": parsed["education_degree"],
                "cgpa":             parsed["cgpa"],
                "has_internship":   parsed["has_internship"],
                "projects_found":   parsed["projects"]
            },
            "guidance": guidance
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500