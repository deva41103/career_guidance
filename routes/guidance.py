"""
Guidance API Route
POST /api/generate-guidance
"""

from flask import Blueprint, request, jsonify
from services.resume_parser import parse_resume
from services.feature_builder import build_feature_text, merge_skills
from services.guidance_engine import get_guidance
import requests
import traceback

guidance_bp = Blueprint("guidance", __name__)


def fetch_resume_text(pdf_url: str) -> str:
    """Download PDF from Supabase URL and extract text"""
    try:
        import pdfplumber, io
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text.strip()
    except ImportError:
        # pdfplumber not available in this env — return empty string
        return ""
    except Exception as e:
        raise ValueError(f"Failed to fetch/parse resume PDF: {str(e)}")


@guidance_bp.route("/generate-guidance", methods=["POST"])
def generate_career_guidance():
    """
    Expected JSON body:
    {
        "resume_url": "https://supabase.../resume.pdf",   ← optional if resume_text provided
        "resume_text": "raw text...",                      ← optional if resume_url provided
        "qa_responses": {
            "interests": "...",
            "known_skills": "python, sql, pandas",
            "career_goal": "...",
            "projects_done": "...",
            "education_branch": "Computer Science",
            "year_of_study": "3rd year",
            "has_internship": false,
            "self_weakness": "..."
        }
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        qa = data.get("qa_responses", {})
        if not qa:
            return jsonify({"error": "qa_responses is required"}), 400

        # ── Get resume text ──────────────────────────────────
        resume_text = data.get("resume_text", "")
        resume_url  = data.get("resume_url", "")

        if not resume_text and resume_url:
            resume_text = fetch_resume_text(resume_url)

        if not resume_text:
            return jsonify({"error": "Provide either resume_url or resume_text"}), 400

        # ── Parse resume ─────────────────────────────────────
        parsed = parse_resume(resume_text)

        # Override with Q&A values if more specific
        if qa.get("education_branch"):
            parsed["education_branch"] = qa["education_branch"]
        if qa.get("has_internship") is not None:
            parsed["has_internship"] = bool(qa["has_internship"])

        # ── Build features ───────────────────────────────────
        feature_text   = build_feature_text(parsed, qa)
        student_skills = merge_skills(parsed["skills"], qa.get("known_skills", ""))

        # ── Run guidance engine ──────────────────────────────
        guidance = get_guidance(feature_text, student_skills)

        return jsonify({
            "status": "success",
            "student_profile": {
                "skills_detected": student_skills,
                "education_branch": parsed["education_branch"],
                "education_degree": parsed["education_degree"],
                "cgpa": parsed["cgpa"],
                "has_internship": parsed["has_internship"],
                "projects_found": parsed["projects"]
            },
            "guidance": guidance
        }), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500
