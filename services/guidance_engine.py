"""
Guidance Engine
Loads trained model + skill taxonomy and produces full career guidance:
- Top 3 career path predictions with confidence %
- Skills the student already has (good skills)
- Skill gaps (required skills they're missing)
- Good-to-have skills for their top career
- Improvement areas
- Recommended courses per gap
"""

import pickle
import json
import os

# ─────────────────────────────────────────────
# Load model and data files once at startup
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

career_model = pickle.load(open(os.path.join(BASE_DIR, "ml/career_classifier.pkl"), "rb"))
skill_data   = json.load(open(os.path.join(BASE_DIR, "ml/skill_data.json")))
course_map   = json.load(open(os.path.join(BASE_DIR, "ml/course_map.json")))


def get_guidance(feature_text: str, student_skills: list) -> dict:
    """
    Core function: predict career paths and generate full guidance.

    Args:
        feature_text   : combined text string from feature_builder
        student_skills : merged list of student's known skills (lowercased)

    Returns:
        Full guidance dict ready to be returned as API response
    """

    # ── Step 1: Predict top 3 career paths ──────────────────
    proba = career_model.predict_proba([feature_text])[0]
    career_labels = career_model.classes_

    top3_indices = proba.argsort()[-3:][::-1]
    top3_careers = [
        {
            "career": career_labels[i],
            "confidence_percent": round(proba[i] * 100, 1)
        }
        for i in top3_indices
    ]

    primary_career = top3_careers[0]["career"]

    # ── Step 2: Skill analysis for primary career ────────────
    career_info = skill_data.get(primary_career, {})
    required_skills  = set(career_info.get("required_skills", []))
    good_to_have     = career_info.get("good_to_have", [])
    improvement_areas = career_info.get("improvement_areas", [])

    student_skill_set = set(s.lower() for s in student_skills)

    # Skills the student already has that are relevant
    skills_you_have = sorted(list(required_skills & student_skill_set))

    # Required skills they are missing = gaps
    skill_gaps = sorted(list(required_skills - student_skill_set))

    # Good to have skills they don't have yet
    missing_good_to_have = [s for s in good_to_have if s.lower() not in student_skill_set]

    # ── Step 3: Map skill gaps to courses ───────────────────
    recommended_courses = []
    for gap in skill_gaps:
        if gap in course_map:
            recommended_courses.append({
                "skill": gap,
                "course": course_map[gap]["course"],
                "platform": course_map[gap]["platform"],
                "url": course_map[gap]["url"]
            })

    # Also suggest 1-2 good-to-have courses
    bonus_courses = []
    for skill in missing_good_to_have[:2]:
        if skill in course_map:
            bonus_courses.append({
                "skill": skill,
                "course": course_map[skill]["course"],
                "platform": course_map[skill]["platform"],
                "url": course_map[skill]["url"]
            })

    # ── Step 4: Skill breakdown for second and third careers ─
    alternative_career_skills = []
    for career_entry in top3_careers[1:]:
        c_name = career_entry["career"]
        c_info = skill_data.get(c_name, {})
        c_required = set(c_info.get("required_skills", []))
        alternative_career_skills.append({
            "career": c_name,
            "confidence_percent": career_entry["confidence_percent"],
            "skills_you_have": sorted(list(c_required & student_skill_set)),
            "skill_gaps": sorted(list(c_required - student_skill_set))
        })

    # ── Step 5: Assemble full guidance response ──────────────
    return {
        "top_career_recommendations": top3_careers,
        "primary_career": {
            "name": primary_career,
            "confidence_percent": top3_careers[0]["confidence_percent"],
            "skills_you_have": skills_you_have,
            "skill_gaps": skill_gaps,
            "good_to_have_skills": missing_good_to_have,
            "improvement_areas": improvement_areas,
            "recommended_courses": recommended_courses,
            "bonus_courses_for_growth": bonus_courses
        },
        "alternative_careers": alternative_career_skills,
        "summary": _build_summary(primary_career, skills_you_have, skill_gaps, improvement_areas)
    }


def _build_summary(career: str, have: list, gaps: list, improve: list) -> str:
    """Build a short human-readable summary text"""
    have_str  = ", ".join(have[:3]) if have else "none matched yet"
    gap_str   = ", ".join(gaps[:3]) if gaps else "you are well-equipped"
    imp_str   = ", ".join(improve[:2]) if improve else "keep building projects"

    return (
        f"Based on your profile, you are well-suited for a career as a {career}. "
        f"Your strongest relevant skills include: {have_str}. "
        f"Key areas to work on: {gap_str}. "
        f"Focus on improving: {imp_str}."
    )
