"""
End-to-End Pipeline Test
Tests the full guidance pipeline with mock student profiles
(no Flask server needed — tests services directly)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.resume_parser import parse_resume
from services.feature_builder import build_feature_text, merge_skills
from services.guidance_engine import get_guidance
import json

# ─────────────────────────────────────────────
# MOCK STUDENT PROFILES
# ─────────────────────────────────────────────
test_cases = [
    {
        "name": "Ravi - Aspiring Data Scientist",
        "resume_text": """
            Ravi Kumar
            B.Tech Computer Science | 3rd Year | CGPA: 8.2

            Skills: Python, SQL, Pandas, NumPy, Data Visualization, Git

            Projects:
            - Movie Recommendation System using collaborative filtering
            - House Price Prediction using linear regression

            Education:
            Anna University | B.Tech Computer Science | 2021-2025
        """,
        "qa": {
            "interests": "machine learning, data science, artificial intelligence",
            "known_skills": "python, sql, pandas, numpy, statistics",
            "career_goal": "want to become a data scientist at a top tech company",
            "projects_done": "movie recommendation system, house price prediction",
            "education_branch": "Computer Science",
            "year_of_study": "3rd year",
            "has_internship": False,
            "self_weakness": "weak in deep learning and big data technologies"
        }
    },
    {
        "name": "Priya - Flutter Mobile Developer",
        "resume_text": """
            Priya Sharma
            B.Tech Information Technology | Final Year | CGPA: 7.8

            Skills: Flutter, Dart, Firebase, Android, REST API, Git, UI Design

            Internship: Mobile App Developer Intern at XYZ Startup (2 months)

            Projects:
            - Food Delivery App built with Flutter and Firebase
            - Expense Tracker App with local SQLite storage
            - Quiz App with Firebase backend

            Education:
            VIT University | B.Tech IT | 2020-2024
        """,
        "qa": {
            "interests": "mobile app development, flutter, cross-platform apps",
            "known_skills": "flutter, dart, firebase, android, git, rest api",
            "career_goal": "flutter developer at a product company",
            "projects_done": "food delivery app, expense tracker, quiz app",
            "education_branch": "Information Technology",
            "year_of_study": "final year",
            "has_internship": True,
            "self_weakness": "need to learn state management better and ios development"
        }
    },
    {
        "name": "Arjun - Security Enthusiast",
        "resume_text": """
            Arjun Nair
            B.Tech Computer Science | 2nd Year | CGPA: 7.1

            Skills: Networking, Linux, Python, Kali Linux, Wireshark, Ethical Hacking

            Projects:
            - Network Vulnerability Scanner using Python
            - CTF Challenge Writeups (5 completed)

            Education:
            NIT Trichy | B.Tech CSE | 2022-2026
        """,
        "qa": {
            "interests": "cybersecurity, ethical hacking, network security, ctf challenges",
            "known_skills": "networking, linux, python, kali linux, wireshark, ethical hacking",
            "career_goal": "cybersecurity analyst or penetration tester",
            "projects_done": "network vulnerability scanner, ctf writeups",
            "education_branch": "Computer Science",
            "year_of_study": "2nd year",
            "has_internship": False,
            "self_weakness": "need to learn cloud security and compliance"
        }
    },
    {
        "name": "Sneha - Frontend Developer",
        "resume_text": """
            Sneha Patel
            B.Tech IT | 3rd Year | CGPA: 8.5

            Skills: HTML, CSS, JavaScript, React, Tailwind CSS, Git, Figma

            Projects:
            - Portfolio Website with animations
            - E-Commerce UI built with React
            - Weather App using public API

            Education:
            DAIICT | B.Tech IT | 2021-2025
        """,
        "qa": {
            "interests": "frontend development, web design, building beautiful user interfaces",
            "known_skills": "html, css, javascript, react, tailwind css, git, figma",
            "career_goal": "frontend developer or react developer at a startup",
            "projects_done": "portfolio website, ecommerce ui, weather app",
            "education_branch": "Information Technology",
            "year_of_study": "3rd year",
            "has_internship": False,
            "self_weakness": "not confident in backend development and system design"
        }
    }
]


def run_test(case: dict):
    print("\n" + "=" * 65)
    print(f"STUDENT: {case['name']}")
    print("=" * 65)

    # Step 1: Parse resume
    parsed = parse_resume(case["resume_text"])
    qa = case["qa"]

    if qa.get("education_branch"):
        parsed["education_branch"] = qa["education_branch"]
    if qa.get("has_internship") is not None:
        parsed["has_internship"] = bool(qa["has_internship"])

    print(f"\n[PARSED RESUME]")
    print(f"  Degree          : {parsed['education_degree']}")
    print(f"  Branch          : {parsed['education_branch']}")
    print(f"  CGPA            : {parsed['cgpa']}")
    print(f"  Has Internship  : {parsed['has_internship']}")
    print(f"  Skills found    : {', '.join(parsed['skills'][:8])}...")

    # Step 2: Build features
    feature_text = build_feature_text(parsed, qa)
    student_skills = merge_skills(parsed["skills"], qa.get("known_skills", ""))

    print(f"\n[FEATURE TEXT SNIPPET]: {feature_text[:120]}...")
    print(f"[ALL MERGED SKILLS]: {', '.join(student_skills)}")

    # Step 3: Run guidance engine
    result = get_guidance(feature_text, student_skills)

    print(f"\n[TOP CAREER RECOMMENDATIONS]")
    for i, rec in enumerate(result["top_career_recommendations"], 1):
        print(f"  {i}. {rec['career']:30s} {rec['confidence_percent']:.1f}%")

    primary = result["primary_career"]
    print(f"\n[PRIMARY CAREER: {primary['name']}]")
    print(f"  ✅ Skills you HAVE    : {', '.join(primary['skills_you_have']) or 'none matched'}")
    print(f"  ❌ Skill GAPS         : {', '.join(primary['skill_gaps']) or 'none'}")
    print(f"  ⭐ Good to have       : {', '.join(primary['good_to_have_skills'][:4])}")
    print(f"  📈 Improvement areas  : {', '.join(primary['improvement_areas'][:3])}")

    print(f"\n[RECOMMENDED COURSES]")
    for c in primary["recommended_courses"][:4]:
        print(f"  → {c['skill']:25s} | {c['course']} ({c['platform']})")

    if primary["bonus_courses_for_growth"]:
        print(f"\n[BONUS COURSES FOR GROWTH]")
        for c in primary["bonus_courses_for_growth"]:
            print(f"  → {c['skill']:25s} | {c['course']} ({c['platform']})")

    print(f"\n[SUMMARY]\n  {result['summary']}")


if __name__ == "__main__":
    print("\n🚀 CAREER GUIDANCE ML — END-TO-END PIPELINE TEST")
    for case in test_cases:
        run_test(case)
    print("\n\n✅ All tests completed successfully.")
