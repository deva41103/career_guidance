"""
Resume Parser Service
Extracts structured information from raw resume text:
- Skills (matched against master keyword list)
- Education details
- Experience / internship presence
- Projects mentioned
"""

import re
from services.skill_keywords import MASTER_SKILLS


def extract_skills(text: str) -> list:
    """Match resume text against the master skill keyword list"""
    text_lower = text.lower()
    found = []
    for skill in MASTER_SKILLS:
        # Use word boundary matching to avoid partial matches
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill)
    return list(set(found))


def extract_education(text: str) -> dict:
    """Extract education branch and degree from resume text"""
    text_lower = text.lower()

    degree = "Unknown"
    if any(kw in text_lower for kw in ["b.tech", "btech", "b tech", "bachelor of technology"]):
        degree = "B.Tech"
    elif any(kw in text_lower for kw in ["b.e.", "be ", "bachelor of engineering"]):
        degree = "B.E."
    elif any(kw in text_lower for kw in ["bsc", "b.sc", "bachelor of science"]):
        degree = "B.Sc"
    elif any(kw in text_lower for kw in ["bca", "b.c.a"]):
        degree = "BCA"
    elif any(kw in text_lower for kw in ["mtech", "m.tech", "m tech"]):
        degree = "M.Tech"
    elif any(kw in text_lower for kw in ["mca", "m.c.a"]):
        degree = "MCA"

    branch = "Computer Science"  # default
    branch_patterns = {
        "Computer Science": ["computer science", "cse", "cs"],
        "Information Technology": ["information technology", "it"],
        "Electronics and Communication": ["electronics", "ece", "eee", "electrical"],
        "Mechanical Engineering": ["mechanical", "mech"],
        "Mathematics": ["mathematics", "math", "statistics"],
        "Data Science": ["data science", "artificial intelligence", "ai & ml"],
        "Business Administration": ["business", "bba", "mba", "commerce"],
    }
    for branch_name, keywords in branch_patterns.items():
        if any(kw in text_lower for kw in keywords):
            branch = branch_name
            break

    cgpa = None
    cgpa_match = re.search(r'\b([0-9]\.[0-9]{1,2})\s*(cgpa|gpa|/10|out of 10)?\b', text_lower)
    if cgpa_match:
        val = float(cgpa_match.group(1))
        if 0.0 <= val <= 10.0:
            cgpa = val

    return {"degree": degree, "branch": branch, "cgpa": cgpa}


def has_internship(text: str) -> bool:
    """Detect internship mentions in resume"""
    text_lower = text.lower()
    keywords = ["intern", "internship", "trainee", "summer training", "industrial training"]
    return any(kw in text_lower for kw in keywords)


def extract_projects(text: str) -> list:
    """Extract project names/descriptions from resume"""
    projects = []
    # Look for project sections
    project_section = re.search(
        r'(projects?|personal projects?|academic projects?)(.*?)(experience|education|skills|certifications|$)',
        text, re.IGNORECASE | re.DOTALL
    )
    if project_section:
        section_text = project_section.group(2)
        # Extract lines that look like project names (capitalized, short lines)
        lines = [l.strip() for l in section_text.split('\n') if l.strip()]
        for line in lines[:5]:  # limit to top 5
            if 5 < len(line) < 100:
                projects.append(line)
    return projects


def parse_resume(text: str) -> dict:
    """Main function: parse resume text into structured dict"""
    skills = extract_skills(text)
    education = extract_education(text)
    internship = has_internship(text)
    projects = extract_projects(text)

    return {
        "skills": skills,
        "education_degree": education["degree"],
        "education_branch": education["branch"],
        "cgpa": education["cgpa"],
        "has_internship": internship,
        "projects": projects,
        "raw_text": text
    }
