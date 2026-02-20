"""
Feature Builder Service
Combines parsed resume data and Q&A responses into a single
text blob that can be fed into the TF-IDF + classifier pipeline.
"""


def build_feature_text(parsed_resume: dict, qa: dict) -> str:
    """
    Merge all available student information into one text string.
    This mirrors exactly how the training data's 'combined_text' was built.
    """
    parts = []

    # Skills from resume
    skills = parsed_resume.get("skills", [])
    if skills:
        parts.append(" ".join(skills))

    # Education branch
    branch = parsed_resume.get("education_branch", "")
    if branch:
        parts.append(branch.lower())

    # Internship signal
    parts.append("internship" if parsed_resume.get("has_internship") else "no internship")

    # Q&A fields
    if qa.get("interests"):
        parts.append(qa["interests"].lower())

    if qa.get("career_goal"):
        parts.append(qa["career_goal"].lower())

    if qa.get("projects_done"):
        parts.append(qa["projects_done"].lower())

    if qa.get("known_skills"):
        # Additional skills from Q&A that may not be in resume
        parts.append(qa["known_skills"].lower())

    if qa.get("self_weakness"):
        parts.append(qa["self_weakness"].lower())

    combined = " ".join(parts)
    return combined


def merge_skills(resume_skills: list, qa_known_skills: str) -> list:
    """
    Merge skills from resume parsing and Q&A-reported skills.
    Deduplicates and lowercases everything.
    """
    all_skills = set(s.lower() for s in resume_skills)

    if qa_known_skills:
        for skill in qa_known_skills.split(","):
            cleaned = skill.strip().lower()
            if cleaned:
                all_skills.add(cleaned)

    return sorted(list(all_skills))
