"""
Synthetic Dataset Generator for Career Guidance ML Model
Generates realistic fresher student profiles mapped to career labels
"""

import random
import pandas as pd
import json
import os

random.seed(42)

# ─────────────────────────────────────────────
# CAREER LABEL CONFIGS
# Each entry defines the skill pool, interest keywords, goals,
# project types and education branches typical for that career
# ─────────────────────────────────────────────
CAREER_CONFIGS = {
    "Software Engineer": {
        "core_skills": ["data structures", "algorithms", "java", "python", "c++", "object oriented programming", "git", "problem solving"],
        "bonus_skills": ["system design", "design patterns", "rest api", "unit testing", "agile", "linux"],
        "interests": ["competitive programming", "open source", "software development", "coding challenges", "hackathons"],
        "goals": ["software engineer at a top product company", "backend developer", "work at a tech startup", "sde at faang", "software developer"],
        "projects": ["url shortener", "library management system", "student result portal", "chat application", "ecommerce website", "task manager app"],
        "education_branches": ["Computer Science", "Information Technology", "Software Engineering"],
        "has_internship_prob": 0.4
    },
    "Frontend Developer": {
        "core_skills": ["html", "css", "javascript", "react", "responsive design", "git"],
        "bonus_skills": ["typescript", "nextjs", "tailwind css", "redux", "figma", "vue", "rest api"],
        "interests": ["web design", "ui development", "building user interfaces", "frontend development", "web animations"],
        "goals": ["frontend developer", "ui developer", "web developer", "react developer", "frontend engineer"],
        "projects": ["portfolio website", "ecommerce ui", "landing page", "weather app", "todo app", "dashboard ui", "blog website"],
        "education_branches": ["Computer Science", "Information Technology", "Electronics"],
        "has_internship_prob": 0.35
    },
    "Backend Engineer": {
        "core_skills": ["python", "java", "node.js", "sql", "rest api", "git", "data structures", "databases"],
        "bonus_skills": ["docker", "redis", "mongodb", "microservices", "aws", "system design", "message queues"],
        "interests": ["backend development", "api design", "databases", "server-side development", "distributed systems"],
        "goals": ["backend developer", "api developer", "server-side engineer", "backend engineer at a product company"],
        "projects": ["rest api backend", "authentication service", "blog api", "inventory management api", "payment gateway integration"],
        "education_branches": ["Computer Science", "Information Technology", "Software Engineering"],
        "has_internship_prob": 0.4
    },
    "Full Stack Developer": {
        "core_skills": ["html", "css", "javascript", "react", "node.js", "sql", "rest api", "git"],
        "bonus_skills": ["typescript", "mongodb", "docker", "nextjs", "aws", "redis", "figma"],
        "interests": ["full stack development", "web development", "end to end development", "building web applications"],
        "goals": ["full stack developer", "web application developer", "mern stack developer", "full stack engineer"],
        "projects": ["full stack ecommerce app", "social media clone", "job portal", "real estate website", "food delivery app"],
        "education_branches": ["Computer Science", "Information Technology"],
        "has_internship_prob": 0.45
    },
    "Mobile Developer": {
        "core_skills": ["flutter", "dart", "android", "kotlin", "java", "git", "mobile ui"],
        "bonus_skills": ["firebase", "rest api", "state management", "ios", "react native", "push notifications", "sqlite"],
        "interests": ["mobile app development", "android development", "flutter development", "cross platform apps", "ios development"],
        "goals": ["mobile app developer", "android developer", "flutter developer", "ios developer", "cross platform developer"],
        "projects": ["fitness tracker app", "food delivery app", "expense tracker", "notes app", "quiz app", "weather app flutter", "e-learning app"],
        "education_branches": ["Computer Science", "Information Technology", "Electronics and Communication"],
        "has_internship_prob": 0.35
    },
    "Data Scientist": {
        "core_skills": ["python", "statistics", "machine learning", "pandas", "numpy", "sql", "data visualization", "probability"],
        "bonus_skills": ["deep learning", "tensorflow", "pytorch", "spark", "tableau", "feature engineering", "nlp", "scikit-learn"],
        "interests": ["data science", "machine learning", "artificial intelligence", "data analysis", "predictive modeling", "statistics"],
        "goals": ["data scientist", "machine learning practitioner", "ai researcher", "data science engineer", "applied scientist"],
        "projects": ["house price prediction", "sentiment analysis", "customer churn prediction", "image classifier", "movie recommendation system", "stock price forecasting"],
        "education_branches": ["Computer Science", "Information Technology", "Mathematics", "Electronics"],
        "has_internship_prob": 0.45
    },
    "ML Engineer": {
        "core_skills": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "numpy", "pandas", "model deployment"],
        "bonus_skills": ["mlops", "docker", "kubernetes", "fastapi", "aws sagemaker", "spark", "data pipelines", "nlp"],
        "interests": ["machine learning engineering", "mlops", "ai model deployment", "deep learning", "neural networks", "production ml systems"],
        "goals": ["ml engineer", "machine learning engineer", "ai engineer", "deep learning engineer", "mlops engineer"],
        "projects": ["deployed ml api", "image recognition model", "nlp pipeline", "real-time inference system", "ml model monitoring", "recommendation engine"],
        "education_branches": ["Computer Science", "Information Technology", "Artificial Intelligence", "Mathematics"],
        "has_internship_prob": 0.5
    },
    "Data Analyst": {
        "core_skills": ["sql", "excel", "python", "data visualization", "statistics", "pandas", "tableau", "power bi"],
        "bonus_skills": ["r", "google analytics", "looker", "etl", "machine learning basics", "dashboard design", "pivot tables"],
        "interests": ["data analysis", "business analytics", "reporting", "data visualization", "insight generation", "dashboards"],
        "goals": ["data analyst", "business analyst", "analytics engineer", "insights analyst", "reporting analyst"],
        "projects": ["sales dashboard", "market analysis report", "excel dashboard", "customer behavior analysis", "sql analytics project", "power bi report"],
        "education_branches": ["Computer Science", "Information Technology", "Commerce", "Mathematics", "Business Administration"],
        "has_internship_prob": 0.4
    },
    "DevOps Engineer": {
        "core_skills": ["linux", "docker", "kubernetes", "ci/cd", "git", "aws", "shell scripting", "networking basics"],
        "bonus_skills": ["terraform", "ansible", "prometheus", "grafana", "azure", "jenkins", "monitoring", "gcp"],
        "interests": ["devops", "infrastructure", "automation", "cloud computing", "system administration", "site reliability"],
        "goals": ["devops engineer", "cloud engineer", "site reliability engineer", "infrastructure engineer", "platform engineer"],
        "projects": ["ci/cd pipeline setup", "dockerized application", "kubernetes cluster", "infrastructure as code", "automated deployment pipeline"],
        "education_branches": ["Computer Science", "Information Technology", "Electronics and Communication"],
        "has_internship_prob": 0.35
    },
    "Cybersecurity Analyst": {
        "core_skills": ["networking", "linux", "security fundamentals", "ethical hacking", "cryptography", "firewalls", "sql"],
        "bonus_skills": ["penetration testing", "kali linux", "python", "wireshark", "siem tools", "cloud security", "compliance"],
        "interests": ["cybersecurity", "ethical hacking", "network security", "penetration testing", "ctf challenges", "information security"],
        "goals": ["cybersecurity analyst", "security engineer", "ethical hacker", "penetration tester", "information security analyst"],
        "projects": ["penetration testing report", "network vulnerability scanner", "ctf writeups", "firewall configuration", "security audit"],
        "education_branches": ["Computer Science", "Information Technology", "Electronics and Communication"],
        "has_internship_prob": 0.3
    },
    "UI/UX Designer": {
        "core_skills": ["figma", "user research", "wireframing", "prototyping", "design thinking", "typography", "color theory"],
        "bonus_skills": ["adobe xd", "html css basics", "motion design", "accessibility", "usability testing", "sketch", "invision"],
        "interests": ["ui design", "ux research", "product design", "user experience", "visual design", "interaction design"],
        "goals": ["ui/ux designer", "product designer", "user experience designer", "visual designer", "interaction designer"],
        "projects": ["app redesign case study", "e-commerce ux design", "mobile app prototype", "design system", "user research report", "website wireframes"],
        "education_branches": ["Computer Science", "Information Technology", "Design", "Arts"],
        "has_internship_prob": 0.4
    },
    "Business Analyst": {
        "core_skills": ["requirement gathering", "sql", "excel", "data analysis", "documentation", "communication", "uml"],
        "bonus_skills": ["power bi", "tableau", "jira", "agile", "process mapping", "python basics", "stakeholder management"],
        "interests": ["business analysis", "process improvement", "requirement analysis", "project management", "consulting"],
        "goals": ["business analyst", "product analyst", "systems analyst", "requirements analyst", "it business analyst"],
        "projects": ["business requirement document", "process flow diagram", "gap analysis report", "use case documentation", "stakeholder analysis"],
        "education_branches": ["Computer Science", "Information Technology", "Business Administration", "Commerce", "Management"],
        "has_internship_prob": 0.4
    },
    "QA Engineer": {
        "core_skills": ["manual testing", "test cases", "bug reporting", "sql", "git", "api testing", "sdlc"],
        "bonus_skills": ["selenium", "python", "jmeter", "postman", "ci/cd", "performance testing", "automation testing"],
        "interests": ["software testing", "quality assurance", "test automation", "bug hunting", "api testing", "performance testing"],
        "goals": ["qa engineer", "software tester", "test automation engineer", "quality assurance analyst", "sdet"],
        "projects": ["test automation framework", "api testing suite", "performance test report", "selenium test scripts", "bug tracking report"],
        "education_branches": ["Computer Science", "Information Technology", "Software Engineering"],
        "has_internship_prob": 0.35
    },
    "Cloud Engineer": {
        "core_skills": ["aws", "azure", "networking", "linux", "docker", "python", "git", "storage services"],
        "bonus_skills": ["kubernetes", "terraform", "serverless", "gcp", "cloud security", "monitoring", "cost optimization"],
        "interests": ["cloud computing", "aws", "azure", "cloud architecture", "infrastructure", "cloud native development"],
        "goals": ["cloud engineer", "aws engineer", "azure engineer", "cloud architect", "cloud developer"],
        "projects": ["aws s3 file storage app", "serverless api on lambda", "cloud deployment pipeline", "multi-region architecture", "cloud cost dashboard"],
        "education_branches": ["Computer Science", "Information Technology", "Electronics and Communication"],
        "has_internship_prob": 0.35
    },
    "Embedded Systems Engineer": {
        "core_skills": ["c", "c++", "microcontrollers", "rtos", "electronics", "arduino", "raspberry pi", "debugging"],
        "bonus_skills": ["python", "linux", "iot", "pcb design", "communication protocols", "matlab", "assembly"],
        "interests": ["embedded systems", "iot", "robotics", "electronics", "hardware programming", "firmware development"],
        "goals": ["embedded systems engineer", "firmware engineer", "iot developer", "robotics engineer", "hardware software engineer"],
        "projects": ["iot home automation", "arduino sensor project", "line following robot", "smart irrigation system", "rtos based system"],
        "education_branches": ["Electronics and Communication", "Electrical Engineering", "Computer Science", "Mechatronics"],
        "has_internship_prob": 0.3
    }
}

YEAR_OF_STUDY = ["1st year", "2nd year", "3rd year", "4th year", "final year"]
CGPA_RANGES = [(6.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 10.0)]
WEAKNESS_POOL = [
    "not confident in public speaking", "weak in mathematics", "need to improve problem solving speed",
    "not good at system design", "need to improve communication skills", "lack of real-world project experience",
    "weak in statistics", "not confident in interviews", "need to work on time management",
    "limited knowledge of cloud platforms", "need to improve debugging skills", "weak in networking concepts"
]
WORK_PREFS = [
    "product company, remote", "service company", "startup environment", "research organization",
    "mnc with good work life balance", "fintech startup", "edtech company", "remote job"
]


def pick_skills(core, bonus, min_core=4, max_core=8, min_bonus=0, max_bonus=4):
    """Pick a random subset of skills to simulate real student knowledge"""
    selected_core = random.sample(core, k=random.randint(min(min_core, len(core)), min(max_core, len(core))))
    selected_bonus = random.sample(bonus, k=random.randint(min_bonus, min(max_bonus, len(bonus))))
    return list(set(selected_core + selected_bonus))


def generate_student_profile(career_label, config):
    """Generate one synthetic student profile"""
    skills = pick_skills(config["core_skills"], config["bonus_skills"])
    interest = random.choice(config["interests"])
    goal = random.choice(config["goals"])
    project = random.choice(config["projects"])
    branch = random.choice(config["education_branches"])
    year = random.choice(YEAR_OF_STUDY)
    cgpa_range = random.choice(CGPA_RANGES)
    cgpa = round(random.uniform(*cgpa_range), 1)
    has_internship = random.random() < config["has_internship_prob"]
    weakness = random.choice(WEAKNESS_POOL)
    work_pref = random.choice(WORK_PREFS)
    open_to_masters = random.choice([True, False])

    # Combine all text for the model's input
    combined_text = " ".join([
        " ".join(skills),
        interest,
        goal,
        project,
        branch.lower(),
        "internship" if has_internship else "no internship",
        weakness
    ])

    return {
        "career_label": career_label,
        "skills": ", ".join(skills),
        "interests": interest,
        "career_goal": goal,
        "projects_done": project,
        "education_branch": branch,
        "year_of_study": year,
        "cgpa": cgpa,
        "has_internship": has_internship,
        "weakness": weakness,
        "preferred_work": work_pref,
        "open_to_masters": open_to_masters,
        "combined_text": combined_text
    }


def generate_dataset(samples_per_label=500):
    """Generate full dataset across all career labels"""
    all_records = []
    for label, config in CAREER_CONFIGS.items():
        print(f"  Generating {samples_per_label} samples for: {label}")
        for _ in range(samples_per_label):
            profile = generate_student_profile(label, config)
            all_records.append(profile)

    df = pd.DataFrame(all_records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    print("Generating synthetic student profile dataset...")
    df = generate_dataset(samples_per_label=500)
    os.makedirs("data", exist_ok=True)
    output_path = "data/student_profiles.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved: {output_path}")
    print(f"Total records  : {len(df)}")
    print(f"Career labels  : {df['career_label'].nunique()}")
    print(f"\nLabel distribution:")
    print(df['career_label'].value_counts().to_string())
    print(f"\nSample row:")
    print(df.iloc[0].to_dict())
