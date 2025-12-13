#!/usr/bin/env python3
"""
Preprocess and combine resume datasets into a single encoded CSV.
- Processes master_resumes.jsonl and dataset9000.csv
- Normalizes skill names and job titles
- Aligns feature columns and encodes skills
- Outputs combined_encoded_master_resumes.csv
"""

import json
import re
import os
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

# ===== CONFIG =====
BASE = "/Users/yiming/Documents/Sunway/Sem 9/FYP2/Code/career_recommendation"
INPUT_JSONL = os.path.join(BASE, "data/datasets/master_resumes.jsonl")
INPUT_9000_CSV = os.path.join(BASE, "data/datasets/dataset9000.csv")
COMBINED_ENCODED_CSV = os.path.join(BASE, "data/cleaned datasets/encoded_combined_datasets.csv")

# ===== Skill normalization mapping =====
# Maps variant skill names to a consistent standard
skill_aliases = {
    r"\bangular\s?js\b": "angularjs",
    r"\bangular\b": "angular",
    r"\bnode\s?js\b": "nodejavascript",
    r"\breact\s?js\b": "react",
    r"\breact\s?native\b": "react native",
    r"\bazure\s?web\s?services\b|\bazurewebservices\b": "azurewebservices",
    r"\bazure\b": "azure",
    r"\bamazon\s?web\s?services\b|\baws\b": "aws",
    r"\bc\+\+\b": "c++",
    r"\bc#\b": "c#",
    r"\b(core\s)?java\b|\bjava/j2ee\b|\bjava j2ee\b|\bjavaee\b|\badvanced java\b": "java",
    r"\bhtml5\b": "html",
    r"\bcss3\b": "css",
    r"\bms[- ]?excel\b": "ms-excel",
    r"\bms[- ]?access\b": "ms-access",
    r"\bmssql\b|\bms[- ]?sql\b": "ms-sql",
    r"\bmy\s?sql\b": "mysql",
    r"\bpython3\b": "python",
    r"\bpytorch\b": "pytorch",
    r"\bkeras\b": "keras",
    r"\btableau\b": "tableau",
    r"\btensorflow\b": "tensorflow",
    r"\br studio\b": "r studio",
    r"\bsap netweaver gateway\b": "sap netweaver gateway",
    r"\bsap abap\b": "sap abap",
    r"\bflask\b": "flask",
    r"\bdjango rest framework\b": "django rest framework",
    r"\bdjango\b": "django",
    r"\bjs\b|\bjavascript\b": "javascript",
    r"\bjsp\b": "jsp",
    r"\bbootstrap\b": "bootstrap",
    r"\bvue\b": "vue",
    r"\bspring boot\b": "springboot",
    r"\bspring mvc\b": "springmvc",
    r"\bspring\b": "spring",
    r"\brest api\b|\brest\b": "restapi",
    r"\bnosql\b": "nosql",
    r"\bmongodb\b": "mongodb",
    r"\boracle\b": "oracle",
    r"\bpl/sql\b|\bplsql\b": "plsql",
    r"\bphp\b": "php",
    r"\bunix\b": "unix",
    r"\bvisual basic 6.0\b|\bvisualbasic60\b": "visualbasic60",
    r"\bsqlit3\b|\bsqlite\b": "sqlite",
    r"\bmachine learning\b": "machinelearning",
    r"\bext\s?js\b": "extjavascript",
    r"\bfastapi\b": "fastapi",
    r"\bfiori\b": "fiori",
    r"\bfirewall and vpn configuration\b": "firewallandvpnconfiguration",
    r"\bhadoop\b": "hadoop",
    r"\bhibernate\b": "hibernate",
    r"\bionic 3\b": "ionic3",
    r"\bj2ee\b": "j2ee",
    r"\bjdbc\b": "jdbc",
    r"\bjquery\b": "jquery",
    r"\bjson\b": "json",
    r"\blogger\b": "logger",
    r"\bsharepoint\b": "sharepoint",
    r"\bnetwork administration\b": "networkadministration",
    r"\bnetwork security\b": "networksecurity",
    r"\bredis\b": "redis",
    r"\brouting and switching\b": "routingandswitching",
    r"\brpg/400\b": "rpg400",
    r"\bruby\b": "ruby",
    r"\bsafe - agile craft\b": "safeagilecraft",
    r"\bscipy\b": "scipy",
    r"\bselenium\b": "selenium",
    r"\bservlet\b": "servlet",
    r"\bshell script\b|\bshell scripting\b": "shellscripting",
    r"\bsklearn\b": "sklearn",
    r"\bsolidity\b": "solidity",
    r"\bspark\b": "spark",
    r"\bsql\b": "sql",
    r"\bstatsmodels\b": "statsmodels",
    r"\bstruts\b": "struts",
    r"\bswift\b": "swift",
    r"\bvisualization\b": "visualization",
    r"\bweb services\b": "webservices",
    r"\bwindows\b": "windows",
    r"\bword processing software\b": "wordprocessingsoftware",
    r"\bnumpy\b": "numpy",
    r"\bdata analysis\b": "dataanalysis",
    r"\bdata visualization\b": "datavisualization",
    r"\bderby\b": "derby",
    r"\bdb2/400\b|\bdb2400\b": "db2400",
    r"\bodata\b": "odata",
    r"\bpostgresql\b": "postgresql",
    r"\bc\b": "c",
    r"\bdart\b": "dart",
    r"\bds\b": "ds",
}

# ===== Job title merge mapping =====
# Standardizes different variations of the same job title
JOB_TITLE_MERGE_MAP = {
    "ai engineer": "AI ML Engineer",
    "ai ml specialist": "AI ML Engineer",
    "machine learning engineer": "AI ML Engineer",
    "machine learning engineer intern": "AI ML Engineer",
    "mlops engineer": "AI ML Engineer",
    "data scientist": "Data Scientist",
    "data science consultant": "Data Scientist",
    "adjunct faculty & data scientist": "Data Scientist",
    "adjunct faculty data scientist": "Data Scientist",
    "software developer": "Software Developer",
    "software engineer": "Software Developer",
    "python developer": "Python Developer",
    "python developer/analyst": "Python Developer",
    "python developeranalyst": "Python Developer",
    "python restful api developer": "Python Developer",
    "python api developer": "Python Developer",
    "java developer": "Java Developer",
    "java web developer": "Java Developer",
    "jr. java developer": "Java Developer",
    "jr java developer": "Java Developer",
    "devops engineer": "DevOps Engineer",
    "cloud operations architect (devops)": "DevOps Engineer",
    "cloud engineer": "Cloud Engineer",
    "solutions architect": "Cloud Engineer",
    "cloud operations architect devops": "Cloud Engineer",
    "infrastructure engineer": "Infrastructure Engineer",
    "platform engineer": "Infrastructure Engineer",
    "systems engineer": "Infrastructure Engineer",
    "site reliability engineer": "Site Reliability Engineer",
    "kubernetes engineer": "Site Reliability Engineer",
    "cyber security specialist": "Cybersecurity Engineer",
    "cybersecurity engineer": "Cybersecurity Engineer",
    "security engineer": "Cybersecurity Engineer",
    "information security specialist": "Cybersecurity Engineer",
    "information security analyst": "Cybersecurity Engineer",
    "networking engineer": "Network Engineer",
    "network security engineer": "Network Engineer",
    "network and security engineer": "Network Engineer",
    "database administrator": "Database Administrator",
    "database engineer": "Database Administrator",
    "sql developer": "SQL Developer",
    "nosql developer": "SQL Developer",
    "application support engineer": "Technical Support Engineer",
    "helpdesk engineer": "Technical Support Engineer",
    "sap technical architect": "Technical Architect",
    "senior business analyst - rpa": "Business Analyst",
    "senior business analyst rpa": "Business Analyst",
    "business analyst": "Business Analyst",
    "advocate": "Business Analyst",
}

# ===== Utility functions =====

def normalize_skill_text(skill: str) -> str:
    """Standardizes skill names across datasets."""
    if skill is None:
        return ""
    s = str(skill).lower().strip()
    s = re.sub(r"[\/\-&\(\)\[\],]", " ", s)
    for pattern, repl in skill_aliases.items():
        if re.search(pattern, s):
            s = re.sub(pattern, repl, s)
    s = re.sub(r"[^a-z0-9+# ]", "", s)
    s = s.strip()
    # Preserve exceptions with spaces
    exceptions = {"r studio", "react native", "django rest framework", "ms-excel", "ms-access", "db2400"}
    if s not in exceptions:
        s = s.replace(" ", "")
    return s

def normalize_column_name(col: str) -> str:
    """Normalize column names to match skill encoding."""
    return normalize_skill_text(col)

def normalize_job_title(title: str) -> str:
    """Standardizes job titles and applies merge mapping."""
    if title is None:
        return "Unknown"
    t = str(title).lower().strip()
    t = t.replace("_", " ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9 ]", "", t)
    if t in JOB_TITLE_MERGE_MAP:
        t = JOB_TITLE_MERGE_MAP[t].lower().replace("_", " ")
    return " ".join([w.capitalize() for w in t.split()])

# ===== 1) Process master_resumes.jsonl =====
master_records = []
with open(INPUT_JSONL, "r", encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        job_title = "Unknown"
        if isinstance(rec.get("experience"), list) and len(rec["experience"]) > 0:
            job_title = rec["experience"][0].get("title", job_title) or job_title
        elif rec.get("title"):
            job_title = rec.get("title")
        job_title = str(job_title).strip()
        job_title = normalize_job_title(job_title)

        # Extract technical skills
        skills_set = set()
        tech = rec.get("skills", {}).get("technical")
        if isinstance(tech, dict):
            for cat in ["programming_languages", "frameworks", "databases", "cloud", "tools", "libraries"]:
                items = tech.get(cat) or []
                for s in items:
                    name = s.get("name") if isinstance(s, dict) else s
                    norm = normalize_skill_text(name)
                    if norm:
                        skills_set.add(norm)
        # Fallback skill extraction
        if not skills_set:
            fallback = rec.get("skills_list") or rec.get("technical_skills") or rec.get("skills")
            if isinstance(fallback, list):
                for s in fallback:
                    norm = normalize_skill_text(s)
                    if norm:
                        skills_set.add(norm)
            elif isinstance(fallback, str):
                for token in re.split(r"[,\n;|/]", fallback):
                    norm = normalize_skill_text(token)
                    if norm:
                        skills_set.add(norm)

        master_records.append({"Job_Title": job_title, "Skills": sorted(skills_set)})

master_df = pd.DataFrame(master_records)
master_df = master_df[master_df["Skills"].map(len) > 0].reset_index(drop=True)
master_df["Skills_str"] = master_df["Skills"].apply(lambda lst: ", ".join(lst))

# Encode master skills into binary columns
mlb = MultiLabelBinarizer(sparse_output=False)
skills_encoded = mlb.fit_transform(master_df["Skills"])
skills_cols_master = [normalize_column_name(c) for c in mlb.classes_]
skills_df_master = pd.DataFrame(skills_encoded, columns=skills_cols_master)
master_encoded = pd.concat([master_df[["Job_Title", "Skills_str"]], skills_df_master], axis=1)

# ===== 2) Process dataset9000.csv =====
df9000 = pd.read_csv(INPUT_9000_CSV)
if "Role" not in df9000.columns:
    if "role" in df9000.columns:
        df9000 = df9000.rename(columns={"role": "Role"})
    else:
        raise ValueError("dataset9000 CSV must contain 'Role'")
role_col = df9000.pop("Role")
df9000.insert(0, "Role", role_col)

# Normalize skill column names
orig_skill_cols_9000 = [c for c in df9000.columns if c != "Role"]
df9000 = df9000.rename(columns={c: normalize_column_name(c) for c in orig_skill_cols_9000})

# Map textual skill proficiency to binary values
proficiency_map = {"poor":0,"beginner":0,"not interested":0,"notinterested":0,"average":1,
                   "intermediate":1,"excellent":1,"professional":1,"yes":1,"no":0,"1":1,"0":0}
for col in df9000.columns:
    if col == "Role": continue
    df9000[col] = df9000[col].apply(lambda x: int(proficiency_map.get(str(x).strip().lower(),0)) if not pd.isna(x) else 0)

# Construct skills list and string
skill_cols_9000_norm = [c for c in df9000.columns if c != "Role"]
df9000["Skills"] = df9000[skill_cols_9000_norm].apply(lambda r: [c for c,v in r.items() if int(v)==1], axis=1)
df9000["Skills_str"] = df9000["Skills"].apply(lambda lst: ", ".join(lst))
df9000_enc = df9000.rename(columns={"Role": "Job_Title"})

# ===== 3) Align skill columns and combine datasets =====
skill_cols_master_set = set(skills_cols_master)
skill_cols_9000_set = set(skill_cols_9000_norm)
all_skill_cols = sorted(skill_cols_master_set | skill_cols_9000_set)

# Add missing columns with zeros
for col in all_skill_cols:
    if col not in master_encoded.columns:
        master_encoded[col] = 0
    if col not in df9000_enc.columns:
        df9000_enc[col] = 0

cols_order = ["Job_Title", "Skills_str"] + all_skill_cols
master_final = master_encoded[cols_order].reset_index(drop=True)
df9000_final = df9000_enc[cols_order].reset_index(drop=True)
combined = pd.concat([master_final, df9000_final], ignore_index=True)

# ===== 4) Normalize job titles after combining =====
combined["Job_Title"] = combined["Job_Title"].apply(normalize_job_title)

# ===== 5) Encode Job_Title as Job_Label =====
le = LabelEncoder()
combined["Job_Label"] = le.fit_transform(combined["Job_Title"].astype(str))
final_cols = ["Job_Title", "Job_Label", "Skills_str"] + all_skill_cols
combined_encoded = combined[final_cols]

# ===== 6) Save combined encoded dataset =====
os.makedirs(Path(COMBINED_ENCODED_CSV).parent, exist_ok=True)
combined_encoded.to_csv(COMBINED_ENCODED_CSV, index=False, encoding="utf-8")
print(f"✅ Saved combined encoded dataset: {COMBINED_ENCODED_CSV} (rows={len(combined_encoded)}, cols={len(final_cols)})")
print(f"✅ Total unique skills: {len(all_skill_cols)}, unique job titles: {len(le.classes_)}")

# ===== 7) Dataset statistics summary =====
print("\n===== DATASET SUMMARY =====")
meta_cols = {"Job_Title", "Job_Label", "Skills_str"}
skill_cols = [c for c in combined_encoded.columns if c not in meta_cols]

N = len(combined_encoded)  # Total samples
C = combined_encoded["Job_Label"].nunique()  # Number of job classes
class_counts = combined_encoded["Job_Label"].value_counts()
min_support = class_counts.min()
median_support = class_counts.median()
max_support = class_counts.max()
avg_skills = combined_encoded[skill_cols].sum(axis=1).mean()  # Average skills per resume
