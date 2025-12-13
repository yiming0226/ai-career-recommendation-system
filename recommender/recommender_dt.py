import re
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

# ===== CONFIG =====
ENCODED_CSV = "/Users/yiming/Documents/Sunway/Sem 9/FYP2/Code/career_recommendation/data/cleaned datasets/encoded_combined_datasets.csv"
MODEL_PATH  = "/Users/yiming/Documents/Sunway/Sem 9/FYP2/Code/career_recommendation/training/models/dt/dt_tfidf_skill_model.pkl"
VECT_PATH   = "/Users/yiming/Documents/Sunway/Sem 9/FYP2/Code/career_recommendation/training/models/dt/tfidf_vectorizer.pkl"
LE_PATH     = "/Users/yiming/Documents/Sunway/Sem 9/FYP2/Code/career_recommendation/training/models/dt/label_encoder.pkl"

PREFERRED_TEXT_COLS = ["cleaned_resume", "Skills_str", "Skills", "Resume_str"]
TOP_K = 5

# ===== Skill normalization map =====
skill_aliases = {
    r"\bangular\s?js\b": "angularjs",
    r"\bangular\b": "angular",
    r"\bnode\s?js\b": "node.js",
    r"\breact\s?js\b": "react",
    r"\breact\s?native\b": "react native",
    r"\bazure\s?web\s?services\b|\bazurewebservices\b": "azurewebservices",
    r"\baws\b": "aws",
    r"\bc\+\+\b": "c++",
    r"\bc#\b": "c#",
    r"\bjava/j2ee\b|\bjava j2ee\b": "java",
    r"\bhtml5\b": "html",
    r"\bcss3\b": "css",
    r"\bms[- ]?excel\b": "ms-excel",
    r"\bmysql\b": "mysql",
    r"\bpython3\b": "python",
    r"\bpytorch\b": "pytorch",
    r"\bkeras\b": "keras",
    r"\btableau\b": "tableau",
    r"\btensorflow\b": "tensorflow",
    r"\br studio\b": "r studio",
    r"\bdjango rest framework\b": "django rest framework",
    r"\bdjango\b": "django",
    r"\bjavascript\b": "javascript",
    r"\bbootstrap\b": "bootstrap",
    r"\bvue\b": "vue",
    r"\bspring boot\b": "spring boot",
    r"\brest api\b|\brest\b": "rest api",
    r"\bnosql\b": "nosql",
    r"\bmongodb\b": "mongodb",
    r"\boracle\b": "oracle",
    r"\bpl/sql\b": "pl/sql",
    r"\bphp\b": "php",
    r"\bunix\b": "unix",
    r"\bsqlite\b": "sqlite",
    r"\bmachine learning\b": "machine learning",
    r"\bhadoop\b": "hadoop",
    r"\bhibernate\b": "hibernate",
    r"\bj2ee\b": "j2ee",
    r"\bjdbc\b": "jdbc",
    r"\bjquery\b": "jquery",
    r"\bjson\b": "json",
    r"\bsharepoint\b": "sharepoint",
    r"\bnetwork administration\b": "network administration",
    r"\bnetwork security\b": "network security",
    r"\bredis\b": "redis",
    r"\bruby\b": "ruby",
    r"\bscipy\b": "scipy",
    r"\bselenium\b": "selenium",
    r"\bshell scripting\b": "shell scripting",
    r"\bsklearn\b": "sklearn",
    r"\bspark\b": "spark",
    r"\bsql\b": "sql",
    r"\bstatsmodels\b": "statsmodels",
    r"\btableau\b": "tableau",
    r"\bnumpy\b": "numpy",
    r"\bdata analysis\b": "data analysis",
    r"\bdata visualization\b": "data visualization",
    r"\bpostgresql\b": "postgresql",
    r"\bdart\b": "dart",
    r"\bds\b": "ds",
}

# ===== Helpers =====
def normalize_text_for_matching(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r'[\n\r\t]', ' ', t)
    t = re.sub(r'[-/,&()]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def apply_skill_aliases(skill: str) -> str:
    s = skill.lower().strip()
    for pat, repl in skill_aliases.items():
        if re.search(pat, s):
            s = re.sub(pat, repl, s)
    s = re.sub(r'[^a-z0-9 +#+\-\.]', ' ', s).strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def extract_skills_from_text(resume_text: str, skill_columns: list) -> list:
    t = normalize_text_for_matching(resume_text)
    found = set()
    for skill in skill_columns:
        cand = str(skill).lower().strip()
        if re.search(r'\b' + re.escape(cand) + r'\b', t):
            found.add(cand)
            continue
        alias_norm = apply_skill_aliases(cand)
        if alias_norm != cand and re.search(r'\b' + re.escape(alias_norm) + r'\b', t):
            found.add(alias_norm)
    return sorted(found)

# ===== Main prediction function =====
def recommend_from_resume(resume_text=None, skills_list=None, top_k=TOP_K):
    """
    Predict top-k career recommendations using the trained Decision Tree model.
    """
    # Load trained artifacts
    clf = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECT_PATH)
    le = joblib.load(LE_PATH)
    enc_df = pd.read_csv(ENCODED_CSV)

    # Identify text column
    text_col = next((c for c in PREFERRED_TEXT_COLS if c in enc_df.columns), None)
    if text_col is None:
        raise RuntimeError("Could not find a valid text column in encoded dataset.")

    # Extract skill columns
    skill_cols = [c for c in enc_df.columns if c not in ("Job_Title", "Job_Label", text_col)]

    # Normalize skills
    if skills_list:
        normalized_skills = [apply_skill_aliases(s) for s in skills_list]
    elif resume_text:
        normalized_skills = extract_skills_from_text(resume_text, skill_cols)
    else:
        raise ValueError("Provide either resume_text or skills_list.")

    # Build skill vector
    skill_vector = np.zeros(len(skill_cols), dtype=int)
    normalized_set = set([s.lower().strip() for s in normalized_skills])
    for i, col in enumerate(skill_cols):
        col_norm = col.lower().strip()
        col_alias = apply_skill_aliases(col_norm)
        if col_norm in normalized_set or col_alias in normalized_set:
            skill_vector[i] = 1

    # TF-IDF transformation
    if text_col.lower() in ("skills_str", "skills"):
        text_for_tfidf = " ".join(normalized_skills)
    else:
        text_for_tfidf = resume_text if resume_text else " ".join(normalized_skills)

    tfidf_vec = tfidf.transform([text_for_tfidf])
    skills_sparse = csr_matrix(skill_vector.reshape(1, -1))
    X = hstack([tfidf_vec, skills_sparse], format="csr")

    # Align feature dimensions
    expected_features = clf.n_features_in_
    if X.shape[1] < expected_features:
        X = hstack([X, csr_matrix(np.zeros((1, expected_features - X.shape[1])))], format="csr")
    elif X.shape[1] > expected_features:
        X = X[:, :expected_features]

    # Predict using Decision Tree
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)
        topk_idx = probs[0].argsort()[-top_k:][::-1]
        topk_labels = le.inverse_transform(topk_idx)
        topk_probs = probs[0][topk_idx]
    else:
        # fallback: deterministic prediction only
        pred_label = clf.predict(X)[0]
        topk_labels = [le.inverse_transform([pred_label])[0]]
        topk_probs = [1.0]

    results = list(zip(topk_labels, np.round(topk_probs, 4)))
    return {
        "extracted_skills": normalized_skills,
        "top_recommendations": results
    }

# # ----------------- Example usage -----------------
# if __name__ == "__main__":
#     # ---------- Option A: Use raw resume text ----------
#     sample_resume_text = """
#     Experienced software engineer with 3 years of experience in Python, Django, REST API, PostgreSQL, AWS, and Docker.
#     Built scalable microservices using FastAPI and containerized applications with Docker and Kubernetes.
#     """

#     # ---------- Option B: Use multiple custom skills lists ----------
#     test_skills_lists = [
#         # 1. Python / Data Analysis
#         ['python', 'pandas', 'numpy', 'dataanalysis', 'sql'],

#         # 2. Java / Backend Development
#         ['java', 'spring', 'hibernate', 'maven', 'sql'],

#         # 3. Frontend Web Development
#         ['react', 'javascript', 'html', 'css', 'node.js'],

#         # 4. Business Intelligence / Analytics
#         ['excel', 'tableau', 'powerbi', 'sql', 'dataanalysis'],

#         # 5. Cloud & DevOps
#         ['aws', 'docker', 'kubernetes', 'terraform', 'ansible'],

#         # 6. Machine Learning / Data Science
#         ['python', 'tensorflow', 'scikit-learn', 'pandas', 'matplotlib'],

#         # 7. Mobile App Development (Android)
#         ['java', 'android', 'kotlin', 'firebase', 'xml'],

#         # 8. Cybersecurity / Network Security
#         ['networking', 'firewall', 'cybersecurity', 'linux', 'penetrationtesting'],

#         # 9. Software Testing / QA Automation
#         ['selenium', 'pytest', 'jira', 'testautomation', 'python'],

#         # 10. Database Administration / Data Engineering
#         ['mysql', 'postgresql', 'etl', 'datawarehouse', 'sql'],

#         # 11. UI/UX Design
#         ['figma', 'adobeXD', 'wireframing', 'prototyping', 'usabilitytesting'],

#         # 12. Project Management / Agile
#         ['scrum', 'agile', 'jira', 'projectmanagement', 'communication'],

#         # 13. Artificial Intelligence / NLP
#         ['python', 'nlp', 'transformers', 'pytorch', 'textclassification'],

#         # 14. Embedded Systems / IoT
#         ['c', 'c++', 'arduino', 'raspberrypi', 'microcontrollers'],

#         # 15. Cloud Architecture / Infrastructure
#         ['azure', 'cloudarchitecture', 'devops', 'docker', 'networking']
#     ]

#     # Choose which input to test
#     use_skills_lists = True  # 🔹 set False to test with raw resume text

#     if use_skills_lists:
#         for i, skills_list in enumerate(test_skills_lists, 1):
#             print(f"\n🔹 Skills List {i}: {skills_list}")
#             result = recommend_from_resume(skills_list=skills_list, top_k=5)
#             print("Top job recommendations:")
#             for lbl, prob in result["top_recommendations"]:
#                 print(f"  {lbl} ({prob})")
#     else:
#         result = recommend_from_resume(resume_text=sample_resume_text, top_k=5)
#         print("\nExtracted skills:", result["extracted_skills"])
#         print("\nTop job recommendations:")
#         for lbl, prob in result["top_recommendations"]:
#             print(f"  {lbl} ({prob})")

