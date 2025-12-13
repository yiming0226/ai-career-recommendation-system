import os
import json
from glob import glob
from career_recommendation.parsers.resume_parser import parse_resume, load_skills

try:
    from career_recommendation.recommender.recommender_svm import recommend_from_resume
except Exception:
    recommend_from_resume = None

# ===== CONFIG =====
SHOW_EXTRACTED_SKILLS = False
PROFILES_PATH = os.path.join(os.path.dirname(__file__), "data", "profiles.json")

# ===== PROFILE IO =====
def load_profiles():
    if os.path.exists(PROFILES_PATH):
        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_profiles(profiles):
    os.makedirs(os.path.dirname(PROFILES_PATH), exist_ok=True)
    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=4, ensure_ascii=False)

def update_profiles(new_profile, profiles):
    resume_file = os.path.basename(new_profile.get("Resume_File", ""))
    for i, profile in enumerate(profiles):
        if os.path.basename(profile.get("Resume_File", "")) == resume_file:
            profiles[i] = new_profile
            return profiles
    profiles.append(new_profile)
    return profiles

# ===== SKILL NORMALIZATION =====
def _normalize_skills_input(parsed_profile):
    skills = parsed_profile.get("Skills")
    if isinstance(skills, list) and skills:
        return [str(s).strip().lower() for s in skills if s is not None and str(s).strip()]
    skills_str = (
        parsed_profile.get("Skills_str")
        or parsed_profile.get("SkillsStr")
        or parsed_profile.get("SkillsString")
        or ""
    )
    if isinstance(skills_str, str) and skills_str.strip():
        return [s.strip().lower() for s in skills_str.split(",") if s.strip()]
    return []

# ===== MAIN PIPELINE =====
def main():
    resumes_dir = os.path.join(os.path.dirname(__file__), "data", "resumes2")
    skills_list = load_skills()
    profiles = load_profiles()

    resume_files = glob(os.path.join(resumes_dir, "*.pdf"))
    if not resume_files:
        print("⚠️ No PDF resumes found in:", resumes_dir)
        return

    if recommend_from_resume is None:
        print("⚠️ recommend_from_resume not importable — predictions will be skipped.")

    for resume_file in resume_files:
        print(f"\n📄 Processing: {os.path.basename(resume_file)}")
        try:
            parsed_data = parse_resume(resume_file, skills_list=skills_list)
            parsed_data["Resume_File"] = os.path.basename(resume_file)

            skills_input = _normalize_skills_input(parsed_data)
            recommendations = []
            normalized_skills_from_model = None

            if recommend_from_resume is not None and skills_input:
                try:
                    rec_result = recommend_from_resume(skills_list=skills_input, top_k=5)
                    recommendations = [
                        {"job_title": label, "score": float(prob)}
                        for label, prob in rec_result["top_recommendations"]
                    ]
                    normalized_skills_from_model = rec_result.get("extracted_skills")
                except Exception as e:
                    print(f"❌ Prediction error for {resume_file}: {e}")

            parsed_data["Recommendations"] = recommendations

            if SHOW_EXTRACTED_SKILLS:
                parsed_data["Extracted_Skills"] = skills_input
                parsed_data["Extracted_Skills_Normalized"] = normalized_skills_from_model or skills_input

            profiles = update_profiles(parsed_data, profiles)
            print("✅ Parsed & predicted successfully.")

        except Exception as e:
            print(f"❌ Error processing {resume_file}: {e}")

    save_profiles(profiles)
    print("\n✅ All profiles saved successfully!")

if __name__ == "__main__":
    main()
