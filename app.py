import gradio as gr
from parsers.resume_parser import parse_resume, load_skills
from recommender.recommender_svm import recommend_from_resume

def process_resume(file):
    skills_list = load_skills()
    parsed = parse_resume(file.name, skills_list=skills_list)

    skills = parsed.get("Skills", [])
    if not skills:
        return parsed, "No skills found. Cannot recommend."

    rec_result = recommend_from_resume(skills_list=skills, top_k=5)

    text = "## 🔮 Top Career Recommendations\n\n"

    for i, rec in enumerate(rec_result["top_recommendations"], 1):

        text += f"### {i}. {rec['job_title']}  \n"
        text += f"**Match Score:** `{rec['score']:.3f}`\n\n"

        desc = rec["description"]

        # Format tasks
        if isinstance(desc, dict) and "Typical daily tasks" in desc:
            text += "**📌 Typical Daily Tasks:**\n"
            for task in desc["Typical daily tasks"]:
                text += f"- {task}\n"

            text += "\n**🧠 Key Skills Required:**\n"
            for skill in desc["Key skills required"]:
                text += f"- {skill}\n"

        else:
            text += f"{desc}\n"

        text += "\n---\n\n"

    return parsed, text



with gr.Blocks() as demo:
    gr.Markdown("# 📘 AI Career Recommendation System")
    gr.Markdown("Upload your resume to get parsed information and detailed career recommendations.")

    with gr.Row():
        resume = gr.File(label="Upload PDF Resume")
        parsed_out = gr.JSON(label="Parsed Resume Data")

    job_out = gr.Markdown()

    btn = gr.Button("Process Resume")
    btn.click(process_resume, inputs=resume, outputs=[parsed_out, job_out])

demo.launch()
