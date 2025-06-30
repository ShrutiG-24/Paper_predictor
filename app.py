from flask import Flask, render_template, request, jsonify
import os
import fitz  # PyMuPDF
from docx import Document
import tempfile
import json
from dotenv import load_dotenv
from openai import OpenAI
import re

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(docx_stream):
    try:
        doc = Document(docx_stream)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def identify_sections(text):
    section_pattern = re.compile(r'(Section\s+[A-Z])[^\n\r]*', re.IGNORECASE)
    matches = section_pattern.findall(text)
    sections = []
    for match in matches:
        full_title = match.strip()
        if full_title not in sections:
            sections.append(full_title)
    return sections

def build_openai_prompt(extracted_text, identified_sections):
    section_list_str = '\n'.join(f'- {s}' for s in identified_sections) if identified_sections else "- Section A\n- Section B\n- Section C"

    prompt = f"""
You are an expert CBSE Class 12 paper setter. Generate a challenging 70-mark practice question paper for CBSE Class 12, matching real board exam difficulty and structure.

Instructions:
- Analyze the uploaded sample papers and past CBSE Class 12 board papers.
- Use the original sections found in the paper:
{section_list_str}
- Create questions under each of those section headings.
- Include a mix of MCQs, short answers, long answers, and case studies as per section type.
- Ensure at least one diagram-based or visual question in each section where appropriate.
- Use only challenging, application-based, and analytical questions.
- Return ONLY valid JSON. No markdown, no explanation.

Format:
{{
  "paper_title": "CBSE Class 12 [Subject] Practice Question Paper (70 Marks)",
  "total_marks": 70,
  "sections": [
    {{
      "section_name": "[From extracted paper]",
      "marks_per_question": X,
      "total_questions": Y,
      "questions": [
        {{
          "question": "...",
          "options": {{"A": "...", "B": "..."}},
          "question_type": "MCQ/Numerical/Diagram/Case Study/etc."
        }}
      ]
    }}
  ]
}}

Extracted text for reference:
{extracted_text}
"""
    return prompt

@app.route("/")
def home():
    return render_template("predictor.html")

@app.route("/paper-predictor", methods=["POST"])
def predictor():
    files = request.files.getlist("pdf_file")
    all_extracted_text = ""
    temp_files = []

    if not files or all(not file.filename for file in files):
        return jsonify({"error": "No files were uploaded."}), 400

    for file in files:
        if file and file.filename:
            filename = file.filename.lower()

            try:
                if filename.endswith(".pdf"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        file.save(tmp.name)
                        temp_files.append(tmp.name)
                        extracted = extract_text_from_pdf(tmp.name)
                        all_extracted_text += extracted + "\n"

                elif filename.endswith(".docx"):
                    extracted = extract_text_from_docx(file.stream)
                    all_extracted_text += extracted + "\n"

                else:
                    return jsonify({"error": f"Unsupported file type: {filename}"}), 400

            except Exception as e:
                return jsonify({"error": f"Error processing {filename}: {str(e)}"}), 500

    for f in temp_files:
        try: os.unlink(f)
        except: pass

    if not all_extracted_text.strip():
        return jsonify({"error": "No text extracted from uploaded files."}), 400

    identified_sections = identify_sections(all_extracted_text)
    prompt = build_openai_prompt(all_extracted_text, identified_sections)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a JSON generator. Only return valid JSON. No markdown or explanation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        content = response.choices[0].message.content
        start = content.find('{')
        end = content.rfind('}') + 1
        json_str = content[start:end].replace('\n', ' ')
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        data = json.loads(json_str)

        if "paper_title" not in data or "sections" not in data:
            raise ValueError("Missing required keys in OpenAI response.")

        return jsonify(data)

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
