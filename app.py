from flask import Flask, render_template, request, jsonify
import os
import fitz  # PyMuPDF
from docx import Document
import tempfile
import json
from dotenv import load_dotenv
from openai import OpenAI
import re
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import io
import openai
load_dotenv()
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set your OpenAI API key (ensure it's set in your environment)
openai.api_key = os.getenv("OPENAI_API_KEY")

def pdf_page_text(page):
    """Robustly extract text from a PyMuPDF page object."""
    try:
        return page.get_text("text")
    except Exception:
        try:
            return page.get_text()
        except Exception:
            try:
                return page.extract_text()
            except Exception:
                return ""

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF. Uses normal extraction first, then OCR if needed.
    """
    try:
        print(f"Attempting to extract text from PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += pdf_page_text(page)
        doc.close()

        # If text is too short or mostly whitespace, try OCR
        if len(text.strip()) < 100:
            print(f"Normal extraction found little text ({len(text.strip())} chars), trying OCR...")
            ocr_text = extract_text_with_ocr(pdf_path)
            if len(ocr_text.strip()) > len(text.strip()):
                print(f"OCR extracted more text ({len(ocr_text.strip())} chars), using OCR result.")
                return ocr_text
            else:
                print("OCR did not improve extraction, returning whatever was found.")
                return text
        else:
            print(f"Normal extraction succeeded ({len(text.strip())} chars).")
            return text
    except Exception as e:
        print(f"Error in normal extraction: {e}. Trying OCR as fallback...")
        try:
            ocr_text = extract_text_with_ocr(pdf_path)
            print(f"OCR fallback extracted {len(ocr_text.strip())} chars.")
            return ocr_text
        except Exception as ocr_e:
            print(f"OCR extraction also failed: {ocr_e}")
            return ""

def extract_text_from_docx(docx_stream):
    try:
        doc = Document(docx_stream)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_with_ocr(pdf_path):
    """
    Extract text from scanned PDF using OCR.
    """
    try:
        print(f"Starting OCR extraction for: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=300)
        print(f"Converted {len(images)} pages to images")
        extracted_text = ""
        for i, image in enumerate(images):
            print(f"OCR on page {i+1}/{len(images)}")
            if image.mode != 'RGB':
                image = image.convert('RGB')
            page_text = pytesseract.image_to_string(image, lang='eng')
            extracted_text += f"\n--- Page {i+1} ---\n{page_text}\n"
        print(f"OCR extraction completed. Total text length: {len(extracted_text.strip())}")
        return extracted_text
    except Exception as e:
        print(f"Error in OCR extraction: {e}")
        return ""

def is_scanned_pdf(pdf_path):
    """
    Check if PDF is scanned (image-based) by attempting to extract text
    Returns True if it's likely a scanned PDF, False if it has selectable text
    """
    try:
        doc = fitz.open(pdf_path)
        text_content = ""
        # Check first few pages for text content
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            text_content += pdf_page_text(page)
        doc.close()
        # If very little text is found, it's likely scanned
        text_length = len(text_content.strip())
        print(f"Text content length: {text_length}")
        # Threshold: if less than 100 characters found, consider it scanned
        return text_length < 100
    except Exception as e:
        print(f"Error checking if PDF is scanned: {e}")
        # If we can't determine, assume it's scanned and use OCR
        return True

def extract_total_marks(text):
    marks_match = re.findall(r"(?:Maximum\s*Marks[:\s]*)(\d{2,3})", text, re.IGNORECASE)
    if marks_match:
        return int(marks_match[0])
    fallback_match = re.search(r"\b(\d{2,3})\s*marks\b", text, re.IGNORECASE)
    if fallback_match:
        return int(fallback_match.group(1))
    return 70  # fallback

def identify_sections(text):
    section_pattern = re.compile(r'(Section\s+[A-Z])[^\n\r]*', re.IGNORECASE)
    matches = section_pattern.findall(text)
    sections = []
    for match in matches:
        full_title = match.strip()
        if full_title not in sections:
            sections.append(full_title)
    return sections

def build_openai_prompt(extracted_text, identified_sections, total_marks):
    section_list_str = '\n'.join(f'- {s}' for s in identified_sections) if identified_sections else "- Section A\n- Section B\n- Section C"

    prompt = f"""
You are an expert CBSE Class 12 paper setter. Generate a challenging {total_marks}-mark practice question paper for CBSE Class 12.

IMPORTANT: Refer to the latest NCERT Class 12 textbooks AND 12thclass.com question bank for all questions and topics. All predictions must be based on NCERT content, structure, and chapter divisions, supplemented with 12thclass.com's comprehensive question database.

CRITICAL INSTRUCTIONS FOR SUBJECT AND REFERENCE:
- Carefully analyze the uploaded question paper for subject, structure, and style.
- The predicted paper MUST match the subject of the uploaded paper (e.g., if the uploaded paper is Physics, the predicted paper must also be Physics).
- Use 12thclass.com as a reference for question patterns, difficulty levels, and topic coverage.

CRITICAL INSTRUCTIONS FOR DIFFICULTY AND STYLE:
- Carefully analyze the uploaded sample paper(s), past CBSE board papers, and 12thclass.com question patterns.
- Match the **difficulty level, language, and style** of the uploaded/board papers and 12thclass.com questions. If the uploaded paper uses higher-order thinking skills (HOTS), application-based, or analytical questions, do the same.
- Avoid generic or overly simple questions. Use real CBSE board exam rigor and complexity as seen on 12thclass.com.
- Use similar wording, context, and structure as seen in the uploaded/board papers and 12thclass.com question bank.

CRITICAL INSTRUCTIONS FOR MARKS AND QUESTION DISTRIBUTION:
- The sum of all question marks in the paper MUST EXACTLY match the extracted total marks: {total_marks}.
- For every section, the number of questions in the 'questions' list MUST EXACTLY match the 'total_questions' field for that section.
- For every MCQ question, always include an 'options' field with at least options A, B, C, and D.
- Before returning the JSON, double-check that the sum of all question marks in all sections and questions is exactly {total_marks}, and that each section's question count matches its 'total_questions'. If not, adjust the questions/marks so the sum and counts match.
- Distribute marks across sections and questions as per CBSE/NCERT board exam style (e.g., mix of 1, 2, 3, 4, 5-mark questions, etc.).
- Clearly indicate the marks for each question and section.

Instructions:
- Use the original sections found in the paper:
{section_list_str}
- Create questions under each of those section headings.
- Include a mix of MCQs (with options), short answers, long answers, and case studies.
- Use only challenging, application-based, and analytical questions similar to those on 12thclass.com.
- Every question must include the exact CBSE NCERT chapter name in a "chapter" field and corresponding "marks" field.
- After generating the paper, give a list of most important **topics** (not just chapter names) based on previous years' papers and 12thclass.com question analysis. Mention topic name and refer ncert textbooks to extract exact page numbers.
- ALSO, provide a list of the 10 most likely questions to appear in the upcoming exam, based on analysis of previous years' papers, CBSE patterns, and 12thclass.com question trends for this subject. For each, include the question text, chapter name, the exact NCERT page number, and the NCERT book name (e.g., 'Part 1' or 'Part 2'). Return this as a 'most_likely_questions' field in the JSON.
- Return ONLY valid JSON. No markdown, no explanation.

Format:
{{
  "paper_title": "CBSE Class 12 [Subject] Practice Question Paper ({total_marks} Marks)",
  "total_marks": {total_marks},
  "sections": [
    {{
      "section_name": "[From extracted paper]",
      "marks_per_question": X,
      "total_questions": Y,
      "questions": [
        {{
          "question": "...",
          "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
          "question_type": "MCQ/Numerical/Case Study/etc.",
          "chapter": "Electrostatics",
          "marks": X
        }}
      ]
    }}
  ],
  "important_topics": [
    {{"topic": "Electric field due to a point charge", "page": 14}},
    {{"topic": "Kirchhoff's rules with circuit analysis", "page": 87}},
    {{"topic": "Lens maker's formula derivation", "page": 312}}
  ],
  "most_likely_questions": [
    {{"question": "...", "chapter": "...", "page": ..., "ncert_book": "Part 1"}},
    ... (10 in total)
  ]
}}

Extracted text for reference:
{extracted_text}
"""
    return prompt


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predictor")
def predictor_page():
    return render_template("predictor.html")

@app.route("/paper-predictor", methods=["POST"])
def predictor():
    files = request.files.getlist("pdf_file")
    all_extracted_text = ""
    temp_files = []
    processing_info = []

    if not files or all(not file.filename for file in files):
        print("No files were uploaded.")
        return jsonify({"error": "No files were uploaded."}), 400

    for file in files:
        if file and file.filename:
            filename = file.filename.lower()
            try:
                print(f"\n--- Processing file: {filename} ---")
                if filename.endswith(".pdf"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        file.save(tmp.name)
                        temp_files.append(tmp.name)
                        file_size = os.path.getsize(tmp.name)
                        print(f"Saved PDF to {tmp.name} ({file_size} bytes)")
                        if file_size == 0:
                            print("File size is 0 bytes! Skipping.")
                            processing_info.append(f"{filename} is empty after upload.")
                            continue
                        # Check if it's a scanned PDF
                        is_scanned = is_scanned_pdf(tmp.name)
                        if is_scanned:
                            processing_info.append(f"Processing {filename} as scanned PDF (using OCR)")
                        else:
                            processing_info.append(f"Processing {filename} as text-based PDF")
                        extracted = extract_text_from_pdf(tmp.name)
                        print(f"Extracted text length: {len(extracted.strip())}")
                        all_extracted_text += extracted + "\n"
                        char_count = len(extracted.strip())
                        processing_info.append(f"Extracted {char_count} characters from {filename}")
                elif filename.endswith(".docx"):
                    processing_info.append(f"Processing {filename} as DOCX")
                    extracted = extract_text_from_docx(file.stream)
                    print(f"Extracted text length: {len(extracted.strip())}")
                    all_extracted_text += extracted + "\n"
                    char_count = len(extracted.strip())
                    processing_info.append(f"Extracted {char_count} characters from {filename}")
                else:
                    print(f"Unsupported file type: {filename}")
                    return jsonify({"error": f"Unsupported file type: {filename}"}), 400
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                return jsonify({"error": f"Error processing {filename}: {str(e)}"}), 500

    for f in temp_files:
        try:
            os.unlink(f)
        except Exception as e:
            print(f"Error deleting temp file {f}: {e}")

    if not all_extracted_text.strip():
        print("No text extracted from uploaded files!")
        return jsonify({"error": "No text extracted from uploaded files. Please check that your files are not password-protected, corrupted, or scanned in a way that is unreadable. If you are uploading scanned PDFs, ensure Tesseract OCR and Poppler are installed and accessible. See server logs for details."}), 400

    print("Processing summary:")
    for info in processing_info:
        print(f"  - {info}")

    identified_sections = identify_sections(all_extracted_text)
    total_marks = extract_total_marks(all_extracted_text)
    prompt = build_openai_prompt(all_extracted_text, identified_sections, total_marks)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a JSON generator. Only return valid JSON. No markdown or explanation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI response content is None. Check API response and prompt.")
        start = content.find('{')
        end = content.rfind('}') + 1
        json_str = content[start:end].replace('\n', ' ')
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        data = json.loads(json_str)

        if "paper_title" not in data or "sections" not in data:
            raise ValueError("Missing required keys in OpenAI response.")

        chapter_weightage = {}
        for section in data["sections"]:
            for q in section.get("questions", []):
                chapter = q.get("chapter", "Unknown")
                marks = q.get("marks", section.get("marks_per_question", 0))
                chapter_weightage[chapter] = chapter_weightage.get(chapter, 0) + marks
        data["chapter_weightage"] = chapter_weightage

        return jsonify(data)

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/12thgpt', methods=['GET', 'POST'])
def twelfth_gpt():
    return render_template('12thgpt.html')

@app.route('/12thgpt_api', methods=['POST'])
def twelfthgpt_api():
    system_prompt = (
        "You are 12thGPT, an expert CBSE Class 12 subject assistant. "
        "You ONLY answer questions that are directly relevant to CBSE Class 12 studies, subjects, or exam preparation. "
        "If a question is not related to studies, politely respond: 'Sorry, I can only help with CBSE Class 12 study-related questions.' "
        "For all valid questions, always answer in well-structured, numbered or bulleted points wherever possible. "
        "Be as precise and concise as possible. "
        "If the question mentions marks (e.g., 1 mark, 2 marks, 3 marks, 5 marks), answer in the style and depth expected in CBSE board papers for that mark value. "
        "For higher-mark questions, provide detailed, stepwise, and well-structured answers. "
        "Always use board-style formatting and clarity."
    )
    if 'text' in request.form and request.form['text'].strip():
        user_text = request.form['text'].strip()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})
    elif 'image' in request.files:
        image_file = request.files['image']
        img = Image.open(image_file.stream)
        extracted_text = pytesseract.image_to_string(img)
        if not extracted_text.strip():
            return jsonify({"answer": "Sorry, I couldn't read any question from the image. Please try again with a clearer image."})
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": extracted_text}
            ]
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})
    return jsonify({"answer": "Please enter a question or upload an image."})

if __name__ == "__main__":
    app.run(debug=True)
