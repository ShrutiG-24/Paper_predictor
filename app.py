from flask import Flask, render_template, request, send_from_directory
import os
import re
import fitz  # PyMuPDF
from docx import Document
from werkzeug.utils import secure_filename
from langdetect import detect, DetectorFactory
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

DetectorFactory.seed = 0

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, "images")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

client = OpenAI()  # Set this in your environment

def clean_lines(lines, char_threshold=30, word_threshold=20):
    allowed_chars = r"[A-Za-z0-9\s\.,\-\+=:\(\)\[\]\{\}/°μΩπσΔλ∞×∑α-ω√]"
    allowed_pattern = re.compile(f"[^{allowed_chars}]")
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    clean = []

    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:
            continue

        if devanagari_pattern.search(line):
            continue

        gibberish_char_count = len(allowed_pattern.findall(line))
        gibberish_word_count = len(re.findall(r'\b[^a-zA-Z\s]{3,}\b', line))

        if gibberish_char_count >= char_threshold or gibberish_word_count >= word_threshold:
            continue

        clean.append(line)

    return clean

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_clean_lines = []
    image_refs = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")
        lines = text.splitlines()
        clean = clean_lines(lines)
        if not clean:
            print(f"⚠ Skipping page {page_number} (no valid lines)")
            continue

        print(f"✅ Including lines from page {page_number}")
        all_clean_lines.extend(clean)

        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            image_filename = f"page-{page_number}-img-{img_index + 1}.{ext}"
            image_path = os.path.join(IMAGE_FOLDER, image_filename)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            image_refs.append(f"/uploads/images/{image_filename}")

    return "\n".join(all_clean_lines), image_refs

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join(para.text for para in doc.paragraphs), []

def is_english(text):
    try:
        cleaned = re.sub(r'[^\w\s]', '', text)
        if len(cleaned.strip()) < 5:
            return False
        return detect(cleaned) == "en"
    except:
        return False

def extract_questions(text, image_refs):
    lines = text.splitlines()
    clean_lines = []

    skip_keywords = [
        "time allowed", "maximum marks", "instructions", "note:", "attempt any",
        "read the following", "you may use", "marking scheme", "guidelines",
        "candidates must", "please check", "visually impaired",
        "this paper consists", "question paper code"
    ]

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(kw in line.lower() for kw in skip_keywords):
            continue
        clean_lines.append(line)

    questions = []
    current_q = ""

    for line in clean_lines:
        if re.match(r'^(Q\.?\s*\d+|Q\s*\d+|Q\.?|^\d{1,2}[.)])', line, re.IGNORECASE):
            if current_q:
                questions.append(current_q.strip())
            current_q = line
        elif re.match(r'^\(?[A-Da-d]\)?\.?', line) or line.startswith("    "):
            current_q += "\n    " + line
        else:
            current_q += " " + line.strip()

    if current_q:
        questions.append(current_q.strip())

    filtered = []
    img_index = 0
    for q in questions:
        if not is_english(q):
            continue
        if any(kw in q.lower() for kw in ["figure", "diagram", "image", "shown below"]):
            img = image_refs[img_index] if img_index < len(image_refs) else None
            q += "\n    [image]"
            filtered.append((q, img))
            img_index += 1
        else:
            filtered.append((q, None))

    return filtered

def generate_predicted_paper(previous_questions):
    prompt = (
        "You are an expert CBSE Physics paper setter. Based on the following previous exam questions, "
        "generate a predicted paper for the upcoming exam. Do not repeat the same questions. "
        "Use related concepts from the syllabus. Format as 10 questions, some objective, some descriptive.\n\n"
        "Previous questions:\n" +
        "\n".join(f"{i+1}. {q}" for i, (q, _) in enumerate(previous_questions)) +
        "\n\nPredicted Paper:\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        
    )

    return response.choices[0].message.content.strip().splitlines()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("pdf_file")
        if not file or file.filename == "":
            return "No file selected"

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        ext = filename.lower().split(".")[-1]
        if ext == "pdf":
            raw_text, image_refs = extract_text_from_pdf(file_path)
        elif ext == "docx":
            raw_text, image_refs = extract_text_from_docx(file_path)
        else:
            return "Unsupported file type. Please upload a PDF or DOCX."

        questions = extract_questions(raw_text, image_refs)
        question_list = [(f"Q{idx + 1}.", q, img) for idx, (q, img) in enumerate(questions)]

        predicted = generate_predicted_paper(questions)

        return render_template("result.html", questions=question_list, predicted=predicted, filename=filename)

    return render_template("index.html")

@app.route('/uploads/images/<path:filename>')
def uploaded_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
