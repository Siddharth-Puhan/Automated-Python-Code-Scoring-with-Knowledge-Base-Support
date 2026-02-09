from flask import Flask, request, render_template, session, redirect, url_for, send_file, make_response, flash
import requests, subprocess, os, re, json, spacy, io, logging
from os import abort
import ast, uuid, base64
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from flask import send_file
from reportlab.platypus import Image
from io import BytesIO
from nltk.corpus import stopwords


nlp = spacy.load("en_core_web_md")

app = Flask(__name__)
app.secret_key = "secret-key"

app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

KB_DIR = os.path.join(os.path.dirname(__file__), "Knowledge_base")
TRUTH_DIR = os.path.join(os.path.dirname(__file__), "Truth_tables")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_analysis_from_disk(analysis_id):
    if not analysis_id: return None
    path = os.path.join("analysis_report", f"{analysis_id}.json")
    if not os.path.exists(path):
        return None
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.exception("Failed to load analysis %s: %s", analysis_id, e)
        return None

# check if a valid python code has been submitted
def is_valid_python(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# AST based structural analysis. Checks for:
# unreachable code
# undefined variables
# missing returns
# empty loops
# syntax errors
def ast_structural_analysis(code):
    issues = []

    try:
        tree = ast.parse(code)
    except Exception as e:
        return [f"AST could not parse code: {e}"]

    assigned = set()

    def check_unreachable(body):
        unreachable = []
        hit_terminal = False
        for node in body:
            if hit_terminal:
                unreachable.append(f"Unreachable code at line {node.lineno}.")
            if isinstance(node, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                hit_terminal = True
        return unreachable

    for node in ast.walk(tree):

        # Track assigned variables
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    assigned.add(t.id)

        # Variable used before assignment
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in assigned and node.id not in dir(__builtins__):
                issues.append(
                    f"Variable `{node.id}` used before assignment (line {node.lineno})."
                )

        # Empty loop detection
        if isinstance(node, (ast.For, ast.While)):
            if not node.body:
                issues.append(f"Empty loop at line {node.lineno} — does nothing.")

        # Missing return
        if isinstance(node, ast.FunctionDef):
            returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
            if not returns:
                issues.append(
                    f"Function `{node.name}` has no return statement (line {node.lineno})."
                )

        # Unreachable code
        if hasattr(node, "body") and isinstance(node.body, list):
            issues.extend(check_unreachable(node.body))

    if not issues:
        return ["No structural issues detected by AST."]

    return issues











# Homepage
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')










# Code explanation using LLM 
@app.route('/explain', methods=['POST'])
def explain_code():
    user_code = request.form.get('code', '').strip()
    uploaded_file = request.files.get('file')

    # If file uploaded
    if uploaded_file and uploaded_file.filename:
        if not uploaded_file.filename.endswith(".py"):
            return render_template("index.html",
                                   explanation="Please upload a valid .py file.",
                                   valid_code=False)

        path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
        uploaded_file.save(path)

        with open(path, "r") as f:
            user_code = f.read().strip()

        os.remove(path)

    if not user_code:
        return render_template("index.html",
                               explanation="Please paste or upload Python code.",
                               valid_code=False)

    # Syntax check
    if not is_valid_python(user_code):
        return render_template(
            "index.html",
            explanation="This is NOT valid Python syntax.",
            code=user_code,
            valid_code=False
        )

    # LLM Explanation Prompt
    payload = {
        "model": "mistral",
        "prompt": f"""
You are a strict Python reviewer.
Find logical issues and explain EXACTLY what the code does.

1. Identify the algorithm or mathematical problem this code solves (e.g., factorial, Fibonacci, prime checking).
2. Give a topic to the code that is no longer than 4 words and captures exactly what the code does. You can refer to the function names to find the topic name as well.
3. Explain in brief exactly what the code does.
4. List ALL logical errors.
5. Show corrected versions, if any.
6. Simulate execution for common inputs.

CODE:
{user_code}
"""
    }

    explanation = ""

    try:
        response = requests.post("http://localhost:11434/api/generate",
                                 json=payload, stream=True)
    except:
        return render_template("index.html",
                               explanation="Ollama is not running.",
                               valid_code=False)

    for raw in response.iter_lines():
        if not raw:
            continue

        line = raw.decode().strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                explanation += data.get("response", "")
            except:
                pass

    explanation = explanation.replace("\\n", "\n").replace('\\"', '"').strip()

    # getting the topic name from the explanation
    def extract_topic(explanation):
        if not explanation:
            return None
        
        for line in explanation.splitlines():
            if line.lower().startswith("topic"):
                return line.split(":", 1)[-1].strip()

        return None
    
    extracted_topic = extract_topic(explanation)

    if not extracted_topic:
        category = "Miscellaneous"
        matched_topic = "Unknown"
        return render_template(
            "index.html",
            code=user_code,
            explanation=explanation,
            category=category,
            topic=matched_topic,
            valid_code=True
        )

    from hard_overrides import match_hard_override

    override_result = match_hard_override(user_code)

    if override_result:
        return render_template(
            "index.html",
            code=user_code,
            explanation=explanation,
            category=override_result["category"],
            topic=override_result["topic"],
            valid_code=True
        )
    

    # LLM tag generation - mainly used to search the KB for the optimal solution
    # kb_index.json contains the categories and the topics within those categories
    with open("kb_index.json") as f:
        KB = json.load(f)


    SIM_THRESHOLD = 0.75
    STOP_WORDS = list(stopwords.words('english'))
    STOP_WORDS.extend(["algorithm", "implementation", "function", "method", "class", "code", "using", "based", "approach", "solution"])

    def normalize_text(text):
        if not text:
            return ""

        doc = nlp(text.lower())
        tokens = [
            token.lemma_
            for token in doc
            if token.is_alpha and token.lemma_ not in STOP_WORDS
        ]
        tokens.sort()
        return " ".join(tokens)

    
    normalized_kb = []  # list of dicts

    for category, topic_list in KB.items():
        for t in topic_list:
            normalized_kb.append({
                "category": category,
                "raw": t,
                "normalized": normalize_text(t)
            })

    def keyword_overlap(a, b):
        return len(set(a.split()) & set(b.split()))


    def match_topic_to_kb(topic_name, normalized_kb):
        norm_topic = normalize_text(topic_name)

        for entry in normalized_kb:
            if norm_topic == entry["normalized"]:
                return entry["category"], entry["raw"]

        topic_doc = nlp(norm_topic)
        best_score = 0
        best_entry = None

        for entry in normalized_kb:
            kb_doc = nlp(entry["normalized"])
            score = topic_doc.similarity(kb_doc)

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= SIM_THRESHOLD:
            if keyword_overlap(norm_topic, best_entry["normalized"]) >= 1:
                return best_entry["category"], best_entry["raw"]


        return "Miscellaneous", topic_name

    category, matched_topic = match_topic_to_kb(extracted_topic, normalized_kb)

    return render_template("index.html",
                           code=user_code,
                           explanation=explanation,
                           category=category,
                           topic=matched_topic,
                           valid_code=True)






# Code analysis route
@app.route('/analyze', methods=['POST', 'GET'])
def analyze_code():
    user_code = request.form.get('code', '').strip()

    if not user_code:
        return render_template("index.html",
                               analysis="No code provided.",
                               valid_code=False)

    with open("temp_code.py", "w") as f:
        f.write(user_code)

    pylint_result = subprocess.run(
        ["pylint", "temp_code.py", "--disable=R,C"],
        capture_output=True, text=True
    ).stdout

    def extract_pylint_score(output):
        match = re.search(r"rated at ([0-9.]+)/10", output)
        if match:
            return float(match.group(1))
        return None
    pylint_score = extract_pylint_score(pylint_result.strip())


    cc_grades = {
        'A': "CC Range (1-5): Very simple and clear.",
        'B': "CC Range (6-10): Slight complexity.",
        'C': "CC Range (11-20): Moderate complexity.",
        'D': "CC Range (21-30): High complexity.",
        'E': "CC Range (31-40): Very high complexity.",
        'F': "CC Range (>40): Unmaintainable."
    }

    radon_cc = subprocess.run(
        ["radon", "cc", "temp_code.py", "-s", "-a"],
        capture_output=True, text=True
    ).stdout
    match = re.search(r"-\s+([A-F])\s*\(([\d.]+)\)", radon_cc)
    if match:
        cc_grade = match.group(1)
        cc_score = float(match.group(2))
        cc_remarks = cc_grades.get(cc_grade, "")
    else:
        cc_grade = "N/A"
        cc_score = 0
        cc_remarks = "Cyclomatic complexity could not be computed."

    # Maintainability Index
    mi_grades = {'A':"MI Range (65-100): Highly maintainable — clean and simple", 
                 'B':"MI Range (35-64): Moderate — acceptable but could be improved", 
                 'C':"MI Range (0-34): Hard to maintain — needs refactoring"} 
    
    radon_mi = subprocess.run(
        ["radon", "mi", "temp_code.py", "-s"],
        capture_output=True, text=True
    ).stdout
    match = re.search(r"-\s+([A-F])\s*\(([\d.]+)\)", radon_mi)
    if match:
        mi_grade = match.group(1)
        mi_score = float(match.group(2))
        mi_remarks = "" 
        if mi_grade in mi_grades.keys(): 
            mi_remarks = mi_grades[mi_grade]
    else:
        mi_grade = "N/A"
        mi_score = 0
        mi_remarks = "Maintainability Index could not be computed."

    # code to calculate the structural score
    def score_ast_issues(issues):
        if not issues or (len(issues) == 1 and "No structural issues" in issues[0]):
            return 10.0  # perfect

        if issues[0].startswith("AST could not parse"):
            return 0.0  # invalid code

        score = 10.0

        for issue in issues:
            txt = issue.lower()

            if "used before assignment" in txt:
                score -= 1.0
            elif "has no return" in txt:
                score -= 0.5
            elif "empty loop" in txt:
                score -= 0.5
            elif "unreachable code" in txt:
                score -= 0.5

        return max(0.0, round(score, 1))

    structural_issues = ast_structural_analysis(user_code)
    structural_output = "\n".join(structural_issues)
    structural_score = score_ast_issues(structural_issues)

    math_prompt = f"""
        You are an expert Python tutor and algorithm analyst.

        Identify ONLY mathematical or algorithmic mistakes such as:
        - wrong initialization (e.g., factorial = 0)
        - incorrect loop logic
        - incorrect base cases
        - wrong return values
        - math errors producing wrong output

        Then simulate execution for sample inputs.

        Give clean, concise output. Do NOT include chain-of-thought. 

        CODE:
        {user_code}
"""

    math_payload = {"model": "deepseek-r1:1.5b", "prompt": math_prompt, "max_tokens": 300, "temperature": 0.2}
    math_feedback_raw = ""
    math_feedback = ""
    try:
        resp = requests.post("http://localhost:11434/api/generate", json=math_payload, timeout=45)
        
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode().strip()
            try:
                data = json.loads(line)
                if "response" in data and data["response"].strip():
                    math_feedback_raw += data["response"]

                if "thinking" in data:
                    pass
            except json.JSONDecodeError:
                continue

        math_feedback = math_feedback_raw.replace("\\n", "\n").replace('\\"', '"').strip()

        if not math_feedback:
            math_feedback = "No mathematical issues found or the model returned an empty response."


    except Exception as e:
        math_feedback = f"Mathematical analysis LLM unavailable: {e}"


    # calculating the logic score using deepseek's response as input to phi3

    logic_scorer_prompt = f"""
        You are scoring the correctness of Python logic.
        Here is an analysis of a piece of code:{math_feedback}

        Based on the analysis:

        - If no mistakes are mentioned → score = 10
        - If minor mistakes but overall correct → score = 7–9
        - If multiple logic issues → score = 4–6
        - If the code is largely incorrect → score = 1–3
        - If the code is completely wrong → score = 0

        Respond in this EXACT format, with NO extra text:
        logic_score: <number>
        """
    
    logic_payload = {"model": "phi3", "prompt": logic_scorer_prompt}
    logic_response = ""

    try:
        resp = requests.post("http://localhost:11434/api/generate", json=logic_payload, timeout=30)
        for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode().strip()
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            logic_response += data["response"]
                    except json.JSONDecodeError:
                        continue

    except Exception as e:
        logic_response = f"Logic Scoring Failed: {e}"
    
    logic_response = logic_response.strip()
    match = re.search(r"logic_score:\s*(\d+)", logic_response)
    logic_score = int(match.group(1)) if match else 0


    # ---------- Complexity & Readability LLM ----------
    complexity_prompt = f"""
        You are an expert in algorithm analysis.

        Analyze the following Python code and provide:

        1. **Time Complexity** (in Big-O notation)
        2. **Space Complexity** (in Big-O notation)
        3. **A brief explanation** for each complexity.
        4. **A score out of 10** for overall efficiency:
        - 10 = extremely efficient (O(1), O(log n))
        - 7–9 = good efficiency
        - 4–6 = average efficiency
        - 1–3 = poor efficiency
        - 0 = extremely inefficient or incorrect complexity

        Return your output in the following format ONLY:

        time_complexity: "...",
        space_complexity: "...",
        explanation: "...",
        efficiency_score: 0-10

        CODE:
        {user_code}
"""


    llm_payload = {"model": "phi3", 
                   "prompt": complexity_prompt,
                   "temperature": 0}
    llm_feedback = ""

    try:
        cplx_response = requests.post("http://localhost:11434/api/generate",
                                      json=llm_payload, stream=True)
        for raw in cplx_response.iter_lines():
            if not raw:
                continue
            line = raw.decode().strip()
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    llm_feedback += data.get("response", "")
                except:
                    pass

    except:
        llm_feedback = "Complexity LLM unavailable."

    match = re.search(r"efficiency_score:\s*(\d+)", llm_feedback)
    complexity_score = int(match.group(1)) if match else 0

    func_count = user_code.count("def ")
    class_count = user_code.count("class ")

    code = request.form.get("code")
    category = request.form.get("category")
    topic = request.form.get("topic") # not required for calculating the loc score


    # calculate LOC
    def count_loc(code):
        loc = 0
        for line in code.split("\n"):
            striped = line.strip()
            if striped and not striped.startswith("#"):
                loc += 1
        return loc

    # function to score the code on the basis of LOC
    LOC_BENCHMARKS = {
        "Basic Python": (5, 35),
        "Dictionary Comprehension": (5, 20),
        "Graph": (30, 120),
        "Linked List": (25, 80),
        "List Comprehension": (3, 15),
        "Miscellaneous": (15, 60),  # Problems like knapsack, ToH
        "Search Algorithms": (10, 40),
        "Sort Algorithms": (15, 100),  # Merge/quick sort 
        "Stack": (10, 70)
    }


    def score_loc_by_category(loc, category):
        if category not in LOC_BENCHMARKS:
            return 5   # default neutral

        low, high = LOC_BENCHMARKS[category]

        # ideal range
        if low <= loc <= high:
            return 10

        # too short
        if loc < low:
            diff = low - loc
            return max(0, 10 - diff * 0.5)

        # too long
        if loc > high:
            diff = loc - high
            return max(0, 10 - diff * 0.3)

    loc = count_loc(code)
    loc_score = score_loc_by_category(loc, category)

    # calculating the composite score for the code
    # map cyclomatic grade to scores
    def map_cc_to_score(cc_value):
        cc_value = float(cc_value)

        if 1 <= cc_value <= 3:
            return 10
        elif 4 <= cc_value <= 5:
            return 9
        elif 6 <= cc_value <= 8:
            return 8
        elif 9 <= cc_value <= 10:
            return 7
        elif 11 <= cc_value <= 15:
            return 6
        elif 16 <= cc_value <= 20:
            return 5
        elif 21 <= cc_value <= 30:
            return 4
        else:
            return 3

    cyclomatic_score = map_cc_to_score(cc_score)
    composite_score = (
        float(pylint_score)*0.25 + 
        float(complexity_score)*0.25 + 
        float(loc_score)*0.05 + 
        float(structural_score)*0.05 + 
        float(logic_score)*0.3 + 
        float(cyclomatic_score)*0.05 + 
        (mi_score/10)*0.05
        )
    composite_score = round(composite_score, 2)

    # ---------- Final Report ----------
    combined = f"""
CODE ANALYSIS REPORT
-----------------------------------------------
LINES OF CODE: {loc}
Functions: {func_count}
Classes: {class_count}

Score: {loc_score}/10

---------- STRUCTURAL ERRORS (AST) ----------
{structural_output}

Score: {structural_score}/10

---------- PYLINT ----------
{pylint_result.strip()}

Score: {pylint_score}/10

---------- CYCLOMATIC COMPLEXITY ----------
Grade: {cc_grade}
Score: {cc_score}
Remarks: {cc_remarks}

---------- MAINTAINABILITY INDEX ----------
Grade: {mi_grade}
Score: {mi_score}
Remarks: {mi_remarks}

---------- MATHEMATICAL / ALGORITHMIC ERRORS (LLM) ----------
{math_feedback}

---------- TIME/SPACE COMPLEXITY & READABILITY ----------
{llm_feedback}

---------- COMPOSITE SCORE FOR YOUR CODE ----------
{composite_score}

--------------------------------------------
"""
    
    # persisting the complete analysis report
    analysis_id = str(uuid.uuid4())

    analysis_payload = {
        "meta": {
        "analysis_id": analysis_id,
        "timestamp": datetime.utcnow().isoformat(),
        "category": category,
        "topic": topic
    },
    "scores": {
        "pylint_score": round(float(pylint_score), 2),
        "complexity_score": round(float(complexity_score), 2),
        "loc_score": round(float(loc_score), 2),
        "structural_score": round(float(structural_score), 2),
        "logic_score": round(float(logic_score), 2),
        "cyclomatic_score": round(float(cyclomatic_score), 2),
        "mi_score": round(float(mi_score / 10), 2),
        "composite_score": composite_score
    },
    "raw_outputs": {
        "pylint": pylint_result.strip(),
        "ast": structural_issues,
        "math_llm": math_feedback,
        "complexity_llm": llm_feedback
    },
    "final_report": combined
    }

    try:
        os.makedirs("analysis_report", exist_ok=True)
        with open(f"analysis_report/{analysis_id}.json", "w", encoding="utf-8") as f:
            json.dump(analysis_payload, f, indent=2)
    except Exception as e:
        logger.exception("Failed to persist analysis: %s", e)

    # save the last analysis in session so compare page can read it
    try:
        session['analysis_id'] = analysis_id
        session['last_analysis'] = {
            "pylint_score": round(float(pylint_score), 2),
            "complexity_score": round(float(complexity_score), 2),
            "loc_score": round(float(loc_score), 2),
            "structural_score": round(float(structural_score), 2),
            "logic_score": round(float(logic_score), 2),
            "cyclomatic_score": round(float(cyclomatic_score), 2),
            "mi_score": round(float(mi_score / 10), 2),
            "composite_score": round(float(composite_score), 2),
        }
        session['last_category'] = category
        session['last_topic'] = topic
    except Exception as e:
        logger.exception("Failed to save analysis to session: %s", e)

    # cleanup
    try:
        os.remove("temp_code.py")
    except:
        pass

    # Pass a flag so index.html can enable the Compare button
    return render_template("index.html",
                           code=user_code,
                           analysis=combined,
                           valid_code=True,
                           compare_available=True)





# displaying the contents of KB
# make the page URLs more readable
def slugify(text):
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text.strip("-")





# pretty name from filename
def pretty_category_name(filename):
    name = os.path.splitext(filename)[0]
    return name.replace("_", " ").title()





# List available category files
def list_category_files():
    files = []
    try:
        for fname in sorted(os.listdir(KB_DIR)):
            if fname.endswith(".json"):
                files.append(fname)
    except FileNotFoundError:
        return []
    return files






# Load full JSON array from a category file, return list of dict entries
def load_category_entries(category_slug):
    filename = f"{category_slug}.json"
    path = os.path.join(KB_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data is expected to be a list of objects
    return data





# Find entry by topic slug
def find_entry_by_topic(entries, topic_slug):
    for entry in entries:
        topic = entry.get("topic", "")
        if slugify(topic) == topic_slug:
            return entry
    return None






@app.route("/kb")
def kb_index():
    files = list_category_files()
    categories = [
        {
            "slug": os.path.splitext(fname)[0],
            "pretty": pretty_category_name(fname)
        } for fname in files
    ]
    return render_template("kb.html", categories=categories)





@app.route("/kb/<category_slug>")
def kb_category(category_slug):
    entries = load_category_entries(category_slug)
    if entries is None:
        abort(404)

    # extract topics and topic slugs
    topics = []
    for e in entries:
        topic = e.get("topic", "")
        topics.append({
            "topic": topic,
            "slug": slugify(topic)
        })

    # optional search q param (search by topic name)
    q = request.args.get("q", "").strip().lower()
    if q:
        topics = [t for t in topics if q in t["topic"].lower()]

    category_pretty = pretty_category_name(category_slug + ".json")
    return render_template("kb_topics.html", category_slug=category_slug,
                           category_pretty=category_pretty, topics=topics, q=q)





@app.route("/kb/<category_slug>/<topic_slug>")
def kb_detail(category_slug, topic_slug):
    entries = load_category_entries(category_slug)
    if entries is None:
        abort(404)
    entry = find_entry_by_topic(entries, topic_slug)
    if entry is None:
        abort(404)

    # Provide safe defaults
    description = entry.get("description", "")
    optimal_code = entry.get("optimal_code", {})
    complexity = entry.get("complexity", {})
    metrics = entry.get("metrics", {})
    tags = entry.get("tags", [])

    return render_template("kb_detail.html",
                           category_slug=category_slug,
                           category_pretty=pretty_category_name(category_slug + ".json"),
                           entry=entry,
                           description=description,
                           optimal_code=optimal_code,
                           complexity=complexity,
                           metrics=metrics,
                           tags=tags,
                           topic_slug=topic_slug)





@app.route("/compare")
def compare_page():
    # Load user analysis from session
    user_scores = session.get("last_analysis")
    category = session.get("last_category")
    topic = session.get("last_topic")   
    analysis_id = session.get("analysis_id")
    full_analysis = load_analysis_from_disk(analysis_id)

    if not full_analysis:
        return render_template(
            "index.html",
            analysis="Stored analysis could not be loaded!",
            valid_code=False
        )

    if not user_scores or not category or not topic:
        return render_template("index.html",
                               analysis="No recent analysis available to compare.",
                               valid_code=False)

    # Only support Basic Python for now
    # Load the truth table file
    truth_file = os.path.join(TRUTH_DIR, "basic_python_truth_table.json")
    if not os.path.exists(truth_file):
        return render_template("index.html",
                               analysis="Truth table file not found.",
                               valid_code=False)

    with open(truth_file, "r", encoding="utf-8") as f:
        truth_entries = json.load(f)

    # find entries for this topic (case-insensitive)
    candidates = [e for e in truth_entries if e.get("topic","").strip().lower() == topic.strip().lower()]

    if not candidates:
        # try fuzzy match via slug
        def slugify(text):
            text = text.lower()
            text = re.sub(r"[^\w\s-]", "", text)
            text = re.sub(r"[\s_]+", "-", text)
            return text.strip("-")
        topic_slug = slugify(topic)
        candidates = [e for e in truth_entries if slugify(e.get("topic","")) == topic_slug]

    if not candidates:
        flash(f"No truth-table entry found for topic '{topic}'.", "warning")
        return redirect(url_for("home"))

    # choose the candidate with the highest composite_score as the 'optimal'
    def safe_comp(e):
        try:
            return float(e.get("metrics", {}).get("composite_score", 0))
        except:
            return 0.0
    candidates.sort(key=safe_comp, reverse=True)
    optimal = candidates[0]

    # Prepare side-by-side metric dicts
    optimal_metrics = optimal.get("metrics", {})
    # normalize keys to the ones we store in user_scores
    # ensure floats
    def norm(val):
        try:
            return float(val)
        except:
            return val

    optimal_scores = {
        "pylint_score": norm(optimal_metrics.get("pylint_score", 0)),
        "complexity_score": norm(optimal_metrics.get("complexity_score", 0)),
        "loc_score": norm(optimal_metrics.get("loc_score", 0)),
        "structural_score": norm(optimal_metrics.get("structural_score", 0)),
        "logic_score": norm(optimal_metrics.get("logic_score", 0)),
        "cyclomatic_score": norm(optimal_metrics.get("cyclomatic_score", optimal_metrics.get("cyclomatic_complexity", 0))),
        "mi_score": norm((optimal_metrics.get("mi_score") or optimal_metrics.get("maintainability_index", 0))/10),
        "composite_score": norm(optimal_metrics.get("composite_score", 0))
    }

    # compute differences
    differences = {}
    for k, u_val in user_scores.items():
        if k in optimal_scores and isinstance(optimal_scores[k], (int, float)) and isinstance(u_val, (int, float)):
            differences[k] = round(u_val - optimal_scores[k], 2)
        else:
            differences[k] = None

    
    # now append the report with the optimal data and difference
    report_path = os.path.join("analysis_report", f"{analysis_id}.json")

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    report["comparison"] = {
        "user_scores": user_scores,
        "optimal_scores": optimal_scores,
        "differences": differences,
        "kb_reference": {
            "category": category,
            "topic": topic
        }
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return render_template(
        "compare.html",
        topic=topic,
        user_scores=user_scores,
        optimal_scores=optimal_scores,
        differences=differences,
        truth_entry=optimal,
        analysis_id=analysis_id,
        full_analysis=full_analysis
        )






@app.route("/report/<analysis_id>")
def report_pages(analysis_id):
    report_path = os.path.join("analysis_report", f"{analysis_id}.json")

    if not os.path.exists(report_path):
        return render_template(
            "index.html",
            analysis="Report not found.",
            valid_code=False
        )

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    return render_template(
        "report.html",
        report=report
    )








@app.route("/report/<analysis_id>/pdf", methods=["POST"])
def download_report_pdf(analysis_id):
    chart_image = request.form.get("chart_image")
    report_path = os.path.join("analysis_report", f"{analysis_id}.json")

    if not os.path.exists(report_path):
        return "Report not found", 404

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("<b>Code Analysis Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Metadata
    meta = report["meta"]
    elements.append(Paragraph(f"<b>Category:</b> {meta['category']}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Topic:</b> {meta['topic']}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Timestamp:</b> {meta['timestamp']}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    scores = report["scores"]

    table_data = [["Metric", "Score (/10)"]]
    for k, v in scores.items():
        table_data.append([k.replace("_", " ").title(), v])

    table = Table(table_data, colWidths=[250, 100])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ]))

    elements.append(Paragraph("<b>Scores</b>", styles["Heading2"]))
    elements.append(Spacer(1, 8))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # text analysis
    elements.append(Paragraph("<b>Detailed Analysis</b>", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    for line in report["final_report"].split("\n"):
        elements.append(Paragraph(line.replace("&", "&amp;"), styles["Normal"]))
        elements.append(Spacer(1, 4))

    if chart_image:
        image_data = chart_image.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        img = Image(BytesIO(image_bytes), width=400, height=250)
        elements.append(img)

    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"analysis_{analysis_id}.pdf",
        mimetype="application/pdf"
    )



# Optional: root redirect to /kb
@app.route("/")
def root():
    return """<script>location='/kb'</script>"""


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
