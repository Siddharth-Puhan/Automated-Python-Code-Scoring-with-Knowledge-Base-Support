DOCUMENTATION

Architecture Diagram

            User Code
               ↓
        Static Analysis
               ↓
      LLM Explanation + Topic Guess
              ↓
      Hybrid Topic Matcher
              ↓
Match Found: Compare with KB and Store Report 
              ↓
            No Match
               ↓
Clarification / Disambiguation Prompt
               ↓
      Provisional Categorization
               ↓
      KB Expansion Candidate
               ↓
     Save as Learnable Entry

This pipeline enables the system to continuously expand its knowledge base by identifying previously unseen programming patterns and storing them as learnable entries for future comparisons.

STEP 1: Code Submission
* User submits Python code via the web UI (index.html)
* Code is sent to /analyze
No assumptions are made about correctness or runtime behavior.

STEP 2: Static Analysis (Non-executing)
Your system extracts metrics using:
* AST parsing
* Cyclomatic complexity
* Maintainability index
* Structural patterns
* Logical heuristics
* Pylint-style checks
This produces a metrics dictionary, for example:

{
  "pylint_score": 7.5,
  "logic_score": 8.0,
  "structural_score": 6.8,
  "cyclomatic_score": 7.2,
  "mi_score": 6.9,
  "composite_score": 7.28
}
Important:
* These metrics are normalized
* Composite score is a weighted aggregate

STEP 3: LLM-Based Explanation & Topic Extraction
The code is sent to a local LLM (Ollama) with a strict prompt:
* Explain exactly what the code does
* Assign a short topic name (≤ 6 words)
* Identify logical issues
* Suggest corrections
* Simulate execution (conceptually)
From the LLM output:
* A human-readable explanation is produced
* A raw topic string is extracted (e.g., "Iterative Factorial")
This step provides semantic understanding, not scoring.

STEP 4: Topic Matching (KB Alignment)
This is where intelligence kicks in.
The extracted topic is matched against the Knowledge Base (KB) using:
1. Hard keyword overrides (planned)
2. Semantic similarity (spaCy embeddings)
3. Fallback categorization
Result:

{
  "category": "Basic Python",
  "topic": "Iterative Factorial"
}
If no match is found:
* The code is temporarily categorized as "Miscellaneous"
* (Later phase: clarification + KB expansion)

STEP 5: Knowledge Base Comparison
If a KB entry exists:
* Optimal metrics are loaded from the truth table
* User metrics are aligned against optimal metrics
* Differences are computed per metric
Example:

"differences": {
  "logic_score": -1.2,
  "cyclomatic_score": 0.5,
  "mi_score": -0.8
}
This enables:
* Side-by-side comparison
* Visual charts
* Feedback generation

STEP 6: Report Generation & Persistence
A single source of truth is created:
📁 analysis_report/<analysis_id>.json
This report contains:
* Metadata (topic, category, timestamp)
* User scores
* LLM explanation
* Comparison data (if available)
All views (/report, /compare, /pdf) read from this file.
This is a very good architectural decision — stable, debuggable, reproducible.

STEP 7: Self-Learning & Knowledge Base Expansion
If the extracted topic cannot be confidently matched to any existing Knowledge Base (KB) entry, the system does not discard the analysis.
Instead, it enters a self-learning mode, where the code is treated as a potential new knowledge candidate.
This process works as follows:
* The system triggers a clarification / disambiguation prompt (optional future extension)
* The code is provisionally categorized under:
    * An existing broad category (e.g., Miscellaneous), or
    * A semantically closest category inferred from embeddings
* Static analysis metrics computed earlier are treated as ideal baseline metrics for this new pattern
* The following information is stored as a learnable KB entry:
    * Topic name inferred from LLM
    * Category
    * User-submitted code (as reference implementation)
    * Static metrics (cyclomatic complexity, MI, LOC, etc.)
    * Tags inferred from structure and keywords
This enables the system to grow its knowledge base over time without manual curation.
Future submissions matching this pattern can then:
* Be recognized automatically
* Be compared against this newly learned optimal reference
* Receive structured, metric-based feedback
📌 Important properties of this approach:
* No code execution is required
* Learning is entirely static-analysis driven
* The KB evolves incrementally and transparently
* Previously “unknown” programs become first-class citizens in the system


Data & Storage Design
Knowledge Base
* kb_index.json → category → topic mapping
* Truth tables store:
    * Optimal code
    * Ideal metrics
    * Tags
    * Complexity notes
Reports
* Stored per analysis ID
* Append-only enrichment (comparison added later)
Session State
Used only for:
* Last analysis
* Navigation convenience
Persistent truth always lives on disk.
Learnable entries generated from unmatched programs are appended incrementally


Deployment & Local Setup

1 System Requirements
* Python 3.10+
* Ollama (running locally)
* spaCy English model
* Git

2 Clone the Repository

git clone https://github.com/Siddharth-Puhan/Automated-Python-Code-Scoring-with-Knowledge-Base-Support.git
cd Automated-Python-Code-Scoring-with-Knowledge-Base-Support

3 Create & Activate Virtual Environment

python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

4 Install Dependencies

pip install -r requirements.txt
If spaCy model is missing:

python -m spacy download en_core_web_sm

5 Start Ollama
Make sure Ollama is running:

ollama run mistral
Or keep it running in background.
The app assumes: http://localhost:11434

6 Run the Flask App

python app.py
Then open:

http://127.0.0.1:5000

7 Common Failure Modes (Document This!)
* Ollama not running → LLM explanation fails
* spaCy model missing → similarity matching breaks
* Missing truth table → comparison unavailable

