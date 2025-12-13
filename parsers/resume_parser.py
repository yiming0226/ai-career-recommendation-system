import os
import re
import string
from typing import List, Dict, Optional, Set

import pandas as pd
import spacy
from pdfminer.high_level import extract_text
from rapidfuzz import fuzz, process

# ===== Load SpaCy model ===== 
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError("SpaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")

# ===== Constants and compiled regexes ===== 
SECTION_HEADERS = {
    "experience", "work experience", "employment history", "projects", "project",
    "education", "skills", "certifications", "contact", "contact info",
    "summary", "profile", "objective", "achievements", "publications",
    "internships", "research", "technical skills", "volunteer", "awards"
}
GENERIC_HEADERS = {"functional resume sample", "resume", "cv", "curriculum vitae"}

_CID_RE = re.compile(r"\(cid:\d+\)")
_EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
_URL_RE = re.compile(r'(https?://\S+|www\.\S+)')
_PHONE_CANDIDATE_RE = re.compile(r'(?:\+|00)?[0-9\-\.\(\)\s]{6,}[0-9]', flags=re.M)
_PDF_ARTIFACT_RE = re.compile(r'cid|page|section|¶|\u00A7', flags=re.I)
_DIGITS_RE = re.compile(r'\D')
_NAME_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z\-\.]'{0,1}[A-Za-z]*$")

_YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')
_YEAR_RANGE_RE = re.compile(r'\b(19|20)\d{2}\s*[-–—]\s*(19|20)\d{2}\b')

_JOB_WORDS = {
    "engineer", "developer", "designer", "manager", "analyst",
    "consultant", "specialist", "coordinator", "intern", "technician",
    "assistant", "associate", "officer", "executive", "administrator", "lead", "sr", "jr"
}

_DECORATIVE_RE = re.compile(r'^[\s\W○•·●•\u2022]{2,}$')
_TOO_MANY_NONALNUM = re.compile(r'^[^A-Za-z0-9]{3,}$')

_FALLBACK_JOB_WORDS = {
    "engineer", "developer", "designer", "manager", "analyst", "consultant",
    "specialist", "coordinator", "intern", "technician", "assistant", "associate",
    "officer", "executive", "administrator", "lead", "scientist", "representative",
    "recruiter", "architect", "director", "supervisor", "sales", "account", "advisor",
    "generalist", "human", "hr", "humanresources", "administrator"
}

TITLE_MODIFIERS = {
    "junior", "senior", "jr", "sr", "lead", "principal", "intern",
    "assistant", "associate", "entry", "mid-level", "trainee", "staff",
    "internship", "head", "manager", "director", "engineer", "developer",
    "summer", "winter", "spring", "fall"
}
_EDU_TERMS = {"university", "college", "b.sc", "bachelor", "bachelor's", "m.sc", "master", "mba", "mca", "phd", "degree", "education", "educational", "school", "institute"}
_SECTION_BOUNDARY_TERMS = {"education", "educational", "achievements", "achievement", "awards", "interests", "extracurricular", "curricular", "projects", "experience", "professional experience", "employment history", "publications", "certifications", "certification", "educational history"}

# ===== Utility text normalization =====
def _merge_spaced_letters(line: str) -> str:
    """Merge spaced-letter patterns like 'J o h n' -> 'John'."""
    parts = re.split(r'\s{2,}', line)
    merged_parts = []
    for part in parts:
        if re.fullmatch(r'(?:[A-Za-z]\s)+[A-Za-z]', part.strip()):
            merged_parts.append(part.replace(' ', ''))
        else:
            merged_parts.append(part)
    return ' '.join(merged_parts)


def _normalize_unicode_phone_chars(text: str) -> str:
    if not text:
        return text
    replacements = {
        '\u2010': '-', '\u2011': '-', '\u2012': '-', '\u2013': '-', '\u2014': '-', '\u2015': '-',
        '\u2212': '-', '\u00A0': ' ', '\u2007': ' ', '\u202F': ' '
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text

def normalize_resume_text(text: str) -> str:
    """Basic cleaning: remove CID artifacts and merge spaced-letter lines."""
    if not text:
        return ""
    lines = []
    for ln in text.splitlines():
        ln = _CID_RE.sub(' ', ln).strip()
        if not ln:
            continue
        ln = _merge_spaced_letters(ln)
        lines.append(ln)
    return "\n".join(lines)

# ===== PDF extraction wrapper ===== 
def extract_text_from_pdf(pdf_path: str) -> str:
    raw_text = extract_text(pdf_path)
    return normalize_resume_text(raw_text)

# ===== Duration helpers ===== 
def _looks_like_year_sequence(raw: str, digits_only: str) -> bool:
    raw_s = raw.strip()
    if re.match(r'^\+?0*\s*\d{4}\s*[-–]\s*\d{4}\s*$', raw_s):
        return True
    if len(digits_only) >= 8 and len(digits_only) % 4 == 0:
        chunks = [digits_only[i:i+4] for i in range(0, len(digits_only), 4)]
        if len(chunks) >= 2 and all(1900 <= int(c) <= 2099 for c in chunks):
            return True
    years_found = re.findall(r'(?:19|20)\d{2}', raw_s)
    return len(years_found) >= 2

def is_duration_line(line: str) -> bool:
    """Heuristic checks for lines containing dates/durations."""
    if not line or not line.strip():
        return False
    line = line.replace("–", "-").replace("—", "-")
    if re.search(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b', line, re.I):
        return True
    if re.search(r'\b(?:[A-Za-z]{3,}\s+\d{4})\s*(?:to|[-–])\s*(?:[A-Za-z]{3,}\s+\d{4}|present|current|\d{4})\b', line, re.I):
        return True
    if re.search(r'\b(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}|present|current\b', line, re.I):
        return True
    if re.search(r'\b(?:19|20)\d{2}\b', line):
        return True
    return False

def extract_duration_from_line(line: str) -> Optional[str]:
    """Extracts the longest matching duration/date substring from a line."""
    if not line:
        return None
    line = line.replace("–", "-").replace("—", "-")
    patterns = [
        r'(?:[A-Za-z]{3,}\s+\d{4}\s*(?:to|[-])\s*(?:[A-Za-z]{3,}\s+\d{4}|present|current))',
        r'\b(?:19|20)\d{2}\s*[-]\s*(?:19|20)\d{2}|present|current\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
        r'\b(?:19|20)\d{2}\b'
    ]
    matches = []
    for pat in patterns:
        for m in re.finditer(pat, line, re.I):
            matches.append(m.group(0).strip())
    return max(matches, key=len) if matches else None

# ===== Name extraction ===== 
def _is_spaced_letter_line(line: str, min_letters: int = 3) -> bool:
    if not line or len(line) < min_letters:
        return False
    cleaned = re.sub(r'[^\w\s]', ' ', line).strip()
    tokens = [t for t in cleaned.split() if t]
    if len(tokens) < min_letters:
        return False
    single_len = sum(1 for t in tokens if len(t) == 1)
    return single_len >= (len(tokens) * 0.6)

def _sanitize_candidate_name(name: str) -> str:
    if not name:
        return name
    prot = re.sub(r'\b([A-Za-z])\.\b', r'\1<DOT>', name)
    prot = re.sub(r'(?:\s+\b[A-Za-z]\b){3,}', '', prot)
    prot = prot.replace('<DOT>', '.')
    return re.sub(r'\s+', ' ', prot).strip() or name

def _titlecase_name(name: str) -> str:
    """Titlecase while preserving initials and hyphen/apostrophe parts."""
    if not name:
        return name

    def cap_part(p: str) -> str:
        if len(p) == 1 and p.isalpha():
            return p.upper()
        if len(p) == 2 and p[1] == '.' and p[0].isalpha():
            return p[0].upper() + '.'
        return p[0].upper() + p[1:].lower() if p else p

    words = name.split()
    out = []
    for w in words:
        pieces = re.split("([-'])", w)
        pieces = [cap_part(piece) if piece not in ("-", "'") else piece for piece in pieces]
        out.append("".join(pieces))
    return " ".join(out)

def _is_section_header(line: str) -> bool:
    low = re.sub(r'[^a-z0-9\s]', '', (line or "").lower()).strip()
    if not low:
        return False
    for h in SECTION_HEADERS:
        if low == h or low.startswith(h + ' ') or low.startswith(h + ' &') or (' ' + h + ' ') in (' ' + low + ' '):
            return True
    return False

def _ner_person_candidate(text: str) -> Optional[str]:
    """Use SpaCy NER to find PERSON entities when available."""
    if not text or 'nlp' not in globals():
        return None
    try:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                ent_clean = re.sub(r'\s+', ' ', ent.text).strip()
                if len(ent_clean.split()) >= 2 and not re.search(r'[\d@/\\]', ent_clean):
                    return ent_clean
    except Exception:
        pass
    return None

def extract_name(resume_text: str) -> Optional[str]:
    """Attempts multiple heuristics and NER to extract the candidate name."""
    if not resume_text or not resume_text.strip():
        return None
    text = resume_text.replace('\r', '\n')
    lines = []
    for ln in text.splitlines():
        ln = _CID_RE.sub(' ', ln).strip()
        if not ln:
            continue
        ln = _merge_spaced_letters(ln)
        lines.append(ln)

    header_block = "\n".join(
        ln for ln in lines[:4]
        if not _is_spaced_letter_line(ln) and not _is_section_header(ln) and ln.lower() not in GENERIC_HEADERS
    )
    candidate = None
    if header_block:
        cand = _ner_person_candidate(header_block)
        if cand:
            candidate = _sanitize_candidate_name(cand)

    for idx, raw in enumerate(lines):
        if not raw or _is_spaced_letter_line(raw) or _is_section_header(raw) or raw.lower() in GENERIC_HEADERS:
            continue
        low = raw.lower().strip()
        if candidate is None:
            cand = _ner_person_candidate(raw)
            if cand:
                candidate = _sanitize_candidate_name(cand)
        if candidate is None and raw.isupper():
            next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
            if _is_spaced_letter_line(next_line):
                next_line = ""
            if next_line and next_line.isupper() and len(raw.split()) == 1 and len(next_line.split()) == 1 and \
               all(w.isalpha() for w in (raw + " " + next_line).split()):
                candidate = f"{raw.title()} {next_line.title()}"
            elif len(raw.split()) >= 2 and all(w.isalpha() for w in raw.split()):
                candidate = " ".join(w.title() for w in raw.split())

        if candidate is None:
            tokens = [w for w in raw.split() if w]
            if 2 <= len(tokens) <= 4 and all(_NAME_TOKEN_RE.match(w) for w in tokens):
                if not any(t.lower().strip('.,') in _JOB_WORDS for t in tokens):
                    candidate = " ".join(t.capitalize() for t in tokens)

        if (_EMAIL_RE.search(raw) or _URL_RE.search(raw) or _PHONE_CANDIDATE_RE.search(raw) or
                any(k in low for k in ("email", "phone", "tel", "contact", "linkedin", "github", "www"))):
            if candidate:
                final = _sanitize_candidate_name(candidate)
                return _titlecase_name(final)

    if candidate is None:
        top_text = "\n".join(ln for ln in lines[:6] if not _is_spaced_letter_line(ln))
        if top_text and 'nlp' in globals():
            try:
                doc = nlp(top_text)
                alpha_tokens = [t.text for t in doc if getattr(t, "is_alpha", False)]
                if len(alpha_tokens) >= 2:
                    candidate = f"{alpha_tokens[0].capitalize()} {alpha_tokens[1].capitalize()}"
            except Exception:
                candidate = None
        elif top_text:
            toks = re.findall(r"[A-Za-z]+", top_text)
            if len(toks) >= 2:
                candidate = f"{toks[0].capitalize()} {toks[1].capitalize()}"

    if candidate:
        final = _sanitize_candidate_name(candidate)
        return _titlecase_name(final)
    return None

# Contact number & email extraction

def _default_looks_like_year_sequence(raw: str, digits_only: str) -> bool:
    if len(digits_only) in (4, 8):
        return bool(re.match(r'^(19|20)\d{2}$', digits_only)) or bool(re.match(r'^(19|20)\d{2}(19|20)\d{2}$', digits_only))
    return False

def extract_contact_number_from_resume(text: str) -> Optional[str]:
    """Return first plausible contact number found, applying several filters."""
    if not text:
        return None
    norm_text = _normalize_unicode_phone_chars(text)
    looks_like_year_seq = globals().get('_looks_like_year_sequence', None) or _default_looks_like_year_sequence
    for m in _PHONE_CANDIDATE_RE.finditer(norm_text):
        raw = m.group().strip()
        digits_only = _DIGITS_RE.sub('', raw)
        if not (7 <= len(digits_only) <= 15):
            continue
        if looks_like_year_seq(raw, digits_only):
            continue
        if _PDF_ARTIFACT_RE.search(raw):
            continue
        if raw.count('(') > 1 or raw.count(')') > 1:
            continue
        if raw.startswith('(') and not re.search(r'\d{2,}', raw):
            continue
        return raw
    return None


def extract_email_from_resume(text: str) -> Optional[str]:
    match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    return match.group() if match else None

# ===== Education extraction ===== 
def extract_education_from_resume(text: str, min_entry_len: int = 6) -> List[Dict[str, str]]:
    """Extract degree/course entries with durations from the education section."""
    if not text or not text.strip():
        return []
    txt = text.replace('\r\n', '\n').replace('\r', '\n')
    edu_header_pattern = re.compile(
        r'(?im)^(?:education|educational background|educational history|education & training|education and training|academic (?:background|qualifications)|academic qualifications|academics|qualifications)\b.*$',
        flags=re.M
    )
    next_header_pattern = re.compile(
        r'(?im)^(?:experience|work experience|professional experience|employment history|projects|skills|technical skills|certifications|certificates|achievements|summary|profile|objective|contact|personal details|publications|awards|activities|interests|references)\b.*$',
        flags=re.M
    )
    start_match = edu_header_pattern.search(txt)
    if not start_match:
        return []
    start_pos = start_match.end()
    next_match = next_header_pattern.search(txt, pos=start_pos)
    end_pos = next_match.start() if next_match else len(txt)
    section_text = txt[start_pos:end_pos].strip()
    if not section_text:
        return []

    raw_lines = [ln.strip(' \t,-•*') for ln in section_text.splitlines() if ln.strip()]
    lines = []
    for ln in raw_lines:
        if re.match(r'(?i)^(relevant coursework|coursework|relevant projects|activities|interests|references|hobbies|major:|minor:)\b', ln):
            continue
        if re.match(r'(?i)^\s*(gpa[:\s]|major[:\s]|minor[:\s])', ln):
            continue
        lines.append(ln)
    if not lines:
        return []

    degree_kw = (
        r'(?:Bachelor|B\.?Sc|BSc|BA|B\.?A\.?|B\.?F\.?A\.?|BS|Master|M\.?Sc|MSc|MS|MBA|Ph\.?D|PhD|'
        r'Diploma|Certificate|Associate)'
    )
    degree_regex = re.compile(degree_kw, flags=re.I)

    degree_indices = []
    for idx, ln in enumerate(lines):
        if degree_regex.search(ln):
            degree_indices.append(idx)
    if not degree_indices:
        for idx, ln in enumerate(lines):
            if re.search(r'(?i)\b(university|college|institute|school|academy|college)\b', ln):
                degree_indices.append(idx)

    all_durations = []
    for ln in lines:
        d = extract_duration_from_line(ln)
        if d:
            ds = d.replace('–', '-').replace('—', '-')
            ds = re.sub(r'\s*-\s*', ' - ', ds).strip()
            all_durations.append(ds)

    entries = []
    seen = set()
    num_degrees = len(degree_indices)

    def find_forward_duration(start_idx: int, window: int = 6) -> Optional[str]:
        for k in range(start_idx + 1, min(len(lines), start_idx + window + 1)):
            if is_duration_line(lines[k]):
                d = extract_duration_from_line(lines[k])
                if d:
                    ds = d.replace('–', '-').replace('—', '-')
                    return re.sub(r'\s*-\s*', ' - ', ds).strip()
        return None

    if num_degrees > 0 and len(all_durations) >= num_degrees:
        start_pos_for_map = len(all_durations) - num_degrees
        for i, deg_idx in enumerate(degree_indices):
            ln = lines[deg_idx]
            m = degree_regex.search(ln)
            if m:
                course_raw = ln[m.start():].strip()
            else:
                if deg_idx > 0:
                    prev_ln = lines[deg_idx - 1].strip()
                    if prev_ln and not is_duration_line(prev_ln) and not re.search(r'(?i)\b(university|college|institute|school|academy)\b', prev_ln):
                        course_raw = prev_ln
                    else:
                        course_raw = ln.strip()
                else:
                    course_raw = ln.strip()
            if '|' in course_raw:
                course_raw = course_raw.split('|', 1)[0].strip()
            course_raw = re.sub(r'(?i)\b(major:|minor:).*', '', course_raw).strip(' ,;:-')
            course = re.sub(r'\s{2,}', ' ', course_raw).strip(' ,;:-')
            duration = all_durations[start_pos_for_map + i]
            key = (course + duration).lower()
            if course and len(course) >= min_entry_len and key not in seen:
                seen.add(key)
                entries.append({"Course": course, "Duration": duration})
        return entries

    for deg_idx in degree_indices:
        ln = lines[deg_idx]
        m = degree_regex.search(ln)
        if m:
            course_raw = ln[m.start():].strip()
        else:
            if deg_idx > 0:
                prev_ln = lines[deg_idx - 1].strip()
                if prev_ln and not is_duration_line(prev_ln) and not re.search(r'(?i)\b(university|college|institute|school|academy)\b', prev_ln):
                    course_raw = prev_ln
                else:
                    course_raw = ln.strip()
            else:
                course_raw = ln.strip()

        if '|' in course_raw:
            course_raw = course_raw.split('|', 1)[0].strip()
        course_raw = re.sub(r'(?i)\b(major:|minor:).*', '', course_raw).strip(' ,;:-')
        course = re.sub(r'\s{2,}', ' ', course_raw).strip(' ,;:-')

        duration = None
        d_same = extract_duration_from_line(ln)
        if d_same:
            duration = d_same.replace('–', '-').replace('—', '-')
            duration = re.sub(r'\s*-\s*', ' - ', duration).strip()
        if not duration:
            duration = find_forward_duration(deg_idx, window=8)
        if not duration and all_durations:
            for cand in reversed(all_durations):
                ds = cand
                if ds and (course + ds).lower() not in seen:
                    duration = ds
                    break
        if not duration:
            duration = "N/A"
        key = (course + duration).lower()
        if course and len(course) >= min_entry_len and key not in seen:
            seen.add(key)
            entries.append({"Course": course, "Duration": duration})
    return entries

# ===== Skills extraction ===== 
def load_skills() -> List[str]:
    skills_path = os.path.join(os.path.dirname(__file__), "../data/datasets/skills.csv")
    skills_path = os.path.abspath(skills_path)
    try:
        df = pd.read_csv(skills_path, header=0)
    except Exception:
        return []
    vals = df.iloc[:, 0].dropna().astype(str).tolist()
    return [s.strip().lower() for s in vals if s.strip()]

def _score_section_for_skills(section_text: str, job_titles_lower: Set[str]) -> float:
    """Score a section to estimate whether it contains skills (higher = more likely)."""
    if not section_text or not section_text.strip():
        return -999
    txt_lower = section_text.lower()
    job_matches = sum(1 for jt in job_titles_lower if jt and len(jt) > 2 and re.search(rf"\b{re.escape(jt)}\b", txt_lower))
    delim_count = txt_lower.count('•') + txt_lower.count(',') + txt_lower.count('/') + txt_lower.count(';') + txt_lower.count('|')
    lines = [ln.strip() for ln in section_text.splitlines() if ln.strip()]
    short_lines = sum(1 for ln in lines if len(ln.split()) <= 6)
    short_line_ratio = short_lines / len(lines) if lines else 0
    employment_terms = ["supervisor", "assistant", "manager", "teacher", "intern", "company",
                        "employment", "experience", "work", "volunteer", "firm", "organization"]
    location_terms = ["street", "road", "avenue", "city", "county", "state", "university",
                      "school", "arkansas", "california", "texas", "london", "malaysia", "india"]
    employment_hits = sum(1 for t in employment_terms if t in txt_lower)
    location_hits = sum(1 for t in location_terms if t in txt_lower)
    penalty = (employment_hits + location_hits) * 2.0
    score = (delim_count * 0.8) + (short_line_ratio * 2.0) - (job_matches * 2.5) - penalty
    return score

def _chunks_from_section(section_text: str) -> List[str]:
    """Split a skills-like block into candidate skill strings."""
    if not section_text or not section_text.strip():
        return []
    normalized = re.sub(r'[•·\u2022]+', '\n', section_text)
    normalized = re.sub(r'[\/;|]+', '\n', normalized)
    normalized = re.sub(r',\s*', '\n', normalized)
    normalized = re.sub(r'\n{2,}', '\n', normalized)
    keep_chars = {'+', '#', '.'}
    strip_chars = ''.join(ch for ch in string.punctuation if ch not in keep_chars)
    chunks = []
    seen = set()
    for raw in [c.strip() for c in normalized.splitlines() if c.strip()]:
        low_raw = raw.lower().strip()
        if any(term in low_raw for term in _SECTION_BOUNDARY_TERMS):
            break
        if _DECORATIVE_RE.match(raw):
            continue
        if _PDF_ARTIFACT_RE.search(raw):
            break
        if _YEAR_RANGE_RE.search(raw) or (_YEAR_RE.search(raw) and len(re.findall(r'\d{4}', raw)) >= 1 and len(raw.split()) <= 3):
            if raw.strip().isdigit() or re.match(r'^\d{4}[-–—]\d{4}$', raw.strip()):
                continue
            break
        if re.search(r'\b(supervisor|assistant|manager|teacher|intern|company|employment|experience)\b', low_raw):
            break
        candidate = raw.strip(strip_chars + " \t")
        candidate = re.sub(r'^[\(\[\{]+', '', candidate)
        candidate = re.sub(r'[\)\]\}]+$', '', candidate)
        if not candidate:
            continue
        if _TOO_MANY_NONALNUM.match(candidate):
            continue
        if len(candidate.split()) > 10:
            break
        if any(term in candidate.lower() for term in _EDU_TERMS):
            break
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        chunks.append(candidate)
    return chunks

def split_resume_sections(text: str) -> Dict[str, str]:
    """Split resume text into named sections using common header keywords."""
    if not text or not text.strip():
        return {}
    txt = text.replace('\r', '\n')
    pattern = re.compile(
        r'\n\s*(skills|technical skills|key skills|core competencies|proficient skills|professional skills|computer skills|experience|work experience|employment history|education|projects|certifications|awards|languages|summary|profile|objective|competencies|achievements|interests|extra curriculars|educational history|other activities|activities)\s*[:\-]?\s*\n',
        flags=re.IGNORECASE
    )
    matches = list(pattern.finditer(txt))
    if not matches:
        return {}
    sections = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(txt)
        name = m.group(1).lower().strip()
        content = txt[start:end].strip()
        sections[name] = content
    return sections

def extract_skills_from_resume(text: str, known_skills: Optional[List[str]] = None) -> List[str]:
    """Return list of extracted skill strings using section heuristics and fallbacks."""
    if not text or not text.strip():
        return []
    sections = split_resume_sections(text)
    job_titles = JOB_TITLES
    extracted = []
    for explicit in [
        "skills", "technical skills", "key skills", "core competencies",
        "proficient skills", "professional skills", "computer skills"
    ]:
        if explicit in sections and sections[explicit].strip():
            extracted = _chunks_from_section(sections[explicit])
            if extracted:
                return extracted
    best_section = None
    best_score = -float("inf")
    for name, sec_text in sections.items():
        score = _score_section_for_skills(sec_text, job_titles)
        if score > best_score:
            best_score = score
            best_section = sec_text
    MIN_ACCEPT_SCORE = -1.0
    if best_section and best_score >= MIN_ACCEPT_SCORE:
        extracted = _chunks_from_section(best_section)
    if not extracted:
        m = re.search(r'(skills\s*(?:[:\-]?)\s*)([\s\S]{0,500})', text, flags=re.IGNORECASE)
        if m:
            extracted = _chunks_from_section(m.group(2))
    if not extracted and sections:
        for name, sec_text in sections.items():
            if sum(sec_text.count(c) for c in [',', '/', '•', '|', ';']) >= 4:
                fallback = _chunks_from_section(sec_text)
                if fallback:
                    extracted = fallback
                    break
    return extracted or []

# ===== Job titles loader & matching ===== 
def load_job_titles() -> Set[str]:
    job_titles_path = os.path.join(os.path.dirname(__file__), "../data/datasets/job_titles.csv")
    job_titles_path = os.path.abspath(job_titles_path)
    titles = set()
    try:
        with open(job_titles_path, "r", encoding="utf-8") as f:
            for line in f:
                for part in re.split(r'[,\n]+', line):
                    t = part.strip().lower()
                    if t:
                        titles.add(t)
    except Exception:
        return set()
    return titles

JOB_TITLES = load_job_titles()

def clean_token(tok: str) -> str:
    return tok.lower().strip(string.punctuation + " ")

def stem_token(tok: str) -> str:
    tok = clean_token(tok)
    tok = re.sub(r"'s$", "", tok)
    for suf in ('ership','ation','tion','ing','ed','er','ers','ist','ian','s'):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            tok = tok[: -len(suf)]
            break
    return tok

def tokens_from_text(s: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z&\.\-']+", s) if t.strip()]

def stem_overlap_score(jt: str, line: str) -> float:
    jt_tokens = tokens_from_text(jt)
    line_tokens = tokens_from_text(line)
    if not jt_tokens or not line_tokens:
        return 0.0
    jt_stems = [stem_token(t) for t in jt_tokens]
    line_stems = [stem_token(t) for t in line_tokens]
    inter = len(set(jt_stems) & set(line_stems))
    return inter / max(1, len(jt_stems))

def fuzzy_job_match_enhanced(line: str, job_titles: Set[str], threshold: int = 90):
    """Use fuzzy matching + stem overlap to find job title matches."""
    if not line or not line.strip():
        return None, 0.0
    line_norm = line.strip()
    best_match = None
    best_score = 0.0
    try:
        res = process.extractOne(line_norm, list(job_titles), scorer=fuzz.token_set_ratio)
        if res and res[1] > best_score:
            best_match, best_score = res[0], float(res[1])
    except Exception:
        pass
    try:
        res = process.extractOne(line_norm, list(job_titles), scorer=fuzz.partial_ratio)
        if res and float(res[1]) > best_score:
            best_match, best_score = res[0], float(res[1])
    except Exception:
        pass
    best_stem_match = None
    best_stem_score = 0.0
    for jt in job_titles:
        s = stem_overlap_score(jt, line_norm)
        if s > best_stem_score:
            best_stem_score = s
            best_stem_match = jt
    stem_score100 = best_stem_score * 100.0
    if stem_score100 > best_score:
        best_match, best_score = best_stem_match, stem_score100
    if best_match and best_score >= threshold:
        return best_match, best_score
    return None, 0.0

def extract_variant_title_from_line(line: str, base_job_lower: str) -> Optional[str]:
    """Extract a nearby variant of a base job title from a line."""
    if not base_job_lower:
        return None
    base_tokens = tokens_from_text(base_job_lower)
    base_stems = [stem_token(t) for t in base_tokens]
    line_tokens = tokens_from_text(line)
    if not line_tokens:
        return None
    best_win = None
    best_score = -1.0
    best_w = 0
    max_window = min(len(line_tokens), len(base_tokens) + 6)
    for w in range(1, max_window + 1):
        for start in range(0, len(line_tokens) - w + 1):
            window = line_tokens[start:start + w]
            window_stems = [stem_token(t) for t in window]
            inter = len(set(window_stems) & set(base_stems))
            score = inter / max(1, len(base_stems))
            lower_window = [t.lower() for t in window]
            if any(m in lower_window for m in TITLE_MODIFIERS):
                score += 0.15
            if score > best_score or (abs(score - best_score) < 1e-6 and w > best_w):
                best_score = score
                best_win = (start, w)
                best_w = w
    if best_win and best_score > 0:
        start, w = best_win
        seq = line_tokens[start:start + w]
        pattern = r'\\b' + r'\\s+'.join(re.escape(tok) for tok in seq) + r'\\b'
        m = re.search(pattern, line, flags=re.I)
        if m:
            candidate = " ".join(m.group(0).split())
            return candidate.strip(string.punctuation + " ")
        else:
            candidate = " ".join(seq)
            return candidate.strip(string.punctuation + " ")
    return None

# ===== Experience extraction ===== 
def is_section_header(line: str) -> bool:
    return bool(re.match(
        r"^(education|skills|projects|certifications|awards|languages|summary|profile|contact|references|achievements|professional profile|educational history)\b",
        line.strip().lower()
    ))

def looks_like_sentence(line: str) -> bool:
    """Heuristic to decide if a line looks like a sentence/description."""
    if not line or len(line.strip()) == 0:
        return False
    verbs = [
        "worked", "created", "built", "developed", "designed", "implemented",
        "given", "managed", "led", "produced", "revamped", "improved",
        "taught", "organized", "enhanced", "supervised", "executed", "analyzed",
        "collaborated", "increased", "boosted", "mentored", "used",
        "want", "interested", "fascinated", "trained", "project", "during",
        "capstone", "contribute", "learning", "looking", "seeking",
        "graduate", "fresh", "freshers", "profile", "responsible", "duties",
        "achieved", "delivered", "coordinated", "assisted", "supported"
    ]
    lower = line.lower()
    if any(v in lower for v in verbs):
        return True
    if re.search(r'[.!?]$', line):
        return True
    if len(line.split()) > 12:
        return True
    if re.match(r'^[•\-\*]\s*(worked|created|developed|managed|led|implemented|designed|built)', lower):
        return True
    if re.search(r'\b(i|we|my|me|our)\b', lower) and re.search(r'\b(want|join|trained|worked|create|contribute|seeking)\b', lower):
        return True
    return False

def is_standalone_job_title_line(line: str) -> bool:
    """Heuristic for lines that are likely job-title-like (short, non-sentence)."""
    if not line or not line.strip():
        return False
    if looks_like_sentence(line):
        return False
    words = line.split()
    if len(words) > 10:
        return False
    lower = line.lower()
    sentence_indicators = [
        'responsible for', 'duties include', 'including', 'such as',
        'worked on', 'developed', 'created', 'managed', 'led',
        'where i', 'where we', 'which', 'that involved'
    ]
    if any(ind in lower for ind in sentence_indicators):
        return False
    if re.search(r'[.!?]\s*$', line):
        return False
    if len(words) <= 8:
        return True
    return False

def extract_experience_from_resume(text: str, duration_window: int = 2) -> List[Dict[str, str]]:
    """Extract job title + duration pairs from experience/work sections."""
    text_norm = text.replace('\r', '\n')
    exp_header_re = re.compile(
        r'(?:^|\n)\s*(experience|work experience|employment history|professional experience|employment)\s*[:\-]?\s*(?:\n|$)',
        re.IGNORECASE
    )
    next_section_re = re.compile(
        r'(?:^|\n)\s*(education|skills|projects|certifications|awards|languages|summary|profile|contact|references|achievements)\s*[:\-]?\s*(?:\n|$)',
        re.IGNORECASE
    )
    m = exp_header_re.search(text_norm)
    if m:
        start = m.end()
        end_m = next_section_re.search(text_norm, start)
        end = end_m.start() if end_m else len(text_norm)
        section_text = text_norm[start:end].strip()
        section_lines = [ln.strip() for ln in section_text.splitlines() if ln.strip()]
    else:
        all_lines = [ln.strip() for ln in text_norm.splitlines() if ln.strip()]
        section_lines = all_lines[8:] if len(all_lines) > 30 else all_lines

    experiences = []
    processed_indices = set()

    for i, raw_line in enumerate(section_lines):
        if i in processed_indices:
            continue
        line = raw_line.strip("•- ").strip()
        if not line:
            continue
        title_candidate = line
        if '|' in line or ',' in line:
            sep = '|' if '|' in line else ','
            parts = re.split(r'\s*' + re.escape(sep) + r'\s*', line)
            left = parts[0].strip()
            right = ' '.join(parts[1:]).strip() if len(parts) > 1 else ''
            _, left_score = fuzzy_job_match_enhanced(left, JOB_TITLES, threshold=0)
            _, right_score = fuzzy_job_match_enhanced(right, JOB_TITLES, threshold=0)

            def side_looks_like_title(s: str, score: float) -> bool:
                if not s:
                    return False
                if score >= 75:
                    return True
                if is_standalone_job_title_line(s):
                    return True
                toks = [t.lower() for t in tokens_from_text(s)]
                if any(w in toks for w in _FALLBACK_JOB_WORDS):
                    return True
                return False

            left_is_title = side_looks_like_title(left, left_score)
            right_is_title = side_looks_like_title(right, right_score)
            if sep == '|':
                if right_is_title and (right_score > left_score or not left_is_title):
                    title_candidate = right
                else:
                    title_candidate = left
            else:
                if left_is_title and not right_is_title:
                    title_candidate = left
                elif right_is_title and not left_is_title:
                    title_candidate = right
                else:
                    title_candidate = left

        if not is_standalone_job_title_line(line):
            if '|' in line:
                left = re.split(r'\s*\|\s*', line)[0].strip()
                if left and not is_duration_line(left) and not looks_like_sentence(left) and len(left.split()) <= 10:
                    title_candidate = left
                else:
                    continue
            else:
                continue

        base_job_lower, base_score = fuzzy_job_match_enhanced(title_candidate, JOB_TITLES, threshold=85)
        matched_title = None
        if base_job_lower:
            matched_title = extract_variant_title_from_line(title_candidate, base_job_lower)
        if not matched_title:
            left = re.split(r'\s*\|\s*', title_candidate)[0].strip()
            left_clean = left.strip(string.punctuation + " ")
            tokens = tokens_from_text(left_clean)
            lower_tokens = [t.lower() for t in tokens]
            contains_job_word = any(w in lower_tokens for w in _FALLBACK_JOB_WORDS)
            looks_like_duration = is_duration_line(left_clean)
            looks_like_loc = bool(re.search(r'\b(city|state|street|avenue|road|ny|il|ca|pa|chicago|albany|boston)\b', left_clean.lower()))
            if (tokens and len(tokens) <= 10 and not looks_like_duration and not looks_like_sentence(left_clean)
                and (contains_job_word or not looks_like_loc)):
                matched_title = left_clean
        if not matched_title:
            continue
        t_tokens = tokens_from_text(matched_title)
        if len(t_tokens) < 1:
            continue

        duration = None
        duration_line_idx = None
        if is_duration_line(line):
            duration = extract_duration_from_line(line)
            duration_line_idx = i
        if not duration:
            search_range = range(max(0, i - duration_window), min(len(section_lines), i + duration_window + 1))
            for k in search_range:
                if k == i or k in processed_indices:
                    continue
                cand_line = section_lines[k].strip("•- ").strip()
                if not cand_line:
                    continue
                if is_section_header(cand_line):
                    continue
                if re.search(r'\b(university|college|school|institute|academy|b\.sc|mca|degree|bachelor|master|ph\.?d)\b', cand_line, re.I):
                    continue
                if is_duration_line(cand_line):
                    duration = extract_duration_from_line(cand_line)
                    if duration:
                        duration_line_idx = k
                        break
        if not duration:
            continue
        duration = duration.replace('–', '-').replace('—', '-').strip()
        duration = re.sub(r'\s*-\s*', ' - ', duration)
        experiences.append({
            "Job_Title": matched_title.strip(string.punctuation + " "),
            "Duration": duration.strip(string.punctuation + " ")
        })
        processed_indices.add(i)
        if duration_line_idx is not None:
            processed_indices.add(duration_line_idx)
    return experiences

# ===== Public parser ===== 
def parse_resume(resume_path: str, skills_list: Optional[List[str]] = None) -> Dict[str, object]:
    text = extract_text_from_pdf(resume_path)
    
    # Education
    education_list = extract_education_from_resume(text)
    
    # Skills
    if skills_list is None:
        skills_list = load_skills()
    skills = extract_skills_from_resume(text, skills_list)
    
    # Experience
    experiences = extract_experience_from_resume(text)
    
    profile = {
        "Name": extract_name(text) or "N/A",
        "Contact_Number": extract_contact_number_from_resume(text) or "N/A",
        "Email": extract_email_from_resume(text) or "N/A",
        "Education": education_list if education_list else [{"Course": "N/A", "Duration": "N/A"}],
        "Skills": skills,
        "Experience": experiences if experiences else ["N/A"],
        "Raw_Text": text[:3000]
    }
    
    return profile
