#!/usr/bin/env python3
"""
PDF outline extraction module.
Implements the multi-signal heading classification algorithm.
"""
import fitz  # PyMuPDF
import re
import unicodedata
# --- Language-Aware Fixes ---
try:
    from unidecode import unidecode
    HAS_UNIDECODE = True
except ImportError:
    HAS_UNIDECODE = False
    def unidecode(text):
        return text # Fallback if not installed
# ----------------------------
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)

# --- Precision Improvement: Less Aggressive Common False Positives ---
# List kept minimal for post-processing filter. Main filtering is upfront.
COMMON_FALSE_POSITIVES = {
    "table of contents", "list of figures", "list of tables",
    "abstract", "acknowledgements", "acknowledgments",
    "references", "bibliography", "index", "glossary",
    "appendix", "annex", "foreword", "preface", "contents"
}
COMMON_FALSE_POSITIVES_PATTERNS = [re.compile(re.escape(fp), re.IGNORECASE) for fp in COMMON_FALSE_POSITIVES]

def _is_potential_false_positive(text: str) -> bool:
    """Checks if the heading text is a potential common false positive (for post-filtering)."""
    text_lower = text.strip().lower()
    return any(pattern.fullmatch(text_lower) for pattern in COMMON_FALSE_POSITIVES_PATTERNS)

# --- Multilingual Handling: CJK Font Heuristic ---
def _is_cjk_font(font_name: str) -> bool:
    """Simple heuristic to detect CJK fonts by name."""
    if not font_name:
        return False
    cjk_indicators = [
        'cid', 'cjk', 'chinese', 'japanese', 'korean', 'simsun', 'simhei', 'fangsong', 'kaiti',
        'ms gothic', 'ms mincho', 'mingliu', 'pmingliu', 'pingfang', 'hiragino', 'noto sans cjk',
        'noto serif cjk', 'source han sans', 'source han serif', 'arphic', 'cwTeX', 'hanazono'
    ]
    font_name_lower = font_name.lower()
    return any(indicator in font_name_lower for indicator in cjk_indicators)

# --- Enhanced Font Weight & Size Ranking ---
def _calculate_font_characteristic_score(span: Dict) -> float:
    """Calculates a combined score for font characteristics (size + flags)."""
    size = span.get('font_size', 0)
    flags = span.get('font_flags', 0)
    is_bold = bool(flags & (1 << 4)) # Bold flag
    weight = 1.2 if is_bold else 1.0
    return size * weight

# --- New: Upfront Heuristic Filter for Obvious Non-Headings ---
def _is_obviously_not_a_heading(text: str) -> bool:
    """
    Applies simple heuristics to quickly discard spans that are almost certainly not headings.
    """
    stripped_text = text.strip()
    len_text = len(stripped_text)

    # 1. Filter out spans shorter than 4 characters
    if len_text < 4:
        return True

    # 2. Filter out text with sentence-ending punctuation, unless it's a clear heading pattern.
    if stripped_text.endswith(('.', '!', '?')):
        # Allow if it looks like an abbreviation/heading (short, caps, dots/special chars)
        # Or contains common heading words.
        if not (re.match(r'^[A-Z0-9\s\.:()/\[\]-]+$', stripped_text) and len_text < 20):
            # Check if it's a common heading ending that might have a period
            common_heading_roots = [
                "appendix", "annex", "figure", "table", "chapter", "section", "part",
                "article", "clause", "schedule", "form", "declaration", "summary",
                "introduction", "conclusion", "preface", "foreword", "acknowledgement",
                "abstract", "name", "title", "designation", "date", "service", "amount",
                "total", "balance", "due", "paid", "order", "number", "id", "reference"
            ]
            if not any(phrase in stripped_text.lower() for phrase in common_heading_roots):
                 return True

    # 3. Filter out spans containing email addresses or likely URLs
    if "@" in stripped_text or ("://" in stripped_text and ("http" in stripped_text or "www" in stripped_text)):
        return True

    # 4. Filter out spans that are entirely lowercase (strong body text indicator)
    #    (This removes all-lowercase acronyms if used as headings, but improves precision significantly)
    if stripped_text.islower() and not stripped_text.isupper(): # isupper check avoids empty strings
        return True

    # 5. Filter out spans that are just repeated characters (e.g., "-----", ".....")
    if len_text > 0 and stripped_text == stripped_text[0] * len_text:
        return True

    # 6. Filter out obvious numeric values or simple dates
    if re.fullmatch(r'^[\$\€\£\¥]?\d{1,3}(,\d{3})*(\.\d+)?%?M?$', stripped_text): # Currency/Numbers: $75M, 100, 3.14%, 1000
        return True
    # Simple date pattern check (be careful not to filter "Chapter 1. Date of Birth")
    if re.search(r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b', stripped_text) or \
       re.search(r'\b\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}\b', stripped_text) or \
       re.search(r'\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b', stripped_text, re.IGNORECASE):
        if not any(phrase in stripped_text.lower() for phrase in ["chapter", "section", "part", "form", "claim", "birth", "service", "order", "date"]):
            return True

    # 7. Filter out common non-heading phrases or fragments
    obvious_non_headings = {
        "page", "of", "to:", "from:", "re:", "subject:", "cc:", "bcc:",
        "i agree", "i understand", "signed:", "signature:", "date:",
        "amount", "total", "subtotal", "balance", "due", "paid",
        "please", "note", "see", "refer to", "as per", "pursuant to",
        "the", "and", "or", "but", "in", "on", "at", "by", "for", "with", "to", "from"
    }
    text_lower = stripped_text.lower()
    # Check for exact match or if it starts/ends with a non-heading phrase (simple heuristic)
    if any(phrase == text_lower or
           text_lower.startswith(phrase + " ") or
           text_lower.endswith(" " + phrase) or
           (" " + phrase + " ") in text_lower for phrase in obvious_non_headings):
        return True

    # 8. Filter out likely form *values* that are single words
    common_value_words = {"date", "name", "title", "amount", "number", "code", "id", "reference", "signature"}
    if text_lower in common_value_words:
        # Allow if it's part of a numbered pattern (likely a label)
        # This check is implicit in the numbering pattern matching, but as a last resort:
        return True # If it's just "Date" and not "1. Date of Birth", filter it.

    return False
# ---------------------------------------------------------------


class PDFOutlineExtractor:
    """Main class for extracting outlines from PDF documents."""

    def __init__(self):
        self.initial_heading_threshold = 0.5
        self.min_text_length = 4 # Enforced upfront now
        self.max_heading_repeats = 3
        self.header_footer_freq_threshold = 0.4
        self.header_footer_y_top_threshold = 1.0 * 72
        self.header_footer_y_bottom_threshold = 1.0 * 72
        self.watermark_size_factor = 2.0
        self.title_size_factor = 1.5
        self.toc_indicators = {"table of contents", "contents", "目次"}
        self.toc_page_window = 3
        self.common_toc_entries = {
             "introduction", "preface", "foreword", "acknowledgements", "abstract",
             "table of contents", "list of figures", "list of tables", "chapter", "section",
             "conclusion", "appendix", "references", "bibliography", "index", "glossary"
        }

        # --- Expanded Number Patterns for Forms and Subheadings ---
        self.number_patterns = [
            r'^\d+(\.\d+){0,3}[\.\)\s]+', # Allow deeper nesting: 1.1.1.1
            r'^[IVXLCDM]+\.?\s+',
            r'^第?\s*[０-９0-9]+[章節部]\s*',
            r'^第[一二三四五六七八九十百千]+[章节部分款]\s*',
            r'^[A-Z]\.(\d+\.?)*\s+',
            r'^Appendix\s+[A-Z]+:?', # Specific pattern for Appendix headings
            r'^Annex\s+[A-Z\d]+:?',   # Specific pattern for Annex headings
            # --- NEW FOR FORMS and SUB-POINTS ---
            r'^\d+\.\s+[A-Z]',        # Form fields: "1. Name", "2. Designation" (must start with capital after number)
            r'^[a-z]\)\s+',           # Lowercase letter lists: "a) Option", "b) Block..."
            r'^\([a-z]\)\s+',         # Parenthesized lowercase: "(a) Option", "(b) Block..."
            r'^\([A-Z]\)\s+',         # Parenthesized uppercase (less common for subpoints but possible)
            # -------------------------------------
        ]
        self.compiled_number_patterns = [re.compile(p) for p in self.number_patterns]

    def extract_raw_spans(self, doc: fitz.Document) -> List[Dict]:
        """Extract raw text spans from all pages."""
        all_spans = []
        page_lines_info = defaultdict(list)

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_dict = page.get_text("dict")

            for block in page_dict.get("blocks", []):
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    line_bbox = line['bbox']
                    page_lines_info[page_num].append(line_bbox)

                    for span in line["spans"]:
                        span_info = {
                            'text': span['text'],
                            'font_size': span['size'],
                            'font_flags': span['flags'],
                            'font_name': span['font'],
                            'bbox': span['bbox'],
                            'page': page_num + 1,
                            'line_bbox': line_bbox,
                            'font_char_score': _calculate_font_characteristic_score(span)
                        }
                        all_spans.append(span_info)

        all_spans_with_gap = self._calculate_vertical_gaps(all_spans, page_lines_info, doc)
        return all_spans_with_gap

    def _calculate_vertical_gaps(self, spans: List[Dict], page_lines_info: Dict, doc: fitz.Document) -> List[Dict]:
        """Calculate vertical gap below each line and add it to spans."""
        if not spans or not page_lines_info:
            return spans

        spans.sort(key=lambda s: (s['page'], s['line_bbox'][1], s['bbox'][0]))
        page_line_spans = defaultdict(lambda: defaultdict(list))
        for span in spans:
            page_line_spans[span['page']][span['line_bbox']].append(span)

        spans_with_gap = []
        all_gaps = []
        for page_num in sorted(page_line_spans.keys()):
            lines_on_page = sorted(page_line_spans[page_num].keys(), key=lambda lb: lb[1])
            page = doc[page_num - 1]
            page_height = page.rect.height

            for i, line_bbox in enumerate(lines_on_page):
                line_spans = page_line_spans[page_num][line_bbox]
                line_bottom = line_bbox[3]

                next_line_top = None
                if i + 1 < len(lines_on_page):
                    next_line_bbox = lines_on_page[i + 1]
                    next_line_top = next_line_bbox[1]
                else:
                    next_line_top = page_height

                if next_line_top is not None:
                    gap = next_line_top - line_bottom
                    if 0 <= gap < page_height / 2: # Ignore huge gaps (page breaks)
                        all_gaps.append(gap)
                else:
                    gap = 0

                for span in line_spans:
                    span['vertical_gap'] = gap
                    spans_with_gap.append(span)

        # Normalize gap score based on document's gap distribution
        if all_gaps:
            median_gap = np.median(all_gaps)
            mean_gap = np.mean(all_gaps)
            normalization_base = max(median_gap, mean_gap, 5.0)
        else:
            normalization_base = 20.0 # Fallback

        for span in spans_with_gap:
            gap = span.get('vertical_gap', 0)
            normalized_gap_score = min(gap / normalization_base, 1.0)
            span['s6_gap_score'] = normalized_gap_score

        return spans_with_gap

    def clean_and_merge_spans(self, spans: List[Dict]) -> List[Dict]:
        """Clean and merge consecutive spans with identical styling."""
        if not spans:
            return []

        cleaned_spans = []
        page_lines = defaultdict(lambda: defaultdict(list))
        for span in spans:
            page_lines[span['page']][span['line_bbox']].append(span)

        for page_num in sorted(page_lines.keys()):
            for line_bbox in page_lines[page_num]:
                line_spans = page_lines[page_num][line_bbox]
                line_spans.sort(key=lambda x: x['bbox'][0])

                merged_spans = self._merge_line_spans(line_spans)
                cleaned_spans.extend(merged_spans)

        filtered_spans = []
        for span in cleaned_spans:
            clean_text = self._clean_text(span['text'])
            # --- Apply the upfront heuristic filter ---
            if len(clean_text) >= self.min_text_length and not _is_obviously_not_a_heading(clean_text):
                span['text'] = clean_text
                filtered_spans.append(span)

        return filtered_spans

    def _merge_line_spans(self, line_spans: List[Dict]) -> List[Dict]:
        """Merge consecutive spans on the same line with identical styling."""
        if not line_spans:
            return []

        merged = []
        current = line_spans[0].copy()

        for span in line_spans[1:]:
            # --- Use Enhanced Font Score for comparison ---
            if (abs(current['font_char_score'] - span['font_char_score']) < 1.0 and
                current['font_flags'] == span['font_flags'] and
                current['font_name'] == span['font_name']):
            # ------------------------------------------------
                current['text'] += span['text']
                current['bbox'] = (
                    min(current['bbox'][0], span['bbox'][0]),
                    min(current['bbox'][1], span['bbox'][1]),
                    max(current['bbox'][2], span['bbox'][2]),
                    max(current['bbox'][3], span['bbox'][3])
                )
            else:
                merged.append(current)
                current = span.copy()

        merged.append(current)
        return merged

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if HAS_UNIDECODE:
            text = unidecode(text)
        text = unicodedata.normalize('NFKC', text)
        text = text.strip()
        text = re.sub(r'-\s*\n\s*', '-', text) # Collapse hyphenated words
        return text

    def analyze_font_statistics(self, spans: List[Dict]) -> Dict:
        """Analyze font statistics across the document."""
        if not spans:
            return {}

        # --- Use Enhanced Font Score ---
        font_char_scores = [span['font_char_score'] for span in spans]
        # -------------------------------
        size_counter = Counter(font_char_scores)
        unique_scores = sorted(set(font_char_scores), reverse=True)
        score_rank = {score: rank for rank, score in enumerate(unique_scores)}
        modal_score = size_counter.most_common(1)[0][0] if size_counter else 0

        # --- Improved Tiering using Enhanced Score ---
        TOLERANCE = 1.5
        sorted_unique_scores = sorted(unique_scores, reverse=True)
        size_tiers = []
        current_tier = []
        current_base_score = sorted_unique_scores[0] if sorted_unique_scores else 0

        for score in sorted_unique_scores:
            if abs(score - current_base_score) <= TOLERANCE:
                current_tier.append(score)
            else:
                size_tiers.append(current_tier)
                current_tier = [score]
                current_base_score = score
        if current_tier:
            size_tiers.append(current_tier)

        # Map each score to its tier index (0 = largest tier)
        score_to_tier_index = {}
        for i, tier in enumerate(size_tiers):
            for score in tier:
                score_to_tier_index[score] = i

        stats = {
            'score_rank': score_rank,
            'max_score_rank': len(unique_scores) - 1,
            'modal_score': modal_score,
            'mean_score': np.mean(font_char_scores) if font_char_scores else 0,
            'std_score': np.std(font_char_scores) if len(font_char_scores) > 1 else 0,
            'unique_scores': unique_scores,
            'size_tiers': size_tiers,
            'score_to_tier_index': score_to_tier_index,
            'max_tier_index': len(size_tiers) - 1 if size_tiers else 0
        }

        return stats

    def calculate_heading_scores(self, spans: List[Dict], font_stats: Dict) -> List[Dict]:
        """Calculate heading scores for all spans."""
        if not spans or not font_stats:
             return spans

        # Count text occurrences for repeatability score
        text_counts = Counter(span['text'] for span in spans)
        x_positions = [span['bbox'][0] for span in spans]
        median_x = np.median(x_positions) if x_positions else 0
        min_x = min(x_positions) if x_positions else 0

        scored_spans = []
        candidate_scores = []
        modal_score = font_stats.get('modal_score', 0)

        for span in spans:
            text = span['text']

            # --- Apply upfront heuristic filter AGAIN (in case cleaning changed text) ---
            if _is_obviously_not_a_heading(text):
                # Assign a very low score and skip detailed calculation
                span_with_score = span.copy()
                span_with_score.update({
                    'heading_score': 0.0,
                    'is_heading_candidate': False,
                    's1_font': 0, 's2_bold_caps': 0, 's3_indent': 0,
                    's4_number': 0, 's5_repeat': 0, 's6_gap': 0
                })
                scored_spans.append(span_with_score)
                continue
            # -------------------------------------------------------------------------

            # --- Ensure all variables used later are initialized ---
            s1 = s2 = s3 = s4 = s5 = s6 = 0
            s1_weight = s2_weight = s3_weight = s4_weight = s6_weight = 0
            s4_boost = 0
            # -------------------------------------------------------

            # S1: Font Score (based on tier using enhanced score)
            tier_index = font_stats['score_to_tier_index'].get(span['font_char_score'], font_stats['max_tier_index'])
            max_tier_index = font_stats['max_tier_index']
            s1 = 1 - (tier_index / max_tier_index) if max_tier_index > 0 else 1
            s1_weight = 0.20 # Font Tier weight

            # S2: Bold/Cap Score
            is_bold = bool(span['font_flags'] & (1 << 4))
            is_all_caps = text.isupper() and bool(re.search(r'[A-Z]', text))
            s2_weight = 0.30 # Strong weight for Bold/Caps
            s2 = 1 if (is_bold or is_all_caps) else 0

            # S3: Indent Score (left-aligned relative to min x)
            s3 = 1 if abs(span['bbox'][0] - min_x) < 2.0 else 0 # Consider aligned if within 2 points
            s3_weight = 0.05 # Lower weight for indent

            # S4: Number Pattern Score
            s4 = 0 # <--- Explicit initialization before the loop
            for pattern in self.compiled_number_patterns:
                if pattern.match(text):
                    s4 = 1
                    break
            s4_weight = 0.35 # Highest weight for Numbering

            # --- Boost for Number + Capital Start ---
            # This calculation must happen AFTER s4 is determined
            s4_boost = 0
            if s4 == 1 and re.match(r'^\S+\s+[A-Z]', text): # Number/symbol followed by space and capital
                s4_boost = 0.05 # Small additional boost
            # ----------------------------------------

            # S5: Repeatability Score (used as penalty multiplier later)
            s5 = 1 if text_counts[text] <= self.max_heading_repeats else 0
            # s5_weight conceptually 0 in the sum, applied as a multiplier

            # S6: Vertical Gap Score
            s6 = span.get('s6_gap_score', 0)
            s6_weight = 0.10 # Solid weight for gap

            # --- Adjusted Weights ---
            # Font (20%), Bold/Caps (30%), Indent (5%), Number (35%), Gap (10%)
            # Repeat (S5) is not added but used to penalize the final score.
            # Include the small boost for numbered items starting with a capital.
            heading_score = (s1_weight * s1 +
                             s2_weight * s2 +
                             s3_weight * s3 +
                             s4_weight * s4 + s4_boost + # Include boost
                             s6_weight * s6) # s5 not added here

            # --- Apply repeatability as a penalty ---
            # If repeated too much, lower the score significantly
            if text_counts[text] > self.max_heading_repeats:
                 heading_score *= 0.5 # Halve the score if it's likely a list item

            span_with_score = span.copy()
            span_with_score.update({
                'heading_score': heading_score,
                's1_font': s1,
                's2_bold_caps': s2,
                's2_weight_used': s2_weight,
                's3_indent': s3,
                's4_number': s4, # Make sure s4 is stored
                's4_boost': s4_boost, # For debugging
                's4_weight_used': s4_weight,
                's5_repeat': s5, # Keep for context/debugging
                's6_gap': s6,
                's6_weight_used': s6_weight,
                'is_cjk_font': _is_cjk_font(span.get('font_name', '')),
            })

            scored_spans.append(span_with_score)
            if heading_score > 0:
                 candidate_scores.append(heading_score)

        # --- Dynamic Threshold: Use 90th Percentile for High Selectivity ---
        if candidate_scores:
            dynamic_threshold = max(self.initial_heading_threshold, np.percentile(candidate_scores, 90))
        else:
            dynamic_threshold = self.initial_heading_threshold

        for span in scored_spans:
             span['is_heading_candidate'] = span['heading_score'] >= dynamic_threshold

        logger.debug(f"Dynamic heading threshold set to: {dynamic_threshold:.3f}")
        return scored_spans

    def assign_heading_levels(self, heading_candidates: List[Dict], font_stats: Dict, all_spans: List[Dict]) -> List[Dict]:
        """Assign hierarchical levels to heading candidates."""
        if not heading_candidates or not font_stats:
            return []

        # Sort candidates primarily by page, then by vertical position (top to bottom)
        candidates = sorted(heading_candidates, key=lambda x: (x['page'], x['bbox'][1]))

        leveled_headings = []
        prev_heading = None

        for candidate in candidates:
            # --- Fix: Use score_to_tier_index ---
            tier_index = font_stats['score_to_tier_index'].get(candidate['font_char_score'], font_stats['max_tier_index'])
            # ------------------------------------
            has_numbering = candidate['s4_number'] == 1
            indent_x = candidate['bbox'][0]
            candidate_font_score = candidate['font_char_score']

            # --- Primary Logic: Combine Tier, Indentation, and Numbering ---
            # 1. Strong Numbering Override (Highest Priority)
            if has_numbering:
                level_from_number = self._infer_level_from_numbering(candidate['text'])
                if level_from_number:
                    level = level_from_number
                else:
                    # If pattern matched but level unclear, fall back to tier but favor high tiers
                    if tier_index == 0:
                        level = "H1"
                    elif tier_index <= 1:
                        level = "H2"
                    else:
                        level = "H3"
            # 2. Font Tier as Base (if no strong numbering)
            else:
                if tier_index == 0:
                    level = "H1"
                elif tier_index == 1:
                    level = "H2"
                else:
                    level = "H3" # Default to H3 for smaller tiers

            # --- Contextual Refinement ---
            if prev_heading:
                prev_level = prev_heading['level']
                prev_indent = prev_heading['bbox'][0]
                prev_font_score = prev_heading['font_char_score']

                # Rule: An H2/H3 should generally be indented compared to its preceding H1/H2
                if level in ["H2", "H3"] and prev_level in ["H1", "H2"]:
                    if indent_x < prev_indent - 2: # If significantly less indented, might be a new H1
                         if tier_index <= 1: # If font is large enough
                              level = "H1"

                # Rule: An H1 should generally NOT be indented compared to its preceding H1/H2
                # And prevent consecutive H1s/H2s of similar size if not indented
                if level == "H1" and prev_level == "H1":
                     # Two consecutive H1s, second one might be H2 if indented
                     if indent_x > prev_indent + 2 and tier_index <= 1:
                          level = "H2"
                     # If very similar font and not indented, downgrade the current one
                     elif (abs(candidate_font_score - prev_font_score) < 1.0 and
                           abs(indent_x - prev_indent) < 5.0):
                          level = "H2" # Assume second is not a main heading

                if level == "H2" and prev_level == "H2":
                     # Similar logic for H2s
                     if (abs(candidate_font_score - prev_font_score) < 1.0 and
                         abs(indent_x - prev_indent) < 5.0):
                          level = "H3"

                # Prevent illogical jumps (H1 after H3 is not ideal)
                if level == "H1" and prev_level in ["H2", "H3"]:
                     # If it's much larger or numbered, okay. Otherwise, promote it less aggressively.
                     if not (has_numbering or (tier_index == 0)):
                          if tier_index <= 1:
                               level = "H2"
                          # else keep H3

            candidate['level'] = level
            leveled_headings.append(candidate)
            prev_heading = candidate

        return leveled_headings

    def _infer_level_from_numbering(self, text: str) -> Optional[str]:
        """Infers heading level from numbering pattern."""
        # Check patterns in order of specificity
        for pattern in self.compiled_number_patterns:
             if pattern.match(text):
                  matched_text = pattern.match(text).group(0)
                  # Count dots for standard numbering (e.g., 1.1.1)
                  dot_count = matched_text.count('.')
                  if "appendix" in matched_text.lower() or "annex" in matched_text.lower():
                       return "H1" # Treat appendix/annex titles as H1
                  elif dot_count == 0: # e.g., "1. ", "A. "
                       return "H1"
                  elif dot_count == 1: # e.g., "1.1 ", "A.1 "
                       return "H2"
                  elif dot_count >= 2: # e.g., "1.1.1 ", "A.1.a "
                       return "H3"
                  # Handle letter lists (a), b), (a), (b))
                  elif pattern.pattern in [r'^[a-z]\)\s+', r'^\([a-z]\)\s+']:
                       return "H3" # Assume these are sub-points
                  elif pattern.pattern == r'^\([A-Z]\)\s+':
                       return "H3" # Assume these are sub-points

        return None # If no clear level can be inferred from pattern

    def extract_title(self, doc: fitz.Document, spans: List[Dict], font_stats: Dict) -> str:
        """Extract document title using prioritized rules, ignoring bad metadata."""
        # Rule 1: PDF metadata (Sanitize first)
        metadata = doc.metadata
        if metadata and metadata.get('title'):
            raw_title = metadata['title'].strip()
            # --- Sanitize Metadata ---
            # Discard if it looks like a filename or MS Word placeholder
            if (raw_title and
                not self._is_generic_title(raw_title) and
                not re.search(r'\.(doc|docx|pdf|txt)\s*$', raw_title, re.IGNORECASE) and
                not re.search(r'Microsoft\s+Word\s*-', raw_title, re.IGNORECASE) and
                not re.search(r'-\s*\d+\s*$', raw_title) # Looks like "Document-1"
                ):
                # Title seems potentially valid based on format, proceed with normalization
                logger.debug(f"Using sanitized metadata title: '{raw_title}'")
                return self._normalize_title(raw_title)
            else:
                logger.debug(f"Discarded suspect metadata title: '{raw_title}'")
        # else: No metadata title or it was discarded, proceed to fallback rules

        # Rule 2: Prominent text on first page (top portion, large, bold)
        first_page_spans = [s for s in spans if s['page'] == 1]
        if first_page_spans and font_stats:
            page = doc[0]
            page_height = page.rect.height
            upper_threshold = page_height * 0.35 # Slightly larger top portion

            # Filter for top portion
            top_spans = [s for s in first_page_spans if s['bbox'][1] <= upper_threshold]

            if top_spans:
                modal_score = font_stats.get('modal_score', 0)
                # Candidate: Large (significantly bigger than body) AND bold/caps
                # --- Use Enhanced Font Score ---
                title_candidates = [
                    s for s in top_spans
                    if s['font_char_score'] > modal_score * self.title_size_factor and
                       (bool(s['font_flags'] & (1 << 4)) or (s['text'].isupper() and bool(re.search(r'[A-Z]', s['text']))))
                ]
                # -----------------------------

                if title_candidates:
                    # Pick the one that is largest and highest (most prominent)
                    # --- Use Enhanced Font Score ---
                    best_candidate = max(title_candidates, key=lambda x: (x['font_char_score'], -x['bbox'][1]))
                    # -----------------------------
                    logger.debug(f"Title from largest bold text on page 1: '{best_candidate['text']}'")
                    return self._normalize_title(best_candidate['text'])

                # Fallback within top portion: just the largest (if significantly large)
                # --- Use Enhanced Font Score ---
                largest_span = max(top_spans, key=lambda x: x['font_char_score'])
                if largest_span['font_char_score'] > modal_score * self.title_size_factor:
                # -----------------------------
                     logger.debug(f"Title from largest text on page 1: '{largest_span['text']}'")
                     return self._normalize_title(largest_span['text'])

            # --- Title extraction fallback: Longest top span (if substantial) ---
            if top_spans:
                 # Find the longest one that's reasonably long
                 substantial_top_spans = [s for s in top_spans if len(s['text']) > 10]
                 if substantial_top_spans:
                      longest_span = max(substantial_top_spans, key=lambda x: len(x['text']))
                      logger.debug(f"Title from longest top span (fallback): '{longest_span['text']}'")
                      return self._normalize_title(longest_span['text'])
            # --------------------------------------------------------------------

        # Rule 3: First strong H1 detected by main logic (if available post-processing)
        # This requires passing `scored_spans` and checking levels after post-processing.
        # For simplicity here, and because H1s are filtered post-process, rely on filename fallback.

        # Ultimate Fallback: use filename stem
        import os
        filename_title = os.path.splitext(os.path.basename(doc.name if hasattr(doc, 'name') else 'untitled.pdf'))[0]
        filename_title = re.sub(r'[_\-]+', ' ', filename_title).strip().title()
        logger.debug(f"Title fallback to filename: '{filename_title}'")
        return filename_title

    def _is_generic_title(self, title: str) -> bool:
        """Check if title is too generic."""
        generic_patterns = [
            r'^untitled',
            r'^document',
            r'^pdf',
            r'^page\s*\d*$',
            r'^\s*$'
        ]
        title_lower = title.lower().strip()
        return any(re.match(pattern, title_lower) for pattern in generic_patterns)

    def _normalize_title(self, title: str) -> str:
        """Normalize title text."""
        # Strip trailing numbers/whitespace that might be page numbers or versions
        title = re.sub(r'[\s\-_]*\d+\s*$', '', title)
        title = re.sub(r'\s+', ' ', title).strip()
        return title

    def post_process_outline(self, headings: List[Dict], all_spans: List[Dict], font_stats: Dict, doc: fitz.Document) -> List[Dict]:
        """Apply post-processing sanity checks."""
        if not headings or not doc:
            return []

        total_pages = len(doc)
        # Initial sort
        headings.sort(key=lambda x: (x['page'], x['bbox'][1]))

        # --- TOC-Aware Level Correction ---
        headings = self._correct_toc_levels(headings, doc)
        # ----------------------------------

        # --- Final Filter for Obvious Non-Headings ---
        # Add one more check after level assignment
        initially_filtered_count = len(headings)
        headings = [h for h in headings if not _is_obviously_not_a_heading(h['text'])]
        if len(headings) < initially_filtered_count:
            logger.debug(f"Post-process filter removed {initially_filtered_count - len(headings)} obvious non-headings (pre-removal).")
        # ---------------------------------------------

        # 1. Remove running headers/footers/watermarks (now with dynamic page size)
        # This also handles the refined false-positive logic.
        filtered_headings = self._remove_running_elements(headings, all_spans, font_stats, total_pages, doc)

        # --- Filter using COMMON_FALSE_POSITIVES list ---
        fp_filtered_count = len(filtered_headings)
        filtered_headings = [h for h in filtered_headings if not _is_potential_false_positive(h['text'])]
        if len(filtered_headings) < fp_filtered_count:
            logger.debug(f"Post-process filter removed {fp_filtered_count - len(filtered_headings)} common false positives.")
        # -----------------------------------------------

        # --- Final Filter for Obvious Non-Headings (AGAIN) ---
        # One last pass after running element removal
        final_filtered_count = len(filtered_headings)
        filtered_headings = [h for h in filtered_headings if not _is_obviously_not_a_heading(h['text'])]
        if len(filtered_headings) < final_filtered_count:
            logger.debug(f"Post-process filter removed {final_filtered_count - len(filtered_headings)} obvious non-headings (post-removal).")
        # ----------------------------------------------------

        # --- Aggressive Final Length Filter ---
        # Last-ditch effort to remove extremely short items that slipped through
        len_filtered_count = len(filtered_headings)
        filtered_headings = [h for h in filtered_headings if len(h['text']) >= 5]
        if len(filtered_headings) < len_filtered_count:
             logger.debug(f"Post-process filter removed {len_filtered_count - len(filtered_headings)} headings for being too short (<5 chars).")
        # -------------------------------------

        # 2. Deduplicate consecutive identical headings (page breaks)
        deduplicated_headings = self._deduplicate_consecutive(filtered_headings)

        # 3. Sort again after filtering
        deduplicated_headings.sort(key=lambda x: (x['page'], x['bbox'][1]))

        # 4. Minor level adjustments
        final_headings = self._adjust_levels_if_needed(deduplicated_headings)

        return final_headings

    # --- TOC-Aware Level Correction (Refined) ---
    def _correct_toc_levels(self, headings: List[Dict], doc: fitz.Document) -> List[Dict]:
        """Detects Table of Contents sections and adjusts heading levels."""
        if not headings:
            return headings

        toc_title_headings = [
            h for h in headings
            if any(indicator in h['text'].lower() for indicator in self.toc_indicators)
        ]

        if not toc_title_headings:
            return headings

        corrected_headings = headings.copy()
        processed_pages = set()

        for toc_title in toc_title_headings:
            toc_page_num = toc_title['page']
            if toc_page_num in processed_pages:
                continue

            start_check_page = max(1, toc_page_num)
            end_check_page = min(len(doc), toc_page_num + self.toc_page_window)

            toc_entries = []
            for h in corrected_headings:
                if (start_check_page <= h['page'] <= end_check_page and
                    h['page'] >= toc_page_num and
                    h != toc_title and
                    h.get('level') in ['H2', 'H3'] # Must be originally H2/H3
                    ):
                    is_numbered = h.get('s4_number', 0) == 1
                    is_common_toc_text = h['text'].strip().lower() in self.common_toc_entries
                    is_below_toc_title = True
                    if h['page'] == toc_page_num and h['bbox'][1] <= toc_title['bbox'][1]:
                        is_below_toc_title = False

                    if is_below_toc_title and (is_numbered or is_common_toc_text):
                        toc_entries.append(h)

            # Correct levels: Set TOC title to H1, entries to H2
            for i, h in enumerate(corrected_headings):
                if h == toc_title:
                    corrected_headings[i]['level'] = 'H1'
                    break

            for entry in toc_entries:
                for i, h in enumerate(corrected_headings):
                    if h == entry:
                        corrected_headings[i]['level'] = 'H2'
                        break

            processed_pages.add(toc_page_num)
            # Assume one main TOC section per document for simplicity
            break

        return corrected_headings
    # ----------------------------------

    # --- Refined False-Positive and Running Element Removal ---
    def _remove_running_elements(self, headings: List[Dict], all_spans: List[Dict], font_stats:Dict, total_pages: int, doc: fitz.Document) -> List[Dict]:
        """Remove headers, footers, and potential watermarks. Refined false-positive logic."""
        if not headings or total_pages == 0 or not doc:
             return headings

        text_page_counts = defaultdict(set)
        text_first_last_pages = defaultdict(lambda: {'first': float('inf'), 'last': -1})

        # Count on how many pages each text appears and track first/last page
        for heading in headings:
            text = heading['text']
            page = heading['page']
            text_page_counts[text].add(page)
            text_first_last_pages[text]['first'] = min(text_first_last_pages[text]['first'], page)
            text_first_last_pages[text]['last'] = max(text_first_last_pages[text]['last'], page)

        filtered = []
        # modal_size = font_stats.get('modal_size', 0)
        modal_score = font_stats.get('modal_score', 0)
        if not all_spans:
             # max_size_in_doc = max((s['font_size'] for s in headings), default=0)
             max_score_in_doc = max((s['font_char_score'] for s in headings), default=0)
        else:
             # max_size_in_doc = max((s['font_size'] for s in all_spans), default=0)
             max_score_in_doc = max((s['font_char_score'] for s in all_spans), default=0)

        for heading in headings:
            text = heading['text']
            page_count = len(text_page_counts[text])
            frequency = page_count / total_pages if total_pages > 0 else 0
            first_page_occurrence = text_first_last_pages[text]['first']
            last_page_occurrence = text_first_last_pages[text]['last']

            # --- Dynamic Page Dimensions ---
            page_obj = doc[heading['page'] - 1] # 0-based index
            page_height = page_obj.rect.height
            page_width = page_obj.rect.width
            # -----------------------------

            # --- Check for Running Elements ---
            is_header = heading['bbox'][1] < self.header_footer_y_top_threshold
            # is_footer = heading['bbox'][3] > (792 - self.header_footer_y_bottom_threshold) # Old hardcoded
            is_footer = heading['bbox'][3] > (page_height - self.header_footer_y_bottom_threshold) # Dynamic

            # Check if it's a potential watermark (very large, central, spans pages)
            # --- Use Enhanced Font Score and Dynamic Page Width ---
            # is_potential_watermark = (
            #     heading['font_size'] > modal_size * self.watermark_size_factor and
            #     heading['font_size'] > max_size_in_doc * 0.8 and
            #     abs(heading['bbox'][0] - (612/2 - (heading['bbox'][2] - heading['bbox'][0])/2)) < 100 and
            is_potential_watermark = (
                heading['font_char_score'] > modal_score * self.watermark_size_factor and
                heading['font_char_score'] > max_score_in_doc * 0.8 and
                abs(heading['bbox'][0] - (page_width/2 - (heading['bbox'][2] - heading['bbox'][0])/2)) < (page_width * 0.2) and # Roughly centered X (20% of page width)
            # ------------------------------------------------------
                frequency > 0.5 and # Appears on more than half the pages
                first_page_occurrence == 1 and # Starts on first page
                last_page_occurrence == total_pages # Ends on last page
            )

            is_running_element = (frequency >= self.header_footer_freq_threshold and (is_header or is_footer)) or is_potential_watermark
            # --------------------------------

            # --- Refined False-Positive Logic (Context-based) ---
            # The main upfront filter `_is_obviously_not_a_heading` should catch most.
            # This check is for things that might score high but are still common FPs or running elements.
            # is_potential_fp = _is_potential_false_positive(text) # Moved to separate post-filter
            is_on_page_1 = heading['page'] == 1
            # Keep if it's on page 1 (titles, main headings are here), or if it's not a running element.
            # The upfront filter should have removed most bad stuff, so this is a safety net.
            # keep_because_of_context = is_on_page_1 or (is_potential_fp and not is_running_element)
            keep_because_of_context = is_on_page_1 # Simplified: keep page 1, remove running elements.

            # Remove if it's a frequent running element
            if is_running_element: # Removed `or (is_potential_fp and not keep_because_of_context)` logic
                logger.debug(f"Filtered out running element: '{text}' (Freq: {frequency:.2f}, Page: {heading['page']})")
                continue

            filtered.append(heading)

        return filtered
    # ---------------------------------------------------------

    def _deduplicate_consecutive(self, headings: List[Dict]) -> List[Dict]:
        """Merge consecutive headings with identical text."""
        if len(headings) <= 1:
            return headings

        deduped = [headings[0]]
        for i in range(1, len(headings)):
            prev = deduped[-1]
            curr = headings[i]
            # Check if text is identical and they are on consecutive pages or very close vertically
            if (prev['text'] == curr['text'] and
                (curr['page'] == prev['page'] + 1 or
                 (curr['page'] == prev['page'] and abs(curr['bbox'][1] - prev['bbox'][3]) < 5)) # Small vertical gap on same page
                ):
                logger.debug(f"Merged consecutive headings: '{curr['text']}'")
                continue # Skip adding curr
            else:
                deduped.append(curr)
        return deduped

    def _adjust_levels_if_needed(self, headings: List[Dict]) -> List[Dict]:
         """Apply minor adjustments, like promoting a single H2 to H1."""
         if not headings:
              return headings
         # Count levels
         level_counts = Counter(h['level'] for h in headings)
         h1_count = level_counts.get('H1', 0)
         h2_count = level_counts.get('H2', 0)

         # Error recovery: If there's only one H1 and many H2s, the first H2 might actually be the main title
         if h1_count == 0 and h2_count >= 1:
              # No H1s found, promote the first H2 to H1
              for heading in headings:
                   if heading['level'] == 'H2':
                        logger.debug(f"Promoting first H2 '{heading['text']}' to H1 due to no H1s found.")
                        heading['level'] = 'H1'
                        break # Only promote the first one

         return headings

# --- Core Extraction Logic ---
def _extract_outline_core(pdf_path: str) -> Dict:
    """The actual core extraction logic."""
    try:
        # Ensure logging is on for debugging in subprocess
        # logging.basicConfig(level=logging.DEBUG)
        logger.debug(f"Starting core extraction for: {pdf_path}")
        # Open PDF document
        doc = fitz.open(pdf_path)

        # Initialize extractor
        extractor = PDFOutlineExtractor()

        # Step 1: Extract raw spans (now includes vertical gap calculation)
        raw_spans = extractor.extract_raw_spans(doc)
        logger.debug(f"Extracted {len(raw_spans)} raw spans (with gap scores).")

        # Step 2: Clean and merge spans
        cleaned_spans = extractor.clean_and_merge_spans(raw_spans)
        logger.debug(f"Cleaned and merged to {len(cleaned_spans)} spans.")

        # Step 3: Analyze font statistics (using enhanced score)
        font_stats = extractor.analyze_font_statistics(cleaned_spans)
        logger.debug(f"Analyzed font stats: {len(font_stats.get('unique_scores', []))} unique font scores.")

        # Step 4: Calculate heading scores (includes S6 gap score, adjusted weights)
        scored_spans = extractor.calculate_heading_scores(cleaned_spans, font_stats)
        heading_candidates = [s for s in scored_spans if s.get('is_heading_candidate', False)]
        logger.debug(f"Identified {len(heading_candidates)} heading candidates (after 90th percentile threshold).")

        # Step 5: Assign heading levels (pass all spans for context)
        leveled_headings_unfiltered = extractor.assign_heading_levels(heading_candidates, font_stats, cleaned_spans)
        logger.debug(f"Assigned levels to headings (before filtering).")

        # --- Precision Improvement: Filtering is now mostly upfront and in post-process ---
        leveled_headings = leveled_headings_unfiltered # Filtering moved
        logger.debug(f"Filtering logic applied in pre-process and post-process. Headings count: {len(leveled_headings)}.")
        # ------------------------------------------------------------

        # Step 6: Extract title (needs to happen after scoring/leveling for full context)
        # Pass scored spans so title extractor can look for H1s if needed.
        title = extractor.extract_title(doc, scored_spans, font_stats)
        # Fallback is now handled inside extract_title
        logger.debug(f"Extracted or determined title: '{title}'")

        # Step 7: Post-process outline (pass doc for dynamic page sizes, refined filtering, TOC correction)
        final_headings = extractor.post_process_outline(leveled_headings, cleaned_spans, font_stats, doc)
        logger.debug(f"Post-processed outline, final count: {len(final_headings)}.")

        # Step 8: Build output format
        outline = []
        for heading in final_headings:
            outline.append({
                'level': heading['level'],
                'text': heading['text'],
                'page': heading['page']
            })

        doc.close()
        logger.debug(f"Core extraction completed successfully for: {pdf_path}")
        return {
            'title': title,
            'outline': outline
        }

    except Exception as e:
        logger.error(f"Error in core processing {pdf_path}: {e}", exc_info=True)
        return {
            'title': '',
            'outline': [],
            'error': str(e)
        }

# --- Main Extraction Function (Handles Timeout Logic via Caller) ---
def extract_outline(pdf_path: str, timeout_seconds: int = 9) -> Dict:
    """
    Main function to extract outline from a PDF file.
    The timeout is handled by the caller (e.g., ProcessPoolExecutor).
    This function just calls the core logic.
    """
    # Configure logging level here if needed for standalone runs or debugging in subprocess
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info(f"Starting extraction for: {pdf_path}")
    # The actual work is done by _extract_outline_core
    # Timeout will be enforced by the executor calling this function
    result = _extract_outline_core(pdf_path)
    logger.info(f"Extraction finished for: {pdf_path}")
    return result

if __name__ == '__main__':
    # Test with a sample PDF
    import sys
    import json
    # Configure logging for standalone run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if len(sys.argv) > 1:
        result = extract_outline(sys.argv[1])
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("Usage: python extractor.py <pdf_file>")
