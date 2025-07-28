# Adobe India Hackathon 2025 - Challenge 1A: PDF Outline Extractor

## Approach

This solution extracts structured outlines (Title, H1, H2, H3 headings) from PDF documents using a multi-signal analysis approach implemented purely in Python, without machine learning models.

### Key Techniques

1.  **Text Extraction:** Utilizes PyMuPDF (fitz) to parse the PDF and extract text spans along with their font properties (size, bold flag), bounding boxes, and page numbers.
2.  **Text Preprocessing:** Merges consecutive spans with identical styling, cleans text (normalization, hyphenation), and applies an upfront heuristic filter (`_is_obviously_not_a_heading`) to remove obvious non-headings (emails, dates, single lowercase words, etc.).
3.  **Multi-Signal Scoring:** For remaining candidates, a scoring function combines six signals:
    *   **S1 (Font Characteristics):** Based on a combined score of font size and boldness, grouped into tiers.
    *   **S2 (Bold/All Caps):** Presence of bold font or all uppercase text.
    *   **S3 (Indentation):** Alignment with the leftmost text on the page.
    *   **S4 (Numbering):** Matches against a comprehensive set of regex patterns for various numbering/list styles (1., 1.1, a), (A), Appendix A, etc.).
    *   **S5 (Repeatability):** Penalizes text that appears too frequently, indicating list items or headers/footers.
    *   **S6 (Vertical Gap):** Measures the whitespace gap below a line, as headings often have more space after them.
4.  **Dynamic Thresholding:** Uses the 90th percentile of candidate scores to select only the most likely headings, improving precision.
5.  **Level Assignment:** Assigns H1/H2/H3 levels primarily based on detected numbering patterns. Font tier and indentation are used contextually for refinement (e.g., preventing consecutive H1s of similar size).
6.  **Post-Processing:**
    *   Applies TOC-aware logic to correct levels of Table of Contents sections.
    *   Filters out running headers/footers/watermarks based on position and frequency.
    *   Filters candidates against a list of common false positives (Abstract, References, etc.).
    *   Performs a final pass with `_is_obviously_not_a_heading` and a length filter to remove any last remnants of noise.
    *   Deduplicates consecutive identical headings (often due to page breaks).
    *   Adjusts levels if no H1 is found (promotes the first H2).
7.  **Title Extraction:** Prioritizes PDF metadata (after sanitizing common placeholders/filenames). Falls back to the largest, boldest text on the first page, then to the first detected H1, and finally to the PDF filename.

### Libraries Used

*   `PyMuPDF (fitz)`: For robust and fast PDF parsing and text extraction.
*   `numpy`: For efficient numerical calculations (percentiles, means, standard deviations).
*   `unidecode`: (Bonus) For normalizing Unicode characters, aiding multilingual text handling.

### Performance & Constraints

*   **No ML Models:** Relies solely on rule-based heuristics and text analysis, keeping the solution lightweight and fast.
*   **Multithreading:** Uses `concurrent.futures.ThreadPoolExecutor` for orchestrating file processing and `concurrent.futures.ProcessPoolExecutor` within each file task to enforce a hard 9-second timeout per PDF, complying with the 10-second limit for 50 pages (assuming < 6 files processed).
*   **Offline:** No external API calls or internet dependencies.
*   **AMD64 Compatibility:** Base Docker image is explicitly set for `linux/amd64`.
*   **Resource Awareness:** Designed to be lightweight. Processing logic is CPU-bound but parallelized.

## How to Build and Run

1.  **Build the Docker Image:**
    ```bash
    docker build --platform linux/amd64 -t pdf-outline-extractor .
    ```

2.  **Run the Solution:**
    Ensure you have an `input` directory containing your PDF files and an `output` directory for the results.
    ```bash
    docker run --rm \
      -v $(pwd)/input:/app/input:ro \
      -v $(pwd)/output:/app/output \
      --network none \
      pdf-outline-extractor
    ```
    This command mounts the local `input` directory to `/app/input` (read-only) and the local `output` directory to `/app/output` within the container. It also disables network access as required.
