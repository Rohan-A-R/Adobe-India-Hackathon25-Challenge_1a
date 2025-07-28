#!/usr/bin/env python3
"""
Main runner script for PDF outline extraction.
Processes all PDFs in /app/input and outputs JSON files to /app/output.
"""

import json
import concurrent.futures
from pathlib import Path
from extractor import extract_outline # Import the function

# Configure logging for the main process
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def worker(pdf_path: str, output_path: str, timeout: int = 9):
    """
    Worker function to process a single PDF file with timeout.
    Uses ProcessPoolExecutor internally to run the extraction in a separate process.
    """
    try:
        logger.info(f"Starting processing for: {pdf_path}")
        # Use ProcessPoolExecutor to run the extraction in a separate process with timeout
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            # Submit the task (extract_outline function)
            future = executor.submit(extract_outline, pdf_path, timeout)

            # Wait for the result with a timeout
            result = future.result(timeout=timeout)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Write the result to the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"Successfully processed: {pdf_path} -> {output_path}")

    except concurrent.futures.TimeoutError:
        error_msg = f"Processing timed out after {timeout} seconds."
        logger.warning(f"Timeout for {pdf_path}: {error_msg}")
        # Write timeout error to output file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"title": "", "outline": [], "error": error_msg}, f, ensure_ascii=False, indent=2)

    except Exception as e:
        error_msg = f"Error in worker process: {str(e)}"
        logger.error(f"Error processing {pdf_path}: {error_msg}", exc_info=True)
        # Write general error to output file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"title": "", "outline": [], "error": error_msg}, f, ensure_ascii=False, indent=2)


def main():
    """Main function to orchestrate PDF processing."""
    # Use the paths expected by the hackathon environment
    input_dir = Path('input')
    output_dir = Path('output')

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all PDF files in input directory (case-insensitive)
    pdf_files = list(input_dir.glob('*.pdf')) + list(input_dir.glob('*.PDF'))

    if not pdf_files:
        logger.info("No PDF files found in /app/input")
        return

    # Prepare list of (input_path, output_path) tuples
    pdf_tasks = []
    for pdf_file in pdf_files:
        output_file = output_dir / f"{pdf_file.stem}.json"
        pdf_tasks.append((str(pdf_file), str(output_file)))

    logger.info(f"Found {len(pdf_tasks)} PDF files to process")

    # Process PDFs using ThreadPoolExecutor for I/O-bound orchestration
    # The actual CPU-bound work (PyMuPDF) happens in separate processes via worker's ProcessPoolExecutor
    # Respect the 8 CPU limit from the hackathon rules
    num_worker_threads = min(8, len(pdf_tasks))
    logger.info(f"Using up to {num_worker_threads} worker threads for orchestration")

    timeout_seconds = 9 # As per requirements and previous discussion
    completed_tasks = 0

    # Use ThreadPoolExecutor for the main loop
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(worker, pdf_path, output_path, timeout_seconds): (pdf_path, output_path)
            for pdf_path, output_path in pdf_tasks
        }

        # Wait for completion and handle results
        for future in concurrent.futures.as_completed(future_to_task):
            pdf_path, output_path = future_to_task[future]
            try:
                # The worker function handles writing the output and errors
                # Getting the result just confirms the worker finished (or raised an exception caught by the worker)
                future.result() # This will raise if the worker itself had an unhandled exception
                completed_tasks += 1
            except Exception as e:
                logger.error(f"Unhandled exception in worker for {pdf_path}: {e}")

    logger.info(f"Processing complete! {completed_tasks}/{len(pdf_tasks)} tasks finished.")

if __name__ == '__main__':
    main()