"""
run_pipeline.py  —  One-Click Full Pipeline Execution
======================================================
Runs the entire data → feature → model → vector store pipeline in sequence.
Run this ONCE before launching the Streamlit dashboard.

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-rag     # Skip vector store build (faster)
    python run_pipeline.py --rebuild-rag  # Force rebuild the ChromaDB store
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def run_step(name: str, script: str, check: bool = True) -> bool:
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  ❌ FAILED in {elapsed:.1f}s — {name}")
        if check:
            sys.exit(1)
        return False
    print(f"  ✅ Done in {elapsed:.1f}s — {name}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-rag",    action="store_true", help="Skip ChromaDB build")
    parser.add_argument("--rebuild-rag", action="store_true", help="Force rebuild ChromaDB")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║   AI-Powered Product Quality Intelligence System             ║
║   Full Data → ML → RAG Pipeline                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    t_start = time.time()

    # 1. Generate synthetic data
    run_step("Generate Synthetic Data",   "src/data/generate_data.py")

    # 2. Preprocess and merge
    run_step("Preprocess & Merge Data",   "src/data/preprocess.py")

    # 3. Sentiment scoring
    run_step("VADER Sentiment Scoring",   "src/features/sentiment.py")

    # 4. Feature engineering
    run_step("Feature Engineering",       "src/features/engineering.py")

    # 5. Train defect predictor
    run_step("Train Defect Predictor",    "src/models/defect_predictor.py")

    # 6. Train return forecaster
    run_step("Train Return Forecaster",   "src/models/return_forecaster.py")

    # 7. Run anomaly detection
    run_step("Run Anomaly Detector",      "src/models/anomaly_detector.py")

    # 8. Build vector store (optional)
    if not args.skip_rag:
        import os
        chroma_dir = ROOT / "chroma_db"
        if chroma_dir.exists() and not args.rebuild_rag:
            print(f"\n  ℹ️  ChromaDB already exists at {chroma_dir}. Use --rebuild-rag to force rebuild.")
        else:
            run_step("Build ChromaDB Vector Store", "src/rag/vectorstore.py")
    else:
        print("\n  ⏭️  Skipping RAG vector store (--skip-rag)")

    total = time.time() - t_start
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ✅ PIPELINE COMPLETE in {total:.0f}s                              
║                                                              ║
║  Launch the dashboard:                                        ║
║    streamlit run app/streamlit_app.py                        ║
║                                                              ║
║  (Optional) Start FastAPI server in a separate terminal:     ║
║    uvicorn src.api.server:app --host 127.0.0.1 --port 8000  ║
╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
