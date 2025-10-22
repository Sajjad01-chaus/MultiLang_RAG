import os, time, json, requests
from tqdm import tqdm

BASE_URL = "http://127.0.0.1:8000"
CONFIG_FILE = "evaluation_config.json"
OUTPUT_FILE = "performance_report.json"

DELAY_BETWEEN_QUERIES = 30
MAX_RETRIES = 3
RETRY_WAIT = 60


def ingest_pdf(path, language, doc_id):
    print(f"\n📘 Ingesting {os.path.basename(path)} ({language})...")
    files = {"file": open(path, "rb")}
    data = {"language": language}
    r = requests.post(f"{BASE_URL}/ingest", files=files, data=data)
    if r.status_code == 200:
        print("✅ Ingested successfully")
        return r.json()
    else:
        print(f"❌ Ingestion failed: {r.text}")
        return None


def query_rag(query_text):
    payload = {"query": query_text}
    start = time.time()

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(f"{BASE_URL}/query", json=payload)
            latency = round((time.time() - start) * 1000, 2)

            if r.status_code == 429:
                print(f"⚠️ Rate limited (attempt {attempt+1}/{MAX_RETRIES}) → waiting {RETRY_WAIT}s...")
                time.sleep(RETRY_WAIT)
                continue

            if r.status_code == 200:
                response = r.json()
                response["latency_ms"] = latency
                return response
            else:
                print(f"❌ Query failed ({r.status_code}): {r.text}")
                break

        except Exception as e:
            print(f"💥 Query error: {e}")
            time.sleep(5)

    return None


def evaluate(config):
    results = []
    print("\n🚀 Starting evaluation...\n")

    # Step 1: Ingest PDFs
    for pdf in config["pdfs"]:
        ingest_pdf(pdf["path"], pdf["language"], pdf["doc_id"])

    # Step 2: Query loop
    for q in tqdm(config["queries"], desc="Evaluating queries"):
        query = q["query"]
        expected_docs = q.get("expected_docs", [])

        print(f"\n🔍 Query: {query}")
        response = query_rag(query)
        if not response:
            print("⚠️ Skipping (no response)")
            continue

        provenance = response.get("provenance", [])
        retrieved_docs = list(set([p.get("doc_id", "") for p in provenance]))
        latency = response.get("latency_ms", 0)
        answer = response.get("answer", "")

        def fuzzy_match(a, b):
            if not a or not b:
                return False
            return a.lower() in b.lower() or b.lower() in a.lower()

        retrieved_docs_base = {r.split("::")[0] for r in retrieved_docs}  # normalize to doc-level
        expected_set = set(expected_docs)

        correct_docs = retrieved_docs_base & expected_set
        precision = len(correct_docs) / max(len(retrieved_docs_base), 1)
        recall = len(correct_docs) / max(len(expected_set), 1)



        results.append({
            "query": query,
            "expected_docs": expected_docs,
            "retrieved_docs": retrieved_docs,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "latency_ms": latency,
            "answer_length": len(answer),
            "answer_preview": answer[:150] + "..." if len(answer) > 150 else answer
        })

        print(f"⏳ Waiting {DELAY_BETWEEN_QUERIES}s before next query...")
        time.sleep(DELAY_BETWEEN_QUERIES)

    if not results:
        print("❌ No results collected.")
        return {"message": "No successful queries."}

    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)

    summary = {
        "total_queries": len(results),
        "avg_latency_ms": round(avg_latency, 2),
        "avg_precision": round(avg_precision, 3),
        "avg_recall": round(avg_recall, 3),
        "timestamp": time.ctime(),
        "results": results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n✅ Evaluation complete!")
    print(f"📁 Saved results to {OUTPUT_FILE}")
    return summary


if __name__ == "__main__":
    if not os.path.exists(CONFIG_FILE):
        print("❌ Config file not found.")
        exit(1)

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    summary = evaluate(config)

    print("\n📊 Summary:")
    print(json.dumps({
        "Avg Latency (ms)": summary.get("avg_latency_ms", 0),
        "Avg Precision": summary.get("avg_precision", 0),
        "Avg Recall": summary.get("avg_recall", 0)
    }, indent=2))
