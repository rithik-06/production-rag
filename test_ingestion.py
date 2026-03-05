from ingestion.loader import load_all

docs = load_all(folder_path="data")

print(f"Total pages loaded: {len(docs)}")
print(f"First 200 chars: {docs[0].page_content[:200]}")