import os

files = ["src/logging/schema.py", "src/logging/db.py"]
for path in files:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            content = f.read()
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Stripped BOM from {path}")
