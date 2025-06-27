import json

def load_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                data.append(json.loads(line))
    return data

file1 = 'data_with_similarity_score.json'     # Your first file (likely JSON array)
file2 = 'datasetwithdup_corrected.json'       # Your second file (likely JSON array)

output_file = 'merged_all_posts.json'

def load_file(file):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"{file} is not a list, trying JSONL loading")
            data = load_jsonl(file)
    except json.JSONDecodeError:
        print(f"{file} is not valid JSON array, trying JSONL loading")
        data = load_jsonl(file)
    return data

data1 = load_file(file1)
data2 = load_file(file2)

print(f"Loaded {len(data1)} posts from {file1}")
print(f"Loaded {len(data2)} posts from {file2}")

merged = data1 + data2

print(f"Total posts after merge: {len(merged)}")

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged, f, indent=2)

print(f"Saved merged posts to {output_file}")
