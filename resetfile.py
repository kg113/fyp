import os

def fast_clean_and_initialize():
    out_dir = "./predic/out"
    
    # 1. Fast recursive traversal and deletion using os.walk
    if os.path.exists(out_dir):
        for root, dirs, files in os.walk(out_dir, topdown=False):
            for file in files:
                if file.lower().endswith(".jpg"):
                    try:
                        os.remove(os.path.join(root, file))
                    except OSError as e:
                        print(f"Error removing {file}: {e}")

    # 2. Ensure directory exists and write CSV
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "csv","large_outputs.csv")
    
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("img_save_path,caption,targetcaption\n")

if __name__ == "__main__":
    fast_clean_and_initialize()