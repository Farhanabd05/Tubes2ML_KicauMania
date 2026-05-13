from pathlib import Path

def print_tree(directory, prefix=""):
    path = Path(directory)
    # Define folders to skip
    ignored_folders = {".git", "__pycache__", "node_modules", "CNN_dataset", "RNN_dataset"}
    
    # Filter out ignored folders and files
    entries = [
        e for e in path.iterdir() 
        if not (e.is_dir() and e.name in ignored_folders)
    ]
    
    # Sort entries: folders first, then files alphabetically
    entries.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
    
    count = len(entries)
    for index, entry in enumerate(entries):
        is_last = index == count - 1
        connector = "└── " if is_last else "├── "
        
        print(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
        
        if entry.is_dir():
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(entry, new_prefix)

if __name__ == "__main__":
    target_path = r"."  
    print(f"Directory Tree for: {Path(target_path).resolve()}")
    print_tree(target_path)
