import os
import uuid

UPLOAD_DIR = "uploads"
VECTOR_DIR = "vectorstores"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

def generate_id(prefix="pdf"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def save_upload_fileobj(fileobj, filename):
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(fileobj.read())
    return path

def vectorstore_dir_for(pdf_id: str):
    return os.path.join(VECTOR_DIR, pdf_id)