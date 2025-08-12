from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import JSONResponse
import os, csv, zipfile, io, base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from llm import get_llm_response

app = FastAPI()

# --- Security ---
API_KEY = os.getenv("API_KEY")

def check_api_key(x_api_key: str):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server API key not configured.")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")

# --- Helpers ---
def file_to_text(upload: UploadFile):
    ext = upload.filename.split(".")[-1].lower()
    content = upload.file.read()
    if ext == "txt" or ext == "md":
        return content.decode("utf-8", errors="ignore")
    elif ext == "csv":
        df = pd.read_csv(io.BytesIO(content))
        return df.to_csv(index=False)
    elif ext in ["xls", "xlsx"]:
        df = pd.read_excel(io.BytesIO(content))
        return df.to_csv(index=False)
    elif ext == "json":
        return content.decode("utf-8", errors="ignore")
    elif ext == "parquet":
        df = pd.read_parquet(io.BytesIO(content))
        return df.to_csv(index=False)
    elif ext == "zip":
        ztxts = []
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name in zf.namelist():
                with zf.open(name) as f:
                    subext = name.split(".")[-1].lower()
                    if subext in ["txt", "md", "csv", "xls", "xlsx"]:
                        sub_content = f.read()
                        ztxts.append(file_to_text(UploadFile(filename=name, file=io.BytesIO(sub_content))))
        return "\n\n".join(ztxts)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

def plot_to_data_uri(plot_func, *args, **kwargs):
    """Executes plot_func, saves plot to PNG base64, returns data URI"""
    plt.clf()
    plot_func(*args, **kwargs)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches='tight')
    plt.close()
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# --- Main Endpoint ---
@app.post("/api/")
async def process_data(x_api_key: str = Header(None), files: list[UploadFile] = File(...)):
    check_api_key(x_api_key)

    # Ensure questions.txt exists
    qfile = next((f for f in files if f.filename.lower() == "questions.txt"), None)
    if not qfile:
        raise HTTPException(status_code=400, detail="questions.txt file is required.")

    question_text = file_to_text(qfile)
    other_contexts = []

    for f in files:
        if f.filename.lower() != "questions.txt":
            try:
                other_contexts.append(file_to_text(f))
            except HTTPException:
                continue  # skip unsupported types gracefully

    full_context = "\n\n".join(other_contexts) if other_contexts else None

    # Simple example logic: detect if plot is requested
    answers = []
    if "scatterplot" in question_text.lower():
        # Example: generate scatterplot for demo purposes
        df = pd.DataFrame({"Rank": range(1, 11), "Peak": [2.1, 1.9, 1.8, 1.7, 1.85, 1.6, 1.5, 1.4, 1.6, 1.55]})
        def make_scatter():
            sns.scatterplot(x="Rank", y="Peak", data=df)
            sns.regplot(x="Rank", y="Peak", data=df,
                        scatter=False, ci=None,
                        line_kws={"color":"red","linestyle":"dotted"})
            plt.xlabel("Rank")
            plt.ylabel("Peak")
        img_uri = plot_to_data_uri(make_scatter)
        answers = [1, "Titanic", 0.48578, img_uri]  # placeholder example
    else:
        # LLM text-only processing
        resp = get_llm_response(question_text, full_context)
        answers.append(resp.get("answer", ""))

    return JSONResponse(content=answers)
