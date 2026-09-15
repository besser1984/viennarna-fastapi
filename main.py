import os
import tempfile
import RNA
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

app = FastAPI()

class RNARequest(BaseModel):
    sequence: str = Field(..., example="GAGUCCUGUGAGAACUGUUGAGUAGAGUGUGAGCUCCCUG")

@app.post("/analyze")
def analyze_rna(req: RNARequest):
    clean_seq = req.sequence.upper().replace('T', 'U')
    mfe_structure, mfe_energy = RNA.fold(clean_seq)
    
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        RNA.file_draw_struct_svg(clean_seq, mfe_structure, tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {
        "sequence": clean_seq,
        "mfe_structure": mfe_structure,
        "mfe_energy_kcal": round(mfe_energy, 2),
        "svg_xml": svg_content
    }