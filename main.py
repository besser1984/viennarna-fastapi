import os
import tempfile
import RNA
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

class RNARequest(BaseModel):
    sequence: str = Field(..., example="GAGUCCUGUGAGAACUGUUGAGUAGAGUGUGAGCUCCCUG")

@app.get("/")
def health_check():
    return {"status": "ok", "service": "ViennaRNA API"}

@app.post("/analyze")
def analyze_rna(req: RNARequest):
    try:
        clean_seq = req.sequence.upper().replace('T', 'U').strip()
        
        # 1. Fold compound & compute MFE structure
        fc = RNA.fold_compound(clean_seq)
        mfe_structure, mfe_energy = fc.mfe()

        # 2. Render SVG file safely to disk using the correct SWIG function
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".svg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # RNA.file_svg writes the structure plot to the specified file path
            RNA.file_svg(clean_seq, mfe_structure, tmp_path)
            
            with open(tmp_path, "r", encoding="utf-8") as f:
                svg_content = f.read()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return {
            "sequence": clean_seq,
            "mfe_structure": mfe_structure,
            "mfe_energy_kcal": round(float(mfe_energy), 2),
            "svg_xml": svg_content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ViennaRNA Error: {str(e)}")
