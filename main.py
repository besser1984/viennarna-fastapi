from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import primer3

app = FastAPI()

class HomodimerRequest(BaseModel):
    sequence: str = Field(..., example="TGACTATAAGTCCTGGCGATTTGATGCA")
    temperature_c: float = Field(60.0)
    na_mM: float = Field(50.0)
    mg_mM: float = Field(3.5)

@app.post("/analyze")
def analyze_homodimer(req: HomodimerRequest):
    try:
        seq = req.sequence.upper().replace('U', 'T')

        # Use primer3.bindings for the thermodynamic calculation functions
        end_res = primer3.bindings.calc_end_dimer(
            seq,
            mv_conc=req.na_mM,
            dv_conc=req.mg_mM,
            temp_c=req.temperature_c
        )

        homo_res = primer3.bindings.calc_homodimer(
            seq,
            mv_conc=req.na_mM,
            dv_conc=req.mg_mM,
            temp_c=req.temperature_c
        )

        # Convert dG from cal/mol to kcal/mol
        end_dg = round(end_res.dg / 1000.0, 2)
        global_dg = round(homo_res.dg / 1000.0, 2)

        # Benchling displays 3'-end dimer energy (-3.55 kcal/mol) as the primary Min ΔG value
        reported_dg = end_dg

        warning = global_dg < -5.0 or end_dg < -3.0

        return {
            "sequence": req.sequence,
            "min_delta_g": reported_dg,
            "global_delta_g": global_dg,
            "end_delta_g": end_dg,
            "temperature_c": req.temperature_c,
            "na_mM": req.na_mM,
            "mg_mM": req.mg_mM,
            "redesign_recommended": warning,
            "alignment": {
                "structure_found": homo_res.structure_found,
                "is_3prime_end": True
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
