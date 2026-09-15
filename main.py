from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import primer3

app = FastAPI()

class HomodimerRequest(BaseModel):
    sequence: str = Field(..., example="TGACTATAAGTCCTGGCGATTTGATGCA")
    temperature_c: float = Field(60.0)
    na_mM: float = Field(50.0)
    mg_mM: float = Field(3.5)
    dntp_mM: float = Field(0.6)

@app.post("/analyze")
def analyze_homodimer(req: HomodimerRequest):
    try:
        seq = req.sequence.upper().replace('U', 'T')
        
        # Calculate 3'-end stability explicitly passing all buffer parameters
        end_res = primer3.bindings.calc_end_stability(
            seq1=seq,
            seq2=seq,
            mv_conc=float(req.na_mM),
            dv_conc=float(req.mg_mM),
            dntp_conc=float(req.dntp_mM),
            dna_conc=50.0,  # 50 nM primer concentration (Benchling default)
            temp_c=float(req.temperature_c)
        )

        # Calculate Global Homodimer stability
        homo_res = primer3.bindings.calc_homodimer(
            seq=seq,
            mv_conc=float(req.na_mM),
            dv_conc=float(req.mg_mM),
            dntp_conc=float(req.dntp_mM),
            dna_conc=50.0,
            temp_c=float(req.temperature_c)
        )

        # Convert dG from cal/mol to kcal/mol
        end_dg = round(end_res.dg / 1000.0, 2)
        global_dg = round(homo_res.dg / 1000.0, 2)

        # Benchling displays 3'-end dimer energy as the primary Min ΔG value
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
            "dntp_mM": req.dntp_mM,
            "redesign_recommended": warning,
            "alignment": {
                "structure_found": getattr(homo_res, 'structure_found', True),
                "is_3prime_end": True
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
