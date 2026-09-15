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

        # Initialize Primer3 ThermoAnalysis engine matching Benchling's default parameters
        thermo = primer3.thermoanalysis.ThermoAnalysis(
            mv_conc=req.na_mM,
            dv_conc=req.mg_mM,
            dntp_conc=req.dntp_mM,
            dna_conc=50.0,
            temp_c=req.temperature_c
        )

        # 3'-End stability (Benchling's primary reported Min ΔG Homodimer value)
        end_res = thermo.calc_end_stability(seq, seq)
        
        # Global Homodimer stability across full sequence
        homo_res = thermo.calc_homodimer(seq)

        end_dg = round(end_res.dg / 1000.0, 2)
        global_dg = round(homo_res.dg / 1000.0, 2)

        # Primary value displayed in Benchling UI
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
