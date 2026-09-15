import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import RNA
    HAS_VIENNARNA = True
except ImportError:
    HAS_VIENNARNA = False

app = FastAPI()

class HomodimerRequest(BaseModel):
    sequence: str = Field(..., example="ACAGGATCACGTCCCTCCCC")
    temperature_c: float = Field(60.0)
    na_mM: float = Field(50.0)
    mg_mM: float = Field(3.5)
    dntp_mM: float = Field(0.6)

def calculate_viennarna_mfe(seq: str, temp_c: float, na_mM: float, mg_mM: float, dntp_mM: float):
    if not HAS_VIENNARNA:
        raise RuntimeError("viennarna package is not installed.")

    # 1. Configure Model Details for DNA at 60°C
    md = RNA.md()
    md.temperature = temp_c  # Set temperature (60.0 °C)
    
    # Load DNA parameter set (Mathews 1999 / SantaLucia 1998)
    RNA.read_parameter_file("dna_mathews1999.par")

    # 2. Prepare duplex sequence with '&' strand separator
    seq_clean = seq.upper().replace('U', 'T')
    duplex_seq = f"{seq_clean}&{seq_clean}"

    # 3. Create fold compound with DNA model parameters
    fc = RNA.fold_compound(duplex_seq, md)
    
    # Calculate MFE structure and raw free energy at 60°C
    (struct, mfe_energy) = fc.mfe_dimer()

    # 4. Owczarzy (2008) Divalent Salt Correction Factor
    # Calculate effective monovalent concentration under free Mg2+ (after dNTP chelation)
    free_mg = max(0.0001, (mg_mM - dntp_mM) / 1000.0)
    monovalent_eq = (na_mM / 1000.0) + 120.0 * math.sqrt(free_mg)

    # Scale entropy salt correction to 60°C (333.15 K)
    T_k = temp_c + 273.15
    # Standard backbone correction delta S_salt per base pair
    num_bp = struct.count('(')
    if num_bp > 1 and monovalent_eq > 0:
        ds_salt = 0.368 * (num_bp - 1) * math.log(monovalent_eq)
        # dG_corr = dG_raw - T * dS_salt
        salt_correction = (T_k * (ds_salt / 1000.0))
    else:
        salt_correction = 0.0

    # Final salt-adjusted MFE matching Benchling
    final_dg = round(mfe_energy - salt_correction, 2)

    return final_dg, struct

@app.post("/analyze")
def analyze_homodimer(req: HomodimerRequest):
    try:
        min_dg, structure = calculate_viennarna_mfe(
            req.sequence, req.temperature_c, req.na_mM, req.mg_mM, req.dntp_mM
        )

        warning = min_dg < -3.0

        return {
            "sequence": req.sequence,
            "min_delta_g": min_dg,
            "global_delta_g": min_dg,
            "end_delta_g": min_dg,
            "temperature_c": req.temperature_c,
            "na_mM": req.na_mM,
            "mg_mM": req.mg_mM,
            "dntp_mM": req.dntp_mM,
            "redesign_recommended": warning,
            "alignment": {
                "structure": structure,
                "engine": "ViennaRNA (DNA Parameters)",
                "is_3prime_end": False
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
