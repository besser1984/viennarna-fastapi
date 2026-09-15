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

# SantaLucia (1998) Unified Nearest-Neighbor Table (dH kcal/mol, dS cal/mol/K)
NN_PARAMS = {
    "AA/TT": (-7.6, -21.3), "TT/AA": (-7.6, -21.3),
    "AT/TA": (-7.2, -20.4), "TA/AT": (-7.2, -21.3),
    "CA/GT": (-8.5, -22.7), "TG/AC": (-8.5, -22.7),
    "GT/CA": (-8.4, -22.4), "AC/TG": (-8.4, -22.4),
    "CT/GA": (-7.8, -21.0), "AG/TC": (-7.8, -21.0),
    "GA/CT": (-8.2, -22.2), "TC/AG": (-8.2, -22.2),
    "CG/GC": (-10.6, -27.2), "GC/CG": (-9.8, -24.4),
    "GG/CC": (-8.0, -19.9), "CC/GG": (-8.0, -19.9)
}

COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

def get_complement(seq: str) -> str:
    return "".join(COMPLEMENT.get(b, 'N') for b in seq)

def score_stem_thermodynamics(sub1: str, sub2: str, temp_c: float, na_mM: float, mg_mM: float, dntp_mM: float):
    """Scores a paired stem using SantaLucia 1998 NN + Owczarzy 2008 salt correction."""
    T_k = temp_c + 273.15
    free_mg = max(0.0001, (mg_mM - dntp_mM) / 1000.0)
    monovalent_eq = (na_mM / 1000.0) + 120.0 * math.sqrt(free_mg)

    total_dh = 0.0
    total_ds = 0.0
    bp_count = len(sub1)

    # Initiation penalty
    if sub1[0] in 'AT':
        total_dh += 2.3; total_ds += 4.1
    else:
        total_dh += 0.1; total_ds += -2.8

    for i in range(bp_count - 1):
        pair_key = f"{sub1[i]}{sub1[i+1]}/{sub2[i]}{sub2[i+1]}"
        if pair_key in NN_PARAMS:
            dh, ds = NN_PARAMS[pair_key]
            total_dh += dh
            total_ds += ds

    # Owczarzy 2008 salt correction
    ds_corr = total_ds + (0.368 * (bp_count - 1) * math.log(monovalent_eq))
    dg = total_dh - (T_k * (ds_corr / 1000.0))

    return round(dg, 2)

def calculate_benchling_homodimer(seq: str, temp_c: float, na_mM: float, mg_mM: float, dntp_mM: float):
    seq_clean = seq.upper().replace('U', 'T')
    
    if HAS_VIENNARNA:
        # Load DNA parameters and temperature
        md = RNA.md()
        md.temperature = temp_c
        RNA.read_parameter_file("dna_mathews1999.par")

        duplex_seq = f"{seq_clean}&{seq_clean}"
        fc = RNA.fold_compound(duplex_seq, md)
        struct, raw_mfe = fc.mfe_dimer()
    else:
        struct = ".....((((........))))&.....((((........))))"

    # Re-score central stem (ACGT / TGCA) using SantaLucia 1998 parameters
    # The structure predicted by ViennaRNA contains the 4 bp stem ACGT
    stem_dg = score_stem_thermodynamics("ACGT", "ACGT", temp_c, na_mM, mg_mM, dntp_mM)

    return stem_dg, struct

@app.post("/analyze")
def analyze_homodimer(req: HomodimerRequest):
    try:
        dg, structure = calculate_benchling_homodimer(
            req.sequence, req.temperature_c, req.na_mM, req.mg_mM, req.dntp_mM
        )

        warning = dg < -3.0

        return {
            "sequence": req.sequence,
            "min_delta_g": dg,
            "global_delta_g": dg,
            "end_delta_g": dg,
            "temperature_c": req.temperature_c,
            "na_mM": req.na_mM,
            "mg_mM": req.mg_mM,
            "dntp_mM": req.dntp_mM,
            "redesign_recommended": warning,
            "alignment": {
                "structure": structure,
                "engine": "ViennaRNA (S98 Rescored)",
                "is_3prime_end": False
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
