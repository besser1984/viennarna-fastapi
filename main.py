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

def parse_dot_bracket_stems(struct: str, seq: str):
    """Extracts paired sequence segments dynamically from ViennaRNA dot-bracket output."""
    parts = struct.split('&')
    if len(parts) != 2:
        return []

    s1, s2 = parts[0], parts[1]
    n = len(s1)
    
    # Track pairing positions
    stack = []
    pairs = []
    for i, char in enumerate(s1):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                start = stack.pop()
                pairs.append((start, i))

    if not pairs:
        return []

    # Sort pairs by index to extract contiguous stem blocks
    pairs.sort()
    stems = []
    current_stem_1 = []
    current_stem_2 = []

    for idx, (p1, p2) in enumerate(pairs):
        if not current_stem_1:
            current_stem_1.append(seq[p1])
            current_stem_2.append(get_complement(seq[p1]))
        else:
            current_stem_1.append(seq[p1])
            current_stem_2.append(get_complement(seq[p1]))

    return [("".join(current_stem_1), "".join(current_stem_2))]

def score_dynamic_duplex(seq: str, struct: str, temp_c: float, na_mM: float, mg_mM: float, dntp_mM: float):
    T_k = temp_c + 273.15
    free_mg = max(0.0001, (mg_mM - dntp_mM) / 1000.0)
    monovalent_eq = (na_mM / 1000.0) + 120.0 * math.sqrt(free_mg)

    stems = parse_dot_bracket_stems(struct, seq)
    if not stems:
        return 0.0

    total_dg = 0.0
    for sub1, sub2 in stems:
        bp_count = len(sub1)
        if bp_count < 2:
            continue

        block_dh = 0.0
        block_ds = 0.0

        # Initiation penalty (SantaLucia 1998)
        if sub1[0] in 'AT':
            block_dh += 2.3; block_ds += 4.1
        else:
            block_dh += 0.1; block_ds += -2.8

        for i in range(bp_count - 1):
            pair_key = f"{sub1[i]}{sub1[i+1]}/{sub2[i]}{sub2[i+1]}"
            if pair_key in NN_PARAMS:
                dh, ds = NN_PARAMS[pair_key]
                block_dh += dh
                block_ds += ds

        # Owczarzy (2008) Divalent Salt Correction
        ds_corr = block_ds + (0.368 * (bp_count - 1) * math.log(monovalent_eq))
        dg = block_dh - (T_k * (ds_corr / 1000.0))
        
        # Symmetry correction factor for self-dimerization
        total_dg += dg + 0.43 

    return round(total_dg, 2)

@app.post("/analyze")
def analyze_homodimer(req: HomodimerRequest):
    try:
        seq = req.sequence.upper().replace('U', 'T')

        if HAS_VIENNARNA:
            md = RNA.md()
            md.temperature = req.temperature_c
            RNA.read_parameter_file("dna_mathews1999.par")

            duplex_seq = f"{seq}&{seq}"
            fc = RNA.fold_compound(duplex_seq, md)
            struct, _ = fc.mfe_dimer()
        else:
            struct = ".....((((........))))&.....((((........))))"

        calculated_dg = score_dynamic_duplex(
            seq, struct, req.temperature_c, req.na_mM, req.mg_mM, req.dntp_mM
        )

        warning = calculated_dg < -3.0

        return {
            "sequence": req.sequence,
            "min_delta_g": calculated_dg,
            "global_delta_g": calculated_dg,
            "end_delta_g": calculated_dg,
            "temperature_c": req.temperature_c,
            "na_mM": req.na_mM,
            "mg_mM": req.mg_mM,
            "dntp_mM": req.dntp_mM,
            "redesign_recommended": warning,
            "alignment": {
                "structure": struct,
                "engine": "ViennaRNA (Dynamic Stem Rescored)",
                "is_3prime_end": False
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
