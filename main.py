import math
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import RNA
    HAS_VIENNARNA = True
except ImportError:
    HAS_VIENNARNA = False

app = FastAPI(title="DNA Homodimer Analyzer Microservice")

class HomodimerRequest(BaseModel):
    sequence: str = Field(..., example="TGACTATAAGTCCTGGCGATTTGATGCA")
    temperature_c: float = Field(60.0)
    na_mM: float = Field(50.0)
    mg_mM: float = Field(3.5)
    dntp_mM: float = Field(0.6)

# SantaLucia (1998) Unified NN parameters (dH in kcal/mol, dS in cal/mol/K)
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

def parse_dot_bracket_pairs(struct: str):
    """Parses duplex structure (strand1&strand2) into aligned base-pairing index maps."""
    parts = struct.split('&')
    if len(parts) != 2:
        return []

    s1, s2 = parts[0], parts[1]
    s1_open = [i for i, c in enumerate(s1) if c == '(']
    s2_close = [j for j, c in enumerate(s2) if c == ')']

    if len(s1_open) != len(s2_close):
        return []

    return list(zip(s1_open, s2_close))

def score_stem_thermodynamics(sub1: str, sub2: str, temp_c: float, na_mM: float, mg_mM: float, dntp_mM: float):
    """Scores a duplex stem using SantaLucia (1998) + Owczarzy (2008) salt scaling."""
    T_k = temp_c + 273.15
    free_mg = max(0.0001, (mg_mM - dntp_mM) / 1000.0)
    monovalent_eq = (na_mM / 1000.0) + 120.0 * math.sqrt(free_mg)

    bp_count = len(sub1)
    if bp_count < 2:
        return 0.0

    # Helix initiation term (SantaLucia 1998)
    block_dh = 2.3 if sub1[0] in 'AT' else 0.1
    block_ds = 4.1 if sub1[0] in 'AT' else -2.8

    # Terminal base pair penalty for helix end
    if sub1[-1] in 'AT':
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
    return dg

def evaluate_duplex_thermodynamics(seq: str, struct: str, raw_mfe: float, temp_c: float, na_mM: float, mg_mM: float, dntp_mM: float):
    seq_clean = seq.upper().replace('U', 'T')
    n = len(seq_clean)

    pairs = parse_dot_bracket_pairs(struct)
    if not pairs:
        return raw_mfe, raw_mfe, 0.0, None

    # Group paired indices into contiguous stem blocks
    stems = []
    curr_stem = [pairs[0]]
    for p in pairs[1:]:
        prev_p1, prev_p2 = curr_stem[-1]
        if p[0] == prev_p1 + 1 and abs(p[1] - prev_p2) == 1:
            curr_stem.append(p)
        else:
            stems.append(curr_stem)
            curr_stem = [p]
    stems.append(curr_stem)

    min_dg = float('inf')
    end_dg = 0.0
    best_align = None

    for stem in stems:
        bp_count = len(stem)
        if bp_count < 2:
            continue

        s1_indices = [p[0] for p in stem]
        s2_indices = [p[1] for p in stem]

        s1_str = "".join(seq_clean[i] for i in s1_indices)
        s2_str = "".join(get_complement(seq_clean[j]) for j in reversed(s2_indices))

        dg = score_stem_thermodynamics(s1_str, s2_str, temp_c, na_mM, mg_mM, dntp_mM)
        is_3prime = (max(s1_indices) == n - 1) or (max(s2_indices) == n - 1)

        if dg < min_dg:
            min_dg = dg
            best_align = {
                "seq1": s1_str,
                "seq2": s2_str,
                "overlap_len": bp_count,
                "is_3prime_end": is_3prime,
                "structure": struct
            }

        if is_3prime and (end_dg == 0.0 or dg < end_dg):
            end_dg = dg

    if min_dg == float('inf'):
        min_dg = 0.0

    return round(raw_mfe, 2), round(min_dg, 2), round(end_dg, 2), best_align

@app.post("/analyze")
def analyze_homodimer(req: HomodimerRequest):
    seq_clean = req.sequence.upper().replace('U', 'T')

    # 6. Sequence character validation
    if not all(c in "ACGT" for c in seq_clean):
        raise HTTPException(status_code=400, detail="Sequence must contain only standard DNA bases (A, C, G, T).")

    if not HAS_VIENNARNA:
        raise HTTPException(status_code=500, detail="viennarna package is not installed.")

    try:
        md = RNA.md()
        md.temperature = req.temperature_c

        # 7. Parameter file path validation
        param_file = "dna_mathews1999.par"
        if os.path.exists(param_file):
            RNA.read_parameter_file(param_file)

        duplex_seq = f"{seq_clean}&{seq_clean}"
        fc = RNA.fold_compound(duplex_seq, md)
        struct, raw_mfe = fc.mfe_dimer()

        global_dg, min_dg, end_dg, alignment = evaluate_duplex_thermodynamics(
            req.sequence, struct, raw_mfe, req.temperature_c, req.na_mM, req.mg_mM, req.dntp_mM
        )

        warning = global_dg < -5.0 or (alignment and alignment.get("is_3prime_end") and end_dg < -3.0)

        return {
            "sequence": req.sequence,
            "min_delta_g": min_dg,
            "global_delta_g": global_dg,
            "end_delta_g": end_dg,
            "temperature_c": req.temperature_c,
            "na_mM": req.na_mM,
            "mg_mM": req.mg_mM,
            "dntp_mM": req.dntp_mM,
            "redesign_recommended": warning,
            "alignment": alignment
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
