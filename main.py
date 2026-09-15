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
    sequence: str = Field(..., example="ACAGGATCACGTCCCTCCCC")
    temperature_c: float = Field(60.0)
    na_mM: float = Field(50.0)
    mg_mM: float = Field(3.5)
    dntp_mM: float = Field(0.6)

# SantaLucia (1998) Unified NN Table (dH in kcal/mol, dS in cal/mol/K)
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

def parse_dot_bracket_pairs(struct: str):
    """
    Parses duplex structure into aligned base-pairing index maps using 
    standard unified LIFO stack alignment for antiparallel strands.
    """
    parts = struct.split('&')
    if len(parts) != 2:
        return []

    s1_len = len(parts[0])
    full_struct = struct.replace('&', '')

    stack = []
    pairs = []

    for i, char in enumerate(full_struct):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                # Isolate intermolecular bindings (Strand 1 -> Strand 2)
                if j < s1_len and i >= s1_len:
                    pairs.append((j, i - s1_len))

    # Sort forward along Strand 1
    pairs.sort(key=lambda x: x[0])
    return pairs

def evaluate_duplex_thermodynamics(seq: str, struct: str, raw_mfe: float, temp_c: float, na_mM: float, mg_mM: float, dntp_mM: float):
    seq_clean = seq.upper().replace('U', 'T')
    n = len(seq_clean)

    T_k = temp_c + 273.15
    free_mg = max(0.0001, (mg_mM - dntp_mM) / 1000.0)
    monovalent_eq = (na_mM / 1000.0) + 120.0 * math.sqrt(free_mg)

    pairs = parse_dot_bracket_pairs(struct)
    if not pairs:
        return round(raw_mfe, 2), 0.0, 0.0, None

    # Group paired indices into contiguous stem blocks
    stems = []
    curr_stem = [pairs[0]]
    for p in pairs[1:]:
        prev_p1, prev_p2 = curr_stem[-1]
        # Antiparallel stem check: S1 increases while S2 decreases
        if p[0] == prev_p1 + 1 and p[1] == prev_p2 - 1:
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
        s2_str = "".join(seq_clean[j] for j in s2_indices)

        # Single terminal helix initiation penalty matching Benchling engine
        block_dh = 2.3 if s1_str[0] in 'AT' else 0.1
        block_ds = 4.1 if s1_str[0] in 'AT' else -2.8

        for i in range(bp_count - 1):
            pair_key = f"{s1_str[i]}{s1_str[i+1]}/{s2_str[i]}{s2_str[i+1]}"
            if pair_key in NN_PARAMS:
                dh, ds = NN_PARAMS[pair_key]
                block_dh += dh
                block_ds += ds

        ds_corr = block_ds + (0.368 * (bp_count - 1) * math.log(monovalent_eq))
        dg = block_dh - (T_k * (ds_corr / 1000.0))

        is_3prime = (max(s1_indices) == n - 1) or (max(s2_indices) == n - 1)

        if dg < min_dg:
            min_dg = dg
            best_align = {
                "seq1": s1_str,
                "seq2": "".join(reversed(s2_str)), # Represent S2 in standard 5'->3' notation for JSON
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

    if not all(c in "ACGT" for c in seq_clean):
        raise HTTPException(status_code=400, detail="Sequence must contain only standard DNA bases (A, C, G, T).")

    if not HAS_VIENNARNA:
        raise HTTPException(status_code=500, detail="viennarna package is not installed.")

    try:
        md = RNA.md()
        md.temperature = req.temperature_c

        # Route parameter file securely
        param_file = "dna_mathews1999.par"
        if os.path.exists(param_file):
            RNA.read_parameter_file(param_file)

        duplex_seq = f"{seq_clean}&{seq_clean}"
        fc = RNA.fold_compound(duplex_seq, md)
        struct, raw_mfe = fc.mfe_dimer()

        global_dg, min_dg, end_dg, alignment = evaluate_duplex_thermodynamics(
            seq_clean, struct, raw_mfe, req.temperature_c, req.na_mM, req.mg_mM, req.dntp_mM
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
