import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

class HomodimerRequest(BaseModel):
    sequence: str = Field(..., example="TGACTATAAGTCCTGGCGATTTGATGCA")
    temperature_c: float = Field(60.0)
    na_mM: float = Field(50.0)
    mg_mM: float = Field(3.5)

# SantaLucia 1998 Unified NN parameters (dH in kcal/mol, dS in cal/mol/K)
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

def get_owczarzy_salt_corrected_ds(ds_base: float, num_bp: int, na_mM: float, mg_mM: float) -> float:
    """Owczarzy et al. (2008) divalent/monovalent salt correction on entropy (dS)."""
    na = na_mM / 1000.0
    mg = mg_mM / 1000.0
    
    # Calculate effective monovalent concentration under divalent excess
    if mg > 0:
        monovalent_eq = na + 120.0 * math.sqrt(mg)
    else:
        monovalent_eq = na

    if monovalent_eq > 0 and num_bp > 1:
        # 0.368 cal/mol/K per phosphate bond
        ds_corr = 0.368 * (num_bp - 1) * math.log(monovalent_eq)
    else:
        ds_corr = 0.0

    return ds_base + ds_corr

def calculate_homodimer(sequence: str, temp_c: float, na_mM: float, mg_mM: float):
    seq1 = sequence.upper().replace('U', 'T')
    seq2 = get_complement(seq1)[::-1]
    n = len(seq1)
    
    T_k = temp_c + 273.15
    min_dg = float('inf')
    best_alignment = None

    # Evaluate all antiparallel overlap positions
    for offset in range(-(n - 1), n):
        s1 = max(0, offset)
        s2 = max(0, -offset)
        overlap_len = min(n - s1, n - s2)

        if overlap_len < 2:
            continue

        sub1 = seq1[s1:s1 + overlap_len]
        sub2 = seq2[s2:s2 + overlap_len]

        # Scan for paired blocks inside alignment region
        i = 0
        while i < overlap_len - 1:
            if f"{sub1[i]}{sub1[i+1]}/{sub2[i]}{sub2[i+1]}" in NN_PARAMS:
                block_dh = 0.0
                block_ds = 0.0
                bp_count = 1

                # Apply helix initiation penalty (SantaLucia 1998)
                if sub1[i] in 'AT':
                    block_dh += 2.3; block_ds += 4.1
                else:
                    block_dh += 0.1; block_ds += -2.8

                while i < overlap_len - 1:
                    pair_key = f"{sub1[i]}{sub1[i+1]}/{sub2[i]}{sub2[i+1]}"
                    if pair_key in NN_PARAMS:
                        dh, ds = NN_PARAMS[pair_key]
                        block_dh += dh
                        block_ds += ds
                        bp_count += 1
                        i += 1
                    else:
                        break

                # Apply terminal penalty on block end
                if sub1[i-1] in 'AT':
                    block_dh += 2.3; block_ds += 4.1
                else:
                    block_dh += 0.1; block_ds += -2.8

                # Correct entropy for buffer cations
                ds_corr = get_owczarzy_salt_corrected_ds(block_ds, bp_count, na_mM, mg_mM)
                dg = block_dh - (T_k * (ds_corr / 1000.0))

                if dg < min_dg:
                    min_dg = dg
                    best_alignment = {
                        "offset": offset,
                        "seq1": sub1,
                        "seq2": sub2,
                        "overlap_len": bp_count,
                        "is_3prime_end": (s1 + overlap_len == n) or (s2 + overlap_len == n)
                    }
            else:
                i += 1

    if min_dg == float('inf'):
        min_dg = 0.0

    return round(min_dg, 2), best_alignment

@app.post("/analyze")
def analyze_homodimer(req: HomodimerRequest):
    try:
        min_dg, alignment = calculate_homodimer(req.sequence, req.temperature_c, req.na_mM, req.mg_mM)
        
        # Heuristics: Flag homodimers with Global ΔG < -5.0 or 3'-end ΔG < -3.0
        warning = False
        if min_dg < -5.0 or (alignment and alignment["is_3prime_end"] and min_dg < -3.0):
            warning = True

        return {
            "sequence": req.sequence,
            "min_delta_g": min_dg,
            "temperature_c": req.temperature_c,
            "na_mM": req.na_mM,
            "mg_mM": req.mg_mM,
            "redesign_recommended": warning,
            "alignment": alignment
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
