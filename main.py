import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

class HomodimerRequest(BaseModel):
    sequence: str = Field(..., example="TGACTATAAGTCCTGGCGATTTGATGCA")
    temperature_c: float = Field(60.0, description="Temperature in Celsius")
    na_mM: float = Field(50.0, description="Monovalent Na+ concentration in mM")
    mg_mM: float = Field(3.5, description="Divalent Mg2+ concentration in mM")

# SantaLucia (1998) Unified NN Parameters (dH in kcal/mol, dS in cal/mol/K at 1 M Na+)
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
    return "".join(COMPLEMENT.get(base, 'N') for base in seq)

def calculate_salt_corrected_ds(ds_base: float, N_pairs: int, na_mM: float, mg_mM: float) -> float:
    """Applies Owczarzy/SantaLucia salt correction to entropy (dS)."""
    Na = na_mM / 1000.0
    Mg = mg_mM / 1000.0

    # Effective monovalent concentration approximation
    if Mg > 0:
        Monovalent_Equivalent = Na + 120 * math.sqrt(Mg)
    else:
        Monovalent_Equivalent = Na

    # Salt correction per phosphate backbone
    if Monovalent_Equivalent > 0:
        ds_correction = 0.368 * (N_pairs - 1) * math.log(Monovalent_Equivalent)
    else:
        ds_correction = 0.0

    return ds_base + ds_correction

def calculate_homodimer(sequence: str, temp_c: float, na_mM: float, mg_mM: float):
    seq1 = sequence.upper().replace('U', 'T')
    seq2_rev = get_complement(seq1)[::-1]
    n = len(seq1)
    
    T_kelvin = temp_c + 273.15
    min_dg = float('inf')
    best_alignment = None

    # Slide antiparallel strands across all offsets
    for offset in range(-(n - 1), n):
        start1 = max(0, offset)
        start2 = max(0, -offset)
        overlap_len = min(n - start1, n - start2)
        
        if overlap_len < 2:
            continue

        sub1 = seq1[start1:start1 + overlap_len]
        sub2 = seq2_rev[start2:start2 + overlap_len]

        # Scan for contiguous matching sub-blocks within the alignment window
        # Truncate single-stranded overhangs to calculate true duplex stability
        i = 0
        while i < overlap_len - 1:
            # Check for standard complementary pairs
            b1, b2 = sub1[i:i+2], sub2[i:i+2]
            pair_key = f"{b1}/{b2}"
            
            if pair_key in NN_PARAMS:
                block_dh = 0.0
                block_ds = 0.0
                
                # Terminal initiation penalty (SantaLucia 1998)
                if sub1[i] in 'AT': 
                    block_dh += 2.3; block_ds += 4.1
                else: 
                    block_dh += 0.1; block_ds += -2.8

                match_len = 1
                while i < overlap_len - 1:
                    pk = f"{sub1[i]:s}{sub1[i+1]:s}/{sub2[i]:s}{sub2[i+1]:s}"
                    if pk in NN_PARAMS:
                        dh, ds = NN_PARAMS[pk]
                        block_dh += dh
                        block_ds += ds
                        match_len += 1
                        i += 1
                    else:
                        break

                # Apply salt correction to the paired duplex block
                ds_corr = calculate_salt_corrected_ds(block_ds, match_len, na_mM, mg_mM)
                dg = block_dh - (T_kelvin * (ds_corr / 1000.0))

                if dg < min_dg:
                    min_dg = dg
                    best_alignment = {
                        "offset": offset,
                        "seq1": sub1,
                        "seq2": sub2,
                        "overlap_len": match_len,
                        "is_3prime_end": (start1 + overlap_len == n) or (start2 + overlap_len == n)
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
        
        # Risk assessment heuristics at elevated PCR annealing temperatures
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
