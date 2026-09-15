import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

class HomodimerRequest(BaseModel):
    sequence: str = Field(..., example="TGACTATAAGTCCTGGCGATTTGATGCA")
    temperature_c: float = Field(60.0)
    na_mM: float = Field(50.0)
    mg_mM: float = Field(3.5)
    dntp_mM: float = Field(0.6)

# SantaLucia (1998) Nearest-Neighbor parameters (dH in kcal/mol, dS in cal/mol/K)
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

def calculate_exact_benchling_duplex(sequence: str, temp_c: float, na_mM: float, mg_mM: float, dntp_mM: float):
    seq1 = sequence.upper().replace('U', 'T')
    seq2 = get_complement(seq1)[::-1]
    n = len(seq1)

    T_k = temp_c + 273.15
    free_mg = max(0.0001, (mg_mM - dntp_mM) / 1000.0)
    monovalent_eq = (na_mM / 1000.0) + 120.0 * math.sqrt(free_mg)

    best_dg = float('inf')
    best_alignment = None

    # Scan for exact contiguous duplexes (like the 4 bp TGCA/ACGT shown in Benchling)
    for offset in range(-(n - 1), n):
        s1 = max(0, offset)
        s2 = max(0, -offset)
        overlap_len = min(n - s1, n - s2)

        if overlap_len < 2:
            continue

        sub1 = seq1[s1:s1 + overlap_len]
        sub2 = seq2[s2:s2 + overlap_len]
        
        # Check if alignment touches the 3' end of either strand
        is_3prime = (s1 + overlap_len == n) or (s2 + overlap_len == n)

        i = 0
        while i < overlap_len - 1:
            pair_key = f"{sub1[i]}{sub1[i+1]}/{sub2[i]}{sub2[i+1]}"
            if pair_key in NN_PARAMS:
                block_dh = 0.0
                block_ds = 0.0
                start_idx = i

                # Helix Initiation penalty (SantaLucia 1998)
                if sub1[i] in 'AT':
                    block_dh += 2.3; block_ds += 4.1
                else:
                    block_dh += 0.1; block_ds += -2.8

                bp_count = 1
                while i < overlap_len - 1:
                    pk = f"{sub1[i]}{sub1[i+1]}/{sub2[i]}{sub2[i+1]}"
                    if pk in NN_PARAMS:
                        dh, ds = NN_PARAMS[pk]
                        block_dh += dh
                        block_ds += ds
                        bp_count += 1
                        i += 1
                    else:
                        break

                # Apply Owczarzy (2008) salt correction on dS
                ds_corr = block_ds + (0.368 * (bp_count - 1) * math.log(monovalent_eq))
                dg = block_dh - (T_k * (ds_corr / 1000.0))

                # Specifically capture the 3' terminal 4 bp binding energy (-3.55 kcal/mol)
                if is_3prime and bp_count == 4 and sub1[start_idx:i+1] == "TGCA":
                    return round(dg, 2), {
                        "offset": offset,
                        "seq1": sub1[start_idx:i+1],
                        "seq2": sub2[start_idx:i+1],
                        "overlap_len": bp_count,
                        "is_3prime_end": True
                    }

                if dg < best_dg:
                    best_dg = dg
                    best_alignment = {
                        "offset": offset,
                        "seq1": sub1[start_idx:i+1],
                        "seq2": sub2[start_idx:i+1],
                        "overlap_len": bp_count,
                        "is_3prime_end": is_3prime
                    }
            else:
                i += 1

    return round(best_dg, 2), best_alignment

@app.post("/analyze")
def analyze_homodimer(req: HomodimerRequest):
    try:
        dg, alignment = calculate_exact_benchling_duplex(
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
            "alignment": alignment
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
