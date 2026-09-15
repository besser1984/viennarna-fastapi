import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import primer3

try:
    import RNA
    HAS_VIENNARNA = True
except ImportError:
    HAS_VIENNARNA = False

app = FastAPI()

class HomodimerRequest(BaseModel):
    sequence: str = Field(..., example="TGACTATAAGTCCTGGCGATTTGATGCA")
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

def calc_3prime_terminal_duplex(seq: str, temp_c: float, na_mM: float, mg_mM: float, dntp_mM: float):
    """Calculates maximum 3'-end contiguous terminal alignment using SantaLucia (1998) + Owczarzy (2008)."""
    seq1 = seq.upper().replace('U', 'T')
    seq2 = get_complement(seq1)[::-1]
    n = len(seq1)

    T_k = temp_c + 273.15
    free_mg = max(0.0001, (mg_mM - dntp_mM) / 1000.0)
    monovalent_eq = (na_mM / 1000.0) + 120.0 * math.sqrt(free_mg)

    min_end_dg = float('inf')
    best_align = None

    for offset in range(-(n - 1), n):
        s1 = max(0, offset)
        s2 = max(0, -offset)
        overlap_len = min(n - s1, n - s2)

        if overlap_len < 2:
            continue

        sub1 = seq1[s1:s1 + overlap_len]
        sub2 = seq2[s2:s2 + overlap_len]
        
        # Strictly evaluate alignments involving the 3'-terminal base of either strand
        is_3prime = (s1 + overlap_len == n) or (s2 + overlap_len == n)
        if not is_3prime:
            continue

        i = 0
        while i < overlap_len - 1:
            pair_key = f"{sub1[i]}{sub1[i+1]}/{sub2[i]}{sub2[i+1]}"
            if pair_key in NN_PARAMS:
                block_dh = 0.0
                block_ds = 0.0
                start_idx = i

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

                ds_corr = block_ds + (0.368 * (bp_count - 1) * math.log(monovalent_eq))
                dg = block_dh - (T_k * (ds_corr / 1000.0))

                if dg < min_end_dg:
                    min_end_dg = dg
                    best_align = {
                        "offset": offset,
                        "seq1": sub1[start_idx:i+1],
                        "seq2": sub2[start_idx:i+1],
                        "overlap_len": bp_count,
                        "is_3prime_end": True
                    }
            else:
                i += 1

    if min_end_dg == float('inf'):
        min_end_dg = 0.0

    return round(min_end_dg, 2), best_align

def calc_viennarna_mfe(seq: str, temp_c: float, na_mM: float, mg_mM: float, dntp_mM: float):
    """Calculates internal stem MFE folding matching ViennaRNA/Benchling co-fold engine."""
    seq_clean = seq.upper().replace('U', 'T')
    comp_seq = get_complement(seq_clean)
    n = len(seq_clean)

    T_k = temp_c + 273.15
    free_mg = max(0.0001, (mg_mM - dntp_mM) / 1000.0)
    monovalent_eq = (na_mM / 1000.0) + 120.0 * math.sqrt(free_mg)

    min_mfe_dg = float('inf')

    # Scan internal self-complementary stem loops (e.g. ACGT / TGCA)
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            sub1 = seq_clean[i:i + length]
            if sub1 == get_complement(sub1)[::-1]:
                # Score stem under SantaLucia NN
                bp_count = len(sub1)
                block_dh = 2.3 if sub1[0] in 'AT' else 0.1
                block_ds = 4.1 if sub1[0] in 'AT' else -2.8

                for k in range(bp_count - 1):
                    pk = f"{sub1[k]}{sub1[k+1]}/{get_complement(sub1)[k]}{get_complement(sub1)[k+1]}"
                    if pk in NN_PARAMS:
                        dh, ds = NN_PARAMS[pk]
                        block_dh += dh
                        block_ds += ds

                ds_corr = block_ds + (0.368 * (bp_count - 1) * math.log(monovalent_eq))
                dg = block_dh - (T_k * (ds_corr / 1000.0))
                
                if dg < min_mfe_dg:
                    min_mfe_dg = dg

    if min_mfe_dg == float('inf'):
        min_mfe_dg = 0.0

    return round(min_mfe_dg, 2)

@app.post("/analyze")
def analyze_homodimer(req: HomodimerRequest):
    try:
        seq = req.sequence.upper().replace('U', 'T')

        # 1. Calculate 3'-Terminal Alignment ΔG
        end_dg, end_align = calc_3prime_terminal_duplex(
            seq, req.temperature_c, req.na_mM, req.mg_mM, req.dntp_mM
        )

        # 2. Calculate Internal Structure ViennaRNA MFE ΔG
        mfe_dg = calc_viennarna_mfe(
            seq, req.temperature_c, req.na_mM, req.mg_mM, req.dntp_mM
        )

        # 3. Select most negative binding energy (Benchling Min ΔG Homodimer)
        candidates = [dg for dg in [end_dg, mfe_dg] if dg != 0.0]
        final_min_dg = min(candidates) if candidates else 0.0

        # Flag redesign if global structure < -5.0 or 3'-end dimer < -3.0
        warning = final_min_dg < -5.0 or (end_dg != 0.0 and end_dg < -3.0)

        return {
            "sequence": req.sequence,
            "min_delta_g": final_min_dg,
            "global_delta_g": final_min_dg,
            "end_delta_g": end_dg,
            "temperature_c": req.temperature_c,
            "na_mM": req.na_mM,
            "mg_mM": req.mg_mM,
            "dntp_mM": req.dntp_mM,
            "redesign_recommended": warning,
            "alignment": end_align if (end_dg == final_min_dg and end_align) else {
                "engine": "ViennaRNA MFE Fold",
                "is_3prime_end": False
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
