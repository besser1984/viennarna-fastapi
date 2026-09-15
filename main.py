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

COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

def get_complement(seq: str) -> str:
    return "".join(COMPLEMENT.get(b, 'N') for b in seq)

def score_stem_thermodynamics(sub1: str, sub2: str, temp_c: float, na_mM: float, mg_mM: float, dntp_mM: float):
    """Scores a duplex stem using SantaLucia 1998 + Owczarzy 2008 salt scaling."""
    T_k = temp_c + 273.15
    free_mg = max(0.0001, (mg_mM - dntp_mM) / 1000.0)
    monovalent_eq = (na_mM / 1000.0) + 120.0 * math.sqrt(free_mg)

    bp_count = len(sub1)
    if bp_count < 2:
        return 0.0

    block_dh = 2.3 if sub1[0] in 'AT' else 0.1
    block_ds = 4.1 if sub1[0] in 'AT' else -2.8

    for i in range(bp_count - 1):
        pair_key = f"{sub1[i]}{sub1[i+1]}/{sub2[i]}{sub2[i+1]}"
        if pair_key in NN_PARAMS:
            dh, ds = NN_PARAMS[pair_key]
            block_dh += dh
            block_ds += ds

    ds_corr = block_ds + (0.368 * (bp_count - 1) * math.log(monovalent_eq))
    dg = block_dh - (T_k * (ds_corr / 1000.0))
    return round(dg, 2)

def calculate_benchling_homodimer(sequence: str, temp_c: float, na_mM: float, mg_mM: float, dntp_mM: float):
    seq_clean = sequence.upper().replace('U', 'T')
    comp_seq = get_complement(seq_clean)
    n = len(seq_clean)

    if not HAS_VIENNARNA:
        raise RuntimeError("viennarna package is required on Railway.")

    # 1. Configure Model Details for DNA parameters at target temperature
    md = RNA.md()
    md.temperature = temp_c
    RNA.read_parameter_file("dna_mathews1999.par")

    # 2. Compute fold topology using fold_compound to prevent index out-of-range slicing
    duplex_seq = f"{seq_clean}&{seq_clean}"
    fc = RNA.fold_compound(duplex_seq, md)
    struct, _ = fc.mfe_dimer()

    # 3. Dynamic target selection matching Benchling structural motifs
    best_dg = float('inf')
    best_align = None

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            sub1 = seq_clean[i:i + length]
            sub2 = comp_seq[i:i + length][::-1]
            is_3prime = (i + length == n)

            # Evaluate contiguous Watson-Crick paired stems
            if sub1 == get_complement(sub1)[::-1]:
                dg = score_stem_thermodynamics(sub1, sub1, temp_c, na_mM, mg_mM, dntp_mM)
                if dg < best_dg:
                    best_dg = dg
                    best_align = {
                        "seq1": sub1,
                        "seq2": sub1,
                        "overlap_len": length,
                        "is_3prime_end": is_3prime,
                        "structure": struct
                    }

    if best_dg == float('inf') or best_dg == 0.0:
        best_dg = 0.0

    return round(best_dg, 2), best_align

@app.post("/analyze")
def analyze_homodimer(req: HomodimerRequest):
    try:
        dg, alignment = calculate_benchling_homodimer(
            req.sequence, req.temperature_c, req.na_mM, req.mg_mM, req.dntp_mM
        )

        warning = dg < -5.0 or (alignment and alignment.get("is_3prime_end") and dg < -3.0)

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
