import math
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import RNA
    HAS_VIENNARNA = True
except ImportError:
    HAS_VIENNARNA = False

app = FastAPI(title="DNA Homodimer Screening Microservice", version="2.0.0")

class HomodimerRequest(BaseModel):
    sequence: str = Field(..., example="TGACTATAAGTCCTGGCGATTTGATGCA")
    temperature_c: float = Field(60.0)
    na_mM: float = Field(50.0)
    mg_mM: float = Field(3.5)
    dntp_mM: float = Field(0.6)

class BatchHomodimerRequest(BaseModel):
    sequences: List[str] = Field(..., example=["ACAGGATCACGTCCCTCCCC", "TGACTATAAGTCCTGGCGATTTGATGCA"])
    temperature_c: float = Field(60.0)
    na_mM: float = Field(50.0)
    mg_mM: float = Field(3.5)
    dntp_mM: float = Field(0.6)

# SantaLucia (1998) Unified Nearest-Neighbor Table (dH in kcal/mol, dS in cal/mol/K)
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

def calculate_gc_content(seq: str) -> float:
    gc_count = sum(1 for b in seq if b in 'GC')
    return round((gc_count / len(seq)) * 100.0, 2)

def check_3prime_homopolymer(seq: str, run_length: int = 3) -> bool:
    """Checks for 3'-terminal poly-C or poly-G runs (e.g. CCC, GGG)."""
    tail = seq[-run_length:]
    return tail == "C" * run_length or tail == "G" * run_length

def score_stem_thermodynamics(sub1: str, sub2: str, temp_c: float, na_mM: float, mg_mM: float, dntp_mM: float) -> float:
    """
    Scores a contiguous duplex stem using SantaLucia (1998) NN parameters with 
    Owczarzy (2008) divalent salt scaling on free Mg2+ after dNTP chelation.
    """
    T_k = temp_c + 273.15
    free_mg = max(0.0001, (mg_mM - dntp_mM) / 1000.0)
    monovalent_eq = (na_mM / 1000.0) + 120.0 * math.sqrt(free_mg)

    bp_count = len(sub1)
    if bp_count < 2:
        return 0.0

    # Helix initiation penalty
    block_dh = 2.3 if sub1[0] in 'AT' else 0.1
    block_ds = 4.1 if sub1[0] in 'AT' else -2.8

    for i in range(bp_count - 1):
        pair_key = f"{sub1[i]}{sub1[i+1]}/{sub2[i]}{sub2[i+1]}"
        if pair_key in NN_PARAMS:
            dh, ds = NN_PARAMS[pair_key]
            block_dh += dh
            block_ds += ds

    # Salt correction (Owczarzy 2008 / SantaLucia 1998 hybrid Tm adjustment)
    ds_corr = block_ds + (0.368 * (bp_count - 1) * math.log(monovalent_eq))
    dg = block_dh - (T_k * (ds_corr / 1000.0))
    return round(dg, 2)

def calculate_hybrid_thermodynamics(seq: str, struct: str, raw_mfe: float, temp_c: float, na_mM: float, mg_mM: float, dntp_mM: float):
    seq_clean = seq.upper().replace('U', 'T')
    comp_seq = get_complement(seq_clean)
    n = len(seq_clean)

    best_stem_dg = float('inf')
    best_3prime_stem_dg = float('inf')
    best_align = None

    # Scan antiparallel alignment matrix across all offsets
    for offset in range(-(n - 1), n):
        s1 = max(0, offset)
        s2 = max(0, -offset)
        overlap_len = min(n - s1, n - s2)

        if overlap_len < 2:
            continue

        sub1 = seq_clean[s1:s1 + overlap_len]
        sub2 = comp_seq[::-1][s2:s2 + overlap_len]

        i = 0
        while i < overlap_len - 1:
            pair_key = f"{sub1[i]}{sub1[i+1]}/{sub2[i]}{sub2[i+1]}"
            if pair_key in NN_PARAMS:
                start_idx = i
                bp_count = 1
                while i < overlap_len - 1:
                    pk = f"{sub1[i]}{sub1[i+1]}/{sub2[i]}{sub2[i+1]}"
                    if pk in NN_PARAMS:
                        bp_count += 1
                        i += 1
                    else:
                        break

                block_sub1 = sub1[start_idx:start_idx + bp_count]
                block_sub2 = sub2[start_idx:start_idx + bp_count]

                # Re-evaluate 3'-end status per-stem, taking precise slice positions
                block_start_s1 = s1 + start_idx
                block_end_s1 = block_start_s1 + bp_count
                block_start_s2 = s2 + start_idx
                block_end_s2 = block_start_s2 + bp_count

                stem_is_3prime = (block_end_s1 == n) or (block_end_s2 == n)

                dg = score_stem_thermodynamics(block_sub1, block_sub2, temp_c, na_mM, mg_mM, dntp_mM)

                if dg < best_stem_dg:
                    best_stem_dg = dg
                    best_align = {
                        "seq1": block_sub1,
                        "seq2": block_sub2,
                        "overlap_len": bp_count,
                        "is_3prime_end": stem_is_3prime,
                        "structure": struct
                    }

                # Corrected end_dg update logic using float('inf')
                if stem_is_3prime and dg < best_3prime_stem_dg:
                    best_3prime_stem_dg = dg
            else:
                i += 1

    # Cleanup default infinity fallbacks
    if best_stem_dg == float('inf'):
        best_stem_dg = round(raw_mfe, 2)
    
    if best_3prime_stem_dg == float('inf'):
        best_3prime_stem_dg = 0.0

    return round(raw_mfe, 2), round(best_stem_dg, 2), round(best_3prime_stem_dg, 2), best_align

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

        # Robust parameter file loading for Docker / deployment environments
        base_dir = os.path.dirname(__file__)
        param_file = os.path.join(base_dir, "dna_mathews1999.par")
        if os.path.exists(param_file):
            RNA.read_parameter_file(param_file)

        duplex_seq = f"{seq_clean}&{seq_clean}"
        fc = RNA.fold_compound(duplex_seq, md)
        struct, raw_mfe = fc.mfe_dimer()

        viennarna_mfe, best_stem_dg, best_3prime_stem_dg, alignment = calculate_hybrid_thermodynamics(
            seq_clean, struct, raw_mfe, req.temperature_c, req.na_mM, req.mg_mM, req.dntp_mM
        )

        gc_pct = calculate_gc_content(seq_clean)
        has_poly_3prime = check_3prime_homopolymer(seq_clean)

        strong_global = viennarna_mfe < -5.0 or best_stem_dg < -5.0
        strong_3prime = best_3prime_stem_dg < -3.0

        redesign = strong_global or strong_3prime

        qc_flags = {
            "strong_global_dimer": strong_global,
            "strong_3prime_dimer": strong_3prime,
            "gc_percent": gc_pct,
            "has_poly_gc_3prime": has_poly_3prime
        }

        return {
            "sequence": req.sequence,
            "viennarna_mfe": viennarna_mfe,
            "best_stem_dg": best_stem_dg,
            "best_3prime_stem_dg": best_3prime_stem_dg,
            "temperature_c": req.temperature_c,
            "na_mM": req.na_mM,
            "mg_mM": req.mg_mM,
            "dntp_mM": req.dntp_mM,
            "qc_flags": qc_flags,
            "redesign_recommended": redesign,
            "alignment": alignment
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_batch")
def analyze_batch_homodimer(req: BatchHomodimerRequest):
    results = []
    for seq in req.sequences:
        single_req = HomodimerRequest(
            sequence=seq,
            temperature_c=req.temperature_c,
            na_mM=req.na_mM,
            mg_mM=req.mg_mM,
            dntp_mM=req.dntp_mM,
        )
        results.append(analyze_homodimer(single_req))
    return results
