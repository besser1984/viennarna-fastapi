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

app = FastAPI(
    title="DNA Homodimer Screening Microservice",
    description="High-throughput primer-dimer analysis distinguishing global MFE cofold from local 3'-end duplex interactions.",
    version="4.0.0"
)

# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------

class HomodimerRequest(BaseModel):
    sequence: str = Field(..., example="TGACTATAAGTCCTGGCGATTTGATGCA")
    temperature_c: float = Field(60.0, description="PCR Reaction temperature in °C")
    na_mM: float = Field(50.0, description="Monovalent cation concentration in mM")
    mg_mM: float = Field(3.5, description="Divalent cation concentration in mM")
    dntp_mM: float = Field(0.6, description="Total dNTP concentration in mM")

class BatchHomodimerRequest(BaseModel):
    sequences: List[str] = Field(..., example=["ACAGGATCACGTCCCTCCCC", "TGACTATAAGTCCTGGCGATTTGATGCA"])
    temperature_c: float = Field(60.0)
    na_mM: float = Field(50.0)
    mg_mM: float = Field(3.5)
    dntp_mM: float = Field(0.6)

class LocalAlignment(BaseModel):
    seq1: str
    seq2: str
    overlap_len: int
    is_3prime_end: bool

class HomodimerResponse(BaseModel):
    sequence: str
    global_cofold_mfe_kcal_mol: float
    local_dimer_dg_kcal_mol: Optional[float]
    three_prime_dimer_dg_kcal_mol: Optional[float]
    mfe_structure: str
    engine: str = "ViennaRNA"
    model: str = "DNA cofold (SantaLucia 1998 / Owczarzy 2008 Hybrid)"
    temperature_c: float
    na_mM: float
    mg_mM: float
    dntp_mM: float
    global_dimer_risk: bool
    three_prime_dimer_risk: bool
    redesign_recommended: bool
    alignment: Optional[LocalAlignment]

# -----------------------------------------------------------------------------
# Thermodynamic Parameters & Salt Scaling
# -----------------------------------------------------------------------------

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

def calculate_monovalent_equivalent(na_mM: float, mg_mM: float, dntp_mM: float) -> float:
    """Calculates equivalent monovalent concentration under free Mg2+ after dNTP chelation."""
    free_mg = max(0.0001, (mg_mM - dntp_mM) / 1000.0)
    return (na_mM / 1000.0) + 120.0 * math.sqrt(free_mg)

def score_stem_thermodynamics(sub1: str, sub2: str, temp_c: float, monovalent_eq: float) -> float:
    """Scores a duplex stem using SantaLucia 1998 NN parameters with Owczarzy 2008 salt scaling."""
    T_k = temp_c + 273.15
    bp_count = len(sub1)
    if bp_count < 2:
        return 0.0

    # SantaLucia 1998 Initiation penalty
    block_dh = 2.3 if sub1[0] in 'AT' else 0.1
    block_ds = 4.1 if sub1[0] in 'AT' else -2.8

    for i in range(bp_count - 1):
        pair_key = f"{sub1[i]}{sub1[i+1]}/{sub2[i]}{sub2[i+1]}"
        if pair_key in NN_PARAMS:
            dh, ds = NN_PARAMS[pair_key]
            block_dh += dh
            block_ds += ds

    # Owczarzy 2008 salt correction on entropy
    ds_corr = block_ds + (0.368 * (bp_count - 1) * math.log(monovalent_eq))
    dg = block_dh - (T_k * (ds_corr / 1000.0))
    return round(dg, 2)

def calculate_local_duplex_dg(seq: str, temp_c: float, na_mM: float, mg_mM: float, dntp_mM: float):
    """Evaluates maximum local contiguous alignment stability and isolates 3'-end interaction."""
    seq_clean = seq.upper().replace('U', 'T')
    comp_seq = get_complement(seq_clean)
    n = len(seq_clean)
    monovalent_eq = calculate_monovalent_equivalent(na_mM, mg_mM, dntp_mM)

    best_local_dg = float('inf')
    best_3prime_dg = float('inf')
    best_align = None

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

                block_end_s1 = s1 + start_idx + bp_count
                block_end_s2 = s2 + start_idx + bp_count
                stem_is_3prime = (block_end_s1 == n) or (block_end_s2 == n)

                dg = score_stem_thermodynamics(block_sub1, block_sub2, temp_c, monovalent_eq)

                if dg < best_local_dg:
                    best_local_dg = dg
                    best_align = LocalAlignment(
                        seq1=block_sub1,
                        seq2=block_sub2,
                        overlap_len=bp_count,
                        is_3prime_end=stem_is_3prime
                    )

                if stem_is_3prime and dg < best_3prime_dg:
                    best_3prime_dg = dg
            else:
                i += 1

    local_dg = round(best_local_dg, 2) if best_local_dg != float('inf') else None
    three_prime_dg = round(best_3prime_dg, 2) if best_3prime_dg != float('inf') else None

    return local_dg, three_prime_dg, best_align

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.post("/analyze", response_model=HomodimerResponse)
def analyze_homodimer(req: HomodimerRequest):
    seq_clean = req.sequence.upper().replace('U', 'T')

    if not all(c in "ACGT" for c in seq_clean):
        raise HTTPException(status_code=400, detail="Sequence must contain only standard DNA bases (A, C, G, T).")

    if not HAS_VIENNARNA:
        raise HTTPException(status_code=500, detail="viennarna package is not installed.")

    try:
        # 1. Configure ViennaRNA Model & Salt Attributes
        md = RNA.md()
        md.temperature = req.temperature_c

        # Calculate effective monovalent salt for ViennaRNA md attributes if supported
        monovalent_eq = calculate_monovalent_equivalent(req.na_mM, req.mg_mM, req.dntp_mM)
        if hasattr(md, "salt"):
            md.salt = monovalent_eq
        elif hasattr(md, "salt_conc"):
            md.salt_conc = monovalent_eq

        # Load DNA parameters
        base_dir = os.path.dirname(__file__)
        param_file = os.path.join(base_dir, "dna_mathews1999.par")
        if os.path.exists(param_file):
            RNA.read_parameter_file(param_file)

        # 2. ViennaRNA Global Cofold MFE
        duplex_seq = f"{seq_clean}&{seq_clean}"
        fc = RNA.fold_compound(duplex_seq, md)
        struct, raw_mfe = fc.mfe_dimer()
        global_mfe = round(raw_mfe, 2)

        # 3. Local Alignment & 3'-End Duplex Evaluation
        local_dg, three_prime_dg, alignment = calculate_local_duplex_dg(
            seq_clean, req.temperature_c, req.na_mM, req.mg_mM, req.dntp_mM
        )

        # 4. Risk Flagging
        global_risk = global_mfe < -5.0
        three_prime_risk = (three_prime_dg is not None) and (three_prime_dg < -3.0)
        redesign = global_risk or three_prime_risk

        return HomodimerResponse(
            sequence=req.sequence,
            global_cofold_mfe_kcal_mol=global_mfe,
            local_dimer_dg_kcal_mol=local_dg,
            three_prime_dimer_dg_kcal_mol=three_prime_dg,
            mfe_structure=struct,
            temperature_c=req.temperature_c,
            na_mM=req.na_mM,
            mg_mM=req.mg_mM,
            dntp_mM=req.dntp_mM,
            global_dimer_risk=global_risk,
            three_prime_dimer_risk=three_prime_risk,
            redesign_recommended=redesign,
            alignment=alignment
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_batch", response_model=List[HomodimerResponse])
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
