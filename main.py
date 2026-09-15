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
    description="High-throughput homodimer evaluation using direct ViennaRNA MFE thermodynamics.",
    version="3.0.0"
)

# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------

class HomodimerRequest(BaseModel):
    sequence: str = Field(..., example="ACAGGATCACGTCCCTCCCC")
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

class QCFlags(BaseModel):
    strong_dimer_risk: bool
    is_3prime_dimer: bool
    gc_percent: float
    has_poly_gc_3prime: bool

class HomodimerResponse(BaseModel):
    sequence: str
    global_delta_g: float
    temperature_c: float
    na_mM: float
    mg_mM: float
    dntp_mM: float
    qc_flags: QCFlags
    redesign_recommended: bool
    structure: str

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def calculate_gc_content(seq: str) -> float:
    gc_count = sum(1 for b in seq if b in 'GC')
    return round((gc_count / len(seq)) * 100.0, 2)

def check_3prime_homopolymer(seq: str, run_length: int = 3) -> bool:
    """Detects GC clamps or poly-C/G runs at the 3' terminal end."""
    tail = seq[-run_length:]
    return tail == "C" * run_length or tail == "G" * run_length

def is_3prime_bound(struct: str) -> bool:
    """
    Checks if the secondary structure fold involves the 3'-terminal base of either strand.
    Duplex structure notation: strand1_structure&strand2_structure
    """
    parts = struct.split('&')
    if len(parts) != 2:
        return False
    # Check if the final character of either strand has a paired bracket '(' or ')'
    return parts[0][-1] != '.' or parts[1][-1] != '.'

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.post("/analyze", response_model=HomodimerResponse)
def analyze_homodimer(req: HomodimerRequest):
    seq_clean = req.sequence.upper().replace('U', 'T')

    # Input validation
    if not all(c in "ACGT" for c in seq_clean):
        raise HTTPException(
            status_code=400, 
            detail="Invalid sequence. Must contain only standard DNA bases (A, C, G, T)."
        )

    if not HAS_VIENNARNA:
        raise HTTPException(
            status_code=500, 
            detail="ViennaRNA library ('viennarna') is not installed on the server environment."
        )

    try:
        # 1. Configure Model Details for DNA parameters at target temperature
        md = RNA.md()
        md.temperature = req.temperature_c

        # Robust parameter file resolution for Docker / Railway environments
        base_dir = os.path.dirname(__file__)
        param_file = os.path.join(base_dir, "dna_mathews1999.par")
        if os.path.exists(param_file):
            RNA.read_parameter_file(param_file)

        # 2. Perform co-fold MFE evaluation
        duplex_seq = f"{seq_clean}&{seq_clean}"
        fc = RNA.fold_compound(duplex_seq, md)
        struct, raw_mfe = fc.mfe_dimer()

        global_dg = round(raw_mfe, 2)
        bound_3prime = is_3prime_bound(struct)
        gc_pct = calculate_gc_content(seq_clean)
        poly_3prime = check_3prime_homopolymer(seq_clean)

        # QC Heuristics for Primer Redesign Recommendation:
        # - Global MFE dimer < -5.0 kcal/mol (Overall high stability)
        # - Any dimer involving 3'-end base < -3.0 kcal/mol (PCR extension risk)
        strong_dimer = global_dg < -5.0
        strong_3prime = bound_3prime and (global_dg < -3.0)
        
        redesign = strong_dimer or strong_3prime

        qc_flags = QCFlags(
            strong_dimer_risk=strong_dimer,
            is_3prime_dimer=strong_3prime,
            gc_percent=gc_pct,
            has_poly_gc_3prime=poly_3prime
        )

        return HomodimerResponse(
            sequence=req.sequence,
            global_delta_g=global_dg,
            temperature_c=req.temperature_c,
            na_mM=req.na_mM,
            mg_mM=req.mg_mM,
            dntp_mM=req.dntp_mM,
            qc_flags=qc_flags,
            redesign_recommended=redesign,
            structure=struct
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Thermodynamic evaluation error: {str(e)}")

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
