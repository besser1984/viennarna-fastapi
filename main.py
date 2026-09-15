import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import primer3
    HAS_PRIMER3 = True
except ImportError:
    HAS_PRIMER3 = False

try:
    import RNA
    HAS_VIENNARNA = True
except ImportError:
    HAS_VIENNARNA = False


app = FastAPI(
    title="DNA Primer QC Microservice",
    description=(
        "High-throughput DNA-primer QC using Primer3 thermodynamic alignment "
        "for dimer/hairpin screening and optional ViennaRNA DNA cofold annotation."
    ),
    version="5.0.0",
)


# =============================================================================
# Configuration
# =============================================================================

DNA_BASES = {"A", "C", "G", "T"}
COMPLEMENT = str.maketrans("ACGT", "TGCA")

DEFAULT_TEMP_C = 60.0
DEFAULT_NA_MM = 50.0
DEFAULT_MG_MM = 3.5
DEFAULT_DNTP_MM = 0.6
DEFAULT_DNA_CONC_NM = 250.0

GLOBAL_DIMER_DG_THRESHOLD = -5.0
THREE_PRIME_DIMER_DG_THRESHOLD = -3.0
HAIRPIN_DG_THRESHOLD = -3.0

TERMINAL_POLY_GC_LENGTH = 3


# =============================================================================
# Request models
# =============================================================================

class ThermoConditions(BaseModel):
    temperature_c: float = Field(
        DEFAULT_TEMP_C,
        ge=0.0,
        le=100.0,
        description="Reaction or analysis temperature in °C.",
    )
    na_mM: float = Field(
        DEFAULT_NA_MM,
        ge=0.0,
        le=2000.0,
        description="Monovalent cation concentration in mM.",
    )
    mg_mM: float = Field(
        DEFAULT_MG_MM,
        ge=0.0,
        le=100.0,
        description="Total Mg2+ concentration in mM.",
    )
    dntp_mM: float = Field(
        DEFAULT_DNTP_MM,
        ge=0.0,
        le=100.0,
        description="Total dNTP concentration in mM.",
    )
    dna_conc_nM: float = Field(
        DEFAULT_DNA_CONC_NM,
        gt=0.0,
        le=100000.0,
        description="Primer strand concentration in nM.",
    )


class HomodimerRequest(ThermoConditions):
    sequence: str = Field(
        ...,
        min_length=2,
        example="AGATTCCTCTGCTTCAGAATTGGCATCT",
        description="DNA oligo sequence, written 5′→3′.",
    )
    include_viennarna_annotation: bool = Field(
        True,
        description=(
            "If true and ViennaRNA is installed, calculate an additional "
            "global DNA-cofold MFE and dot-bracket structure."
        ),
    )


class BatchHomodimerRequest(ThermoConditions):
    sequences: List[str] = Field(
        ...,
        min_length=1,
        max_length=10000,
        example=[
            "ACAGGATCACGTCCCTCCCC",
            "TGACTATAAGTCCTGGCGATTTGATGCA",
        ],
    )
    include_viennarna_annotation: bool = True


class HeterodimerRequest(ThermoConditions):
    sequence_1: str = Field(
        ...,
        min_length=2,
        example="AGATTCCTCTGCTTCAGAATTGGCATCT",
        description="First DNA oligo, written 5′→3′.",
    )
    sequence_2: str = Field(
        ...,
        min_length=2,
        example="TGACTATAAGTCCTGGCGATTTGATGCA",
        description="Second DNA oligo, written 5′→3′.",
    )


class PrimerPairRequest(ThermoConditions):
    forward_primer: str = Field(
        ...,
        min_length=2,
        example="AGATTCCTCTGCTTCAGAATTGGCATCT",
    )
    reverse_primer: str = Field(
        ...,
        min_length=2,
        example="TGACTATAAGTCCTGGCGATTTGATGCA",
    )
    include_viennarna_annotation: bool = False


# =============================================================================
# Response models
# =============================================================================

class ThermodynamicStructure(BaseModel):
    delta_g_kcal_mol: Optional[float] = None
    delta_h_kcal_mol: Optional[float] = None
    delta_s_cal_mol_k: Optional[float] = None
    melting_temp_c: Optional[float] = None
    structure_found: bool
    ascii_structure: Optional[str] = None


class ViennaRNAAnnotation(BaseModel):
    available: bool
    global_cofold_mfe_kcal_mol: Optional[float] = None
    mfe_structure: Optional[str] = None
    parameter_file_loaded: bool = False
    warning: Optional[str] = None


class SequenceQC(BaseModel):
    sequence: str
    length_nt: int
    gc_percent: float
    has_terminal_poly_gc_run: bool
    terminal_sequence: str


class HomodimerQCFlags(BaseModel):
    strong_global_dimer_risk: bool
    strong_three_prime_dimer_risk: bool
    strong_hairpin_risk: bool
    has_poly_gc_3prime: bool


class HomodimerResponse(BaseModel):
    sequence_qc: SequenceQC
    conditions: ThermoConditions
    engine: str
    model: str

    homodimer: ThermodynamicStructure
    three_prime_homodimer: ThermodynamicStructure
    hairpin: ThermodynamicStructure

    viennarna_global_cofold: Optional[ViennaRNAAnnotation] = None

    qc_flags: HomodimerQCFlags
    redesign_recommended: bool


class HeterodimerResponse(BaseModel):
    sequence_1_qc: SequenceQC
    sequence_2_qc: SequenceQC
    conditions: ThermoConditions
    engine: str
    model: str

    heterodimer: ThermodynamicStructure
    heterodimer_risk: bool
    redesign_recommended: bool


class PrimerPairResponse(BaseModel):
    forward_primer: HomodimerResponse
    reverse_primer: HomodimerResponse
    heterodimer: HeterodimerResponse

    pair_recommended: bool
    pair_qc_summary: List[str]


# =============================================================================
# Sequence helper functions
# =============================================================================

def normalize_dna(sequence: str) -> str:
    """
    Normalize an input oligo to uppercase DNA.

    Whitespace is removed. Uracil is converted to thymine so DNA/RNA-like
    input does not silently fail. Ambiguous IUPAC bases are deliberately
    rejected later because Primer3 dimer scoring is intended here for A/C/G/T.
    """
    return "".join(sequence.upper().split()).replace("U", "T")


def validate_dna(sequence: str, label: str = "Sequence") -> str:
    seq = normalize_dna(sequence)

    if len(seq) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"{label} must contain at least 2 nucleotides.",
        )

    invalid = sorted(set(seq) - DNA_BASES)
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label} contains unsupported characters: {', '.join(invalid)}. "
                "Only A, C, G, and T are accepted."
            ),
        )

    return seq


def calculate_gc_percent(sequence: str) -> float:
    return round(
        100.0 * sum(base in {"G", "C"} for base in sequence) / len(sequence),
        2,
    )


def has_terminal_poly_gc_run(
    sequence: str,
    run_length: int = TERMINAL_POLY_GC_LENGTH,
) -> bool:
    """
    Return True if the extreme 3′ terminus ends in C...C or G...G.

    Example with run_length=3:
      AGCTCCC -> True
      AGCTGGG -> True
      AGCTCCG -> False
    """
    if len(sequence) < run_length:
        return False

    tail = sequence[-run_length:]
    return tail == ("C" * run_length) or tail == ("G" * run_length)


def build_sequence_qc(sequence: str) -> SequenceQC:
    terminal_len = min(5, len(sequence))

    return SequenceQC(
        sequence=sequence,
        length_nt=len(sequence),
        gc_percent=calculate_gc_percent(sequence),
        has_terminal_poly_gc_run=has_terminal_poly_gc_run(sequence),
        terminal_sequence=sequence[-terminal_len:],
    )


# =============================================================================
# Primer3 wrappers
# =============================================================================

def ensure_primer3() -> None:
    if not HAS_PRIMER3:
        raise HTTPException(
            status_code=500,
            detail=(
                "Primer3 is not installed. Install primer3-py, for example: "
                "pip install primer3-py"
            ),
        )


def primer3_thermo_kwargs(conditions: ThermoConditions) -> dict:
    """
    Build common arguments for Primer3 thermodynamic functions.

    primer3-py uses:
      mv_conc  = monovalent cations, mM
      dv_conc  = divalent cations, mM
      dntp_conc = total dNTPs, mM
      dna_conc = oligo concentration, nM
      temp_c   = temperature, °C
    """
    return {
        "mv_conc": conditions.na_mM,
        "dv_conc": conditions.mg_mM,
        "dntp_conc": conditions.dntp_mM,
        "dna_conc": conditions.dna_conc_nM,
        "temp_c": conditions.temperature_c,
    }


def result_to_structure(result) -> ThermodynamicStructure:
    """
    Convert primer3-py's ThermoResult-like object to our API response.

    The exact attribute set can vary modestly with primer3-py version.
    getattr prevents an API crash if an optional attribute is unavailable.
    """
    dg = getattr(result, "dg", None)
    dh = getattr(result, "dh", None)
    ds = getattr(result, "ds", None)
    tm = getattr(result, "tm", None)
    structure_found = getattr(result, "structure_found", False)
    ascii_structure = getattr(result, "ascii_structure", None)

    return ThermodynamicStructure(
        delta_g_kcal_mol=round(dg / 1000.0, 2) if dg is not None else None,
        delta_h_kcal_mol=round(dh / 1000.0, 2) if dh is not None else None,
        delta_s_cal_mol_k=round(ds, 2) if ds is not None else None,
        melting_temp_c=round(tm, 2) if tm is not None else None,
        structure_found=bool(structure_found),
        ascii_structure=ascii_structure,
    )


def calc_homodimer(
    sequence: str,
    conditions: ThermoConditions,
) -> ThermodynamicStructure:
    """
    Calculate overall homodimer thermodynamics with Primer3 thal.

    This is the appropriate primary dimer score for DNA primer QC.
    """
    result = primer3.bindings.calc_homodimer(
        sequence,
        **primer3_thermo_kwargs(conditions),
    )
    return result_to_structure(result)


def calc_end_stability_homodimer(
    sequence: str,
    conditions: ThermoConditions,
) -> ThermodynamicStructure:
    """
    Calculate 3′-anchored homodimer thermodynamics with Primer3 thal.

    calc_end_stability specifically evaluates interactions relevant to
    extendable end-associated dimer formation.
    """
    result = primer3.bindings.calc_end_stability(
        sequence,
        sequence,
        **primer3_thermo_kwargs(conditions),
    )
    return result_to_structure(result)


def calc_hairpin(
    sequence: str,
    conditions: ThermoConditions,
) -> ThermodynamicStructure:
    """Calculate intramolecular hairpin thermodynamics with Primer3 thal."""
    result = primer3.bindings.calc_hairpin(
        sequence,
        **primer3_thermo_kwargs(conditions),
    )
    return result_to_structure(result)


def calc_heterodimer(
    sequence_1: str,
    sequence_2: str,
    conditions: ThermoConditions,
) -> ThermodynamicStructure:
    """Calculate two-primer cross-dimer thermodynamics with Primer3 thal."""
    result = primer3.bindings.calc_heterodimer(
        sequence_1,
        sequence_2,
        **primer3_thermo_kwargs(conditions),
    )
    return result_to_structure(result)


# =============================================================================
# Optional ViennaRNA annotation
# =============================================================================

def calculate_viennarna_global_cofold(
    sequence: str,
    temperature_c: float,
) -> ViennaRNAAnnotation:
    """
    Calculate global two-strand DNA cofold MFE with ViennaRNA.

    This value is intentionally reported separately from Primer3's
    thermodynamic-alignment dimer score. It is a global cofold metric,
    not a Benchling/Primer3-compatible dimer threshold value.
    """
    if not HAS_VIENNARNA:
        return ViennaRNAAnnotation(
            available=False,
            warning="ViennaRNA is not installed; no global cofold annotation was generated.",
        )

    parameter_loaded = False

    try:
        md = RNA.md()
        md.temperature = temperature_c

        base_dir = os.path.dirname(os.path.abspath(__file__))
        parameter_file = os.path.join(base_dir, "dna_mathews1999.par")

        if os.path.isfile(parameter_file):
            RNA.read_parameter_file(parameter_file)
            parameter_loaded = True

        fold_compound = RNA.fold_compound(f"{sequence}&{sequence}", md)
        structure, mfe = fold_compound.mfe_dimer()

        warning = None
        if not parameter_loaded:
            warning = (
                "dna_mathews1999.par was not found next to this application. "
                "ViennaRNA used its currently active/default energy parameters."
            )

        return ViennaRNAAnnotation(
            available=True,
            global_cofold_mfe_kcal_mol=round(mfe, 2),
            mfe_structure=structure,
            parameter_file_loaded=parameter_loaded,
            warning=warning,
        )

    except Exception as exc:
        return ViennaRNAAnnotation(
            available=True,
            warning=f"ViennaRNA global cofold annotation failed: {str(exc)}",
        )


# =============================================================================
# QC scoring
# =============================================================================

def is_strong_interaction(
    result: ThermodynamicStructure,
    threshold_kcal_mol: float,
) -> bool:
    """
    Negative ΔG values indicate more stable interactions.

    A result is only called risky if Primer3 found a structure and the
    reported ΔG is at or below the selected threshold.
    """
    return (
        result.structure_found
        and result.delta_g_kcal_mol is not None
        and result.delta_g_kcal_mol < threshold_kcal_mol
    )


def analyze_one_homodimer(
    sequence_input: str,
    conditions: ThermoConditions,
    include_viennarna_annotation: bool,
) -> HomodimerResponse:
    ensure_primer3()

    sequence = validate_dna(sequence_input)
    sequence_qc = build_sequence_qc(sequence)

    homodimer = calc_homodimer(sequence, conditions)
    three_prime_homodimer = calc_end_stability_homodimer(sequence, conditions)
    hairpin = calc_hairpin(sequence, conditions)

    global_risk = is_strong_interaction(
        homodimer,
        GLOBAL_DIMER_DG_THRESHOLD,
    )
    three_prime_risk = is_strong_interaction(
        three_prime_homodimer,
        THREE_PRIME_DIMER_DG_THRESHOLD,
    )
    hairpin_risk = is_strong_interaction(
        hairpin,
        HAIRPIN_DG_THRESHOLD,
    )

    qc_flags = HomodimerQCFlags(
        strong_global_dimer_risk=global_risk,
        strong_three_prime_dimer_risk=three_prime_risk,
        strong_hairpin_risk=hairpin_risk,
        has_poly_gc_3prime=sequence_qc.has_terminal_poly_gc_run,
    )

    viennarna_result = None
    if include_viennarna_annotation:
        viennarna_result = calculate_viennarna_global_cofold(
            sequence=sequence,
            temperature_c=conditions.temperature_c,
        )

    redesign = (
        global_risk
        or three_prime_risk
        or hairpin_risk
        or sequence_qc.has_terminal_poly_gc_run
    )

    return HomodimerResponse(
        sequence_qc=sequence_qc,
        conditions=conditions,
        engine="Primer3 thal; optional ViennaRNA cofold annotation",
        model=(
            "Primer3 DNA thermodynamic alignment for dimer/hairpin scores; "
            "ViennaRNA global DNA cofold MFE reported separately."
        ),
        homodimer=homodimer,
        three_prime_homodimer=three_prime_homodimer,
        hairpin=hairpin,
        viennarna_global_cofold=viennarna_result,
        qc_flags=qc_flags,
        redesign_recommended=redesign,
    )


# =============================================================================
# API endpoints
# =============================================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "primer3_available": HAS_PRIMER3,
        "viennarna_available": HAS_VIENNARNA,
        "service_version": "5.0.0",
    }


@app.post("/analyze", response_model=HomodimerResponse)
def analyze_homodimer(req: HomodimerRequest):
    """
    Analyze one oligo for homodimer, 3′-anchored dimer, and hairpin risk.
    """
    conditions = ThermoConditions(
        temperature_c=req.temperature_c,
        na_mM=req.na_mM,
        mg_mM=req.mg_mM,
        dntp_mM=req.dntp_mM,
        dna_conc_nM=req.dna_conc_nM,
    )

    try:
        return analyze_one_homodimer(
            sequence_input=req.sequence,
            conditions=conditions,
            include_viennarna_annotation=req.include_viennarna_annotation,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Homodimer evaluation failed: {str(exc)}",
        )


@app.post("/analyze_batch", response_model=List[HomodimerResponse])
def analyze_batch_homodimer(req: BatchHomodimerRequest):
    """
    Analyze a batch of oligos under shared reaction conditions.

    Invalid individual sequences return a 400 response for the full request.
    For very large batches, use a worker queue or split requests into chunks.
    """
    conditions = ThermoConditions(
        temperature_c=req.temperature_c,
        na_mM=req.na_mM,
        mg_mM=req.mg_mM,
        dntp_mM=req.dntp_mM,
        dna_conc_nM=req.dna_conc_nM,
    )

    try:
        return [
            analyze_one_homodimer(
                sequence_input=sequence,
                conditions=conditions,
                include_viennarna_annotation=req.include_viennarna_annotation,
            )
            for sequence in req.sequences
        ]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Batch homodimer evaluation failed: {str(exc)}",
        )


@app.post("/analyze_heterodimer", response_model=HeterodimerResponse)
def analyze_heterodimer(req: HeterodimerRequest):
    """
    Analyze the cross-dimer interaction between two primers.
    """
    ensure_primer3()

    sequence_1 = validate_dna(req.sequence_1, "sequence_1")
    sequence_2 = validate_dna(req.sequence_2, "sequence_2")

    conditions = ThermoConditions(
        temperature_c=req.temperature_c,
        na_mM=req.na_mM,
        mg_mM=req.mg_mM,
        dntp_mM=req.dntp_mM,
        dna_conc_nM=req.dna_conc_nM,
    )

    try:
        heterodimer = calc_heterodimer(sequence_1, sequence_2, conditions)
        heterodimer_risk = is_strong_interaction(
            heterodimer,
            GLOBAL_DIMER_DG_THRESHOLD,
        )

        return HeterodimerResponse(
            sequence_1_qc=build_sequence_qc(sequence_1),
            sequence_2_qc=build_sequence_qc(sequence_2),
            conditions=conditions,
            engine="Primer3 thal",
            model="Primer3 DNA thermodynamic alignment",
            heterodimer=heterodimer,
            heterodimer_risk=heterodimer_risk,
            redesign_recommended=heterodimer_risk,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Heterodimer evaluation failed: {str(exc)}",
        )


@app.post("/analyze_pair", response_model=PrimerPairResponse)
def analyze_primer_pair(req: PrimerPairRequest):
    """
    Analyze a forward/reverse primer pair.

    This endpoint evaluates:
      - forward-primer homodimer, 3′ dimer, and hairpin;
      - reverse-primer homodimer, 3′ dimer, and hairpin;
      - forward/reverse heterodimer.
    """
    conditions = ThermoConditions(
        temperature_c=req.temperature_c,
        na_mM=req.na_mM,
        mg_mM=req.mg_mM,
        dntp_mM=req.dntp_mM,
        dna_conc_nM=req.dna_conc_nM,
    )

    try:
        forward = analyze_one_homodimer(
            sequence_input=req.forward_primer,
            conditions=conditions,
            include_viennarna_annotation=req.include_viennarna_annotation,
        )

        reverse = analyze_one_homodimer(
            sequence_input=req.reverse_primer,
            conditions=conditions,
            include_viennarna_annotation=req.include_viennarna_annotation,
        )

        forward_sequence = validate_dna(req.forward_primer, "forward_primer")
        reverse_sequence = validate_dna(req.reverse_primer, "reverse_primer")

        heterodimer_structure = calc_heterodimer(
            forward_sequence,
            reverse_sequence,
            conditions,
        )

        heterodimer_risk = is_strong_interaction(
            heterodimer_structure,
            GLOBAL_DIMER_DG_THRESHOLD,
        )

        heterodimer = HeterodimerResponse(
            sequence_1_qc=build_sequence_qc(forward_sequence),
            sequence_2_qc=build_sequence_qc(reverse_sequence),
            conditions=conditions,
            engine="Primer3 thal",
            model="Primer3 DNA thermodynamic alignment",
            heterodimer=heterodimer_structure,
            heterodimer_risk=heterodimer_risk,
            redesign_recommended=heterodimer_risk,
        )

        summary = []

        if forward.redesign_recommended:
            summary.append("Forward primer failed one or more individual QC criteria.")

        if reverse.redesign_recommended:
            summary.append("Reverse primer failed one or more individual QC criteria.")

        if heterodimer_risk:
            summary.append("Forward/reverse heterodimer ΔG exceeded the configured risk threshold.")

        if not summary:
            summary.append("No configured homodimer, 3′-dimer, hairpin, or heterodimer risk was detected.")

        pair_recommended = not (
            forward.redesign_recommended
            or reverse.redesign_recommended
            or heterodimer_risk
        )

        return PrimerPairResponse(
            forward_primer=forward,
            reverse_primer=reverse,
            heterodimer=heterodimer,
            pair_recommended=pair_recommended,
            pair_qc_summary=summary,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Primer-pair evaluation failed: {str(exc)}",
        )
