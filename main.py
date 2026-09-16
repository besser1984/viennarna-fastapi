import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import RNA
    HAS_VIENNARNA = True
except ImportError:
    HAS_VIENNARNA = False


APP_VERSION = "6.0.0"

DEFAULT_TEMPERATURE_C = 60.0
DEFAULT_NA_MM = 50.0
DEFAULT_MG_MM = 3.5
DEFAULT_DNTP_MM = 0.6

GLOBAL_DIMER_RISK_THRESHOLD_KCAL_MOL = -5.0
THREE_PRIME_DIMER_RISK_THRESHOLD_KCAL_MOL = -3.0
TERMINAL_POLY_GC_RUN_LENGTH = 3

DNA_BASES = {"A", "C", "G", "T"}

BASE_DIR = Path(__file__).resolve().parent
DNA_PARAMETER_FILE = BASE_DIR / "dna_mathews1999.par"


app = FastAPI(
    title="DNA Homodimer Screening Microservice",
    description=(
        "High-throughput DNA oligo homodimer screening with ViennaRNA global "
        "DNA cofold MFE. This endpoint reports a global two-strand cofold energy "
        "and must not be interpreted as an exact Benchling local Min ΔG Homodimer value."
    ),
    version=APP_VERSION,
)


# =============================================================================
# Pydantic models
# =============================================================================

class ThermoConditions(BaseModel):
    temperature_c: float = Field(
        DEFAULT_TEMPERATURE_C,
        ge=0.0,
        le=100.0,
        description="Analysis temperature in °C.",
    )
    na_mM: float = Field(
        DEFAULT_NA_MM,
        ge=0.0,
        le=2000.0,
        description=(
            "Monovalent salt input retained for metadata. "
            "It is not applied as a custom mixed-salt correction by this ViennaRNA endpoint."
        ),
    )
    mg_mM: float = Field(
        DEFAULT_MG_MM,
        ge=0.0,
        le=100.0,
        description=(
            "Mg2+ input retained for metadata. "
            "It is not applied as a custom mixed-salt correction by this ViennaRNA endpoint."
        ),
    )
    dntp_mM: float = Field(
        DEFAULT_DNTP_MM,
        ge=0.0,
        le=100.0,
        description=(
            "Total dNTP input retained for metadata. "
            "It is not applied as a custom mixed-salt correction by this ViennaRNA endpoint."
        ),
    )


class HomodimerRequest(ThermoConditions):
    sequence: str = Field(
        ...,
        min_length=2,
        max_length=500,
        example="ACAGGATCACGTCCCTCCCC",
        description="DNA oligo, written 5′→3′. Only A, C, G, and T are accepted.",
    )


class BatchHomodimerRequest(ThermoConditions):
    sequences: List[str] = Field(
        ...,
        min_length=1,
        max_length=5000,
        example=[
            "ACAGGATCACGTCCCTCCCC",
            "TGACTATAAGTCCTGGCGATTTGATGCA",
        ],
        description="List of DNA oligos, each written 5′→3′.",
    )


class SequenceQC(BaseModel):
    sequence: str
    length_nt: int
    gc_percent: float
    terminal_sequence: str
    has_terminal_poly_gc_run: bool


class HomodimerQCFlags(BaseModel):
    global_dimer_risk: bool
    extreme_3prime_base_paired: bool
    potential_three_prime_dimer_risk: bool
    has_poly_gc_3prime: bool


class ViennaRNASettings(BaseModel):
    library_available: bool
    parameter_file_path: str
    parameter_file_loaded: bool
    parameter_set: Optional[str] = None
    warning: Optional[str] = None


class HomodimerResponse(BaseModel):
    sequence_qc: SequenceQC
    conditions: ThermoConditions

    engine: str
    model: str

    global_cofold_mfe_kcal_mol: float
    mfe_structure: str

    qc_flags: HomodimerQCFlags
    redesign_recommended: bool

    viennarna_settings: ViennaRNASettings


# =============================================================================
# Sequence functions
# =============================================================================

def normalize_dna(sequence: str) -> str:
    """
    Convert to uppercase DNA and remove whitespace.

    RNA U is converted to T. Inputs with ambiguity bases such as N, R, Y,
    or non-nucleotide characters are rejected by validate_dna().
    """
    return "".join(sequence.upper().split()).replace("U", "T")


def validate_dna(sequence: str, label: str = "sequence") -> str:
    seq = normalize_dna(sequence)

    if len(seq) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"{label} must contain at least two nucleotides.",
        )

    invalid_bases = sorted(set(seq) - DNA_BASES)
    if invalid_bases:
        invalid_text = ", ".join(invalid_bases)
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label} contains unsupported characters: {invalid_text}. "
                "Only standard DNA bases A, C, G, and T are accepted."
            ),
        )

    return seq


def calculate_gc_percent(sequence: str) -> float:
    gc_count = sequence.count("G") + sequence.count("C")
    return round((100.0 * gc_count) / len(sequence), 2)


def has_terminal_poly_gc_run(
    sequence: str,
    run_length: int = TERMINAL_POLY_GC_RUN_LENGTH,
) -> bool:
    """
    Detect exactly terminal C or G homopolymers.

    Examples:
      AGCTCCC -> True
      AGCTGGG -> True
      AGCTCCG -> False
      AGCTCCA -> False
    """
    if len(sequence) < run_length:
        return False

    terminal_bases = sequence[-run_length:]
    return terminal_bases in {
        "C" * run_length,
        "G" * run_length,
    }


def build_sequence_qc(sequence: str) -> SequenceQC:
    terminal_length = min(5, len(sequence))

    return SequenceQC(
        sequence=sequence,
        length_nt=len(sequence),
        gc_percent=calculate_gc_percent(sequence),
        terminal_sequence=sequence[-terminal_length:],
        has_terminal_poly_gc_run=has_terminal_poly_gc_run(sequence),
    )


# =============================================================================
# ViennaRNA functions
# =============================================================================

def require_viennarna() -> None:
    if not HAS_VIENNARNA:
        raise HTTPException(
            status_code=500,
            detail=(
                "ViennaRNA is not installed. Install it in the active virtual "
                "environment with: python -m pip install viennarna"
            ),
        )


def load_dna_parameters() -> ViennaRNASettings:
    """
    Load the DNA Mathews 1999 parameters.

    The parameter file must be located beside main.py. The function reloads
    the parameter file on every request to avoid stale/default RNA parameters.
    For a very large production workload, move this into a controlled startup
    function after validating thread/process behavior for your deployment model.
    """
    if not DNA_PARAMETER_FILE.is_file():
        return ViennaRNASettings(
            library_available=HAS_VIENNARNA,
            parameter_file_path=str(DNA_PARAMETER_FILE),
            parameter_file_loaded=False,
            warning=(
                "Required DNA parameter file was not found. "
                "Expected: dna_mathews1999.par beside main.py. "
                "No MFE calculation was performed."
            ),
        )

    try:
        RNA.read_parameter_file(str(DNA_PARAMETER_FILE))

        return ViennaRNASettings(
            library_available=True,
            parameter_file_path=str(DNA_PARAMETER_FILE),
            parameter_file_loaded=True,
            parameter_set="Mathews 1999 DNA",
        )

    except Exception as exc:
        return ViennaRNASettings(
            library_available=True,
            parameter_file_path=str(DNA_PARAMETER_FILE),
            parameter_file_loaded=False,
            warning=f"DNA parameter loading failed: {str(exc)}",
        )


def terminal_base_is_paired(mfe_structure: str) -> bool:
    """
    Return True when the extreme 3′ nucleotide of either sequence is paired
    in the reported global MFE dot-bracket structure.

    This is a structural flag only. It is not an independent 3′-dimer ΔG.
    """
    strands = mfe_structure.split("&")

    if len(strands) != 2 or not strands[0] or not strands[1]:
        return False

    first_strand, second_strand = strands
    return first_strand[-1] in "()" or second_strand[-1] in "()"


def calculate_global_homodimer_mfe(
    sequence: str,
    conditions: ThermoConditions,
) -> tuple[float, str, ViennaRNASettings]:
    """
    Return ViennaRNA global MFE for sequence&sequence.

    The returned ΔG is the complete two-strand global cofold MFE. It is not
    necessarily equal to a local dimer score reported by another software tool.
    """
    require_viennarna()

    settings = load_dna_parameters()

    if not settings.parameter_file_loaded:
        raise HTTPException(
            status_code=500,
            detail=settings.warning,
        )

    try:
        md = RNA.md()
        md.temperature = conditions.temperature_c

        duplex_sequence = f"{sequence}&{sequence}"
        fold_compound = RNA.fold_compound(duplex_sequence, md)

        mfe_structure, mfe_energy = fold_compound.mfe_dimer()

        return round(float(mfe_energy), 2), mfe_structure, settings

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"ViennaRNA cofold calculation failed: {str(exc)}",
        )


# =============================================================================
# Single-oligo analysis
# =============================================================================

def analyze_one_homodimer(
    sequence_input: str,
    conditions: ThermoConditions,
) -> HomodimerResponse:
    sequence = validate_dna(sequence_input)
    sequence_qc = build_sequence_qc(sequence)

    global_mfe, structure, vienna_settings = calculate_global_homodimer_mfe(
        sequence=sequence,
        conditions=conditions,
    )

    three_prime_paired = terminal_base_is_paired(structure)

    global_dimer_risk = global_mfe < GLOBAL_DIMER_RISK_THRESHOLD_KCAL_MOL

    potential_three_prime_risk = (
        three_prime_paired
        and global_mfe < THREE_PRIME_DIMER_RISK_THRESHOLD_KCAL_MOL
    )

    redesign_recommended = (
        global_dimer_risk
        or potential_three_prime_risk
        or sequence_qc.has_terminal_poly_gc_run
    )

    return HomodimerResponse(
        sequence_qc=sequence_qc,
        conditions=conditions,
        engine="ViennaRNA",
        model=(
            "Mathews 1999 DNA energy parameters; global two-strand cofold MFE. "
            "Na+, Mg2+, and dNTP inputs are retained as reaction metadata and "
            "are not used for an unvalidated custom mixed-salt correction."
        ),
        global_cofold_mfe_kcal_mol=global_mfe,
        mfe_structure=structure,
        qc_flags=HomodimerQCFlags(
            global_dimer_risk=global_dimer_risk,
            extreme_3prime_base_paired=three_prime_paired,
            potential_three_prime_dimer_risk=potential_three_prime_risk,
            has_poly_gc_3prime=sequence_qc.has_terminal_poly_gc_run,
        ),
        redesign_recommended=redesign_recommended,
        viennarna_settings=vienna_settings,
    )


# =============================================================================
# API endpoints
# =============================================================================

@app.get("/health")
def health() -> dict:
    """
    Report package and DNA-parameter availability.
    """
    parameter_file_exists = DNA_PARAMETER_FILE.is_file()

    return {
        "status": "ok",
        "service_version": APP_VERSION,
        "viennarna_available": HAS_VIENNARNA,
        "dna_parameter_file_expected_path": str(DNA_PARAMETER_FILE),
        "dna_parameter_file_exists": parameter_file_exists,
        "recommended_parameter_set": "Mathews 1999 DNA",
    }


@app.post("/analyze", response_model=HomodimerResponse)
def analyze_homodimer(req: HomodimerRequest) -> HomodimerResponse:
    """
    Calculate a global ViennaRNA DNA homodimer cofold MFE for one oligo.
    """
    conditions = ThermoConditions(
        temperature_c=req.temperature_c,
        na_mM=req.na_mM,
        mg_mM=req.mg_mM,
        dntp_mM=req.dntp_mM,
    )

    return analyze_one_homodimer(
        sequence_input=req.sequence,
        conditions=conditions,
    )


@app.post("/analyze_batch", response_model=List[HomodimerResponse])
def analyze_batch_homodimer(
    req: BatchHomodimerRequest,
) -> List[HomodimerResponse]:
    """
    Evaluate a list of oligos using shared experimental conditions.

    For very large projects, send batches in chunks and deploy multiple worker
    processes rather than submitting tens of thousands of oligos in one request.
    """
    conditions = ThermoConditions(
        temperature_c=req.temperature_c,
        na_mM=req.na_mM,
        mg_mM=req.mg_mM,
        dntp_mM=req.dntp_mM,
    )

    return [
        analyze_one_homodimer(
            sequence_input=sequence,
            conditions=conditions,
        )
        for sequence in req.sequences
    ]
