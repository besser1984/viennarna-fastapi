import RNA
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

class RNARequest(BaseModel):
    sequence: str = Field(..., example="GAGUCCUGUGAGAACUGUUGAGUAGAGUGUGAGCUCCCUG")

@app.get("/")
def health_check():
    return {"status": "ok", "service": "ViennaRNA API"}

def generate_svg_string(sequence: str, structure: str) -> str:
    """Generates a clean SVG plot string by extracting NAView coordinates."""
    n = len(sequence)
    coords_obj = RNA.naview_xy_coordinates(structure)
    
    # Safely extract X and Y values from ViennaRNA COORDINATE object
    x_coords = []
    y_coords = []
    
    for i in range(1, n + 1):  # ViennaRNA coordinate arrays are 1-indexed
        try:
            pt = coords_obj.get(i)
            x_coords.append(float(pt.X))
            y_coords.append(float(pt.Y))
        except AttributeError:
            # Fallback if get() returns a tuple or direct float
            x_coords.append(float(coords_obj.X[i]))
            y_coords.append(float(coords_obj.Y[i]))

    # Calculate bounding box
    margin = 40
    min_x, max_x = min(x_coords) - margin, max(x_coords) + margin
    min_y, max_y = min(y_coords) - margin, max(y_coords) + margin
    width = max_x - min_x
    height = max_y - min_y

    # Build base-pairing pairs from dot-bracket structure
    stack = []
    pairs = []
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                pairs.append((stack.pop(), i))

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{min_x} {min_y} {width} {height}" width="100%" height="100%">',
        '<style>',
        '  .bg { fill: #ffffff; }',
        '  .bond { stroke: #cbd5e1; stroke-width: 2; }',
        '  .backbone { stroke: #94a3b8; stroke-width: 1.5; stroke-dasharray: 2,2; }',
        '  .node { fill: #3b82f6; stroke: #1d4ed8; stroke-width: 1.5; }',
        '  .label { fill: #ffffff; font-family: monospace; font-size: 10px; font-weight: bold; text-anchor: middle; dominant-baseline: central; }',
        '</style>',
        f'<rect x="{min_x}" y="{min_y}" width="{width}" height="{height}" class="bg" />'
    ]

    # Draw backbone connection lines
    for i in range(n - 1):
        svg_lines.append(
            f'<line x1="{x_coords[i]}" y1="{y_coords[i]}" x2="{x_coords[i+1]}" y2="{y_coords[i+1]}" class="backbone" />'
        )

    # Draw base-pair hydrogen bonds
    for i, j in pairs:
        svg_lines.append(
            f'<line x1="{x_coords[i]}" y1="{y_coords[i]}" x2="{x_coords[j]}" y2="{y_coords[j]}" class="bond" />'
        )

    # Draw nucleotide nodes & text labels
    for i, nuc in enumerate(sequence):
        svg_lines.append(
            f'<circle cx="{x_coords[i]}" cy="{y_coords[i]}" r="8" class="node" />'
        )
        svg_lines.append(
            f'<text x="{x_coords[i]}" y="{y_coords[i]}" class="label">{nuc}</text>'
        )

    svg_lines.append('</svg>')
    return "\n".join(svg_lines)

@app.post("/analyze")
def analyze_rna(req: RNARequest):
    try:
        clean_seq = req.sequence.upper().replace('T', 'U').strip()
        
        # 1. Fold compound & MFE structure calculation
        fc = RNA.fold_compound(clean_seq)
        mfe_structure, mfe_energy = fc.mfe()

        # 2. Build SVG string directly in memory
        svg_content = generate_svg_string(clean_seq, mfe_structure)

        return {
            "sequence": clean_seq,
            "mfe_structure": mfe_structure,
            "mfe_energy_kcal": round(float(mfe_energy), 2),
            "svg_xml": svg_content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ViennaRNA Error: {str(e)}")
