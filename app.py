import os
import json
import re
import base64
import pdfplumber
from flask import Flask, request, jsonify, render_template
from anthropic import Anthropic

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def extract_from_image(image_bytes: bytes, media_type: str) -> list[dict]:
    """Use Claude vision to extract inventory data from handwritten image."""
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    prompt = """Ovo je ručno pisana lista zaliha pića u magacinu ("List dnevnog prometa pića").
Tabela ima sledeće kolone (mogu biti ponovljene u više blokova):
- Naziv jela i pića (naziv artikla)
- Jed. mere (jedinica mere: kom, ml, gr, br...)
- Preneta količina
- Stigla u smeni
- Zalihe na kraju dana (ovo je najvažnije - stvarno stanje)

Izvuci SVE stavke iz slike. Za svaku stavku vrati:
- "naziv": tačan naziv artikla (onako kako piše)
- "jed_mere": jedinica mere
- "zalihe_kraj": broj koji stoji u koloni "Zalihe na kraju dana" (može biti decimalan, može biti prazan - tada null)

Vrati SAMO JSON niz objekata, bez ikakvog drugog teksta.
Primer formata:
[
  {"naziv": "Espreso", "jed_mere": "kom", "zalihe_kraj": 23},
  {"naziv": "Votka", "jed_mere": "kom", "zalihe_kraj": 1000},
  {"naziv": "Smirnoff", "jed_mere": "ml", "zalihe_kraj": 2160}
]

Ako je polje prazno ili nečitljivo, stavi null za "zalihe_kraj".
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    text = response.content[0].text.strip()
    # Extract JSON array from response
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def extract_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """Extract bookkeeping data from PDF text (no table structure)."""
    import io

    # Known units of measure in the system
    UNITS = r"(?:kom|lit|kg|gr)"

    # Pattern: everything before the last <number> <unit> on the line is the name.
    # Using greedy (.*) so it matches the LAST number+unit pair.
    # Handles: decimals with comma or dot, negative numbers, optional trailing prices.
    LINE_RE = re.compile(
        r"^(.*?)\s+([-\d]+(?:[,\.]\d+)?)\s+(" + UNITS + r")\b",
        re.IGNORECASE,
    )

    items = []
    seen = set()

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                # Skip header and title lines
                if line.startswith("Aqua") or "stanja magacina" in line.lower():
                    continue
                if line.lower().startswith("naziv"):
                    continue
                # Skip totals line (only numbers)
                if re.match(r"^[\d,\.\s]+$", line):
                    continue

                m = LINE_RE.match(line)
                if not m:
                    continue

                naziv = m.group(1).strip()
                stanje_raw = m.group(2).strip()
                jm = m.group(3).strip().lower()

                if not naziv:
                    continue

                # Deduplicate (same name may appear on multiple pages via header repeat)
                key = naziv.lower()
                if key in seen:
                    continue
                seen.add(key)

                try:
                    stanje = float(stanje_raw.replace(",", "."))
                except ValueError:
                    stanje = None

                items.append({"naziv": naziv, "jm": jm, "stanje": stanje})

    return items


def normalize_name(name: str) -> str:
    """Normalize beverage name for fuzzy matching."""
    if not name:
        return ""
    n = name.lower().strip()
    # Remove common noise
    n = re.sub(r"\s+", " ", n)
    # Normalize common abbreviations
    replacements = {
        "dz": "dz",
        "đ": "dj",
        "č": "c",
        "ć": "c",
        "š": "s",
        "ž": "z",
        "lj": "lj",
        "nj": "nj",
    }
    for old, new in replacements.items():
        n = n.replace(old, new)
    return n


def find_match(naziv: str, pdf_items: list[dict]) -> dict | None:
    """Find best matching PDF item for a given name."""
    norm = normalize_name(naziv)
    best = None
    best_score = 0

    for item in pdf_items:
        item_norm = normalize_name(item["naziv"])
        # Exact match
        if norm == item_norm:
            return item
        # Check if all words of the shorter name appear in the longer
        words_a = set(norm.split())
        words_b = set(item_norm.split())
        if not words_a or not words_b:
            continue
        common = words_a & words_b
        score = len(common) / max(len(words_a), len(words_b))
        if score > best_score and score >= 0.5:
            best_score = score
            best = item

    return best


def reconcile(image_items: list[dict], pdf_items: list[dict]) -> list[dict]:
    """Compare physical inventory (image) with bookkeeping (PDF)."""
    results = []

    for img_item in image_items:
        naziv = img_item.get("naziv", "").strip()
        if not naziv:
            continue
        zalihe_kraj = img_item.get("zalihe_kraj")
        jed_mere = img_item.get("jed_mere", "")

        pdf_match = find_match(naziv, pdf_items)

        if pdf_match is None:
            status = "nije_u_knjig"
            stanje_knjig = None
            razlika = None
        else:
            stanje_knjig = pdf_match.get("stanje")
            if zalihe_kraj is None or stanje_knjig is None:
                status = "nedostaju_podaci"
                razlika = None
            else:
                fizicko = float(zalihe_kraj)
                knjig = float(stanje_knjig)
                jm_img = (jed_mere or "").strip().lower()
                jm_pdf = (pdf_match.get("jm") or "").strip().lower()

                # Unit conversion: ml <-> lit
                if jm_img in ("ml",) and jm_pdf in ("lit",):
                    fizicko = fizicko / 1000.0
                elif jm_img in ("lit",) and jm_pdf in ("ml",):
                    fizicko = fizicko * 1000.0

                razlika = round(fizicko - knjig, 3)
                tolerance = 0.05  # 50ml / 0.05 lit tolerance for measurement imprecision
                if abs(razlika) <= tolerance:
                    status = "ok"
                elif razlika > 0:
                    status = "visak"
                else:
                    status = "manjak"

        results.append(
            {
                "naziv": naziv,
                "jed_mere": jed_mere,
                "fizicko_stanje": zalihe_kraj,
                "knjig_stanje": stanje_knjig,
                "razlika": razlika,
                "naziv_knjig": pdf_match["naziv"] if pdf_match else None,
                "status": status,
            }
        )

    # Sort: discrepancies first, then OK
    order = {"manjak": 0, "visak": 1, "nije_u_knjig": 2, "nedostaju_podaci": 3, "ok": 4}
    results.sort(key=lambda x: (order.get(x["status"], 5), x["naziv"]))
    return results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/reconcile", methods=["POST"])
def api_reconcile():
    if "image" not in request.files or "pdf" not in request.files:
        return jsonify({"error": "Morate upload-ovati i sliku i PDF"}), 400

    image_file = request.files["image"]
    pdf_file = request.files["pdf"]

    image_bytes = image_file.read()
    pdf_bytes = pdf_file.read()

    # Detect image media type
    filename = image_file.filename.lower()
    if filename.endswith(".png"):
        media_type = "image/png"
    elif filename.endswith((".jpg", ".jpeg")):
        media_type = "image/jpeg"
    elif filename.endswith(".webp"):
        media_type = "image/webp"
    else:
        media_type = "image/jpeg"

    try:
        image_items = extract_from_image(image_bytes, media_type)
    except Exception as e:
        return jsonify({"error": f"Greška pri čitanju slike: {str(e)}"}), 500

    try:
        pdf_items = extract_from_pdf(pdf_bytes)
    except Exception as e:
        return jsonify({"error": f"Greška pri čitanju PDF-a: {str(e)}"}), 500

    results = reconcile(image_items, pdf_items)

    summary = {
        "ukupno": len(results),
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "manjak": sum(1 for r in results if r["status"] == "manjak"),
        "visak": sum(1 for r in results if r["status"] == "visak"),
        "nije_u_knjig": sum(1 for r in results if r["status"] == "nije_u_knjig"),
        "nedostaju_podaci": sum(1 for r in results if r["status"] == "nedostaju_podaci"),
    }

    return jsonify({"summary": summary, "items": results})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
