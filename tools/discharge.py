# tools/discharge.py
from typing import Dict, Any

def discharge_orchestrator(bill_text: str = "", hospital: str = "") -> Dict[str, Any]:
    checklist = [
        "Collect final bill and discharge summary signed by treating physician.",
        "Obtain pharmacy receipt and itemized list of medicines dispensed.",
        "Get OT notes and implant documentation (if any).",
        "Collect diagnostics and radiology reports on a CD/USB or PDF.",
        "Request a stamped, itemized invoice for insurance claims.",
        "Obtain doctor's contact details and follow-up instructions."
    ]
    # quick estimate of documents required
    docs = ["Final bill", "Discharge summary", "Pharmacy receipts", "Investigations (reports)", "ID proof", "Insurance card (if applicable)"]
    claim_email = (
        f"To: insurance@provider\nSubject: Claim submission for patient - {hospital}\n\n"
        f"Dear Claims Team,\n\nPlease find attached the final bill and supporting documents for claim processing. Kindly confirm the documents required and the expected timeline.\n\nRegards,\nPatient"
    )
    return {"status":"ok","checklist":checklist,"required_documents":docs,"claim_email":claim_email}
