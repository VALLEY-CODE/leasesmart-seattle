import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
from datetime import datetime
import io

# -----------------------------
# App config + styling
# -----------------------------
st.set_page_config(
    page_title="LeaseSmart Seattle",
    page_icon="🏙️",
    layout="centered"
)

st.markdown(
    """
    <style>
      .risk-card {padding: 18px; border-radius: 14px; border: 1px solid rgba(0,0,0,0.08);}
      .tiny {opacity: 0.75; font-size: 0.9rem;}
      .pill {display:inline-block; padding:4px 10px; border-radius: 999px; border:1px solid rgba(0,0,0,0.12); margin-right:6px; margin-top:6px;}
      .section {margin-top: 14px;}
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Constants + links
# -----------------------------
RESOURCE_LINKS = {
    "Opening a Business in Seattle – Step-by-Step Guide":
    "https://www.seattle.gov/economic-development/start-a-business/how-to-start-a-business",

    "Seattle SDCI – Permit Coaching for Small Businesses":
    "https://www.seattle.gov/economic-development/grow-a-business/consulting-services/commercial-space-permit-coaching",

    "Seattle Permit & Property Records (Lookup Tool)":
    "https://services.seattle.gov/Portal/Customization/SEATTLE/welcome.aspx",

    "Seattle Office of Economic Development":
    "https://www.seattle.gov/economicdevelopment",

    "Seattle Open Data Portal":
    "https://data.seattle.gov"
}

BUSINESS_TYPES = [
    "Restaurant / Café",
    "Retail",
    "Salon / Personal Services",
    "Office",
    "Light Manufacturing / Maker Space",
    "Childcare / Learning Center",
    "Gym / Fitness Studio",
    "Other"
]

# A simple mapping from "business type" -> typical occupancy/use label in data
# (This is just for demo. You can refine later.)
USE_MAPPING = {
    "Restaurant / Café": ["Restaurant", "Cafe", "Food", "Bar"],
    "Retail": ["Retail", "Store", "Shop"],
    "Salon / Personal Services": ["Salon", "Personal Services", "Barber"],
    "Office": ["Office"],
    "Light Manufacturing / Maker Space": ["Light Industrial", "Maker", "Studio", "Workshop"],
    "Childcare / Learning Center": ["Childcare", "School", "Learning"],
    "Gym / Fitness Studio": ["Gym", "Fitness", "Recreation"],
    "Other": []
}

# -----------------------------
# Helpers
# -----------------------------
def normalize(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().lower().replace(",", "").split())

def best_address_match(address_input: str, df: pd.DataFrame, limit: int = 5):
    """
    Fuzzy match the user-entered address to the demo dataset.
    Returns: (best_row, best_score, candidates_df)
    """
    address_input_n = normalize(address_input)
    if not address_input_n:
        return None, 0, pd.DataFrame()

    # Compare normalized input against raw addresses (good enough for demo)
    choices = df["address"].astype(str).tolist()
    matches = process.extract(
        address_input_n,
        choices,
        scorer=fuzz.WRatio,
        limit=limit
    )

    rows = []
    for addr, score, idx in matches:
        r = df.iloc[idx].to_dict()
        r["match_score"] = score
        rows.append(r)

    candidates = pd.DataFrame(rows).sort_values("match_score", ascending=False)
    if candidates.empty:
        return None, 0, candidates

    best = candidates.iloc[0]
    best_row = df[df["property_id"] == best["property_id"]].iloc[0]
    return best_row, int(best["match_score"]), candidates


def use_matches_business_type(certificate_use: str, business_type: str) -> bool:
    cert = normalize(certificate_use)
    targets = USE_MAPPING.get(business_type, [])
    if business_type == "Other":
        return False
    if not cert:
        return False
    for t in targets:
        if normalize(t) in cert:
            return True
    return False


def compute_risk(row: pd.Series, business_type: str, wants_alcohol: bool, wants_kitchen_hood: bool):
    """
    Explainable, rule-based risk scoring.
    Returns a dict with: risk_label, risk_emoji, score, reasons[], flags[], approvals[]
    """
    score = 0
    reasons = []
    flags = []
    approvals = []

    # Certificate of occupancy / existing use
    cert_use = str(row.get("certificate_use", ""))
    cert_ok = use_matches_business_type(cert_use, business_type)

    if cert_ok:
        reasons.append("The location’s existing Certificate of Occupancy appears compatible with your business type (lower change-of-use risk).")
    else:
        score += 2
        flags.append("Change of use / occupancy may be required")
        approvals.append("Confirm Certificate of Occupancy and whether a change-of-use permit is needed")
        reasons.append("The location’s existing Certificate of Occupancy may NOT match your business type, which can trigger a change-of-use process and upgrades.")

    # Historic district
    is_hist = bool(row.get("is_historic", False))
    if is_hist:
        score += 2
        flags.append("Historic district / special review likely")
        approvals.append("Check if Historic/Special Review applies to signage, storefront, exterior changes")
        reasons.append("This address is flagged as being in or near a historic/special review area, which can add review steps and cost/time.")
    else:
        reasons.append("This address is not flagged as historic in the demo dataset (fewer preservation constraints).")

    # Zoning (coarse demo logic)
    zoning = str(row.get("zoning", ""))
    zoning_n = normalize(zoning)
    if "commercial" in zoning_n or "mixed" in zoning_n:
        reasons.append("Zoning appears compatible with typical business activity (still verify exact use category).")
    else:
        score += 3
        flags.append("Zoning may limit business uses")
        approvals.append("Verify zoning and whether your specific use is permitted at this address")
        reasons.append("Zoning may not be aligned with typical commercial uses, increasing the risk of restrictions or needing special approvals.")

    # Food / alcohol signals (demo)
    if business_type == "Restaurant / Café":
        if wants_kitchen_hood:
            score += 1
            flags.append("Commercial kitchen ventilation/hood requirements")
            approvals.append("Confirm whether hood/grease duct/suppression upgrades are required")
            reasons.append("A commercial kitchen can trigger ventilation/hood/fire suppression requirements depending on cooking type.")
        else:
            reasons.append("A limited food concept (no hood) can reduce build-out complexity compared to full cooking operations.")

    if wants_alcohol:
        score += 1
        flags.append("Alcohol service licensing + possible code implications")
        approvals.append("Plan for alcohol licensing; confirm occupancy/egress impacts if needed")
        reasons.append("Serving alcohol adds licensing steps and may affect occupancy/egress requirements depending on the layout and use.")

    # Permit history (soft flag)
    permit_year = row.get("last_major_permit_year", None)
    try:
        permit_year = int(permit_year)
    except Exception:
        permit_year = None

    current_year = datetime.now().year
    if permit_year and (current_year - permit_year) >= 12:
        score += 1
        flags.append("Older permit history (possible hidden upgrade needs)")
        approvals.append("Ask about older building systems (electrical, plumbing, ADA) and required upgrades")
        reasons.append("This building’s last major permit in the demo data is older, which can correlate with more upgrade needs when you remodel.")
    else:
        reasons.append("Recent permit activity (in the demo data) may indicate some upgrades were done previously (still confirm).")

    # Convert score to label
    if score <= 1:
        risk_label = "Low"
        risk_emoji = "🟢"
    elif score <= 4:
        risk_label = "Moderate"
        risk_emoji = "🟡"
    else:
        risk_label = "High"
        risk_emoji = "🔴"

    return {
        "score": score,
        "risk_label": risk_label,
        "risk_emoji": risk_emoji,
        "reasons": reasons,
        "flags": flags,
        "approvals": approvals,
        "cert_ok": cert_ok
    }


def build_report(address_input, business_type, match_score, row, assessment):
    lines = []
    lines.append("LeaseSmart Seattle — 'Know Before You Sign' Report")
    lines.append("=" * 55)
    lines.append(f"Input address: {address_input}")
    lines.append(f"Matched address: {row.get('address','')} (match score: {match_score}%)")
    lines.append(f"Neighborhood: {row.get('neighborhood','')}")
    lines.append(f"Business type: {business_type}")
    lines.append("")
    lines.append(f"Risk assessment: {assessment['risk_emoji']} {assessment['risk_label']} (score: {assessment['score']})")
    lines.append("")
    lines.append("Key flags:")
    if assessment["flags"]:
        for f in assessment["flags"]:
            lines.append(f"- {f}")
    else:
        lines.append("- No major flags detected in this demo dataset (still verify with City resources).")
    lines.append("")
    lines.append("What to check before signing:")
    if assessment["approvals"]:
        for a in assessment["approvals"]:
            lines.append(f"- {a}")
    else:
        lines.append("- Verify exact permitted use, occupancy, and required permits.")
    lines.append("")
    lines.append("Plain-language explanation:")
    for r in assessment["reasons"]:
        lines.append(f"- {r}")
    lines.append("")
    lines.append("Free resources:")
    for name, url in RESOURCE_LINKS.items():
        lines.append(f"- {name}: {url}")
    lines.append("")
    lines.append("Disclaimer: This is a prototype decision-support tool, not legal advice. Always confirm details with SDCI/OED resources.")
    return "\n".join(lines)


# -----------------------------
# Demo data (embedded to avoid CSV permission issues)
# -----------------------------
@st.cache_data
def load_demo_data() -> pd.DataFrame:
    csv_text = """property_id,address,neighborhood,certificate_use,zoning,is_historic,permit_notes,last_major_permit_year
P-001,504 S King St,Chinatown-International District,Restaurant,Commercial,TRUE,"Older building; prior tenant had kitchen hood installed",2018
P-002,1520 2nd Ave,Downtown,Retail,Commercial,FALSE,"Recent interior remodel",2022
P-003,900 Jackson St,Chinatown-International District,Office,Mixed,TRUE,"Historic district; storefront changes may require review",2016
P-004,2800 NW Market St,Ballard,Retail,Commercial,FALSE,"No major permit history shown in demo",2010
P-005,1101 E Yesler Way,Yesler Terrace,Restaurant,Mixed,FALSE,"Change of use may trigger upgrades; verify occupancy",2014
P-006,600 1st Ave S,Pioneer Square,Warehouse,Industrial,TRUE,"Historic area; verify allowed uses for tenant improvements",2009
"""
    df = pd.read_csv(io.StringIO(csv_text))
    df["is_historic"] = df["is_historic"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    return df

df = load_demo_data()


# -----------------------------
# UI
# -----------------------------
st.title("🏙️ LeaseSmart Seattle")
st.write("**Know before you sign.** Enter an address, choose your business type, and get a plain-language risk check plus next steps.")

with st.expander("What this prototype does (for judges)"):
    st.write(
        "- Flags potential risks *before* lease signing (certificate-of-occupancy mismatch, historic review, zoning conflicts).\n"
        "- Generates a simple checklist of what to verify.\n"
        "- Directs owners to free City resources.\n\n"
        "**Note:** Demo dataset is intentionally small. The value is the workflow + explainable logic."
    )

# Inputs
st.markdown("### Step 1 — Your business + location")
business_type = st.selectbox("Business type", BUSINESS_TYPES)
address_input = st.text_input("Property address", placeholder="Example: 504 S King St")

colA, colB = st.columns(2)
with colA:
    wants_kitchen_hood = st.checkbox(
        "Requires cooking hood (restaurant only)",
        value=False,
        help="If you expect heavy cooking, hood requirements may apply."
    )
with colB:
    wants_alcohol = st.checkbox("Plans to serve alcohol", value=False)

st.markdown("### Step 2 — Analyze")
analyze = st.button("Analyze location", type="primary")

if analyze:
    if not address_input.strip():
        st.error("Please enter an address.")
        st.stop()

    best_row, best_score, candidates = best_address_match(address_input, df)

    if best_row is None or best_score < 70:
        st.warning("We couldn’t confidently match that address in the demo dataset.")
        if not candidates.empty:
            st.write("Closest matches in demo data:")
            st.dataframe(
                candidates[["address", "neighborhood", "certificate_use", "zoning", "is_historic", "match_score"]],
                use_container_width=True
            )
        st.info("Tip: For the demo, try: 504 S King St, 900 Jackson St, 1520 2nd Ave")
        st.stop()

    assessment = compute_risk(best_row, business_type, wants_alcohol, wants_kitchen_hood)

    # Risk card
    st.markdown("### Results")
    st.markdown(
        f"""
        <div class="risk-card">
          <div class="tiny">Matched address</div>
          <h3 style="margin-top:6px;">{best_row.get('address','')} — {best_row.get('neighborhood','')}</h3>
          <div class="tiny">Certificate use: <b>{best_row.get('certificate_use','')}</b> &nbsp;|&nbsp;
          Zoning: <b>{best_row.get('zoning','')}</b> &nbsp;|&nbsp;
          Historic: <b>{'Yes' if bool(best_row.get('is_historic', False)) else 'No'}</b></div>
          <hr style="border:none;border-top:1px solid rgba(0,0,0,0.08); margin:14px 0;">
          <div class="tiny">Risk assessment</div>
          <h2 style="margin:6px 0 0 0;">{assessment['risk_emoji']} {assessment['risk_label']} Risk</h2>
          <div class="tiny">Explainable score: {assessment['score']} &nbsp;|&nbsp; Address match confidence: {best_score}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Flags
    st.markdown('<div class="section"></div>', unsafe_allow_html=True)
    st.subheader("Key renovation / review flags")
    if assessment["flags"]:
        for f in assessment["flags"]:
            st.markdown(f"<span class='pill'>⚑ {f}</span>", unsafe_allow_html=True)
    else:
        st.success("No major flags detected in this demo dataset. Still verify with City resources.")

    # Plain-language explanation
    st.markdown('<div class="section"></div>', unsafe_allow_html=True)
    st.subheader("Plain-language explanation")
    for r in assessment["reasons"]:
        st.write(f"- {r}")

    # Checklist
    st.markdown('<div class="section"></div>', unsafe_allow_html=True)
    st.subheader("Before you sign — checklist")
    if assessment["approvals"]:
        for a in assessment["approvals"]:
            st.checkbox(a, value=False)
    else:
        st.checkbox("Verify permitted use, occupancy, and required permits", value=False)

    # Resource links
    st.markdown('<div class="section"></div>', unsafe_allow_html=True)
    st.subheader("Free City resources")
    for name, url in RESOURCE_LINKS.items():
        st.write(f"- [{name}]({url})")

    # Download report
    report_text = build_report(address_input, business_type, best_score, best_row, assessment)
    st.download_button(
        label="Download 'Before You Sign' Report (TXT)",
        data=report_text,
        file_name="LeaseSmartSeattle_Report.txt",
        mime="text/plain"
    )

    st.caption("Disclaimer: Prototype decision-support tool. Not legal advice. Confirm details with SDCI/OED resources.")

# Footer
st.markdown("---")
st.caption("Demo dataset is embedded for reliability in demos. You can swap in real city datasets later.")

# To run: python -m streamlit run .\streamlit_app.py
#Test1 with high risk: Restaurant / Café, 900 Jackson St,(Click on options)
#Test2 with low risk: Retail / Retail, 1520 2nd Ave
# How
# elevate the fuzzmatching


#  Add alcohol license
#  Improve the webpage to match the city of seattle webpage style:
# ChatGPT Prompt:
#
# Update the layout, design, colors, fonts and any additional design elements to match the url below for City of Seattle.
#
# URL: <add the URL here>
#
# Feel free to add more detail and specifications. The more context you provide the prompt the better the return and more precise changes to the code