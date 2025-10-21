# app/home.py
import streamlit as st
import  torch, json
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
from pathlib import Path
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt

# Optional PDF export (reportlab)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.units import mm
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

st.set_page_config(page_title="Pink October AI", page_icon="🎀", layout="centered")

# ── Pink theme + labels above inputs
st.markdown("""
<style>
  .stApp { background-color:#fff6fa; }
  h1, h2, h3, h4, h5, h6 { color:#e75480 !important; }
  .stTabs [data-baseweb="tab-list"] button[aria-selected="true"]{
    background:#ffd6e8 !important; border-bottom:3px solid #e75480 !important;
  }
  .wdbc-label { font-weight:700; color:#e75480; margin-bottom:2px; }
  .wdbc-normal { font-size:0.9rem; color:#c7376a; margin-bottom:6px; }
  div[data-testid="stMetricValue"]{ color:#e75480 !important; }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent           # app/
MODELS_DIR = ROOT.parent / "models"              # models/

st.title("🎀 Pink October AI — Parashikime me AI")
st.caption("Demonstrim edukativ — jo mjet diagnostik.")

# Tabs
tab_general, tab_wdbc, tab_busi = st.tabs([
    "Pyetje te pergjithshme",
    "🧮 Modeli Numerik (WDBC)",
    "🩻 Modeli me Ultratinguj (BUSI)",
])

# ─────────────────────────────────────────────────────────────
# Session state (persist results across tabs)
# ─────────────────────────────────────────────────────────────
st.session_state.setdefault("wdbc_inputs", {})
st.session_state.setdefault("wdbc_proba", None)
st.session_state.setdefault("wdbc_label", None)

st.session_state.setdefault("busi_pred", None)
st.session_state.setdefault("busi_conf", None)
st.session_state.setdefault("busi_image_bytes", None)

st.session_state.setdefault("bcsc_prob", None)
st.session_state.setdefault("bcsc_label", None)
st.session_state.setdefault("bcsc_inputs", {})
st.session_state.setdefault("bcsc_chart_png", None)

# ----------------------------------------------------------
# 1️⃣ WDBC MODEL (Numerical)
# ----------------------------------------------------------
with tab_wdbc:
    st.subheader("Parashikimi nga të dhënat diagnostikuese (Modeli Numerik)")
    st.write("Fushat mbeten në anglisht; **Normal** tregon vlera tipike të dataset-it (jo pragje klinike).")

    # Load model quietly
    try:
        bundle = joblib.load(MODELS_DIR / "wdbc_model.joblib")
        wdbc_model, wdbc_scaler = bundle["model"], bundle["scaler"]
        wdbc_ready = True
    except Exception:
        wdbc_ready = False
        wdbc_model = None
        wdbc_scaler = None

    # Helper: label + normal above input
    def field(title: str, normal: str, *, minv, maxv, value, step, key):
        st.markdown(f"<div class='wdbc-label'>{title}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='wdbc-normal'>Normal: {normal}</div>", unsafe_allow_html=True)
        return st.number_input(" ", min_value=minv, max_value=maxv, value=value, step=step,
                               key=key, label_visibility="collapsed")

    c1, c2 = st.columns(2)
    with c1:
        mean_radius = field("Mean Radius (mm)", "6–25",
                            minv=6.0,  maxv=30.0,  value=14.0, step=0.1,  key="mean_radius")
        mean_texture = field("Mean Texture", "9–40",
                             minv=5.0,  maxv=40.0,  value=20.0, step=0.1,  key="mean_texture")
        mean_perimeter = field("Mean Perimeter (mm)", "45–170",
                               minv=40.0, maxv=200.0, value=90.0, step=1.0, key="mean_perimeter")
        mean_area = field("Mean Area (mm²)", "100–2000",
                          minv=100.0, maxv=2500.0, value=600.0, step=10.0, key="mean_area")
        mean_smoothness = field("Mean Smoothness", "0.05–0.15",
                                minv=0.05, maxv=0.20, value=0.10, step=0.005, key="mean_smoothness")
    with c2:
        mean_compactness = field("Mean Compactness", "0.02–0.25",
                                 minv=0.01, maxv=0.40, value=0.10, step=0.005, key="mean_compactness")
        mean_concavity = field("Mean Concavity", "0.02–0.40",
                               minv=0.00, maxv=0.60, value=0.10, step=0.01, key="mean_concavity")
        mean_concave_points = field("Mean Concave Points", "0.01–0.20",
                                    minv=0.00, maxv=0.30, value=0.05, step=0.005, key="mean_concave_points")
        mean_symmetry = field("Mean Symmetry", "0.12–0.35",
                              minv=0.10, maxv=0.50, value=0.20, step=0.005, key="mean_symmetry")
        mean_fractal_dimension = field("Mean Fractal Dimension", "0.04–0.09",
                                       minv=0.03, maxv=0.15, value=0.06, step=0.001, key="mean_fractal_dimension")

    if st.button("💗 Parashiko", key="btn_wdbc"):
        if not wdbc_ready:
            st.info("Modeli numerik nuk u ngarkua.")
        else:
            X10 = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area,
                             mean_smoothness, mean_compactness, mean_concavity,
                             mean_concave_points, mean_symmetry, mean_fractal_dimension]])
            # Model trained on 30 features → pad 20 zeros
            X30 = np.pad(X10, ((0,0),(0,20)), mode="constant")
            Xs = wdbc_scaler.transform(X30)
            proba = float(wdbc_model.predict_proba(Xs)[0,1])

            st.metric("Probabiliteti i Malinjitetit", f"{proba:.1%}")
            if proba < 0.4:
                wdbc_label = "Beninj (jo kancerogjen)"
                st.success("💖 Prapësi Beninje — brenda kufijve tipikë.")
            elif proba < 0.7:
                wdbc_label = "Kufitare"
                st.warning("💞 Kufitare — disa vlera jashtë diapazonit tipik.")
            else:
                wdbc_label = "Malinj (i dyshuar)"
                st.error("💔 Modeli tregon tipare të mundshme malinje.")

            # Save to state for reports
            st.session_state.wdbc_inputs = {
                "Mean Radius (mm)": mean_radius,
                "Mean Texture": mean_texture,
                "Mean Perimeter (mm)": mean_perimeter,
                "Mean Area (mm²)": mean_area,
                "Mean Smoothness": mean_smoothness,
                "Mean Compactness": mean_compactness,
                "Mean Concavity": mean_concavity,
                "Mean Concave Points": mean_concave_points,
                "Mean Symmetry": mean_symmetry,
                "Mean Fractal Dimension": mean_fractal_dimension,
            }
            st.session_state.wdbc_proba = proba
            st.session_state.wdbc_label = wdbc_label

# ----------------------------------------------------------
# 2️⃣ BUSI MODEL (Ultrasound)
# ----------------------------------------------------------
with tab_busi:
    st.subheader("Ngarko një imazh me ultratinguj")
    st.write("Ngarko një imazh nga **BUSI dataset**. AI do ta klasifikojë si **Normal**, **Beninj**, ose **Malinj**.")

    # Load BUSI model quietly
    try:
        labels = json.load(open(MODELS_DIR / "busi_labels.json"))
        idx_to_class = {v: k for k, v in labels.items()}
        busi_model = models.efficientnet_b0(weights=None)
        busi_model.classifier[1] = nn.Linear(busi_model.classifier[1].in_features, len(labels))
        busi_model.load_state_dict(torch.load(MODELS_DIR / "busi_model.pth", map_location="cpu"))
        busi_model.eval()
        busi_ready = True
    except Exception:
        busi_ready = False
        busi_model, idx_to_class = None, None

    uploaded = st.file_uploader("Zgjidh një imazh", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("L")
        # keep a copy for report
        buff = BytesIO()
        img.save(buff, format="PNG")
        st.session_state.busi_image_bytes = buff.getvalue()

        st.image(img, caption="Imazhi i ngarkuar", use_container_width=True)

        tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        x = tf(img).unsqueeze(0)

        if not busi_ready:
            st.info("Modeli i ultratingullit nuk u ngarkua.")
        else:
            with torch.no_grad():
                probs = torch.softmax(busi_model(x), dim=1)[0].tolist()
            pred_idx = int(np.argmax(probs))
            pred_class = idx_to_class[pred_idx]
            conf = probs[pred_idx]

            color = "#e75480" if pred_class=="normal" else "#FFB6C1" if pred_class=="benign" else "#ff4f8b"
            st.markdown(
                f"<h3 style='color:{color};text-align:center;'>🎀 Rezultati: {pred_class.upper()} "
                f"({conf*100:.1f}% besueshmëri)</h3>", unsafe_allow_html=True
            )
            st.progress(conf)

            # store for report
            st.session_state.busi_pred = pred_class
            st.session_state.busi_conf = conf
    else:
        st.info("📤 Ju lutem ngarkoni një imazh për të filluar.")

# ----------------------------------------------------------
# 3️⃣ GENERAL MODEL (BCSC-like) — pink pie + pretty PDF
# ----------------------------------------------------------
with tab_general:
    st.header("🧠 Modeli i Stilit të Jetës - BCSC Risk Factors Dataset")
    st.write("Ky model përdor faktorë të rrezikut si mosha, historia familjare dhe stilin e jetës për të vlerësuar rrezikun e kancerit të gjirit.")

    # Info to include in the report
    colA, colB = st.columns(2)
    with colA:
        person_name = st.text_input("Emri (opsionale)", "")
        person_id   = st.text_input("ID / Kodi (opsionale)", "")
    with colB:
        person_age  = st.text_input("Mosha (opsionale)", "")
        extra_notes = st.text_area("Shënime (opsionale)", "")

    try:
        general = joblib.load(MODELS_DIR / "bcsc_general_model.joblib")
        model = general["model"]
        scaler = general["scaler"]
        features = general["features"]

        # Options for each feature (labels shown, numeric used)
        feature_options = {
            'age_group_5_years': {'Më pak se 40 vjeç': 0, '40-50 vjeç': 1, 'Mbi 50 vjeç': 2},
            'race_eth': {'Kaukaziane': 0, 'Afro-Amerikane': 1, 'Aziatike': 2, 'Hispanike': 3, 'Tjetër': 4},
            'first_degree_hx': {'Jo': 0, 'Po (histori familjare)': 1},
            'age_menarche': {'Para 12 vjeç': 0, '12-13 vjeç': 1, '14-15 vjeç': 2, 'Pas 15 vjeç': 3},
            'age_first_birth': {'Para 20 vjeç': 0, '20-25 vjeç': 1, '26-30 vjeç': 2, 'Mbi 30 ose pa fëmijë': 3},
            'BIRADS_breast_density': {'E ulët': 1, 'Mesatare': 2, 'E lartë': 3, 'E dendur shumë': 4},
            'current_hrt': {'Jo (hormone)': 0, 'Po': 1},
            'menopaus': {'Pre-menopauzë': 0, 'Post-menopauzë': 1, 'E pasigurt': 2},
            'bmi_group': {'Nën 18.5 (i dobët)': 0, '18.5-25 (normal)': 1, '25-30 (mbipeshë)': 2, 'Mbi 30 (obez)': 3},
            'biophx': {'Jo (histori biopsie)': 0, 'Po': 1}
        }

        feature_labels = {
            'age_group_5_years': 'Sa vjeç jeni?',
            'race_eth': 'Cila është etnia juaj?',
            'first_degree_hx': 'Histori familjare (nëna/motra/vajza)?',
            'age_menarche': 'Mosha e menstruacioneve të para?',
            'age_first_birth': 'Mosha e fëmijës së parë (ose pa fëmijë)?',
            'BIRADS_breast_density': 'Dendësia e gjirit (BIRADS)?',
            'current_hrt': 'Përdorim i hormoneve zëvendësuese?',
            'menopaus': 'Statusi menopauzal?',
            'bmi_group': 'Kategoria e BMI-së?',
            'biophx': 'Biopsi e mëparshme e gjirit?',
        }

        cols = st.columns(3)
        inputs = {}
        selections_human = {}
        for i, f in enumerate(features):    
            with cols[i % 3]:
                options_dict = feature_options.get(f, {'N/A': 0})
                options = list(options_dict.keys())
                label = feature_labels.get(f, f.replace('_', ' ').title())
                selected_label = st.selectbox(label, options=options, key=f)
                selections_human[f] = selected_label
                inputs[f] = options_dict[selected_label]

        # Predict
        pred_prob = None
        if st.button("💡 Parashiko (BCSC General)"):
            X = np.array([[inputs[f] for f in features]])
            X_scaled = scaler.transform(X)
            pred_prob = float(model.predict_proba(X_scaled)[0][1] * 100.0)  # %
            risk_threshold = 5.0
            is_high_risk = pred_prob > risk_threshold

            st.session_state.bcsc_prob = pred_prob
            st.session_state.bcsc_label = "Rrezik i lartë" if is_high_risk else "Rrezik i ulët"
            st.session_state.bcsc_inputs = selections_human

            if is_high_risk:
                st.error(f"🔮 Rezultati: **Rrezik i lartë** ({pred_prob:.1f}%)")
                st.warning("Sugjerim: Konsultohu me mjekun për mamografi/screening të avancuar.")
            else:
                st.success(f"🔮 Rezultati: **Rrezik i ulët** ({pred_prob:.1f}%)")
                st.info("Sugjerim: Vazhdo me screening rutinë çdo 1–2 vjet.")



                # ── Pink PIE chart: your risk vs remaining (0 - 11.5)
                fig, ax = plt.subplots(figsize=(4.8, 3.2))
                risk_value = float(pred_prob)
                max_possible = 11.5  # maksimumi i modelit

                # Siguro që rreziku nuk e kalon maksimumin
                risk_capped = min(risk_value, max_possible)
                remain = max_possible - risk_capped

                # Labels dhe ngjyrat
                labels_pie = [f"Rreziku juaj ({risk_capped:.2f})", f"Pjesa e mbetur ({remain:.2f})"]
                sizes = [risk_capped, remain]
                colors_pie = ["#e75480", "#ffd6e8"]  # pink + light pink
                explode = (0.06, 0.0)

                # Krijo chart   
                wedges, texts = ax.pie(
                    sizes, explode=explode, labels=labels_pie, autopct=None,
                    startangle=90, colors=colors_pie, textprops={"fontsize": 10}
                )
                ax.axis("equal")
                ax.set_title(f"Përqindja e rrezikut (0 - {max_possible})", fontsize=12)
                st.pyplot(fig)

                # Save chart to memory for PDF
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
                st.session_state.bcsc_chart_png = buf.getvalue()
                plt.close(fig)





        # ── Pretty PDF export for the General tab
        def build_pretty_pdf_bcsc(
            person_name: str, person_id: str, person_age: str, extra_notes: str,
            features, feature_labels, selections_human,
            bcsc_label: str, bcsc_prob: float, chart_png: bytes | None
        ) -> bytes:
            W, H = A4
            M = 22
            PINK = colors.HexColor("#e75480")
            PINK_LIGHT = colors.HexColor("#ffe4ee")
            PINK_SOFT = colors.HexColor("#ffd6e8")
            TEXT = colors.HexColor("#222222")
            ACCENT = colors.HexColor("#9c2e59")

            buf = BytesIO()
            c = canvas.Canvas(buf, pagesize=A4)

            # header banner
            c.setFillColor(PINK)
            c.roundRect(0, H-80, W, 80, 0, stroke=0, fill=1)
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 22)
            c.drawString(M, H-50, "Pink October AI — Raport (BCSC)")
            c.setFont("Helvetica", 11)
            c.drawString(M, H-68, datetime.now().strftime("Data: %Y-%m-%d  %H:%M"))
            y = H - 100

            def title(text):
                nonlocal y
                y -= 12
                c.setFillColor(PINK)
                c.setFont("Helvetica-Bold", 14)
                c.drawString(M, y, text)
                y -= 6
                c.setFillColor(ACCENT)
                c.setLineWidth(1)
                c.line(M, y, W-M, y)
                y -= 10
                c.setFillColor(TEXT)

            def info_card(pairs: list[tuple[str, str]], height=36):
                nonlocal y
                c.setFillColor(PINK_LIGHT)
                c.roundRect(M, y-height, W-2*M, height, 10, stroke=0, fill=1)
                c.setFillColor(TEXT)
                c.setFont("Helvetica", 11)
                x = M + 10
                yy = y - 12
                for k, v in pairs:
                    c.setFont("Helvetica-Bold", 11); c.drawString(x, yy, f"{k}:")
                    c.setFont("Helvetica", 11);      c.drawString(x+115, yy, v if v else "-")
                    yy -= 14
                y -= (height + 10)

            def result_card(label: str, prob: float):
                nonlocal y
                h = 72
                c.setFillColor(PINK_LIGHT)
                c.roundRect(M, y-h, W-2*M, h, 10, stroke=0, fill=1)
                c.setFillColor(PINK)
                c.setFont("Helvetica-Bold", 16)
                c.drawString(M+14, y-22, "Rezultati (BCSC)")
                c.setFont("Helvetica-Bold", 28)
                c.setFillColor(ACCENT)
                c.drawRightString(W-M-14, y-22, f"{prob:.1f}%")
                c.setFont("Helvetica-Bold", 12)
                c.setFillColor(TEXT)
                c.drawString(M+14, y-44, f"Vlerësim: {label}")
                y -= (h + 10)

            def factors_grid():
                nonlocal y
                title("Faktorët e zgjedhur")
                row_h = 18
                padding = 8
                col_w = (W - 2*M - 20) / 2
                content_h = min(14 + (len(features) * row_h) + padding, 220)
                c.setFillColor(colors.white)
                c.roundRect(M, y-content_h, W-2*M, content_h, 10, stroke=1, fill=1)
                c.setFillColor(TEXT)
                c.setFont("Helvetica", 10)
                yy = y - padding - 14
                x1 = M + 10
                x2 = M + 10 + col_w

                half = (len(features)+1)//2
                left_feats = features[:half]
                right_feats = features[half:]

                for f in left_feats:
                    name = feature_labels.get(f, f)
                    val  = selections_human.get(f, "-")
                    c.setFont("Helvetica-Bold", 10); c.drawString(x1, yy, f"{name}:")
                    c.setFont("Helvetica", 10);      c.drawRightString(M+10+col_w-5, yy, str(val))
                    yy -= row_h

                yy = y - padding - 14
                for f in right_feats:
                    name = feature_labels.get(f, f)
                    val  = selections_human.get(f, "-")
                    c.setFont("Helvetica-Bold", 10); c.drawString(x2, yy, f"{name}:")
                    c.setFont("Helvetica", 10);      c.drawRightString(W-M-10-5, yy, str(val))
                    yy -= row_h

                y -= (content_h + 12)

            def note_box(text: str):
                nonlocal y
                if not text:
                    return
                title("Shënime")
                import textwrap
                wrapper = textwrap.TextWrapper(width=92)
                lines = wrapper.wrap(text)
                h = min(160, 18 + len(lines)*13)
                c.setFillColor(colors.white)
                c.roundRect(M, y-h, W-2*M, h, 10, stroke=1, fill=1)
                c.setFillColor(TEXT)
                c.setFont("Helvetica", 11)
                yy = y - 20
                for line in lines:
                    c.drawString(M+12, yy, line)
                    yy -= 13
                    if yy < 80:
                        c.showPage(); y = H-100; yy = y - 20
                y -= (h + 12)

            def insert_chart(png: bytes | None):
                nonlocal y
                if not png:
                    return
                title("Grafiku i rrezikut (pajë rozë)")
                try:
                    img = ImageReader(BytesIO(png))
                    img_w, img_h = 360, 250
                    c.setFillColor(colors.white)
                    c.roundRect(M, y-img_h-20, W-2*M, img_h+20, 10, stroke=1, fill=1)
                    c.drawImage(img, M+12, y-img_h-12, width=img_w, height=img_h,
                                preserveAspectRatio=True, mask='auto')
                    y -= (img_h + 28)
                except Exception:
                    pass

            def footer():
                c.setFillColor(colors.HexColor("#9c2e59"))
                c.setFont("Helvetica", 8.7)
                c.drawCentredString(W/2, 18,
                  "Ky raport është demonstrim edukativ dhe NUK zëvendëson vlerësimin klinik. Konsultohu me një profesionist shëndetësor.")
                c.setFont("Helvetica", 8)
                c.drawRightString(W-16, 18, "Faqe 1")

            # Compose
            c.setFillColor(colors.HexColor("#ffd6e8"))
            c.roundRect(M, y-28, 140, 28, 14, stroke=0, fill=1)
            c.setFillColor(colors.HexColor("#9c2e59"))
            c.setFont("Helvetica-Bold", 11)
            c.drawCentredString(M+70, y-18, "Raport Vlerësimi")
            y -= (28 + 8)

            info_card([
                ("Emri", person_name),
                ("ID", person_id),
                ("Mosha", person_age),
            ], height=42)

            result_card(st.session_state.bcsc_label or "-", st.session_state.bcsc_prob or 0.0)
            insert_chart(st.session_state.get("bcsc_chart_png", None))
            factors_grid()
            note_box(extra_notes)

            footer()
            c.showPage()
            c.save()
            pdf = buf.getvalue()
            buf.close()
            return pdf

        # Show PDF button after a prediction exists
        if REPORTLAB_OK and st.session_state.bcsc_prob is not None:
            if st.button("📄 Gjenero Raport (PDF) "):
                pdf_bytes = build_pretty_pdf_bcsc(
                    person_name, person_id, person_age, extra_notes,
                    features, feature_labels, st.session_state.bcsc_inputs,
                    st.session_state.bcsc_label or "-", st.session_state.bcsc_prob or 0.0,
                    st.session_state.get("bcsc_chart_png", None)
                )
                st.download_button(
                    "⬇️ Shkarko Raportin (PDF)",
                    data=pdf_bytes,
                    file_name=f"PinkOctoberAI_BCSC_Raport_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf"
                )
                st.success("Raporti u gjenerua.")
    except Exception as e:
        st.error(f"Modeli BCSC General nuk u gjet ose s’mund të ngarkohet: {e}")
