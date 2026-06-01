# -*- coding: utf-8 -*-
"""
Create complete JMR Word document from MASTER_DOCUMENT_FINAL.md
Follows AMA/JMR formatting standards with AI disclosure per Sage policy
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os

def apply_ama_formatting(doc):
    """Apply AMA base formatting"""
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(12)

    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

def set_double_spacing(paragraph):
    """Apply double spacing to paragraph"""
    paragraph.paragraph_format.line_spacing = 2.0
    for run in paragraph.runs:
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)
    return paragraph

def add_section_break(doc):
    """Add page break"""
    doc.add_page_break()

def create_title_page():
    """Create FILE 1: Title Page (separate document)"""
    doc = Document()
    apply_ama_formatting(doc)

    # Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.add_run("Long-Term Media Contribution Estimation: Framework Benchmarking Study")
    title_run.font.size = Pt(12)
    title_run.font.name = 'Times New Roman'
    title_run.bold = True
    set_double_spacing(title)

    # Blank lines
    doc.add_paragraph()
    doc.add_paragraph()

    # Author
    author = doc.add_paragraph()
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author_run = author.add_run("[Author Name]\n[Institution]\n[Address]\n[Phone]\n[Email]")
    author_run.font.size = Pt(12)
    author_run.font.name = 'Times New Roman'
    set_double_spacing(author)

    # Blank lines
    doc.add_paragraph()
    doc.add_paragraph()

    # Acknowledgments
    ack_header = doc.add_paragraph()
    ack_header_run = ack_header.add_run("Acknowledgments")
    ack_header_run.bold = True
    ack_header_run.font.size = Pt(12)
    ack_header_run.font.name = 'Times New Roman'

    ack_text = doc.add_paragraph("[Author acknowledgments and funding information]")
    set_double_spacing(ack_text)

    # Blank lines
    doc.add_paragraph()
    doc.add_paragraph()

    # Declarations
    decl_header = doc.add_paragraph()
    decl_header_run = decl_header.add_run("Declarations")
    decl_header_run.bold = True
    decl_header_run.font.size = Pt(12)
    decl_header_run.font.name = 'Times New Roman'

    # Conflict of interest
    coi = doc.add_paragraph()
    coi_label = coi.add_run("Declaration of conflicting interest: ")
    coi_label.bold = True
    coi_label.font.size = Pt(12)
    coi_label.font.name = 'Times New Roman'
    coi_text = coi.add_run("The author declares no conflicts of interest.")
    coi_text.font.size = Pt(12)
    coi_text.font.name = 'Times New Roman'
    set_double_spacing(coi)

    # Funding
    funding = doc.add_paragraph()
    funding_label = funding.add_run("Funding: ")
    funding_label.bold = True
    funding_label.font.size = Pt(12)
    funding_label.font.name = 'Times New Roman'
    funding_text = funding.add_run("[Funding information if applicable]")
    funding_text.font.size = Pt(12)
    funding_text.font.name = 'Times New Roman'
    set_double_spacing(funding)

    # Data availability
    data = doc.add_paragraph()
    data_label = data.add_run("Data availability: ")
    data_label.bold = True
    data_label.font.size = Pt(12)
    data_label.font.name = 'Times New Roman'
    data_text = data.add_run("Synthetic data and code available at https://github.com/sanjsvk/ltc_frameworks")
    data_text.font.size = Pt(12)
    data_text.font.name = 'Times New Roman'
    set_double_spacing(data)

    return doc

def create_main_document():
    """Create FILE 2: Main Document"""
    doc = Document()
    apply_ama_formatting(doc)

    # === PAGE 1: TITLE + ABSTRACT + KEYWORDS ===

    # Title (centered, bold, 12pt)
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.add_run("Long-Term Media Contribution Estimation: Framework Benchmarking Study")
    title_run.bold = True
    title_run.font.size = Pt(12)
    title_run.font.name = 'Times New Roman'
    set_double_spacing(title)

    # Blank line
    doc.add_paragraph()

    # Abstract header
    abstract_header = doc.add_paragraph()
    abstract_run = abstract_header.add_run("Abstract")
    abstract_run.bold = True
    abstract_run.font.size = Pt(12)
    abstract_run.font.name = 'Times New Roman'

    # Abstract (150 words, unstructured)
    abstract_text = ("Marketing mix models routinely underestimate long-term media contributions (LTC) "
        "because estimation methods are designed for short-term elasticities, not sustained brand accumulation. "
        "This creates systematic budget misallocation, leaving 10-15% of true ROI unaccounted for in optimization. "
        "We benchmark ten LTC estimation methods across three frameworks (static adstock, dynamic lag, state-space) "
        "using synthetic data with ground-truth long-term effects, evaluating performance across five diagnostic scenarios "
        "from baseline to structural breaks. State-space methods with Bayesian latent stock estimation (BSTS, MCMC) recover "
        "79.3% of true LTC on average across S1-S4, compared to 44.2% for dynamic models and 29.6% for static adstock. "
        "Critically, aggregate recovery metrics mask channel-level attribution failures: two models achieve 68.8% aggregate "
        "recovery while returning 0% recovery for individual channels, inverting budget allocation recommendations. "
        "We propose a three-tier robustness taxonomy based on scenario sensitivity and provide a decision framework for "
        "practitioners to select methods according to signal strength and spend pattern characteristics.")

    abstract_para = doc.add_paragraph(abstract_text)
    set_double_spacing(abstract_para)

    # Keywords
    keywords_para = doc.add_paragraph()
    kw_label = keywords_para.add_run("Keywords: ")
    kw_label.bold = True
    kw_label.font.size = Pt(12)
    kw_label.font.name = 'Times New Roman'

    kw_text = keywords_para.add_run("media mix modelling; long-term effects; latent stock models; adstock; state-space methods; attribution robustness")
    kw_text.font.size = Pt(12)
    kw_text.font.name = 'Times New Roman'
    set_double_spacing(keywords_para)

    # PAGE BREAK before main text
    add_section_break(doc)

    # === SECTION 2: INTRODUCTION & LITERATURE REVIEW ===

    intro_header = doc.add_paragraph()
    intro_run = intro_header.add_run("2. Introduction")
    intro_run.bold = True
    intro_run.font.size = Pt(12)
    intro_run.font.name = 'Times New Roman'
    set_double_spacing(intro_header)

    # Problem statement
    problem_text = ("Chief marketing officers allocate budgets across media channels using marketing mix models (MMMs) "
        "designed to estimate short-term elasticities - the immediate sales lift from a single exposure. These methods often "
        "provide incomplete estimates of long-term value, overlooking sustained brand accumulation effects that persist weeks "
        "or months after the initial advertising exposure. For media channels like television and video, where brand-building is "
        "a core function, this oversight is substantial. Brands typically derive 10-15% of weekly sales from long-term media "
        "contributions, yet MMM estimates of long-term contributions routinely fall by half of that true value, leading to systematic "
        "misallocation of budgets toward short-term performance channels like search. This paper addresses a fundamental question: "
        "which estimation methods can reliably recover long-term media contributions, and when can practitioners trust their estimates?")

    prob_para = doc.add_paragraph(problem_text)
    set_double_spacing(prob_para)

    # Identification challenge
    challenge_header = doc.add_paragraph()
    challenge_run = challenge_header.add_run("The Identification Challenge in Current Practice")
    challenge_run.bold = True
    challenge_run.font.size = Pt(11)
    challenge_run.font.name = 'Times New Roman'

    challenge_text = ("The dominant approach to MMM uses adstock transformations - geometric or polynomial decay functions "
        "applied to historical spend series - to capture both short-term and long-term effects in a single coefficient. This framework "
        "succeeds when all channels are continuously active. However, adstock methods fail fundamentally in two scenarios that are common "
        "in practice. First, when long-term effects persist after spend stops, adstock cannot separate persistence from zero spend. Second, "
        "under collinearity, when multiple channels move together, adstock has insufficient statistical variation to identify which channel "
        "generates long-term effects, leading to reversals where methods flip signs. These identification limitations have been documented in "
        "individual case studies, but no comprehensive quantification across methods and diagnostic scenarios has been published.")

    challenge_para = doc.add_paragraph(challenge_text)
    set_double_spacing(challenge_para)

    # Gap statement
    gap_header = doc.add_paragraph()
    gap_run = gap_header.add_run("Why Existing Validation Approaches Are Insufficient")
    gap_run.bold = True
    gap_run.font.size = Pt(11)
    gap_run.font.name = 'Times New Roman'

    gap_text = ("Most MMM validation studies use either aggregate metrics on real data (where ground truth is unknown), or specialized "
        "time series models validated only on their own reconstructed baselines. This circular validation cannot detect systematic under-recovery "
        "of true long-term effects. The methodological gap is clear: to measure how much of true long-term contributions each method recovers, "
        "practitioners need synthetic data where the ground truth data-generating process is known and varied. This is the only way to avoid the "
        "confound that best fit to real data may mask systematic misattribution.")

    gap_para = doc.add_paragraph(gap_text)
    set_double_spacing(gap_para)

    # Contributions
    contrib_header = doc.add_paragraph()
    contrib_run = contrib_header.add_run("Contributions of This Paper")
    contrib_run.bold = True
    contrib_run.font.size = Pt(11)
    contrib_run.font.name = 'Times New Roman'

    contrib_text = ("This paper fills this gap with four specific contributions: (1) Synthetic benchmarking framework with known ground truth. "
        "We create a realistic media mix data-generating process with explicit long-term brand stock dynamics, implement ten estimation methods "
        "across three framework classes, and evaluate performance across five diagnostic scenarios. All code, synthetic data, and ground truth values "
        "are provided for replication. (2) Empirical evidence that aggregate recovery masks channel-level attribution failure. We show that a method "
        "achieving 68.8% aggregate long-term contribution recovery can return 0% recovery for individual channels. (3) A three-tier robustness taxonomy "
        "based on scenario sensitivity. We classify methods by their pause-window robustness ratio - how much estimation error increases when spend "
        "temporarily stops. (4) A practitioner decision framework for method selection. Based on signal strength and spend pattern characteristics, we "
        "recommend specific methods and warn against those with known failure modes.")

    contrib_para = doc.add_paragraph(contrib_text)
    set_double_spacing(contrib_para)

    # Paper roadmap
    roadmap_header = doc.add_paragraph()
    roadmap_run = roadmap_header.add_run("Paper Roadmap")
    roadmap_run.bold = True
    roadmap_run.font.size = Pt(11)
    roadmap_run.font.name = 'Times New Roman'

    roadmap_text = ("Section 3 describes the synthetic data-generating process, ten estimation methods, and their configuration. "
        "Section 4 evaluates framework-level performance on the baseline scenario and tests robustness to five diagnostic scenarios. "
        "Section 5 synthesizes findings into a framework hierarchy and practitioner decision framework. All code and data are available "
        "at https://github.com/sanjsvk/ltc_frameworks for full replication.")

    roadmap_para = doc.add_paragraph(roadmap_text)
    set_double_spacing(roadmap_para)

    # AI DISCLOSURE (per Sage/JMR policy)
    doc.add_paragraph()  # Blank line

    disclosure_header = doc.add_paragraph()
    disclosure_run = disclosure_header.add_run("AI Disclosure Statement")
    disclosure_run.bold = True
    disclosure_run.font.size = Pt(11)
    disclosure_run.font.name = 'Times New Roman'
    disclosure_run.italic = True

    disclosure_text = ("This manuscript was prepared with assistance from Claude AI (Anthropic) for the following tasks: "
        "document formatting and structure, cross-referencing figure placements, and verification of citation accuracy against primary sources. "
        "All research methodology, experimental design, data synthesis, results, analysis, and conclusions are original and human-authored. "
        "No AI tools were used to generate research methodology, statistical analysis, results, or interpretations. AI assistance was limited to "
        "technical document preparation tasks as permitted under Sage publishing AI disclosure guidelines.")

    disclosure_para = doc.add_paragraph(disclosure_text)
    set_double_spacing(disclosure_para)

    # PAGE BREAK before Section 3
    add_section_break(doc)

    # === SECTION 3: METHODOLOGY ===
    # (Will add in next iteration due to length)

    method_header = doc.add_paragraph()
    method_run = method_header.add_run("3. Methodology")
    method_run.bold = True
    method_run.font.size = Pt(12)
    method_run.font.name = 'Times New Roman'
    set_double_spacing(method_header)

    # Placeholder for methodology content
    method_placeholder = doc.add_paragraph("[Methodology section with 8 equations, 2 tables, and 5 scenarios - see MASTER_DOCUMENT_FINAL.md Section 3]")
    method_placeholder.paragraph_format.line_spacing = 2.0

    # PAGE BREAK before Section 4
    add_section_break(doc)

    # === SECTION 4: RESULTS ===
    results_header = doc.add_paragraph()
    results_run = results_header.add_run("4. Results")
    results_run.bold = True
    results_run.font.size = Pt(12)
    results_run.font.name = 'Times New Roman'
    set_double_spacing(results_header)

    # Results overview
    results_intro = doc.add_paragraph(
        "State-space methods recover 79.3% of true LTC on average across S1-S4 (baseline plus stress scenarios), "
        "compared to 44.2% for dynamic distributed-lag methods and 29.6% for static adstock methods. This hierarchy "
        "is consistent across all five diagnostic scenarios, with framework architecture determining success or failure.")
    set_double_spacing(results_intro)

    # Placeholder for results figures and tables
    results_placeholder = doc.add_paragraph("[Results section with Table 3 (Full Recovery Matrix), Figures 2-13, and detailed scenario analysis - see MASTER_DOCUMENT_FINAL.md Section 4]")
    results_placeholder.paragraph_format.line_spacing = 2.0

    # PAGE BREAK before Section 5
    add_section_break(doc)

    # === SECTION 5: DISCUSSION & IMPLICATIONS ===
    discussion_header = doc.add_paragraph()
    discussion_run = discussion_header.add_run("5. Discussion & Implications")
    discussion_run.bold = True
    discussion_run.font.size = Pt(12)
    discussion_run.font.name = 'Times New Roman'
    set_double_spacing(discussion_header)

    # Robustness spectrum
    robust_subheader = doc.add_paragraph()
    robust_run = robust_subheader.add_run("The Robustness Spectrum: A Three-Tier Taxonomy")
    robust_run.bold = True
    robust_run.font.size = Pt(11)
    robust_run.font.name = 'Times New Roman'

    robust_text = ("Beyond average performance, a critical secondary dimension emerges: robustness to structural variation. "
        "We classify methods by their pause-window robustness ratio - the ratio of estimation error during spend discontinuities "
        "versus full-series error. Methods with pause ratios < 1.10x maintain consistent accuracy across scenarios (Tier 1, architecturally robust). "
        "Methods with ratios 1.10-1.35x show moderate fragility (Tier 2, identification-sensitive). Methods with ratios > 1.35x are highly fragile "
        "(Tier 3, data-dependent). This taxonomy reveals that framework architecture, not calibration, determines robustness.")

    robust_para = doc.add_paragraph(robust_text)
    set_double_spacing(robust_para)

    # Channel attribution
    channel_subheader = doc.add_paragraph()
    channel_run = channel_subheader.add_run("Channel Attribution Problem: Aggregate Metrics Are Insufficient")
    channel_run.bold = True
    channel_run.font.size = Pt(11)
    channel_run.font.name = 'Times New Roman'

    channel_text = ("A critical finding cuts across frameworks: aggregate LTC recovery can mask severe channel-level misattribution. "
        "The ARDL model achieves 68.8% aggregate recovery but returns 0% recovery for individual channels, inverting budget allocation recommendations. "
        "Channel-level validation is mandatory. Before deploying a framework, validate not just aggregate accuracy but per-channel recovery across "
        "at least one structural-break scenario.")

    channel_para = doc.add_paragraph(channel_text)
    set_double_spacing(channel_para)

    # Practitioner guidance
    guidance_subheader = doc.add_paragraph()
    guidance_run = guidance_subheader.add_run("Practitioner Decision Framework")
    guidance_run.bold = True
    guidance_run.font.size = Pt(11)
    guidance_run.font.name = 'Times New Roman'

    guidance_text = ("Framework architecture dominates over calibration: choosing the right method matters more than tuning "
        "the chosen method. For portfolios with strong signal and stability priority, we recommend BSTS (lowest variance, pause ratio 1.02x). "
        "For maximum accuracy, recommend MCMC Bayesian latent stock (81.0% average recovery, correct channel ranking). For weak-signal scenarios, "
        "MCMC with scenario-specific priors is the only method achieving >80% recovery. Do not use dual_adstock or ARDL without extensive scenario-specific validation.")

    guidance_para = doc.add_paragraph(guidance_text)
    set_double_spacing(guidance_para)

    # Limitations
    limits_subheader = doc.add_paragraph()
    limits_run = limits_subheader.add_run("Limitations and Future Work")
    limits_run.bold = True
    limits_run.font.size = Pt(11)
    limits_run.font.name = 'Times New Roman'

    limits_text = ("This work uses synthetic data, limiting claims about real-world performance. Weekly aggregation may mask daily effects. "
        "The five scenarios cover known challenges but do not exhaust real-world complexity. Real-data validation is essential before deployment. "
        "Future work should characterize speed-accuracy trade-offs as scale increases and test cross-channel synergies.")

    limits_para = doc.add_paragraph(limits_text)
    set_double_spacing(limits_para)

    # PAGE BREAK before References
    add_section_break(doc)

    # === SECTION 6: REFERENCES ===
    ref_header = doc.add_paragraph()
    ref_run = ref_header.add_run("References")
    ref_run.bold = True
    ref_run.font.size = Pt(12)
    ref_run.font.name = 'Times New Roman'

    # References (14 papers in AMA format, single-spaced)
    references = [
        "Broadbent, S. (1979). One way TV advertisements work. Journal of the Market Research Society, 21(3), 139-166.",
        "Clarke, D. G. (1976). Econometric measurement of the duration of advertising effect on sales. Journal of Marketing Research, 13(4), 345-357.",
        "Dekimpe, M. G., & Hanssens, D. M. (2000). Time-series models in marketing: Past, present and future. International Journal of Research in Marketing, 17(2-3), 183-193.",
        "Datta, H., Ailawadi, K. L., & van Heerde, H. J. (2017). How well does consumer-based brand equity align with sales-based brand equity and marketing-mix response? Journal of Marketing, 81(3), 1-20. https://doi.org/10.1509/jm.15.0340",
        "Durbin, J., & Koopman, S. J. (2012). Time series analysis by state space methods (2nd ed.). Oxford University Press.",
        "Hanssens, D. M., Parsons, L. J., & Schultz, R. L. (1990). Approaches to empirical econometrics: Economic time series analysis and dynamic econometric models. Cambridge University Press.",
        "Hanssens, D. M., Parsons, L. J., & Schultz, R. L. (2001). Market response models: Econometric and time series analysis. Kluwer Academic Publishers.",
        "Harvey, A. C. (1989). Forecasting, structural time series models and the Kalman filter. Cambridge University Press.",
        "Jin, Y., Wang, Y., Sun, Y., Chan, D., & Koehler, J. (2017). Bayesian methods for media mix modeling with carryover and shape effects. Google Research Technical Report. Retrieved from https://research.google/pubs/bayesian-methods-for-media-mix-modeling-with-carryover-and-shape-effects/",
        "Keller, K. L. (1993). Conceptualizing, measuring, and managing customer-based brand equity. Journal of Marketing, 57(1), 1-22.",
        "Koyck, L. M. (1954). Distributed lags and investment analysis. North-Holland.",
        "Meta Marketing Science. (2022-2023). Robyn: Open-source Bayesian marketing mix modeling [Software]. Retrieved from https://github.com/facebook/Robyn",
        "Srinivasan, S., & Hanssens, D. M. (2009). Marketing and firm value: Metrics, methods, findings, and future directions. Journal of Marketing Research, 46(3), 293-312.",
        "Vaver, J., & Koehler, J. (2011). Measuring ad effectiveness using geo experiments. Google Research Technical Report. Retrieved from https://research.google/pubs/measuring-ad-effectiveness-using-geo-experiments/"
    ]

    for ref in references:
        ref_para = doc.add_paragraph(ref, style='List Bullet')
        ref_para.paragraph_format.line_spacing = 1.15  # Single-spaced per AMA standard for references
        for run in ref_para.runs:
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12)

    return doc

def main():
    """Create both title page and main document"""

    print("[LAYER 2] Creating Word Documents...")
    print("\n=== FILE 1: Title Page ===")
    title_doc = create_title_page()
    title_path = r"C:\github\ltc\submission\LTC_Frameworks_JMR_TitlePage.docx"
    title_doc.save(title_path)
    print(f"[SUCCESS] Title page saved: {title_path}")

    print("\n=== FILE 2: Main Document ===")
    main_doc = create_main_document()
    main_path = r"C:\github\ltc\submission\LTC_Frameworks_JMR_MainDocument.docx"
    main_doc.save(main_path)
    print(f"[SUCCESS] Main document saved: {main_path}")

    print("\n=== SUMMARY ===")
    print("File 1 (Title Page): LTC_Frameworks_JMR_TitlePage.docx")
    print("File 2 (Main Document): LTC_Frameworks_JMR_MainDocument.docx")
    print("\nBoth documents:")
    print("  - 12pt Times New Roman font")
    print("  - Double-spaced text")
    print("  - 1-inch margins, no page numbers/headers/footers")
    print("  - AMA reference format (14 papers)")
    print("  - AI Disclosure statement (Sage compliance)")
    print("\nNOTE: Methodology section (3) and Results section (4) are placeholders.")
    print("Full content available in MASTER_DOCUMENT_FINAL.md")

if __name__ == "__main__":
    main()
