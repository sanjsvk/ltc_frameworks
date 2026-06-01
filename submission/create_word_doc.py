# -*- coding: utf-8 -*-
"""Create Word document from MASTER_DOCUMENT_FINAL.md using AMA formatting"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
import re

def set_double_spacing(paragraph):
    """Set paragraph to double spacing"""
    paragraph.paragraph_format.line_spacing = 2.0
    return paragraph

def apply_ama_formatting(doc):
    """Apply base AMA formatting to document"""
    # Set default font
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(12)

    # Set margins to 1 inch
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        # Clear headers/footers
        section.header.is_linked_to_previous = False
        section.footer.is_linked_to_previous = False

def create_jmr_word_doc():
    """Create JMR-formatted Word document from master content"""

    doc = Document()
    apply_ama_formatting(doc)

    # TITLE PAGE (FILE 1 - SEPARATE)
    # Will be created separately

    # MAIN DOCUMENT (FILE 2)

    # Page 1: Title + Abstract + Keywords
    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_para.add_run("Long-Term Media Contribution Estimation:")
    title_run.font.size = Pt(12)
    title_run.font.name = 'Times New Roman'
    title_run.bold = True

    subtitle_para = doc.add_paragraph()
    subtitle_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_run = subtitle_para.add_run("Framework Benchmarking Study")
    subtitle_run.font.size = Pt(12)
    subtitle_run.font.name = 'Times New Roman'
    subtitle_run.bold = True

    # Blank line
    doc.add_paragraph()

    # Abstract header
    abstract_header = doc.add_paragraph()
    abstract_run = abstract_header.add_run("Abstract")
    abstract_run.font.size = Pt(12)
    abstract_run.font.name = 'Times New Roman'
    abstract_run.bold = True

    # Abstract text (150 words - unstructured single paragraph)
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
    for run in abstract_para.runs:
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)

    # Keywords
    keywords_para = doc.add_paragraph()
    kw_label = keywords_para.add_run("Keywords: ")
    kw_label.font.size = Pt(12)
    kw_label.font.name = 'Times New Roman'
    kw_label.bold = True

    kw_text = keywords_para.add_run("media mix modelling; long-term effects; latent stock models; adstock; state-space methods; attribution robustness")
    kw_text.font.size = Pt(12)
    kw_text.font.name = 'Times New Roman'

    set_double_spacing(keywords_para)

    # Page break before Section 2
    doc.add_page_break()

    # SECTION 2: INTRODUCTION & LITERATURE REVIEW
    section2_header = doc.add_paragraph()
    section2_run = section2_header.add_run("2. Introduction & Literature Review")
    section2_run.font.size = Pt(12)
    section2_run.font.name = 'Times New Roman'
    section2_run.bold = True
    set_double_spacing(section2_header)

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
    for run in prob_para.runs:
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)

    # Identification challenge
    challenge_header = doc.add_paragraph()
    challenge_header.add_run("2.1 The Identification Challenge in Current Practice")
    challenge_header.runs[0].font.size = Pt(12)
    challenge_header.runs[0].font.name = 'Times New Roman'
    challenge_header.runs[0].bold = True
    set_double_spacing(challenge_header)

    challenge_text = ("The dominant approach to MMM uses adstock transformations - geometric or polynomial decay functions "
    "applied to historical spend series - to capture both short-term and long-term effects in a single coefficient. This framework "
    "succeeds when all channels are continuously active: the adstock function can infer long-term persistence by observing how sales "
    "respond when one channel's spend fluctuates while others remain constant. However, adstock methods fail fundamentally in two scenarios "
    "that are common in practice. First, when long-term effects persist after spend stops - such as a spending pause to measure brand equity "
    "decay - adstock cannot separate persistence from zero spend, and the estimated coefficient becomes unreliable. Second, under collinearity, "
    "when multiple channels move together, adstock has insufficient statistical variation to identify which channel generates long-term effects, "
    "leading to reversals where methods flip the sign and magnitude of channel attribution across scenarios. These identification limitations "
    "have been well-documented in individual case studies, but no comprehensive quantification of their prevalence across methods and diagnostic "
    "scenarios has been published.")

    challenge_para = doc.add_paragraph(challenge_text)
    set_double_spacing(challenge_para)
    for run in challenge_para.runs:
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)

    # Save checkpoint
    doc.save(r"C:\github\ltc\submission\LTC_Frameworks_JMR_DRAFT_v1.docx")
    print("[PROGRESS] Section 1-2.1 complete. Saved to submission folder.")

    return doc

if __name__ == "__main__":
    doc = create_jmr_word_doc()
    print("[SUCCESS] Word document creation started")
