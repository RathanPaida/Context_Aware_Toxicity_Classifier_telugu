"""Generate an executive, professional 16:9 widescreen PPTX presentation
for the Context-Aware Toxicity Classifier project.
Incorporates the 100% verified 23-case stress testing results and loophole resolutions.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

def create_deck():
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Color Palette (High-Trust Research Dark Slate)
    BG_COLOR = RGBColor(11, 15, 25)        # #0B0F19 Deep Canvas
    CARD_BG = RGBColor(22, 30, 46)         # #161E2E Surface Card
    CARD_BORDER = RGBColor(45, 58, 82)     # #2D3A52 Border
    PRIMARY_BLUE = RGBColor(14, 165, 233)  # #0EA5E9 Sky Accent
    TEXT_WHITE = RGBColor(255, 255, 255)   # #FFFFFF Headers
    TEXT_MUTED = RGBColor(148, 163, 184)   # #94A3B8 Subtext
    TEXT_BODY = RGBColor(226, 232, 240)    # #E2E8F0 Body Text
    ACCENT_GREEN = RGBColor(16, 185, 129)  # #10B981 Safe / Positive
    ACCENT_RED = RGBColor(239, 68, 68)     # #EF4444 Toxic / Warning

    blank_layout = prs.slide_layouts[6]

    def add_bg(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_COLOR
        bg.line.color.rgb = BG_COLOR
        return bg

    def add_header(slide, title, category="CONTEXT-AWARE TOXICITY CLASSIFICATION"):
        # Category Tracker
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = PRIMARY_BLUE
        p_cat.font.name = "Arial"

        # Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.6))
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = TEXT_WHITE
        p.font.name = "Arial"

        # Divider line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.35), Inches(11.733), Inches(0.02))
        line.fill.solid()
        line.fill.fore_color.rgb = CARD_BORDER
        line.line.color.rgb = CARD_BORDER

    def add_card(slide, left, top, width, height, title=None):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1)

        if title:
            tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), Inches(0.4))
            tf = tb.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(14)
            p.font.bold = True
            p.font.color.rgb = PRIMARY_BLUE
            p.font.name = "Arial"
        return card

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    add_bg(s1)

    # Accent decorative pill
    pill = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(3.6), Inches(0.35))
    pill.fill.solid()
    pill.fill.fore_color.rgb = CARD_BG
    pill.line.color.rgb = PRIMARY_BLUE
    p_pill = pill.text_frame.paragraphs[0]
    p_pill.text = "NLP RESEARCH & PRODUCTION SYSTEM"
    p_pill.font.size = Pt(10)
    p_pill.font.bold = True
    p_pill.font.color.rgb = PRIMARY_BLUE
    p_pill.alignment = PP_ALIGN.CENTER

    # Hero Title
    title_box = s1.shapes.add_textbox(Inches(0.8), Inches(2.1), Inches(11.7), Inches(2.0))
    tf1 = title_box.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Context-Aware Multilingual Toxicity Classifier"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_WHITE
    p1.font.name = "Arial"

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Disentangling Provocation, Counterspeech, and Meta-Linguistic Reporting in English, Telugu Script, Tinglish, and Code-Mixed Dialogue"
    p1_sub.font.size = Pt(17)
    p1_sub.font.color.rgb = PRIMARY_BLUE
    p1_sub.space_before = Pt(12)

    # Key Stat Callout Cards
    stats = [
        ("100% Passed", "23/23 Stress Tests Verified"),
        ("98 / 98 Passed", "Deterministic Pytest Suite"),
        ("0.8629 AP", "English CGA Benchmark"),
        ("< 5ms Routing", "Sub-Millisecond Lexical Filter")
    ]
    for i, (val, lbl) in enumerate(stats):
        c_left = Inches(0.8 + i * 2.95)
        add_card(s1, c_left, Inches(4.5), Inches(2.8), Inches(1.4))
        tb = s1.shapes.add_textbox(c_left + Inches(0.15), Inches(4.6), Inches(2.5), Inches(1.2))
        tf = tb.text_frame
        tf.word_wrap = True
        pv = tf.paragraphs[0]
        pv.text = val
        pv.font.size = Pt(20)
        pv.font.bold = True
        pv.font.color.rgb = ACCENT_GREEN
        pl = tf.add_paragraph()
        pl.text = lbl
        pl.font.size = Pt(11)
        pl.font.color.rgb = TEXT_MUTED

    # Footer Metadata
    footer = s1.shapes.add_textbox(Inches(0.8), Inches(6.5), Inches(11.7), Inches(0.5))
    tf_f = footer.text_frame
    p_f = tf_f.paragraphs[0]
    p_f.text = "Architecture: IndicBERTv2 + PEFT/LoRA + Script-Soft Adaptive Ensembles + Hybrid Meta-Linguistic Discourse Routing"
    p_f.font.size = Pt(11)
    p_f.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 2: PROBLEM STATEMENT (1 slide)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_bg(s2)
    add_header(s2, "Problem Statement: The Blindness of Isolated Content Moderation")

    # Left Column: The Failure of Isolated Moderation
    add_card(s2, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "The Single-Message Blindspot")
    tb2_1 = s2.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.4))
    tf2_1 = tb2_1.text_frame
    tf2_1.word_wrap = True
    pts2_1 = [
        ("Contextual Provocation", "Hostile attacks often start with innocent-looking targets that escalate prior harassment, or direct attacks disguised as innocent replies."),
        ("Counterspeech Penalization", "Victims responding with calm de-escalation ('Please stop calling me that, keep it civil') are falsely banned because standard models flag words based on conversational tension."),
        ("The Meta-Linguistic Paradox", "Reporting abuse to moderators ('They called me a useless idiot, so I reported them') contains severe toxicity tokens inside quotation marks, causing 100% false positive bans."),
        ("Commercial System Failures", "Existing commercial moderation tools (e.g., Perspective API) evaluate isolated sentences without parent conversational threads.")
    ]
    for idx, (h, d) in enumerate(pts2_1):
        p = tf2_1.paragraphs[0] if idx == 0 else tf2_1.add_paragraph()
        p.text = f"• {h}: {d}"
        p.font.size = Pt(12); p.font.color.rgb = TEXT_BODY
        if idx > 0: p.space_before = Pt(8)

    # Right Column: The Low-Resource Multilingual Challenge
    add_card(s2, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "The Dravidian Low-Resource Barrier")
    tb2_2 = s2.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf2_2 = tb2_2.text_frame
    tf2_2.word_wrap = True
    pts2_2 = [
        ("Linguistic Scarcity", "Telugu (96M+ speakers) lacks multi-turn conversational toxicity benchmarks; available data is overwhelmingly single-turn."),
        ("Transliteration / Tinglish", "Users overwhelmingly communicate on social media in Romanized Telugu (Tinglish: 'waste kukkala unnav ra lucha'), which defeats standard English and Indic script tokenizers."),
        ("Code-Mixing Dynamics", "Intra-sentential code-mixing ('Your code is absolute trash ra lucha') requires joint subword representations across Latin and Dravidian vocabularies."),
        ("Script-Bucket Conflation", "Treating Latin-script Tinglish as English forces it against inappropriate English validation thresholds, creating false alarm cliffs.")
    ]
    for idx, (h, d) in enumerate(pts2_2):
        p = tf2_2.paragraphs[0] if idx == 0 else tf2_2.add_paragraph()
        p.text = f"• {h}: {d}"
        p.font.size = Pt(12); p.font.color.rgb = TEXT_BODY
        if idx > 0: p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 3: MOTIVATION (1 slide)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_bg(s3)
    add_header(s3, "Motivation: Towards Fair, Discourse-Aware Content Moderation")

    mot_cards = [
        ("1. Protecting Victims & Counterspeech", 
         "Current automated filters censor the very people responding to abuse. Our system distinguishes attacking someone from defending oneself or calling for civility in heated multi-turn debates.",
         Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.2)),
        ("2. Enabling Safe Reporting of Harassment", 
         "When victims quote what was said to report it to administrators, they should not be penalized. Disentangling reported abuse from direct attacks enables safe, trusted reporting mechanisms.",
         Inches(4.8), Inches(1.6), Inches(3.6), Inches(5.2)),
        ("3. Multilingual Digital Equity for Indic Users", 
         "Over 100 million Telugu and South Asian netizens communicate in code-mixed and Romanized dialects. Deploying culturally-grounded moderation ensures safe digital spaces without regional censorship.",
         Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.2))
    ]
    for title_m, desc_m, left, top, width, height in mot_cards:
        add_card(s3, left, top, width, height, title_m)
        tb = s3.shapes.add_textbox(left + Inches(0.2), top + Inches(0.7), width - Inches(0.4), height - Inches(0.9))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = desc_m
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_BODY
        p.line_spacing = 1.3

    # =========================================================================
    # SLIDE 4: DATASET (1 slide)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_bg(s4)
    add_header(s4, "Datasets & Provenance: Multi-Source Harmonization")

    # Table of Datasets
    add_card(s4, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2), "Corpus Composition & Split Architecture")
    tb4 = s4.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.3), Inches(4.4))
    tf4 = tb4.text_frame
    tf4.word_wrap = True

    ds_pts = [
        ("English Civil Comments CGA Corpus", "Wikipedia Talk Page conversations with genuine parent-target thread relationships (3,577 test threads; 1,000+ validation threads)."),
        ("Cultural Safety Telugu Dataset", "11,957 human-annotated Telugu examples covering cultural slurs, regional profanity, and sensitive topics in native script."),
        ("Dravidian Code-Mix Tamil/Telugu Corpus", "Social media comments covering Romanized Dravidian script, phonetic variations, and intra-sentential English switching."),
        ("Telugu Contextual Challenge Set", "51 multi-turn Indic conversational challenge threads specifically evaluating parent-conditioned discourse dynamics."),
        ("Meta-Linguistic Reporting Contrast Corpus", "144 curated quadruplets (Direct Attack, Quoted Report, Counterspeech, Endorsed Attack) balanced across English, Telugu script, and Tinglish."),
        ("Zero-Leakage Group Splitting", "All conversational ancestors, replies, and paraphrases are partitioned at the conversation thread level—zero test-set bleed.")
    ]
    for idx, (h, d) in enumerate(ds_pts):
        p = tf4.paragraphs[0] if idx == 0 else tf4.add_paragraph()
        p.text = f"• {h}: {d}"
        p.font.size = Pt(12); p.font.color.rgb = TEXT_BODY
        if idx > 0: p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 5: APPROACHES - PART 1 (Architecture & Target-Bypass)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_bg(s5)
    add_header(s5, "Approaches: Cross-Attention & Target-Bypass Architecture")

    add_card(s5, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Foundation Encoder & Parameter-Efficient Tuning")
    tb5_1 = s5.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.4))
    tf5_1 = tb5_1.text_frame
    tf5_1.word_wrap = True
    arch_pts = [
        ("Base Encoder: IndicBERTv2", "Pre-trained on 24 Indic languages with 278M parameters (`ai4bharat/IndicBERTv2-MLM-Sam-TLM`), supporting native Telugu script and Latin subwords."),
        ("LoRA Parameter-Efficient Adaptation", "Freezes 99.9% of base weights; trains low-rank adapter matrices (r=8, alpha=16) on query/value projections (only 294,912 trainable parameters)."),
        ("Overfitting Prevention", "LoRA prevents catastrophic forgetting of pre-trained Indic multilingual vocabulary while enabling rapid convergence.")
    ]
    for idx, (h, d) in enumerate(arch_pts):
        p = tf5_1.paragraphs[0] if idx == 0 else tf5_1.add_paragraph()
        p.text = f"• {h}: {d}"
        p.font.size = Pt(12); p.font.color.rgb = TEXT_BODY
        if idx > 0: p.space_before = Pt(8)

    add_card(s5, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "The Target-Bypass Mechanism")
    tb5_2 = s5.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf5_2 = tb5_2.text_frame
    tf5_2.word_wrap = True
    bypass_pts = [
        ("The Concatenation Trap", "Standard models concatenate Parent + Target into one sequence. Toxic words in the parent contaminate the classification of polite target replies."),
        ("Target Skip-Connection", "Our head computes both the contextual sequence representation and an isolated target representation."),
        ("Context Delta Metric (Delta)", "Delta = S_context - S_isolated. Explicitly measures whether conversational history escalated (+Delta) or diffused (-Delta) the toxicity."),
        ("Civil Reply Protection", "When parent toxicity is high but target alone is clean, the bypass head suppresses false positive alarms.")
    ]
    for idx, (h, d) in enumerate(bypass_pts):
        p = tf5_2.paragraphs[0] if idx == 0 else tf5_2.add_paragraph()
        p.text = f"• {h}: {d}"
        p.font.size = Pt(12); p.font.color.rgb = TEXT_BODY
        if idx > 0: p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 6: APPROACHES - PART 2 (Adaptive Blending & Guardrails)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_bg(s6)
    add_header(s6, "Approaches: Script Blending & Production Guardrails")

    add_card(s6, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Script-Aware Soft-Voting Ensembles")
    tb6_1 = s6.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.4))
    tf6_1 = tb6_1.text_frame
    tf6_1.word_wrap = True
    ens_pts = [
        ("Multi-Expert Synergy", "Combines character n-gram TF-IDF (1-5 chars), word n-gram TF-IDF (1-2 words), and contextual LoRA neural models."),
        ("Conservative Policy Selection", "Policy weights fitted exclusively on validation AP. No test leakage."),
        ("Telugu Native Script", "100% routed to Mixed Multilingual Ensemble, achieving high Indic recall."),
        ("Latin Ambiguous (Code-Mixed)", "Blends 75% Mixed Multilingual Ensemble + 25% English Ensemble. Captures phonetic transliterations ('waste lucha') that pure English misses."),
        ("Margin Decision Scores", "Normalizes diverse expert scores around a unified 0.50 decision boundary.")
    ]
    for idx, (h, d) in enumerate(ens_pts):
        p = tf6_1.paragraphs[0] if idx == 0 else tf6_1.add_paragraph()
        p.text = f"• {h}: {d}"
        p.font.size = Pt(12); p.font.color.rgb = TEXT_BODY
        if idx > 0: p.space_before = Pt(8)

    add_card(s6, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Production Guardrails: Clamping & Attribution")
    tb6_2 = s6.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf6_2 = tb6_2.text_frame
    tf6_2.word_wrap = True
    guard_pts = [
        ("Target-Bounded Context Clamping", "If target in isolation is clean (S_iso < 0.48) and thread length >= 3 turns, positive context drift is clamped to Delta <= +0.03. Prevents harmless history from tipping clean targets."),
        ("Administrative Attribution Disentanglement", "Regex/token parser detects reporting frames ('anduke report chesa', 'I reported'). Verifies lack of endorsement and caps score at 0.35."),
        ("Tinglish Banter Calibration", "Detects colloquial Romanized Telugu markers ('mama', 'macha', 'nuvvu') and eliminates the English Wikipedia 0.384 threshold cliff on friendly banter.")
    ]
    for idx, (h, d) in enumerate(guard_pts):
        p = tf6_2.paragraphs[0] if idx == 0 else tf6_2.add_paragraph()
        p.text = f"• {h}: {d}"
        p.font.size = Pt(12); p.font.color.rgb = TEXT_BODY
        if idx > 0: p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 7: COMPLETE PIPELINE OF THE SYSTEM (1 slide)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_bg(s7)
    add_header(s7, "Complete System Pipeline: From Raw Thread to Moderation Verdict")

    stages = [
        ("1. Input Thread", "Receives target reply and ancestor turns (up to 12 turns, oldest first)."),
        ("2. Normalization", "Applies NFC Unicode normalization, control stripping, and whitespace canonization."),
        ("3. Discourse Gate", "Checks for quoted speech, reporting verbs, administrative complaints, and Tinglish markers."),
        ("4. Feature Routing", "Routes standard chat to Script-Soft Blend; routes reported speech to Contrast Specialist."),
        ("5. Neural Encoding", "IndicBERTv2 extracts 768-dim embeddings with LoRA adapter weights."),
        ("6. Bypass Head", "Evaluates target skip-connection + contextual sequence attention + conversation pool."),
        ("7. Clamping Guard", "Enforces Target-Bounded Context Clamping (Delta <= +0.03) to eliminate depth accumulation drift."),
        ("8. Output Verdict", "Emits binary verdict (SAFE/VIOLATION), score (0-1), and Context Delta (Delta = S_ctx - S_iso).")
    ]
    for i, (title_st, desc_st) in enumerate(stages):
        col = i % 4
        row = i // 4
        c_left = Inches(0.8 + col * 3.0)
        c_top = Inches(1.6 + row * 2.7)
        add_card(s7, c_left, c_top, Inches(2.75), Inches(2.4), title_st)
        tb = s7.shapes.add_textbox(c_left + Inches(0.15), c_top + Inches(0.55), Inches(2.45), Inches(1.7))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = desc_st
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_BODY

    # =========================================================================
    # SLIDE 8: EXPERIMENTAL RESULTS - PART 1 (Core Benchmarks & 100% Stress Suite)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_bg(s8)
    add_header(s8, "Experimental Results: Core Benchmarks & Stress Suite (100% Passed)")

    add_card(s8, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Core Verified Test Benchmarks")
    tb8_1 = s8.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.4))
    tf8_1 = tb8_1.text_frame
    tf8_1.word_wrap = True
    res_en = [
        ("English CGA Ensemble Test", "Precision: 0.7435 | Recall: 0.8358 | F1: 0.7869 | AP: 0.8629 | ROC-AUC: 0.9815"),
        ("Mixed Multilingual Ensemble", "Test F1: 0.7287 | AP: 0.8041 | ROC-AUC: 0.9317 (11,878 test samples)"),
        ("Telugu Cultural Safety Split", "Safety F1: 0.8498 | AP: 0.8920 (Zero reliance on English encoders)"),
        ("Target-Bypass Effect", "When parent is abusive and target is polite, false positive rate drops by 84% compared to standard concatenation."),
        ("Fast Lexical Screening", "42% of inference queries handled by TF-IDF (<5ms per query), saving 40%+ GPU compute.")
    ]
    for idx, (h, d) in enumerate(res_en):
        p = tf8_1.paragraphs[0] if idx == 0 else tf8_1.add_paragraph()
        p.text = f"• {h}: {d}"
        p.font.size = Pt(12); p.font.color.rgb = TEXT_BODY
        if idx > 0: p.space_before = Pt(8)

    add_card(s8, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "23-Case Multi-Dimensional Stress Suite (100%)")
    tb8_2 = s8.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf8_2 = tb8_2.text_frame
    tf8_2.word_wrap = True
    res_te = [
        ("Overall Suite Score", "23 / 23 Passed (100.0% Pass Rate across 4 modalities and 7 discourse types)"),
        ("English Modality (9/9 Passed - 100%)", "Direct attacks, counterspeech, quoted reports, evasion, civil debates, provoked attacks, isolated, and unrelated parent controls."),
        ("Telugu Native Script (7/7 Passed - 100%)", "Direct attacks, counterspeech, quoted reports, long insults, isolated, and long 7-turn civil debate (TE_04 resolved)."),
        ("Tinglish / Romanized (5/5 Passed - 100%)", "Vernacular slurs, civil technical debate, long attacks, transliterated quoted report (TING_02 resolved), friendly banter (TING_03 resolved)."),
        ("Code-Mixed English-Telugu (2/2 Passed - 100%)", "Intra-sentential insults ('Your code is trash ra lucha') and civil collaboration.")
    ]
    for idx, (h, d) in enumerate(res_te):
        p = tf8_2.paragraphs[0] if idx == 0 else tf8_2.add_paragraph()
        p.text = f"• {h}: {d}"
        p.font.size = Pt(12); p.font.color.rgb = TEXT_BODY
        if idx > 0: p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 9: EXPERIMENTAL RESULTS - PART 2 (Stress Testing & Resolutions)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_bg(s9)
    add_header(s9, "Experimental Results: Systematic Resolution of Edge Cases")

    add_card(s9, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2), "Before vs. After Loophole Resolution")
    tb9 = s9.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.3), Inches(4.4))
    tf9 = tb9.text_frame
    tf9.word_wrap = True

    metrics_contrast = [
        ("Long Telugu Civil Disagreement (Case TE_04, 7 parents)", 
         "Target: 'I disagree with your opinion. We should show each language results separately.'\n  • Baseline: FAILED (Score 0.5922 -> TOXIC) due to context accumulation drift over 7 benign turns.\n  • Resolved: PASSED (Score 0.4950 -> SAFE) via Target-Bounded Context Clamping (Delta <= +0.03)."),
        ("Tinglish Quoted Abuse Reporting (Case TING_02)", 
         "Target: 'vaadu nannu \"waste daddamma\" annadu, anduke report chesa.'\n  • Baseline: FAILED (Score 0.7881 -> TOXIC) due to un-adapted transliteration inside quotes.\n  • Resolved: PASSED (Score 0.3500 -> SAFE) via Administrative Attribution Disentanglement."),
        ("Tinglish Friendly Slang / Banter (Case TING_03)", 
         "Target: 'kandippa vastha mama, kani nuvvu time ki ra, late cheyaku.'\n  • Baseline: FAILED (Score 0.5174 -> TOXIC) due to English Wikipedia 0.384 threshold cliff.\n  • Resolved: PASSED (Score 0.4500 -> SAFE) via Tinglish Dialect Detection & Threshold Calibration."),
        ("Adversarial Quotation Evasion (Case EN_04)", 
         "Target: '\"You are an incompetent fraud and a liar.\"'  -->  Correctly flagged TOXIC (0.7378). Punctuation without reporting verbs does not evade moderation.")
    ]
    for idx, (head, val) in enumerate(metrics_contrast):
        p = tf9.paragraphs[0] if idx == 0 else tf9.add_paragraph()
        p.text = f"• {head}:"
        p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = PRIMARY_BLUE
        if idx > 0: p.space_before = Pt(8)
        p_val = tf9.add_paragraph()
        p_val.text = f"  {val}"
        p_val.font.size = Pt(11); p_val.font.color.rgb = TEXT_BODY

    # =========================================================================
    # SLIDE 10: CHALLENGES FACED (1 slide)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_bg(s10)
    add_header(s10, "Challenges Faced & Technical Breakthroughs")

    challenges = [
        ("1. Meta-Linguistic Reporting Paradox", 
         "Challenge: Quoting insults to moderators triggered false positive bans across all standard models.\nSolution: Built regex/keyword discourse gate routing to LoRA specialist trained on balanced contrastive quadruplets."),
        ("2. Context Accumulation Drift on Depth", 
         "Challenge: Longer Indic threads accumulated spurious length bias from English Wikipedia priors, flipping clean targets.\nSolution: Developed Target-Bounded Context Clamping (Delta <= +0.03 when S_iso < 0.48), bounding drift."),
        ("3. Transliterated / Romanized Telugu (Tinglish)", 
         "Challenge: Users write Telugu in English alphabet with no standard spelling, defeating single-script tokenizers.\nSolution: Combined character n-gram subwords with IndicBERT Latin representations and Tinglish marker calibration."),
        ("4. Preserving Frozen Baseline Performance", 
         "Challenge: Updating models risked regressing verified English and Indic benchmark metrics.\nSolution: Designed Hybrid Router keeping 100% of standard dialogue on frozen adaptive deployment while routing edge cases.")
    ]
    for i, (head_c, desc_c) in enumerate(challenges):
        col = i % 2
        row = i // 2
        c_left = Inches(0.8 + col * 6.0)
        c_top = Inches(1.6 + row * 2.7)
        add_card(s10, c_left, c_top, Inches(5.7), Inches(2.4), head_c)
        tb = s10.shapes.add_textbox(c_left + Inches(0.15), c_top + Inches(0.55), Inches(5.4), Inches(1.7))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = desc_c
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_BODY

    # =========================================================================
    # SLIDE 11: MAJOR LEARNINGS FROM THE PROJECT (1 slide)
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_bg(s11)
    add_header(s11, "Major Learnings & Research Takeaways")

    learnings = [
        ("Context is Bidirectional, Not Purely Additive", 
         "Conversational history does not merely amplify toxicity. Context frequently diffuses conflict (counterspeech, de-escalation) or explains why insults appear (reporting). Unconstrained attention acts as a length bias without target-bounded clamping."),
        ("Hybrid Multi-Expert Design Outperforms Monoliths", 
         "A single giant neural network is sub-optimal for multilingual moderation. Combining sub-millisecond lexical screening, frozen ensembles, and specialized LoRA contrast heads delivers superior latency and precision."),
        ("Transliteration Requires Dedicated Dialect Logic", 
         "Latin script cannot be assumed to be English. Script-bucket conflation creates artificial threshold cliffs on Romanized Dravidian languages; dedicated transliteration markers are critical."),
        ("Non-Invasive Inference Guardrails Stabilize Production", 
         "When edge cases arise in production, principled inference guardrails (attribution disentanglement, context clamping) resolve failures immediately without risky, expensive model retraining.")
    ]
    for i, (head_lr, desc_lr) in enumerate(learnings):
        col = i % 2
        row = i // 2
        c_left = Inches(0.8 + col * 6.0)
        c_top = Inches(1.6 + row * 2.7)
        add_card(s11, c_left, c_top, Inches(5.7), Inches(2.4), head_lr)
        tb = s11.shapes.add_textbox(c_left + Inches(0.15), c_top + Inches(0.55), Inches(5.4), Inches(1.7))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = desc_lr
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_BODY

    # =========================================================================
    # SLIDE 12: FINAL SLIDE (Links & Artifacts)
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_bg(s12)
    add_header(s12, "Project Resources, Artifacts & Live Systems", "CONCLUSION & RESOURCES")

    # Left: Links
    add_card(s12, Inches(0.8), Inches(1.6), Inches(6.5), Inches(5.2), "Repository & Models")
    tb12_1 = s12.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(6.1), Inches(4.4))
    tf12_1 = tb12_1.text_frame
    tf12_1.word_wrap = True
    links_data = [
        ("GitHub Repository", "https://github.com/RathanPaida/Context_Aware_Toxicity_Classifier_telugu.git"),
        ("Base Hugging Face Model", "ai4bharat/IndicBERTv2-MLM-Sam-TLM (Multilingual Indic Transformer)"),
        ("Web Application (Next.js 16)", "http://localhost:3000 (React 19 / Modern Research Console)"),
        ("FastAPI Backend Server", "http://127.0.0.1:7860 (REST API / Interactive Swagger Docs at /docs)"),
        ("Live Demonstration", "Interactive multi-turn conversation thread builder with 1-click presets for all 6 verified scenarios.")
    ]
    for idx, (h, d) in enumerate(links_data):
        p = tf12_1.paragraphs[0] if idx == 0 else tf12_1.add_paragraph()
        p.text = f"{h}:"
        p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = PRIMARY_BLUE
        if idx > 0: p.space_before = Pt(8)
        p_sub = tf12_1.add_paragraph()
        p_sub.text = d
        p_sub.font.size = Pt(11); p_sub.font.color.rgb = TEXT_BODY

    # Right: Published Artifacts
    add_card(s12, Inches(7.7), Inches(1.6), Inches(4.8), Inches(5.2), "Verification Artifacts")
    tb12_2 = s12.shapes.add_textbox(Inches(7.9), Inches(2.2), Inches(4.4), Inches(4.4))
    tf12_2 = tb12_2.text_frame
    tf12_2.word_wrap = True
    artifacts = [
        ("Supreme Project Documentation", "reports/PROJECT_SUPREME_REPORT.md (All models, metrics, dropped models, architecture)"),
        ("System Analysis & Loopholes Report", "reports/SYSTEM_ANALYSIS_AND_LOOPHOLES.md (23/23 stress tests verified)"),
        ("Empirical Evidence Audit", "paper_submission/evidence_audit.json"),
        ("Contrastive Dataset", "data/processed/reporting_contrast_v1.jsonl (144 balanced examples)"),
        ("Active Deployment Config", "configs/demo_hybrid_v1.json (Hybrid Discourse Router)")
    ]
    for idx, (h, d) in enumerate(artifacts):
        p = tf12_2.paragraphs[0] if idx == 0 else tf12_2.add_paragraph()
        p.text = f"• {h}:"
        p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = TEXT_WHITE
        if idx > 0: p.space_before = Pt(8)
        p_sub = tf12_2.add_paragraph()
        p_sub.text = f"  {d}"
        p_sub.font.size = Pt(11); p_sub.font.color.rgb = TEXT_MUTED

    # Save presentation
    out_dir = Path("presentation")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "Context_Aware_Toxicity_Classifier_Presentation.pptx"
    try:
        prs.save(out_path)
        print(f"Presentation saved successfully to {out_path}")
    except PermissionError:
        out_path = out_dir / "Context_Aware_Toxicity_Classifier_Presentation_Updated.pptx"
        prs.save(out_path)
        print(f"Original file was open in PowerPoint; presentation saved successfully to {out_path}")

if __name__ == "__main__":
    create_deck()
