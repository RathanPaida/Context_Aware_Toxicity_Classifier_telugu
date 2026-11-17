"""Generate an executive, professional 16:9 widescreen PPTX presentation

for the Context-Aware Toxicity Classifier project.
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

    # Main Presentation Title
    title_box = s1.shapes.add_textbox(Inches(0.8), Inches(2.1), Inches(11.7), Inches(1.8))
    tf1 = title_box.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Context-Aware Toxicity Classifier"
    p1.font.size = Pt(38)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_WHITE
    p1.font.name = "Arial"

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Hierarchical Attention, Target-Bypass Disentanglement, and Multilingual Adaptive Routing for English & Telugu Conversations"
    p1_sub.font.size = Pt(16)
    p1_sub.font.color.rgb = TEXT_MUTED
    p1_sub.space_before = Pt(12)

    # Overview Cards on Title Slide
    cards_meta = [
        ("Core Novelty", "Target-bypass residual architecture preventing parent contamination traps"),
        ("Multilingual Scale", "English, Native Telugu script, and Romanized code-mixed vernacular"),
        ("Engineered Defense", "Discourse routing for reported speech + script soft-blending ensembles")
    ]
    for i, (k, v) in enumerate(cards_meta):
        c_left = Inches(0.8 + i * 4.0)
        add_card(s1, c_left, Inches(4.3), Inches(3.7), Inches(1.8), k)
        tb = s1.shapes.add_textbox(c_left + Inches(0.2), Inches(4.8), Inches(3.3), Inches(1.1))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = v
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_BODY

    # Author / Date Footer
    ft = s1.shapes.add_textbox(Inches(0.8), Inches(6.5), Inches(11.7), Inches(0.4))
    p_ft = ft.text_frame.paragraphs[0]
    p_ft.text = "Author: Rathan Paida  |  Repository: Context_Aware_Toxicity_Classifier_telugu  |  Date: November 2026"
    p_ft.font.size = Pt(11)
    p_ft.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 2: PROBLEM STATEMENT (1 slide)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_bg(s2)
    add_header(s2, "Problem Statement: The Limits of Isolated Toxicity Classification")

    # Card 1: Context Blindness
    add_card(s2, Inches(0.8), Inches(1.6), Inches(5.6), Inches(2.4), "1. Context Blindness (False Negatives)")
    tb2_1 = s2.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.2), Inches(1.8))
    tf2_1 = tb2_1.text_frame
    tf2_1.word_wrap = True
    p = tf2_1.paragraphs[0]
    p.text = "• Traditional classifiers evaluate target messages as single, isolated sentences."
    p.font.size = Pt(13); p.font.color.rgb = TEXT_BODY
    p2 = tf2_1.add_paragraph()
    p2.text = "• Subtle harassment, sarcasm, micro-aggressions, and provoked hostility are completely missed because their hostile intent only emerges in reference to earlier turns."
    p2.font.size = Pt(13); p2.font.color.rgb = TEXT_BODY; p2.space_before = Pt(6)

    # Card 2: The Parent Contamination Trap
    add_card(s2, Inches(6.8), Inches(1.6), Inches(5.7), Inches(2.4), "2. The Parent Contamination Trap (False Positives)")
    tb2_2 = s2.shapes.add_textbox(Inches(7.0), Inches(2.1), Inches(5.3), Inches(1.8))
    tf2_2 = tb2_2.text_frame
    tf2_2.word_wrap = True
    p = tf2_2.paragraphs[0]
    p.text = "• Naive concatenated models concatenate [Parent + Target] into a single transformer input."
    p.font.size = Pt(13); p.font.color.rgb = TEXT_BODY
    p2 = tf2_2.add_paragraph()
    p2.text = "• If an abusive troll attacks an innocent user, and the victim replies politely ('Please keep it civil'), the model sees toxic parent tokens and falsely bans the victim!"
    p2.font.size = Pt(13); p2.font.color.rgb = TEXT_BODY; p2.space_before = Pt(6)

    # Card 3: The Indic Cross-Lingual Failure
    add_card(s2, Inches(0.8), Inches(4.3), Inches(11.7), Inches(2.5), "3. Cross-Lingual & Code-Mixed Blindness in Indic Discourse")
    tb2_3 = s2.shapes.add_textbox(Inches(1.0), Inches(4.8), Inches(11.3), Inches(1.8))
    tf2_3 = tb2_3.text_frame
    tf2_3.word_wrap = True
    p = tf2_3.paragraphs[0]
    p.text = "• Pure English models achieve 0.0000 F1 on Indic scripts (total cross-lingual blindness)."
    p.font.size = Pt(13); p.font.color.rgb = TEXT_BODY
    p2 = tf2_3.add_paragraph()
    p2.text = "• Code-Mixed Evasion: Modern chat users write Telugu phonetically in Latin script ('waste kukkala unnav ra lucha'). Standard English vocabularies treat these vernacular insults as benign typos or out-of-vocabulary words."
    p2.font.size = Pt(13); p2.font.color.rgb = TEXT_BODY; p2.space_before = Pt(6)
    p3 = tf2_3.add_paragraph()
    p3.text = "• Quoted Abuse Trap: Users reporting harassment ('They called me an idiot, so I reported them') get falsely banned because surface n-grams trigger on the quoted word."
    p3.font.size = Pt(13); p3.font.color.rgb = TEXT_BODY; p3.space_before = Pt(6)

    # =========================================================================
    # SLIDE 3: MOTIVATION (1 slide)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_bg(s3)
    add_header(s3, "Motivation: Building High-Trust Multilingual Online Spaces")

    cards_s3 = [
        ("Real-World Chat Dynamics", "Real-world conversations are fundamentally sequential trees. Communication occurs through dialogue threads (social media, Wikipedia talk pages, gaming voice/text chat, community forums). Evaluating messages in a vacuum distorts meaning."),
        ("Protecting Innocent Users", "Moderation systems must protect victims who engage in counterspeech or report abuse. A system that bans victims who defend themselves against bullies creates toxic chilling effects and destroys platform trust."),
        ("Indic Digital Inclusion", "Over 80 million people speak Telugu, and billions speak Indic languages. Most commercial moderation APIs are heavily biased toward Western English norms, leaving Indic vernacular and code-mixed spaces unprotected."),
        ("Data-Centric Robustness", "Moving beyond brittle black-box models to transparent, architecturally sound systems where context influence is measurable (Context Delta metric), explainable, and resilient to adversarial quote evasion.")
    ]
    for i, (k, v) in enumerate(cards_s3):
        col = i % 2
        row = i // 2
        c_left = Inches(0.8 + col * 6.0)
        c_top = Inches(1.6 + row * 2.7)
        add_card(s3, c_left, c_top, Inches(5.7), Inches(2.4), f"{i+1}. {k}")
        tb = s3.shapes.add_textbox(c_left + Inches(0.2), c_top + Inches(0.55), Inches(5.3), Inches(1.7))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = v
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_BODY

    # =========================================================================
    # SLIDE 4: DATASET (1 slide)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_bg(s4)
    add_header(s4, "Dataset: Multi-Source Curated & Group-Isolated Corpora")

    # Left: Corpus Table Card
    add_card(s4, Inches(0.8), Inches(1.6), Inches(7.5), Inches(5.2), "Corpora Breakdown & Split Topology")
    tb4 = s4.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(7.1), Inches(4.5))
    tf4 = tb4.text_frame
    tf4.word_wrap = True
    
    datasets_info = [
        ("Wikipedia Conversations Gone Awry (CGA)", "14,563 multi-turn English talk page threads with annotated personal attacks and dialogue tree provenance."),
        ("Indic Cultural Safety (Telugu)", "11,957 culturally sensitive native Telugu prompts and comments labeled for toxicity and social offense."),
        ("Dravidian Code-Mixed Corpus", "Code-mixed Tamil/Telugu/English comments capturing phonetic transliterations and vernacular slang."),
        ("Curated Reporting Contrast Corpus (v1)", "144 balanced quadruplets (English, Telugu script, Romanized Telugu) designed for direct insult vs. reported insult discrimination."),
        ("Split Integrity (No Leakage)", "Strict group-level partitioning: all paraphrases and thread ancestors belong to the same split. Train: 54,674 | Val: 11,800 | Test: 11,902.")
    ]
    for idx, (head, desc) in enumerate(datasets_info):
        p = tf4.paragraphs[0] if idx == 0 else tf4.add_paragraph()
        p.text = f"• {head}: {desc}"
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_BODY
        if idx > 0: p.space_before = Pt(8)

    # Right: Data Safety Principles
    add_card(s4, Inches(8.6), Inches(1.6), Inches(3.9), Inches(5.2), "Integrity & Anti-Leakage")
    tb4_r = s4.shapes.add_textbox(Inches(8.8), Inches(2.1), Inches(3.5), Inches(4.5))
    tf4_r = tb4_r.text_frame
    tf4_r.word_wrap = True
    principles = [
        ("Group-Level Partitioning", "Entire conversation subtrees are assigned to exactly one split. Zero cross-split message sharing."),
        ("Lossless Normalization", "NFC Unicode normalization with control character stripping and whitespace canonization."),
        ("SHA-256 Provenance", "Every prepared split is locked with cryptographic hash verification to guarantee 100% reproducibility.")
    ]
    for idx, (h, d) in enumerate(principles):
        p = tf4_r.paragraphs[0] if idx == 0 else tf4_r.add_paragraph()
        p.text = f"{h}:"
        p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = PRIMARY_BLUE
        if idx > 0: p.space_before = Pt(10)
        p_sub = tf4_r.add_paragraph()
        p_sub.text = d
        p_sub.font.size = Pt(11); p_sub.font.color.rgb = TEXT_BODY

    # =========================================================================
    # SLIDE 5: APPROACHES - PART 1 (Architecture & Target-Bypass)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_bg(s5)
    add_header(s5, "Approaches: Hierarchical Architecture & Target-Bypass Head")

    add_card(s5, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "1. Hierarchical Attention Network")
    tb5_1 = s5.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.4))
    tf5_1 = tb5_1.text_frame
    tf5_1.word_wrap = True
    arch_pts = [
        ("Sentence-Level Embedding", "Base multilingual encoders (ai4bharat/IndicBERTv2-MLM-Sam-TLM, XLM-RoBERTa) encode each turn independently up to 128 tokens."),
        ("Parameter-Efficient Fine-Tuning", "Low-Rank Adaptation (LoRA: r=8, alpha=16, dropout=0.1) on query/value projection layers, training only 294K parameters out of 278M."),
        ("Hierarchical Context Transformer", "A 2-layer, 4-head Transformer encoder processes the sequence of sentence vectors with learned position embeddings up to 12 turns."),
        ("Adaptive Contextual Pooling", "Attention pooling computes dynamic attention weights across ancestors, identifying which preceding turn is relevant.")
    ]
    for idx, (h, d) in enumerate(arch_pts):
        p = tf5_1.paragraphs[0] if idx == 0 else tf5_1.add_paragraph()
        p.text = f"• {h}: {d}"
        p.font.size = Pt(12); p.font.color.rgb = TEXT_BODY
        if idx > 0: p.space_before = Pt(8)

    add_card(s5, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "2. The Target-Bypass Residual Head")
    tb5_2 = s5.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf5_2 = tb5_2.text_frame
    tf5_2.word_wrap = True
    bypass_pts = [
        ("The Architectural Innovation", "Standard models concatenate all turns into one text or pool indiscriminately, causing parent contamination."),
        ("Triple-Vector Concatenation", "The classification head explicitly concatenates three distinct representations:\n[ h_target_linear ; h_target_contextual ; h_conversation_pooled ]"),
        ("Independent Target Skip-Connection", "The isolated projection h_target_linear completely bypasses the conversation self-attention layer."),
        ("Mathematical Guarantee", "Because the target has a direct un-tainted residual path to the classifier, a clean target reply to an intensely abusive parent will NOT be tainted into a false positive!"),
        ("Context Delta Metric", "Enables real-time measurement:\nDelta = Score(in context) - Score(in isolation)")
    ]
    for idx, (h, d) in enumerate(bypass_pts):
        p = tf5_2.paragraphs[0] if idx == 0 else tf5_2.add_paragraph()
        p.text = f"• {h}: {d}"
        p.font.size = Pt(12); p.font.color.rgb = TEXT_BODY
        if idx > 0: p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 6: APPROACHES - PART 2 (Adaptive Blending & Discourse Routing)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_bg(s6)
    add_header(s6, "Approaches: Conservative Script Blending & Discourse Routing")

    add_card(s6, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Script-Based Soft-Voting Ensembles")
    tb6_1 = s6.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.4))
    tf6_1 = tb6_1.text_frame
    tf6_1.word_wrap = True
    ens_pts = [
        ("Multi-Expert Synergy", "Combines character n-gram TF-IDF (2-5 chars), word n-gram TF-IDF (1-2 words), and contextual LoRA neural models."),
        ("Conservative Policy Selection", "Policy weights fitted exclusively on validation AP. No test leakage:"),
        ("Telugu Native Script", "100% routed to Mixed Multilingual Ensemble (0% English reliance), achieving high Indic recall."),
        ("Latin Ambiguous (Code-Mixed)", "Blends 75% Mixed Multilingual Ensemble + 25% English Ensemble. Captures phonetic transliterations ('waste lucha') that pure English miss."),
        ("Margin Decision Scores", "Normalizes diverse expert scores around a unified 0.50 decision boundary.")
    ]
    for idx, (h, d) in enumerate(ens_pts):
        p = tf6_1.paragraphs[0] if idx == 0 else tf6_1.add_paragraph()
        p.text = f"• {h}: {d}"
        p.font.size = Pt(12); p.font.color.rgb = TEXT_BODY
        if idx > 0: p.space_before = Pt(8)

    add_card(s6, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Discourse Specialist Routing (Reported Speech)")
    tb6_2 = s6.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf6_2 = tb6_2.text_frame
    tf6_2.word_wrap = True
    disc_pts = [
        ("The Meta-Linguistic Challenge", "Standard toxicity models treat any mention of an insult token as toxic, failing on quotes ('They said: You are an idiot. I reported them')."),
        ("Discourse Pattern Gate", "A fast regex/keyword analyzer identifies quoted abuse, reporting verbs ('said', 'called me', 'annadu'), and administrative actions ('reported', 'flagged')."),
        ("Targeted Specialist Model", "When reporting discourse is identified, routes to the Contrast-Adapted Specialist trained on balanced quadruplets."),
        ("Preserving Old Deployment", "All standard dialogue routes 100% through the frozen adaptive ensemble—zero disruption to existing verified performance.")
    ]
    for idx, (h, d) in enumerate(disc_pts):
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
        ("3. Discourse Gate", "Checks for quoted speech, reporting verbs, and administrative complaint markers."),
        ("4. Feature Routing", "Routes standard chat to Script-Soft Blend; routes reported speech to Contrast Specialist."),
        ("5. Neural Encoding", "IndicBERTv2 extracts 768-dim embeddings with LoRA adapter weights."),
        ("6. Bypass Head", "Evaluates target skip-connection + contextual sequence attention + conversation pool."),
        ("7. Soft Ensemble", "Blends lexical character TF-IDF and neural probability based on validation policy."),
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
    # SLIDE 8: EXPERIMENTAL RESULTS - PART 1 (Core Benchmarks)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_bg(s8)
    add_header(s8, "Experimental Results: Core Benchmark Evaluation")

    add_card(s8, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "English CGA Benchmark (Verified Test Set)")
    tb8_1 = s8.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.4))
    tf8_1 = tb8_1.text_frame
    tf8_1.word_wrap = True
    res_en = [
        ("Isolated Baseline (No Context)", "Precision: 0.6970 | Recall: 0.7404 | F1: 0.7180"),
        ("Contextual Transformer (LoRA)", "Precision: 0.7350 | Recall: 0.7975 | F1: 0.7650 (+4.7% F1 gain)"),
        ("Validation Average Precision", "Improved from 0.710 to 0.792, verifying context utility."),
        ("Target-Bypass Effect", "When parent is abusive and target is polite, false positive rate drops by 84% compared to standard concatenation."),
        ("Direct Hostility Retention", "Recall on provoked direct attacks remains 99.8% decisive.")
    ]
    for idx, (h, d) in enumerate(res_en):
        p = tf8_1.paragraphs[0] if idx == 0 else tf8_1.add_paragraph()
        p.text = f"• {h}: {d}"
        p.font.size = Pt(12); p.font.color.rgb = TEXT_BODY
        if idx > 0: p.space_before = Pt(8)

    add_card(s8, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Cross-Lingual Telugu & Code-Mixed Challenge")
    tb8_2 = s8.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf8_2 = tb8_2.text_frame
    tf8_2.word_wrap = True
    res_te = [
        ("Pure English Model Blindness", "F1 Score: 0.0000 (Recall: 0.0000). Completely blind to Telugu native script."),
        ("Multilingual Mixed Ensemble", "Telugu Safety F1: 0.8498 | AP: 0.8920."),
        ("Telugu Challenge Set", "Adaptive Policy F1: 0.6222 vs. English 0.0000."),
        ("Code-Mixed Transliteration", "Mixed Ensemble routes Latin-ambiguous script with 75% multilingual weight, capturing vernacular insults ('waste lucha', 'daddamma') that pure English filters ignore."),
        ("Combined Multi-Domain Test", "Test F1: 0.7299 | Test AP: 0.8007 across English, Telugu, and Tamil code-mix.")
    ]
    for idx, (h, d) in enumerate(res_te):
        p = tf8_2.paragraphs[0] if idx == 0 else tf8_2.add_paragraph()
        p.text = f"• {h}: {d}"
        p.font.size = Pt(12); p.font.color.rgb = TEXT_BODY
        if idx > 0: p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 9: EXPERIMENTAL RESULTS - PART 2 (Quoted Speech & Reporting)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_bg(s9)
    add_header(s9, "Experimental Results: Quoted Abuse & Reporting Disentanglement")

    add_card(s9, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2), "Untouched Test Evaluation: Reporting vs. Using Insults")
    tb9 = s9.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.3), Inches(4.4))
    tf9 = tb9.text_frame
    tf9.word_wrap = True

    metrics_contrast = [
        ("Reporting Insult Accuracy ('He called me an idiot, so I reported him')", "Baseline: 0.00% (100% False Positive)  -->  Adapted Specialist: 100.00% SAFE (+100% gain)"),
        ("Direct Attack Recall ('You are a useless idiot')", "Baseline: 100.00% (Score: 0.9955)  -->  Adapted Specialist: 100.00% TOXIC (Score: 0.9951) [Zero Recall Loss]"),
        ("Endorsed Attack Recall ('He called you an idiot, and he was right')", "Baseline: 100.00%  -->  Adapted Specialist: 100.00% TOXIC (Score: 0.7759) [Decisive]"),
        ("Admonishment / Counterspeech ('Please stop calling people idiots')", "Baseline: 0.00%  -->  Adapted Specialist: 66.67% SAFE (Mean score dropped from 0.8928 to 0.5387)"),
        ("Telugu Quoted Abuse ('అతను నన్ను చెత్త మనిషి అని దూషించాడు, అందుకే రిపోర్ట్ చేశాను')", "Baseline: TOXIC (0.9856)  -->  Adapted Specialist: SAFE (0.1384) [Decisive]"),
        ("Adversarial Evasion Resilience", "Direct attacks disguised with quotation marks ('\"You are a useless idiot.\"') are caught with 98.23% - 99.09% confidence; punctuation alone does not trick the model.")
    ]
    for idx, (head, val) in enumerate(metrics_contrast):
        p = tf9.paragraphs[0] if idx == 0 else tf9.add_paragraph()
        p.text = f"• {head}:"
        p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = PRIMARY_BLUE
        if idx > 0: p.space_before = Pt(8)
        p_val = tf9.add_paragraph()
        p_val.text = f"  {val}"
        p_val.font.size = Pt(12); p_val.font.color.rgb = TEXT_BODY

    # =========================================================================
    # SLIDE 10: CHALLENGES FACED (1 slide)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_bg(s10)
    add_header(s10, "Challenges Faced & Technical Solutions")

    challenges = [
        ("Parent Contamination Trap", "Early models allowed abusive parent tokens to directly contaminate clean target replies.", "Engineered the Target-Bypass residual head, guaranteeing an un-tainted linear path from the target message directly to the classifier."),
        ("Lexical Surface Matching Bias", "Character TF-IDF subword models fired on subwords ('idio', 'usel') even when quoted innocently.", "Combined word TF-IDF with the contextual transformer, allowing syntactic reporting frames ('they said', 'so I reported') to dominate."),
        ("Dataset Multi-Turn Asymmetry", "English data had 14K deep conversation threads, but Telugu datasets were 100% single-turn prompts.", "Discovered thread-length escalation bias causing attention collapse on Turn 2; established sliding context window guidelines (k=2-3)."),
        ("Phonetic Vernacular Code-Mixing", "Users write Telugu slang in English letters ('waste lucha'), which pure English models treat as typos.", "Implemented script-based soft-voting policy (script_soft_blend), allocating 75% ensemble weight to multilingual IndicBERT.")
    ]
    for i, (title_ch, prob_ch, sol_ch) in enumerate(challenges):
        col = i % 2
        row = i // 2
        c_left = Inches(0.8 + col * 6.0)
        c_top = Inches(1.6 + row * 2.7)
        add_card(s10, c_left, c_top, Inches(5.7), Inches(2.4), f"{i+1}. {title_ch}")
        tb = s10.shapes.add_textbox(c_left + Inches(0.2), c_top + Inches(0.55), Inches(5.3), Inches(1.7))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = f"Challenge: {prob_ch}"
        p1.font.size = Pt(12); p1.font.color.rgb = ACCENT_RED
        p2 = tf.add_paragraph()
        p2.text = f"Solution: {sol_ch}"
        p2.font.size = Pt(12); p2.font.color.rgb = ACCENT_GREEN; p2.space_before = Pt(6)

    # =========================================================================
    # SLIDE 11: MAJOR LEARNINGS FROM THE PROJECT (1 slide)
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_bg(s11)
    add_header(s11, "Major Learnings & Research Insights")

    learnings = [
        ("Data Distribution Trumps Architecture", "You cannot fix missing training paradigms with post-hoc threshold tuning. The model's failure on quoted speech was only solved by creating a balanced contrastive dataset showing insult tokens in non-hostile roles."),
        ("Target Residual Skips Are Essential", "Hierarchical attention without target-bypass leads to severe false positives. Explicitly preserving the isolated target embedding is essential for fair, non-punitive moderation."),
        ("Cross-Lingual Discourse Transfer Gap", "Cross-lingual transformers easily transfer vocabulary semantics across languages, but they do NOT automatically transfer multi-turn conversational discourse dynamics if the target language lacks deep threads."),
        ("Hybrid Routing Beats Monolithic Models", "A single neural model cannot optimally handle native Indic script, Latin code-mixing, and meta-linguistic reporting. A conservative hybrid router delivers production reliability without catastrophic forgetting.")
    ]
    for i, (title_lr, desc_lr) in enumerate(learnings):
        col = i % 2
        row = i // 2
        c_left = Inches(0.8 + col * 6.0)
        c_top = Inches(1.6 + row * 2.7)
        add_card(s11, c_left, c_top, Inches(5.7), Inches(2.4), f"{i+1}. {title_lr}")
        tb = s11.shapes.add_textbox(c_left + Inches(0.2), c_top + Inches(0.55), Inches(5.3), Inches(1.7))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = desc_lr
        p.font.size = Pt(13)
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
        ("Live Demonstration", "Interactive multi-turn conversation thread builder with 1-click presets for all tested scenarios.")
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
        ("LaTeX Paper Source", "paper_submission/main.tex (10 verified tables, full bibliography)"),
        ("Empirical Evidence Audit", "paper_submission/evidence_audit.json"),
        ("Reporting Diagnosis Report", "reports/REPORTING_INSULTS_DIAGNOSIS.md"),
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
    prs.save(out_path)
    print(f"Presentation saved successfully to {out_path}")

if __name__ == "__main__":
    create_deck()
