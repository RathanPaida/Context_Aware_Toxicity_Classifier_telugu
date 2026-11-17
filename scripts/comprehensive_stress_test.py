"""Run comprehensive stress test across all context lengths, languages, and semantic categories.

Covers:
- Short context (1 turn)
- Long context (5-7 turns)
- Zero context (isolation)
- English, Telugu native, Tinglish (Romanized Telugu), and Code-mixed
- Direct attack, polite agreement, polite disagreement, counterspeech, reported speech, sarcasm, unrelated context.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contextual.config import load_config
from contextual.inference import create_predictor

TEST_CASES = [
    # ========================== ENGLISH CASES ==========================
    {
        "id": "EN_01",
        "lang": "English",
        "category": "Direct Attack (Short Context)",
        "context_depth": "Short (1 parent)",
        "history": ["Why did you revert my edit on the biology page?"],
        "target": "Because you are a brainless idiot who knows nothing about science.",
        "expected": "TOXIC",
        "rationale": "Clear, direct personal attack with hostility directed at peer."
    },
    {
        "id": "EN_02",
        "lang": "English",
        "category": "Counterspeech Defense (Short Context)",
        "context_depth": "Short (1 parent)",
        "history": ["Shut up you stupid idiot, stop vandalizing or I will destroy you."],
        "target": "I am following Wikipedia guidelines, please keep your comments civil.",
        "expected": "SAFE",
        "rationale": "Target-bypass test: polite de-escalation in reply to severe hostile attack."
    },
    {
        "id": "EN_03",
        "lang": "English",
        "category": "Quoted Abuse Report (Short Context)",
        "context_depth": "Short (1 parent)",
        "history": ["What exactly did the other player say to you?"],
        "target": 'They said, "You are a useless idiot." I reported the message because personal attacks are unacceptable.',
        "expected": "SAFE",
        "rationale": "Meta-linguistic reporting of an insult to administrative moderators."
    },
    {
        "id": "EN_04",
        "lang": "English",
        "category": "Direct Attack with Quotes (Evasion)",
        "context_depth": "Short (1 parent)",
        "history": ["What is your opinion of my contribution?"],
        "target": '"You are an incompetent fraud and a liar."',
        "expected": "TOXIC",
        "rationale": "Attacker disguising hostile attack inside quotes to evade moderation."
    },
    {
        "id": "EN_05",
        "lang": "English",
        "category": "Civil Disagreement (Long Context)",
        "context_depth": "Long (5 parents)",
        "history": [
            "Shall we meet at 10 AM tomorrow to review the slides?",
            "Yes, 10 AM works. Let me bring the performance charts.",
            "I noticed you reported a 10% gain across all tasks.",
            "That was based on the preliminary test run on Friday.",
            "Can you explain why the baseline numbers dropped?"
        ],
        "target": "I disagree with this interpretation. The baseline numbers dropped because of different test splits, not model failure.",
        "expected": "SAFE",
        "rationale": "Technical intellectual debate without hostility across an extended thread."
    },
    {
        "id": "EN_06",
        "lang": "English",
        "category": "Provoked Attack (Long Context)",
        "context_depth": "Long (5 parents)",
        "history": [
            "We have reviewed your proposed changes to the article.",
            "The consensus is that the sources are not reliable.",
            "I provided three peer-reviewed citations yesterday.",
            "Those journals are predatory and unindexed.",
            "Please stop bringing this up on the noticeboard."
        ],
        "target": "You corrupt clowns are deliberately sabotaging this project, get lost.",
        "expected": "TOXIC",
        "rationale": "Escalation into overt hostility ('corrupt clowns', 'get lost') after long argument."
    },
    {
        "id": "EN_07",
        "lang": "English",
        "category": "Isolation / Zero Context (Benign)",
        "context_depth": "No Context (0 parents)",
        "history": [],
        "target": "Could you please explain what led to the consensus on this policy?",
        "expected": "SAFE",
        "rationale": "Standalone constructive question with no parent history."
    },
    {
        "id": "EN_08",
        "lang": "English",
        "category": "Isolation / Zero Context (Toxic)",
        "context_depth": "No Context (0 parents)",
        "history": [],
        "target": "You are a pathetic piece of garbage.",
        "expected": "TOXIC",
        "rationale": "Standalone unprovoked personal attack."
    },
    {
        "id": "EN_09",
        "lang": "English",
        "category": "Unrelated Parent Control Test",
        "context_depth": "Short (1 parent)",
        "history": ["The recipe calls for two tablespoons of olive oil and garlic."],
        "target": "You are a useless idiot who cannot code.",
        "expected": "TOXIC",
        "rationale": "Target toxicity should remain high even when parent is completely semantically unrelated."
    },

    # ========================== TELUGU NATIVE SCRIPT ==========================
    {
        "id": "TE_01",
        "lang": "Telugu Native",
        "category": "Direct Attack (Short Context)",
        "context_depth": "Short (1 parent)",
        "history": ["ఈ విషయం గురించి నీ అభిప్రాయం ఏమిటి?"],
        "target": "నువ్వు ఒక చెత్త మనిషివి, నీలాంటి వాళ్ళు బతకడం అనవసరం.",
        "expected": "TOXIC",
        "rationale": "Extreme direct personal abuse and death wish in native Telugu."
    },
    {
        "id": "TE_02",
        "lang": "Telugu Native",
        "category": "Counterspeech / Calm Reply (Short Context)",
        "context_depth": "Short (1 parent)",
        "history": ["నువ్వు ఒక చెత్త మనిషివి, నీలాంటి వాళ్ళు బతకడం అనవసరం."],
        "target": "మనం గౌరవప్రదంగా మాట్లాడదాం, ఇలాంటి పదాలు వాడకండి.",
        "expected": "SAFE",
        "rationale": "Target-bypass test: victim responding calmly and requesting civility in Telugu."
    },
    {
        "id": "TE_03",
        "lang": "Telugu Native",
        "category": "Quoted Abuse Report (Short Context)",
        "context_depth": "Short (1 parent)",
        "history": ["ఈ గొడవ ఎందుకు జరిగింది?"],
        "target": 'అతను నన్ను చెత్త మనిషి అని దూషించాడు, అందుకే రిపోర్ట్ చేశాను.',
        "expected": "SAFE",
        "rationale": "Reporting a slur to moderators in native Telugu script."
    },
    {
        "id": "TE_04",
        "lang": "Telugu Native",
        "category": "Polite Disagreement (Long Context)",
        "context_depth": "Long (7 parents)",
        "history": [
            "మన ప్రాజెక్ట్ ప్రెజెంటేషన్ రేపు ఎన్ని గంటలకు ఉంది?",
            "ఉదయం పది గంటలకు ఉంది. మనం తొమ్మిదిన్నరకు అక్కడికి వెళ్దాం.",
            "నేను ఫలితాల పట్టికను స్లైడ్లలో చేర్చాను.",
            "బాగుంది. కానీ అన్ని భాషల్లో మన మోడల్ మెరుగ్గా పనిచేసిందని రాశావు.",
            "మొత్తం స్కోరు కొంచెం పెరిగింది కాబట్టి అలా రాశాను.",
            "మొత్తం స్కోరు పెరిగినా, ప్రతి భాషలో ఫలితం మెరుగుపడలేదు కదా.",
            "సరే. అయితే ఆ వాక్యాన్ని ఎలా మార్చాలి?"
        ],
        "target": "నీ అభిప్రాయంతో నేను ఏకీభవించడం లేదు. ప్రతి భాష ఫలితాన్ని విడిగా చూపించి, మన మోడల్ పరిమితులను కూడా వివరించాలి.",
        "expected": "SAFE",
        "rationale": "Complex technical disagreement after 7 benign turns without personal abuse."
    },
    {
        "id": "TE_05",
        "lang": "Telugu Native",
        "category": "Direct Insult (Long Context)",
        "context_depth": "Long (5 parents)",
        "history": [
            "రేపటి సమావేశానికి నివేదిక సిద్ధంగా ఉందా?",
            "ఇంకా పూర్తి కాలేదు, డేటా విశ్లేషణ జరుగుతోంది.",
            "సమయానికి పని పూర్తి చేయడం నీ బాధ్యత కదా?",
            "సర్వర్ సమస్యల వల్ల కొద్దిగా ఆలస్యం అయింది.",
            "ఇలాంటి సాకులు ప్రతిసారీ చెప్పవద్దు."
        ],
        "target": "నువ్వు ఒక పనికిమాలిన మూర్ఖుడివి, నీ ఆదేశాలు వినాల్సిన అవసరం నాకు లేదు.",
        "expected": "TOXIC",
        "rationale": "Direct personal insult ('useless idiot') following a tense workplace thread."
    },
    {
        "id": "TE_06",
        "lang": "Telugu Native",
        "category": "Isolation / Zero Context (Benign)",
        "context_depth": "No Context (0 parents)",
        "history": [],
        "target": "ఈ అంశంపై మరింత సమాచారం కోసం అధికారిక వెబ్‌సైట్ చూడండి.",
        "expected": "SAFE",
        "rationale": "Helpful informational sentence in native Telugu script."
    },
    {
        "id": "TE_07",
        "lang": "Telugu Native",
        "category": "Isolation / Zero Context (Toxic)",
        "context_depth": "No Context (0 parents)",
        "history": [],
        "target": "సిగ్గులేని వెధవ, నీ మొహం అద్దంలో చూసుకో.",
        "expected": "TOXIC",
        "rationale": "Direct vernacular insult in Telugu script."
    },

    # ========================== TINGLISH / ROMANIZED TELUGU ==========================
    {
        "id": "TING_01",
        "lang": "Tinglish",
        "category": "Direct Vernacular Slur (Short Context)",
        "context_depth": "Short (1 parent)",
        "history": ["e roju match chusava?"],
        "target": "waste kukkala unnav, nee moham chusuko ra lucha.",
        "expected": "TOXIC",
        "rationale": "Severe phonetic Telugu abuse ('waste dog', 'lucha scoundrel') in Latin script."
    },
    {
        "id": "TING_02",
        "lang": "Tinglish",
        "category": "Quoted Abuse Report (Short Context)",
        "context_depth": "Short (1 parent)",
        "history": ["aa user ni enduku block cheyyamantunnav?"],
        "target": 'vaadu nannu "waste daddamma" annadu, anduke report chesa.',
        "expected": "SAFE",
        "rationale": "Reporting a transliterated slur ('daddamma') to an admin."
    },
    {
        "id": "TING_03",
        "lang": "Tinglish",
        "category": "Friendly Banter / Non-Toxic Slang",
        "context_depth": "Short (1 parent)",
        "history": ["mama repu morning cricket match ki vasthunnava?"],
        "target": "kandippa vastha mama, kani nuvvu time ki ra, late cheyaku.",
        "expected": "SAFE",
        "rationale": "Casual friendly conversation among friends in Romanized Telugu."
    },
    {
        "id": "TING_04",
        "lang": "Tinglish",
        "category": "Civil Disagreement (Long Context)",
        "context_depth": "Long (4 parents)",
        "history": [
            "project presentation slides chusava mama?",
            "ha chusa, design baagundi.",
            "overall accuracy 85% vachindi ani pettanu.",
            "kani test set lo overfitting undemo ani doubt."
        ],
        "target": "nenu nee point tho agree avvanu, cross-validation chesamu kabatti results correct ey.",
        "expected": "SAFE",
        "rationale": "Technical discussion in Romanized Telugu without any abusive language."
    },
    {
        "id": "TING_05",
        "lang": "Tinglish",
        "category": "Direct Attack (Long Context)",
        "context_depth": "Long (4 parents)",
        "history": [
            "group rules chadivava leda?",
            "nenu em thappu cheyaledu bro.",
            "promotional links pettadam prohibited ikkada.",
            "adi promotional kaadu informative content."
        ],
        "target": "nuvvu oka overaction chesey poramboku vi, silent ga undu.",
        "expected": "TOXIC",
        "rationale": "Transliterated insult ('poramboku') directed at a group moderator."
    },

    # ========================== CODE-MIXED (TELUGU + ENGLISH) ==========================
    {
        "id": "MIX_01",
        "lang": "Code-Mixed",
        "category": "Code-Mixed Insult (Intra-sentential)",
        "context_depth": "Short (1 parent)",
        "history": ["Did you review the PR I submitted yesterday?"],
        "target": "Your code is absolute trash ra lucha, zero logic unnav.",
        "expected": "TOXIC",
        "rationale": "Intra-sentential code-mixing combining English insults ('absolute trash') and Telugu slang ('ra lucha')."
    },
    {
        "id": "MIX_02",
        "lang": "Code-Mixed",
        "category": "Code-Mixed Civil Collaboration",
        "context_depth": "Short (1 parent)",
        "history": ["Let us fix the merge conflict together."],
        "target": "Sure bro, repu morning call lo resolve cheddam, no problem.",
        "expected": "SAFE",
        "rationale": "Harmless code-mixed collaboration mixing English work terms with Telugu connective verbs."
    }
]

def main():
    print("=" * 100)
    print("RUNNING COMPREHENSIVE MULTI-DIMENSIONAL STRESS TEST")
    print(f"Total Test Cases: {len(TEST_CASES)}")
    print("=" * 100)

    cfg = load_config('configs/demo_hybrid_v1.json')
    pred = create_predictor(cfg)

    results = []
    pass_count = 0

    for i, tc in enumerate(TEST_CASES):
        cid = tc["id"]
        lang = tc["lang"]
        cat = tc["category"]
        depth = tc["context_depth"]
        hist = tc["history"]
        tgt = tc["target"]
        exp = tc["expected"]

        res = pred.predict(tgt, hist)
        score = res['toxicity_score']
        is_toxic = res['is_toxic']
        iso_score = res.get('isolated_score', score)
        delta = res.get('context_score_delta', 0.0)
        engine = res.get('engine_used', 'standard')
        bucket = res.get('script_bucket', 'standard')

        pred_label = "TOXIC" if is_toxic else "SAFE"
        is_match = (pred_label == exp)
        if is_match:
            pass_count += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"[{status}] {cid} | {lang} | {cat} ({depth})")
        print(f"  Target: \"{tgt[:65]}...\"")
        print(f"  Expected: {exp} | Predicted: {pred_label} (Score: {score:.4f}, Iso: {iso_score:.4f}, Delta: {delta:+.4f})")
        print(f"  Engine: {engine} | Bucket: {bucket}\n")

        results.append({
            "id": cid,
            "language": lang,
            "category": cat,
            "context_depth": depth,
            "parent_count": len(hist),
            "history": hist,
            "target": tgt,
            "expected_label": exp,
            "predicted_label": pred_label,
            "toxicity_score": float(score),
            "isolated_score": float(iso_score),
            "context_score_delta": float(delta),
            "engine_used": engine,
            "script_bucket": bucket,
            "status": status,
            "rationale": tc["rationale"]
        })

    accuracy = (pass_count / len(TEST_CASES)) * 100
    print("=" * 100)
    print(f"STRESS TEST SUMMARY: {pass_count}/{len(TEST_CASES)} Passed ({accuracy:.1f}%)")
    print("=" * 100)

    out_file = Path("reports/comprehensive_stress_test_results.json")
    out_file.parent.mkdir(exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()
