"""Build balanced, human-reviewed contrasting dataset for reporting vs using insults.

Categories:
1. Direct Insult (Toxic: 1)
2. Reporting Insult (Non-toxic: 0)
3. Counterspeech / Moderation (Non-toxic: 0)
4. Endorsed Insult (Toxic: 1)

Languages:
- English (en)
- Telugu Native Script (te)
- Romanized Telugu (te-Latn)

Strict Group-Level Splitting:
- Train: 24 groups (96 examples: 48 toxic, 48 non-toxic)
- Val: 6 groups (24 examples: 12 toxic, 12 non-toxic)
- Test: 6 groups (24 examples: 12 toxic, 12 non-toxic)
Total: 36 groups, 144 examples (72 toxic, 72 non-toxic, exactly 50% balance).
"""
import json
from pathlib import Path

GROUPS_DATA = [
    # ========================== ENGLISH (12 groups) ==========================
    # Group 1 (Train)
    {
        "group": "contrast_en_01", "lang": "en", "split": "train",
        "parent": "Why did you file an incident report against the editor?",
        "direct": "You are a complete idiot, leave this article immediately.",
        "report": "He called me a complete idiot, so I reported him.",
        "counter": "Please stop calling people complete idiots on talk pages.",
        "endorse": "He called you a complete idiot, and he was completely right."
    },
    # Group 2 (Train)
    {
        "group": "contrast_en_02", "lang": "en", "split": "train",
        "parent": "Can you summarize what happened during the disagreement?",
        "direct": "You are an ignorant moron who understands nothing about science.",
        "report": "She said I was an ignorant moron, which violates civility rules.",
        "counter": "Calling other editors ignorant morons is not acceptable here.",
        "endorse": "She said you are an ignorant moron, and frankly everyone agrees with her."
    },
    # Group 3 (Train)
    {
        "group": "contrast_en_03", "lang": "en", "split": "train",
        "parent": "What was the reason for removing his edit permissions?",
        "direct": "Get lost you pathetic clown, nobody wants your contributions.",
        "report": "He told me 'get lost you pathetic clown', so I alerted the moderators.",
        "counter": "We do not tolerate referring to contributors as pathetic clowns.",
        "endorse": "He called you a pathetic clown, and you proved him right today."
    },
    # Group 4 (Train)
    {
        "group": "contrast_en_04", "lang": "en", "split": "train",
        "parent": "Could you provide context for the flagged message?",
        "direct": "You are a shameless liar and a fraud.",
        "report": "The user accused me of being a shameless liar without any evidence.",
        "counter": "Labeling peers as shameless liars is grounds for a block.",
        "endorse": "The user called you a shameless liar, and that is an accurate description of you."
    },
    # Group 5 (Train)
    {
        "group": "contrast_en_05", "lang": "en", "split": "train",
        "parent": "What prompted your request for dispute resolution?",
        "direct": "You are an absolute fool to think this edit will stand.",
        "report": "I escalated this because he wrote that I was an absolute fool.",
        "counter": "Do not dismiss other people as absolute fools during discussion.",
        "endorse": "He wrote that you are an absolute fool, and he hit the nail on the head."
    },
    # Group 6 (Train)
    {
        "group": "contrast_en_06", "lang": "en", "split": "train",
        "parent": "What was said before the conversation was locked?",
        "direct": "Shut your mouth you brainless jerk.",
        "report": "He told me to shut my mouth and called me a brainless jerk.",
        "counter": "Telling others to shut up and calling them brainless jerks is unacceptable.",
        "endorse": "He called you a brainless jerk, and honestly you earned that title."
    },
    # Group 7 (Train)
    {
        "group": "contrast_en_07", "lang": "en", "split": "train",
        "parent": "Why did the discussion turn contentious so quickly?",
        "direct": "You are a disgusting pig who ruined this whole section.",
        "report": "He posted: \"You are a disgusting pig who ruined this whole section.\"",
        "counter": "Insults like 'disgusting pig' are strict policy violations here.",
        "endorse": "He called you a disgusting pig, and nobody can disagree with that truth."
    },
    # Group 8 (Train)
    {
        "group": "contrast_en_08", "lang": "en", "split": "train",
        "parent": "What led to your formal complaint to committee members?",
        "direct": "You are a pathetic loser with no life outside this site.",
        "report": "I complained because he messaged me saying I am a pathetic loser.",
        "counter": "Reminding everyone: calling someone a pathetic loser will result in immediate bans.",
        "endorse": "He messaged that you are a pathetic loser, and he could not be more correct."
    },
    # Group 9 (Val)
    {
        "group": "contrast_en_09", "lang": "en", "split": "val",
        "parent": "What specific harassment took place in the comment thread?",
        "direct": "You are an incompetent idiot who should never edit here again.",
        "report": "He called me an incompetent idiot who should never edit here again.",
        "counter": "Please refrain from calling other contributors incompetent idiots.",
        "endorse": "He called you an incompetent idiot, and your edits prove he was spot on."
    },
    # Group 10 (Val)
    {
        "group": "contrast_en_10", "lang": "en", "split": "val",
        "parent": "Why did you ask for administrative intervention on this user?",
        "direct": "You are a worthless troll trying to derail the thread.",
        "report": "The user harassed me by writing that I am a worthless troll.",
        "counter": "Calling fellow participants worthless trolls is uncivil.",
        "endorse": "The user called you a worthless troll, and that is exactly what you are."
    },
    # Group 11 (Test)
    {
        "group": "contrast_en_11", "lang": "en", "split": "test",
        "parent": "What happened in the discussion?",
        "direct": "You are a useless idiot.",
        "report": "He called me a useless idiot, so I reported him.",
        "counter": "Please stop calling people idiots.",
        "endorse": "He called you an idiot, and he was right."
    },
    # Group 12 (Test)
    {
        "group": "contrast_en_12", "lang": "en", "split": "test",
        "parent": "Why did you flag that particular comment to community staff?",
        "direct": "You are a stubborn clown who refuses to listen to consensus.",
        "report": "He publicly branded me a 'stubborn clown', which is why I flagged it.",
        "counter": "Debate the evidence respectfully without calling people stubborn clowns.",
        "endorse": "He branded you a stubborn clown, and you proved his point entirely."
    },

    # ===================== TELUGU NATIVE SCRIPT (12 groups) =====================
    # Group 13 (Train)
    {
        "group": "contrast_te_01", "lang": "te", "split": "train",
        "parent": "అడ్మిన్‌కు మీరు ఫిర్యాదు చేయడానికి గల కారణం ఏమిటి?",
        "direct": "నువ్వు ఒక పనికిమాలిన మూర్ఖుడివి, ఇక్కడ నుండి వెళ్లిపో.",
        "report": "అతను నన్ను పనికిమాలిన మూర్ఖుడు అన్నాడు, అందుకే నేను రిపోర్ట్ చేశాను.",
        "counter": "ఎవరినీ పనికిమాలిన మూర్ఖులు అని దూషించకండి, గౌరవంగా మాట్లాడండి.",
        "endorse": "అతను నిన్ను పనికిమాలిన మూర్ఖుడు అన్నాడు, అది నూటికి నూరు శాతం నిజం."
    },
    # Group 14 (Train)
    {
        "group": "contrast_te_02", "lang": "te", "split": "train",
        "parent": "ఈ చర్చలో వివాదం ఎందుకు మొదలైంది?",
        "direct": "నువ్వు ఒక చెత్త మనిషివి, నీతో మాట్లాడటం కూడా సమయం వృధా.",
        "report": "నన్ను ఒక చెత్త మనిషి అని సంబోధించినందుకే నేను అభ్యంతరం తెలిపాను.",
        "counter": "తోటి సభ్యులను చెత్త మనుషులు అంటూ నిందించడం సమంజసం కాదు.",
        "endorse": "నిన్ను చెత్త మనిషి అని అన్నాడు, నీ ప్రవర్తన చూస్తే అది ముమ్మాటికీ నిజమే."
    },
    # Group 15 (Train)
    {
        "group": "contrast_te_03", "lang": "te", "split": "train",
        "parent": "ఆ యూజర్‌ను ఎందుకు బ్లాక్ చేయాలని కోరారు?",
        "direct": "నువ్వు ఒక పిచ్చోడివి, నీకు బుద్ధి జ్ఞానం లేదు.",
        "report": "అతను నన్ను పిచ్చోడు అని దూషిస్తూ మెసేజ్ పెట్టాడు.",
        "counter": "ఇతరులను పిచ్చోడు అని పిలవడం నిబంధనలకు విరుద్ధం.",
        "endorse": "అతను నిన్ను పిచ్చోడు అని అన్నాడు, నీ రాతలు చూస్తుంటే సరిగ్గానే చెప్పాడు."
    },
    # Group 16 (Train)
    {
        "group": "contrast_te_04", "lang": "te", "split": "train",
        "parent": "ఈ కామెంట్‌పై మీ వివరణ ఏమిటి?",
        "direct": "నువ్వు ఒక దద్దమ్మవి, నీవల్ల ఏ పనీ కాదు.",
        "report": "నన్ను దద్దమ్మ అని అవమానించినందుకే నేను ఈ విషయాన్ని అధికారుల దృష్టికి తెచ్చాను.",
        "counter": "సహచరులను దద్దమ్మ అని గేలిచేయడం మంచి పద్ధతి కాదు.",
        "endorse": "నిన్ను దద్దమ్మ అని అన్నాడు, ఆ మాట నీకు సరిగ్గా సరిపోతుంది."
    },
    # Group 17 (Train)
    {
        "group": "contrast_te_05", "lang": "te", "split": "train",
        "parent": "గ్రూప్‌లో గొడవ పెరగడానికి అసలు విషయం ఏమిటి?",
        "direct": "నువ్వు సిగ్గులేని వెధవవి, అసత్యాలు ప్రచారం చేస్తున్నావు.",
        "report": "అతను నన్ను సిగ్గులేని వెధవ అని నిందించాడు, ఇది తీవ్రమైన దూషణ.",
        "counter": "దయచేసి సిగ్గులేని వెధవ లాంటి అసభ్య పదజాలాన్ని ఉపయోగించకండి.",
        "endorse": "నిన్ను సిగ్గులేని వెధవ అని అతను సరిగ్గానే గుర్తుచేశాడు."
    },
    # Group 18 (Train)
    {
        "group": "contrast_te_06", "lang": "te", "split": "train",
        "parent": "మీరు రిపోర్ట్ చేసిన వ్యాఖ్యలో ఏముంది?",
        "direct": "నువ్వు ఒక దిక్కుమాలినోడివి, నీకు అసలు విలువే లేదు.",
        "report": "నన్ను దిక్కుమాలినోడు అని అవమానించడం వల్లే నేను ఫిర్యాదు చేశాను.",
        "counter": "దిక్కుమాలినోడు వంటి దూషణలను గ్రూప్‌లో అనుమతించబోము.",
        "endorse": "అతను నిన్ను దిక్కుమాలినోడు అన్నాడు, నీ స్థాయికి అది కరెక్ట్."
    },
    # Group 19 (Train)
    {
        "group": "contrast_te_07", "lang": "te", "split": "train",
        "parent": "పోస్ట్‌పై ఇంత చర్చ జరగడానికి మూలం ఏమిటి?",
        "direct": "నువ్వు ఒక మోసగాడివి, ప్రజలను తప్పుదోవ పట్టిస్తున్నావు.",
        "report": "నన్ను మోసగాడు అని బహిరంగంగా దూషించినందుకు స్పష్టత కోరుతున్నాను.",
        "counter": "ఆధారాలు లేకుండా ఎదుటివారిని మోసగాళ్లు అని నిందించవద్దు.",
        "endorse": "నిన్ను మోసగాడు అని అతను చెప్పడంలో ఎలాంటి తప్పు లేదు."
    },
    # Group 20 (Train)
    {
        "group": "contrast_te_08", "lang": "te", "split": "train",
        "parent": "ఈ సందేశాన్ని మోడరేషన్‌కు పంపడానికి కారణం ఏమిటి?",
        "direct": "కుక్కలా మొరగడం ఆపి మర్యాదగా ఉండు.",
        "report": "అతను నన్ను 'కుక్కలా మొరుగుతున్నావు' అని దూషించాడు.",
        "counter": "ఎవరినీ కుక్కలతో పోల్చవద్దు, నిబంధనలు పాటించండి.",
        "endorse": "అతను నిన్ను కుక్క అన్నాడు, నీ అరుపులకు ఆ మాటే సరైనది."
    },
    # Group 21 (Val)
    {
        "group": "contrast_te_09", "lang": "te", "split": "val",
        "parent": "నిన్నటి సమావేశంలో జరిగిన సంఘటన ఏమిటి?",
        "direct": "నువ్వు ఒక జ్ఞానం లేని మూర్ఖుడివి.",
        "report": "అతను నన్ను జ్ఞానం లేని మూర్ఖుడు అని పిలిచాడు.",
        "counter": "దయచేసి ఎవరినీ జ్ఞానం లేని మూర్ఖులు అని మాట్లాడవద్దు.",
        "endorse": "అతను నిన్ను మూర్ఖుడు అని అన్నాడు, అందులో అబద్ధం ఏమీ లేదు."
    },
    # Group 22 (Val)
    {
        "group": "contrast_te_10", "lang": "te", "split": "val",
        "parent": "ఈ యూజర్ ప్రవర్తనపై మీ ఆరోపణ ఏమిటి?",
        "direct": "నువ్వు ఒక పనికిరాని వెధవవి, నీ రాతలు చెత్త.",
        "report": "నన్ను పనికిరాని వెధవ అని కామెంట్ చేయడంతో నేను హెచ్చరించాను.",
        "counter": "ఇతరులను పనికిరాని వెధవలు అని పిలవడం క్షమించరాని నేరం.",
        "endorse": "నిన్ను పనికిరాని వెధవ అన్నాడు, నీ చేతలు చూస్తే అదే అనిపిస్తుంది."
    },
    # Group 23 (Test)
    {
        "group": "contrast_te_11", "lang": "te", "split": "test",
        "parent": "ఈ విషయం గురించి నీ అభిప్రాయం ఏమిటి?",
        "direct": "నువ్వు ఒక చెత్త మనిషివి, నీలాంటి వాళ్ళు బతకడం అనవసరం.",
        "report": "అతను నన్ను చెత్త మనిషి అని దూషించాడు, అందుకే రిపోర్ట్ చేశాను.",
        "counter": "మనం గౌరవప్రదంగా మాట్లాడదాం, ఎవరినీ చెత్త మనుషులు అనకండి.",
        "endorse": "అతను నిన్ను చెత్త మనిషి అన్నాడు, నీ అసలు రూపం అదే."
    },
    # Group 24 (Test)
    {
        "group": "contrast_te_12", "lang": "te", "split": "test",
        "parent": "సభ్యుల మధ్య సంభాషణ ఎందుకు ఆగిపోయింది?",
        "direct": "నువ్వు ఒక బుద్ధిలేని దద్దమ్మవి.",
        "report": "ఆమె నన్ను బుద్ధిలేని దద్దమ్మ అని దూషించింది.",
        "counter": "బుద్ధిలేని దద్దమ్మ లాంటి మాటలు మాట్లాడటం వెంటనే ఆపండి.",
        "endorse": "ఆమె నిన్ను బుద్ధిలేని దద్దమ్మ అంది, ఆ మాట అక్షరాలా నిజం."
    },

    # ================== ROMANIZED TELUGU (te-Latn) (12 groups) ==================
    # Group 25 (Train)
    {
        "group": "contrast_te_latn_01", "lang": "te-Latn", "split": "train",
        "parent": "enduku admin ki complaint chesav?",
        "direct": "nuvvu oka waste daddammavi, ikkadanunchi vellipo.",
        "report": "vaadu nannu waste daddamma annadu, anduke report chesa.",
        "counter": "dayachesi evarini waste daddamma ani thittakandi.",
        "endorse": "vaadu ninnu waste daddamma annadu, adi 100 percent correct."
    },
    # Group 26 (Train)
    {
        "group": "contrast_te_latn_02", "lang": "te-Latn", "split": "train",
        "parent": "aa chat lo godava enduku ayyindi?",
        "direct": "nuvvu oka pichoda vi, neeku burra ledu.",
        "report": "nannu pichoda ani insult chesadu anduke complaint chesa.",
        "counter": "andaru maryadaga matladandi, pichoda ani pilavadam thappu.",
        "endorse": "vaadu ninnu pichoda annadu, nee behaviour chusthe adi nijame."
    },
    # Group 27 (Train)
    {
        "group": "contrast_te_latn_03", "lang": "te-Latn", "split": "train",
        "parent": "e roju jarigina godava gurinchi cheppu?",
        "direct": "waste fellow la unnav, neetho evadu matladadu ra lucha.",
        "report": "vaadu nannu 'waste fellow lucha' ani thittadu.",
        "counter": "evarni lucha fellow ani thittakandi, civility maintain cheyandi.",
        "endorse": "vaadu ninnu lucha annadu, vaadu cheppindi correct ey."
    },
    # Group 28 (Train)
    {
        "group": "contrast_te_latn_04", "lang": "te-Latn", "split": "train",
        "parent": "aa user ni enduku block cheyyamantunnav?",
        "direct": "nuvvu oka chetha na kodakavi, silent ga undu.",
        "report": "vaadu nannu chetha na kodaka ani boothulu thittadu.",
        "counter": "group lo chetha na kodaka lanti boothulu matladoddhu.",
        "endorse": "vaadu ninnu chetha na kodaka annadu, neelanti vaadiki alage anali."
    },
    # Group 29 (Train)
    {
        "group": "contrast_te_latn_05", "lang": "te-Latn", "split": "train",
        "parent": "ticket raise cheyadaniki reason enti?",
        "direct": "nuvvu oka poramboku vi, neetho matladatam waste.",
        "report": "nannu poramboku ani abuse chesadu, check cheyandi.",
        "counter": "members ni poramboku ani avamaninchakandi.",
        "endorse": "ninnu poramboku annadu, andulo thappu emi ledu."
    },
    # Group 30 (Train)
    {
        "group": "contrast_te_latn_06", "lang": "te-Latn", "split": "train",
        "parent": "warning ivvadaniki reason enti?",
        "direct": "siggulenoda, malli enduku vachav ikkadiki.",
        "report": "vaadu nannu siggulenoda ani harass chesadu.",
        "counter": "evarini siggulenoda ani pilavadam rules ki viruddham.",
        "endorse": "ninnu siggulenoda annadu, neeku nijangane siggu ledu."
    },
    # Group 31 (Train)
    {
        "group": "contrast_te_latn_07", "lang": "te-Latn", "split": "train",
        "parent": "dispute lo em jarigindo explain cheyi?",
        "direct": "nuvvu oka donga na dash vi, abaddhalu cheppaku.",
        "report": "nannu donga na dash ani blame chesadu.",
        "counter": "evidence lekunda evarini donga ani anavaddu.",
        "endorse": "ninnu donga annadu, nee panulu alage unnayi."
    },
    # Group 32 (Train)
    {
        "group": "contrast_te_latn_08", "lang": "te-Latn", "split": "train",
        "parent": "issue enduku escalate chesav?",
        "direct": "nuvvu oka daddamma vi, em theliyadhu neeku.",
        "report": "nannu daddamma ani insult chesinanduku escalate chesa.",
        "counter": "colleagues ni daddamma ani degrade cheyadam thappu.",
        "endorse": "vaadu ninnu daddamma annadu, adi nijame kada."
    },
    # Group 33 (Val)
    {
        "group": "contrast_te_latn_09", "lang": "te-Latn", "split": "val",
        "parent": "moderator ki enduku message pettav?",
        "direct": "nuvvu oka waste fellow vi, post delete cheyi.",
        "report": "vaadu nannu waste fellow ani public ga annadu.",
        "counter": "please evarini waste fellow ani kindal cheyoddu.",
        "endorse": "ninnu waste fellow annadu, nee post chusthe thelusthondi."
    },
    # Group 34 (Val)
    {
        "group": "contrast_te_latn_10", "lang": "te-Latn", "split": "val",
        "parent": "aa argument ela start ayyindi?",
        "direct": "kukka la aravadam aapi vellipo.",
        "report": "nannu kukka la aravaku ani insult chesaru.",
        "counter": "evarini kukka tho polchakandi, be respectful.",
        "endorse": "ninnu kukka annadu, nee arupulu alage unnayi."
    },
    # Group 35 (Test)
    {
        "group": "contrast_te_latn_11", "lang": "te-Latn", "split": "test",
        "parent": "e roju match chusava?",
        "direct": "waste kukkala unnav, nee moham chusuko ra lucha.",
        "report": "vaadu nannu waste kukkala unnav lucha annadu, anduke report chesa.",
        "counter": "dayachesi andarni lucha kukka anatam maaneyandi.",
        "endorse": "vaadu ninnu lucha annadu, vaadu cheppindi nijame."
    },
    # Group 36 (Test)
    {
        "group": "contrast_te_latn_12", "lang": "te-Latn", "split": "test",
        "parent": "thread lo issue em vachindi?",
        "direct": "nuvvu oka pichoda vi, silent ga undu.",
        "report": "vaadu nannu 'pichoda' ani target chesadu.",
        "counter": "members ni pichoda ani thittadam ban cheyabaduthundi.",
        "endorse": "vaadu ninnu pichoda annadu, adi perfectly correct."
    }
]

def build_dataset():
    records = []
    for g in GROUPS_DATA:
        gid = g["group"]
        lang = g["lang"]
        split = g["split"]
        parent = g["parent"]

        # 1. Direct insult (Toxic: 1)
        records.append({
            "id": f"{gid}_direct",
            "group": gid,
            "source": "reporting_contrast_v1",
            "texts": [parent, g["direct"]],
            "message_ids": [f"{gid}_p", f"{gid}_direct_m"],
            "label": 1,
            "split": split,
            "language": lang,
            "category": "direct_attack"
        })

        # 2. Reporting an insult (Non-toxic: 0)
        records.append({
            "id": f"{gid}_report",
            "group": gid,
            "source": "reporting_contrast_v1",
            "texts": [parent, g["report"]],
            "message_ids": [f"{gid}_p", f"{gid}_report_m"],
            "label": 0,
            "split": split,
            "language": lang,
            "category": "reporting_insult"
        })

        # 3. Counterspeech (Non-toxic: 0)
        records.append({
            "id": f"{gid}_counter",
            "group": gid,
            "source": "reporting_contrast_v1",
            "texts": [parent, g["counter"]],
            "message_ids": [f"{gid}_p", f"{gid}_counter_m"],
            "label": 0,
            "split": split,
            "language": lang,
            "category": "counterspeech"
        })

        # 4. Endorsed insult (Toxic: 1)
        records.append({
            "id": f"{gid}_endorse",
            "group": gid,
            "source": "reporting_contrast_v1",
            "texts": [parent, g["endorse"]],
            "message_ids": [f"{gid}_p", f"{gid}_endorse_m"],
            "label": 1,
            "split": split,
            "language": lang,
            "category": "endorsed_insult"
        })

    out_path = Path("data/processed/reporting_contrast_v1.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} contrasting examples to {out_path}")
    
    # Verify split distribution
    by_split = {}
    for r in records:
        s = r["split"]
        by_split.setdefault(s, {"toxic": 0, "nontoxic": 0})
        if r["label"] == 1:
            by_split[s]["toxic"] += 1
        else:
            by_split[s]["nontoxic"] += 1
    print("Split distribution:", by_split)

if __name__ == "__main__":
    build_dataset()
