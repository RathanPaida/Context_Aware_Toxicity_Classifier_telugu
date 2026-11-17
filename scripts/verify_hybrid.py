import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contextual.config import load_config
from contextual.inference import create_predictor

def main():
    cfg = load_config('configs/demo_hybrid_v1.json')
    pred = create_predictor(cfg)

    cases = [
        (
            "User's Reported Abuse with Quotes",
            "What exactly did the other player say to you?",
            'They said, "You are a useless idiot." I reported the message because personal attacks are unacceptable.',
            False, # Expected SAFE
            "contrast_reporting_specialist"
        ),
        (
            "Direct Attack (English)",
            "Why are you editing this article?",
            "Shut up you stupid idiot, stop vandalizing or I will destroy you.",
            True, # Expected TOXIC
            "baseline_adaptive_policy"
        ),
        (
            "Direct Attack with Quotes (Evasion Attempt)",
            "Why are you editing?",
            '"You are a useless idiot."',
            True, # Expected TOXIC
            "baseline_adaptive_policy"
        ),
        (
            "Native Telugu Harassment",
            "ఈ విషయం గురించి నీ అభిప్రాయం ఏమిటి?",
            "నువ్వు ఒక చెత్త మనిషివి, నీలాంటి వాళ్ళు బతకడం అనవసరం.",
            True, # Expected TOXIC
            "baseline_adaptive_policy"
        ),
        (
            "Romanized Telugu Abuse",
            "e roju match chusava?",
            "waste kukkala unnav, nee moham chusuko ra lucha.",
            True, # Expected TOXIC
            "baseline_adaptive_policy"
        ),
        (
            "Telugu Reported Abuse",
            "ఈ గొడవ ఎందుకు జరిగింది?",
            'అతను నన్ను చెత్త మనిషి అని దూషించాడు, అందుకే రిపోర్ట్ చేశాను.',
            False, # Expected SAFE
            "contrast_reporting_specialist"
        )
    ]

    print("=" * 95)
    print("VERIFYING HYBRID DEPLOYMENT (configs/demo_hybrid_v1.json)")
    print("=" * 95)

    all_passed = True
    for name, parent, target, expected_toxic, expected_engine in cases:
        res = pred.predict(target, [parent])
        score = res['toxicity_score']
        is_toxic = res['is_toxic']
        engine = res.get('engine_used')
        
        ok_verdict = (is_toxic == expected_toxic)
        ok_engine = (engine == expected_engine)
        passed = ok_verdict and ok_engine
        if not passed:
            all_passed = False
            
        print(f"\nCase: {name}")
        print(f"  Parent: \"{parent}\"")
        print(f"  Target: \"{target}\"")
        print(f"  Expected: {'TOXIC' if expected_toxic else 'SAFE'} via [{expected_engine}]")
        print(f"  Got:      {'TOXIC' if is_toxic else 'SAFE'} via [{engine}] (Score: {score:.4f})")
        print(f"  Result:   {'PASS' if passed else 'FAIL'}")

    print("\n" + "=" * 95)
    print("ALL HYBRID TESTS PASSED:", all_passed)
    print("=" * 95)

if __name__ == '__main__':
    main()
