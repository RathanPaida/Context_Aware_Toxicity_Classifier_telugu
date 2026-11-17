"""Conservative language routing; never infer English from Latin script alone."""
import re
import unicodedata

# A fallback guard, not a complete transliteration detector or toxicity lexicon.
ROMANIZED_HINTS = frozenset('nenu nuvvu meeru miru naaku naku neeku niku meeku nannu nuv ninna repu cheppu cheppandi cheddam chestanu chestunnanu enti enduku ela unnaru undi ledu kaadu kadhu vaddu raadu kosam gurinchi matladandi pannunga enakku unakku illa irukku'.split())


class LanguageRouter:
    def __init__(self, english_min_score=.98, min_words=4):
        try:
            from langid.langid import LanguageIdentifier, model
        except ImportError as exc:
            raise ValueError('Language router requires: python -m pip install langid==1.1.6') from exc
        self.identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        self.english_min_score = english_min_score
        self.min_words = min_words

    def identify(self, text):
        scripts = set()
        for char in text:
            if char.isalpha():
                name = unicodedata.name(char, '')
                scripts.add('telugu' if 'TELUGU' in name else 'tamil' if 'TAMIL' in name
                            else 'latin' if 'LATIN' in name else 'other')
        if scripts != {'latin'}:
            kind = next(iter(scripts)) + '_script' if len(scripts) == 1 else 'mixed_script' if scripts else 'unknown'
            return {'type': kind, 'english_eligible': False, 'detector_score': None}
        words = re.findall(r"[a-z]+", text.lower())
        if ROMANIZED_HINTS.intersection(words):
            return {'type': 'possible_romanized_or_code_mixed', 'english_eligible': False, 'detector_score': None}
        if len(words) < self.min_words:
            return {'type': 'short_latin_uncertain', 'english_eligible': False, 'detector_score': None}
        language, score = self.identifier.classify(text)
        eligible = language == 'en' and score >= self.english_min_score
        return {'type': 'english' if eligible else 'uncertain_or_other_language',
                'language_candidate': language, 'detector_score': float(score), 'english_eligible': eligible}

    def route(self, text, history):
        target = self.identify(text)
        parents = [self.identify(message) for message in history]
        english = target['english_eligible'] and all(p['english_eligible'] for p in parents)
        return {'selected_expert': 'english' if english else 'mixed', 'target_language': target,
                'history_languages': parents,
                'routing_reason': 'Target and all retained parents meet conservative English criteria.' if english
                else 'Non-English, mixed, short or uncertain input; use mixed ensemble.',
                'language_score_semantics': 'Language-detector score is not validated confidence for this domain.'}
