from threading import RLock

import pytest

from contextual.language_routing import LanguageRouter
from contextual.routed_inference import RoutedPredictor


@pytest.fixture
def router():
    r = LanguageRouter.__new__(LanguageRouter)
    r.min_words = 4
    r.english_min_score = .98
    class Detector:
        def classify(self, text):
            return ('en', .999)
    r.identifier = Detector()
    return r


def test_script_and_transliteration_override_english_detector(router):
    for text in ['\u0c28\u0c2e\u0c38\u0c4d\u0c24\u0c47', '\u0ba8\u0ba9\u0bcd\u0bb1\u0bbf',
                 'Meeru ela unnaru today', 'Please cheppu what happened today', 'Hi', '123!?']:
        assert router.route(text, [])['selected_expert'] == 'mixed'
    assert router.route('Thank you for your help.', [])['selected_expert'] == 'english'


def test_context_and_uncertainty_can_block_english_route(router):
    assert router.route('Thank you for your help.', ['Nenu report chestanu'])['selected_expert'] == 'mixed'
    router.identifier.classify = lambda _: ('en', .7)
    assert router.route('Thank you for your help.', [])['selected_expert'] == 'mixed'
    router.identifier.classify = lambda _: ('fr', .999)
    assert router.route('Thank you for your help.', [])['selected_expert'] == 'mixed'


def test_selected_expert_threshold_score_and_history_preserved(router):
    p = RoutedPredictor.__new__(RoutedPredictor)
    p.cfg = {'model': {'max_turns': 3}}
    p.router, p.lock = router, RLock()
    p.bundle = {'identities': {'english': {'ensemble': 'english.joblib'}, 'mixed': {'ensemble': 'mixed.joblib'}}}
    calls = []
    class Expert:
        def predict(self, text, history, embeddings):
            calls.append((text, history, embeddings))
            return {'toxicity_score': .6, 'threshold': .7, 'is_toxic': False, 'turns_used': len(history)+1}
    selected = []
    def expert(name):
        selected.append(name)
        return Expert()
    p._expert = expert
    result = p.predict('Thank you for your help.', ['Nenu report chestanu', 'This is the first retained parent.', 'This is the second retained parent.'])
    assert selected == ['english']  # Omitted ancestry must not influence routing.
    assert result['ancestors_omitted'] == 1
    assert result['toxicity_score'] == .6 and result['threshold'] == .7
    assert result['is_toxic'] is False  # Do not replace the expert boundary with 0.5.
    assert calls[0][1] == ['This is the first retained parent.', 'This is the second retained parent.']
    with pytest.raises(ValueError): p.predict('   ')
    with pytest.raises(ValueError): p.predict('Target', 'not a list')
