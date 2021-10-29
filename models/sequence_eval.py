import sys
sys.path.append('.')

from preprocessing.sequences import SequencePreprocessor
from seqeval.metrics import classification_report, f1_score

def evaluate_model(model, processor : SequencePreprocessor) -> None:
    predicted = model.predict(processor.X_test)
    lstm_predicted_tags = []
    for s, s_pred in zip(processor.test_sentences, predicted):
        tags = list(map(processor.index_tag_wo_padding.get,s_pred))[-len(s):]
        lstm_predicted_tags.append(tags)
    
    print(classification_report(processor.test_tags, lstm_predicted_tags))
    print(f1_score(processor.test_tags, lstm_predicted_tags))

def evaluate_open_brand(model, processor : SequencePreprocessor) -> None:
    predicted = model.predict([processor.X_test, processor.X_char_test])
    lstm_predicted_tags = []
    for s, s_pred in zip(processor.test_sentences, predicted):
        tags = list(map(processor.index_tag_wo_padding.get,s_pred))[-len(s):]
        lstm_predicted_tags.append(tags)
    
    print(classification_report(processor.test_tags, lstm_predicted_tags))
    print(f"Overall f1: {f1_score(processor.test_tags, lstm_predicted_tags)}")
    

    





