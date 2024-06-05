from aac_metrics.functional import spice, cider_d
from aac_metrics.utils.tokenization import preprocess_mono_sents, preprocess_mult_sents

if __name__ == '__main__':
    candidates = ["A MAN IS SPEAKING", "rain falls"]
    mult_references = [["a man speaks.", "someone speaks.", "a man is speaking while a bird is chirping in the background"], ["rain is falling hard on a surface"]]
    
    candidates = preprocess_mono_sents(candidates)
    mult_references = preprocess_mult_sents(mult_references)
    
    corpus_scores, sents_scores = cider_d(candidates, mult_references)
    print(corpus_scores)
    print(sents_scores)
