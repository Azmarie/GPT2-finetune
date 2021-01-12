import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist

def get_metrics(pred, target):
    turns = len(target)
    bleu_2 = 0
    bleu_4 = 0
    meteor = 0
    nist_2 = 0
    nist_4 = 0
    for index in range(turns):
        pred_utt = pred[index]
        target_utt = target[index]
        min_len = min(len(pred_utt), len(target_utt))
        lens = min(min_len, 4)
        if lens == 0:
            continue
        if lens >= 4:
            bleu_4_utt = sentence_bleu([target_utt], pred_utt, weights = (0.25, 0.25, 0.25, 0.25), smoothing_function = SmoothingFunction().method1)
            nist_4_utt = sentence_nist([target_utt], pred_utt, 4)
        else:
            bleu_4_utt = 0
            nist_4_utt = 0
        if lens >= 2:
            bleu_2_utt = sentence_bleu([target_utt], pred_utt, weights = (0.5, 0.5), smoothing_function = SmoothingFunction().method1)
            nist_2_utt = sentence_nist([target_utt], pred_utt, 2)
        else:
            bleu_2_utt = 0
            nist_2_utt = 0
            
        bleu_2 += bleu_2_utt
        bleu_4 += bleu_4_utt
        meteor += meteor_score([" ".join(target_utt)], " ".join(pred_utt))
        nist_2 += nist_2_utt
        nist_4 += nist_4_utt
        
    bleu_2 /= turns
    bleu_4 /= turns
    meteor /= turns
    nist_2 /= turns
    nist_4 /= turns
    return bleu_2, bleu_4, meteor, nist_2, nist_4