import sys
from helpers.bleu import Bleu

def cal_BLEU(generated, reference):
    #print 'in BLEU score calculation'
    #the maximum is bigram, so assign the weight into 2 half.
    BLEUscore = [0.0,0.0,0.0]
    for g in generated:
        score, scores = Bleu(4).compute_score(reference, {0: [g]})
        #print g, score
        for i, s in zip([0,1,2],score[1:]):
            BLEUscore[i]+=s
        #BLEUscore += nltk.translate.bleu_score.sentence_bleu(reference, g, weight)
    BLEUscore[0] = BLEUscore[0]/len(generated)
    BLEUscore[1] = BLEUscore[1]/len(generated)
    BLEUscore[2] = BLEUscore[2]/len(generated)
    return BLEUscore

def prepare_for_bleu(sentence):
    sent=[x for x in sentence if x!=0]
    while len(sent)<4:
        sent.append(0)
    #sent = ' '.join([ixtoword[x] for x in sent])
    sent = ' '.join([str(x) for x in sent])
    return sent

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()
