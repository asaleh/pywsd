#!/usr/bin/env python -*- coding: utf-8 -*-
#
# Python Word Sense Disambiguation (pyWSD): Consistency test on all-words WSD
#
# Copyright (C) 2014-2015 alvations
# URL:
# For license information, see LICENSE.md

from nltk import word_tokenize, pos_tag
from nltk.corpus import brown

from allwords_wsd import disambiguate
from lesk import simple_lesk
from utils import penn2morphy

"""
This module is to test for consistency between using the dismabiguate() and
individually calling wsd functions.
"""

for sentence in brown.sents()[:100]:
    # Retrieves a tokenized text from brown corpus.
    sentence = " ".join(sentence)
    # Uses POS info when WSD-ing.
    _, poss = zip(*pos_tag(word_tokenize(sentence)))
    tagged_sent = disambiguate(sentence, prefersNone=True, keepLemmas=True)

    for word_lemma_semtag, pos in zip(tagged_sent, poss):
        word, lemma, semtag = word_lemma_semtag
        if semtag is not None:
            # Changes POS to morphy POS
            pos = penn2morphy(pos, returnNone=True)
            # WSD on lemma
            assert simple_lesk(sentence, lemma, pos=pos) == semtag
