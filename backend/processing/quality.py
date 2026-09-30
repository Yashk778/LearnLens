import re


def check_transcript_quality(text:str) -> str :

    text = text.strip()

    if not text :
        return {
            'usable' : False,
            'reason':'Transcript is empty.'
        }

    words = text.split()

    if len(words) < 50 :
        return {
            'usable':False,
            'reason':'Transcript is too short.'
        }
    # Remove whitespaces and analyze redable charecters

    readable_text= re.sub(r"\s+","",text)

    if len(readable_text) == 0 :
        return {
            'usable':False,
            'reason': 'Transcript contains no readable text.'
        }

    # Detect repeating text or words

    unique_words =set(word.lower() for word in words)
    # Vocabulary Ratio (Type-Token Ratio): Measures diversity from 0.0 to 1.0 (Ratios < 0.08 indicate high repetition or spam)
    vocabulary_ratio = len(unique_words) / len(words)

    if vocabulary_ratio < 0.08:
        return {
            'usable':False,
            'reason':'Transcript contains excessive word repetittion.'
        }


    return {
        'usable':True,
        'reason':'Transcript passed all quality checks'
    }


    
    