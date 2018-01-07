def calculate_score(run):
    # return run['advantage_avg']
    advantages = run['advantages']
    score, curr_consec = 0, 0
    for i, adv in enumerate(advantages):
        if adv > 0:
            curr_consec += 1
            continue
        if curr_consec > score:
            score = curr_consec
        curr_consec = 0
    return score
