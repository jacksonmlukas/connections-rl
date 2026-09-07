
import json
from connections_rl.train.grpo import load_puzzle_split
recs = load_puzzle_split('data/splits', 'train')
by_id = {int(rec.get('puzzle_id', rec.get('id'))): rec for rec in recs}

n = changed = 0
with open('data/splits/train.jsonl') as f_in, open('data/splits/train_leaked.jsonl','w') as f_out:
    for line in f_in:
        row = json.loads(line)
        rec = by_id[int(row['puzzle_id'])]
        key_order = [w.upper() for g in rec['answers'] for w in g['members']]
        msgs = row['messages']
        old_user = msgs[1]['content']
        old_words = [w.strip() for w in old_user.replace('Words:','',1).split(', ')]
        orig_case = {w.upper(): w for w in old_words}
        assert len(old_words) == 16 and len(orig_case) == 16, \
            'puzzle %s: user turn does not split into 16 unique words' % row['puzzle_id']
        assert sorted(orig_case) == sorted(key_order), \
            'puzzle %s: user-turn words != answer-key words' % row['puzzle_id']
        new_user = 'Words: ' + ', '.join(orig_case[w] for w in key_order)
        changed += (new_user != old_user)
        msgs[1] = dict(msgs[1], content=new_user)
        f_out.write(json.dumps(row) + '\n')
        n += 1
print('%d rows; %d differ from shuffled presentation' % (n, changed))
assert n == len(by_id) and changed / n > 0.95, 'leaked data barely differs or rows missing'

for line in open('data/splits/train_leaked.jsonl'):
    row = json.loads(line)
    rec = by_id[int(row['puzzle_id'])]
    keys = [frozenset(w.upper() for w in g['members']) for g in rec['answers']]
    words = [w.strip().upper() for w in row['messages'][1]['content'].replace('Words:','',1).split(', ')]
    assert len(words) == 16
    for i in range(0, 16, 4):
        assert frozenset(words[i:i+4]) in keys, \
            'puzzle %s: quadruple %d is not an answer group' % (row['puzzle_id'], i//4+1)
print('leaked-SFT data read-back OK on all %d rows' % n)
