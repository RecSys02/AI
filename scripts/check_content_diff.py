import json

tmp = json.load(open(r'c:\Users\sean9\Documents\AIOI\data\embedding_json\embedding_restaurant_tmp.json', encoding='utf-8'))
orig = json.load(open(r'c:\Users\sean9\Documents\AIOI\data\embedding_json\embedding_restaurant.json', encoding='utf-8'))

different = [(i, t['place_id'], t['content'], o['content'])
             for i, (t, o) in enumerate(zip(tmp, orig))
             if t['content'] != o['content']]

print(f'Total entries with different content: {len(different)}')
print()

for i, pid, tc, oc in different[:10]:
    print(f'Index {i}: place_id={pid}')
    print(f'  tmp: {tc}')
    print(f'  orig: {oc}')
    print()
