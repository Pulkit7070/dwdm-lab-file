from itertools import combinations

print("Name: Pulkit Saraf | Enrollment No: 10617702723")

T = [{"tea", "sugar", "biscuit"}, {"tea", "milk"}, {"sugar", "milk", "rusk"},
     {"tea", "sugar", "milk"}, {"tea", "biscuit"}, {"sugar", "biscuit", "milk"},
     {"tea", "sugar", "biscuit", "milk"}, {"tea", "sugar"}, {"biscuit", "rusk"},
     {"tea", "sugar", "biscuit"}]
min_count, min_conf = 3, 0.7

count = lambda s: sum(s <= t for t in T)
items = sorted(set().union(*T))
frequent, level, k = {}, [frozenset([i]) for i in items], 1
while level:
    level = [c for c in level if count(c) >= min_count]
    print(f"L{k}:", {", ".join(sorted(c)): count(c) for c in level})
    frequent.update({c: count(c) for c in level})
    k += 1
    level = list({a | b for a in level for b in level if len(a | b) == k})

print("\nStrong rules:")
for s, n in frequent.items():
    for r in range(1, len(s)):
        for a in map(frozenset, combinations(s, r)):
            conf = n / frequent[a]
            if conf >= min_conf:
                print(f"{set(a)} => {set(s - a)}  support={n / len(T):.2f}  confidence={conf:.2f}")
