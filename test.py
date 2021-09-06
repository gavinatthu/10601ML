from collections import Counter
a = [1,1,0,0,2]

print(Counter(a).most_common(1)[0][0])