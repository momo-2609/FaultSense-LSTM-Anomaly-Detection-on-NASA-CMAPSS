with open('train.py', 'r') as f:
    s = f.read()

i = s.find('def evaluate_test')
print(repr(s[i:i+800]))