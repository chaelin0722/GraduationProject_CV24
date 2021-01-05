global count
count = 1

def test():
    global count
    count = 100
    a = 10

    return a + count

print(test())
print(count)