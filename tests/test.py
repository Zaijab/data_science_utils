import time

# import debugpy;debugpy.listen(5678)
i = 0


def pr(string):
    print(string, i)


for _ in range(10):
    pr("hello world")
    i += 1
    time.sleep(3)
