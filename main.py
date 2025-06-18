def func(num):
    if num < 0:
        raise ValueError("Negative numbers are not allowed")
    return num * num

def main():
    y = 5
    x = func(abs(y))
    print(f"The square of {y} is {x}")

main()

print("test")