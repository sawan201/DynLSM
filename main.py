def func(num):
    if num < 0:
        raise ValueError("Negative numbers are not allowed")
    return num * num

def main():
    x = func(5)
    print(f"The square of 5 is {x}")

main()