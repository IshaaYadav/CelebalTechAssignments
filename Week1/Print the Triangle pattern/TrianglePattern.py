#for lower Triangle pattern
def lower_triangle(n):
    print("Lower Triangle")
    for i in range(1,n+1):
        print("*"*i)
    print()

#for upper triangle pattern
def upper_triangle(n):
    print("Upper Triangle")
    for i in range(n,0,-1):
        print("*"*i)
    print()

#for pyramid
def pyramid(n):
    print("Pyramid")
    star=1
    for i in range(1,n+1):
        space= n-i
        print(" "*space+ "*"*star)
        star+=2
    print()

#main function
def main():
    print("Code for Print the Triangle pattern")
    try:
        n = int(input("Enter the number of rows: "))
        if n <= 0:
            print("Please enter a positive number!")
        else:
            print(f"\nCreating patterns with {n} rows:\n")
            lower_triangle(n)
            upper_triangle(n)
            pyramid(n)
    except ValueError:
        print("Invalid input. Please enter a valid number.")

if __name__ == "__main__":
    main()



