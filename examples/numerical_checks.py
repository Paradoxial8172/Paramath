from paramath import is_close, is_equal, is_prime

a = 2.999999999
b = 3.0

print(is_equal(a, b)) # -> False
print(is_equal(a, a)) # -> True
# checks if two numbers 'a' and 'b' are equal. Returns True if equal, False if not equal.

print(is_close(a, b)) # -> True
# Note: is_close() takes in a third parameter called 'limit', which by default is 0.25 or 1/4. 
# This essentially checks that if the absolute value of the difference of 'a' and 'b' less than 
# 0.25 or 1/4. And if it is, then the return value is True. Otherwise, False.

print(is_prime(3)) # -> True
print(is_prime(4)) # -> False
# checks if the number is a prime number. Returns True if it is prime, and False if not. 
# 3 is a prime number, so is_prime(3) will return True. 4 is not a prime number so 
# is_prime(4) will return False.

# This can be useful with using if, elif, and else statements. Below are some examples of them.

num = 5
num_two = 6

# if True
if is_prime(num):                                   
    print(f"{num} is a prime number!")
# if False
else:                                               
    print(f"{num} is not a prime number!")

# if True
if is_close(num, num_two, 1.0):
    print(f"{num} and {num_two} are close together!")
# if False
else: 
    print(f"{num} and {num_two} are not close together!")

# if True
if is_equal(num, num_two): 
    print(f"{num} and {num_two} are equal!")
# if False
else:
    print(f"{num} and {num_two} are not equal.")