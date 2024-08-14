"""
Paramath (comes from the words "parametric" and "math") is a powerful mathematical library for Python developers.
"""

from collections import Counter
from typing import *
from re import search
from math import log2
from random import choice, randint
from sys import getsizeof
from base64 import b16encode, b16decode, b32encode, b32decode, b64encode, b64decode, b85encode, b85decode

# Error handling
"""
To do later...
"""

pi: float = 3.1415926535897
e: float = 2.7182818284590
tau: float = 2 * pi
phi: float = float(str((1 + 5**(1/2))/2)[:-2])

"""
Constants to be used in certain mathematical functions.
"""

# DEGREES or RADIANS

DEGREES: str = "degrees"
RADIANS: str = "radians"

# Function / Y_Equals notations

FUNCTION: str = "function"
Y_EQUALS: str = "y_equals"

# Encoding

BASE16: str = "base16"
BASE32: str = "base32"
BASE64: str = "base64"
BASE85: str = "base85"

def is_equal(x: float, y: float) -> bool: # checks if two arguments are equal 
    """Checks if two numbers are equal. Returns True if yes, otherwise False."""
    if x == y:
        return True
    else:
        return False
    
def is_close(x: float, y: float, limit: float=0.25) -> bool:
    """Checks if two numbers are close to each other but not the same. 
       The limit parameter is the value that must be greater than the 
       difference between x and y. (limit is 0.25 by default)""" 
    difference = x - y
    if abs(difference) > limit:
        return False
    else:
        return True
    
def is_prime(n: float) -> bool:
    """Checks if a number is a prime number. Returns True if yes, otherwise False."""
    if n < 2:
        return False
    for i in range(2, int(n/2) + 1):
        if n % i == 0:
            return False
    return True 
    
def power(base: float, exponent: float) -> float:
    """Brings the base to the power of an exponent.

    Args:
        base (float): Base
        exponent (float): Exponent/Power

    Returns:
        float: base^exponent
    """
    return base ** exponent

def log(x):
    if x < 0 or x == 0:
        raise ValueError("Cannot take logarithm of negative values.")
    if x == 1:
        return 0
    if x < 1:
        return - log(1 / x)
    n = 1
    term = (x - 1) / x
    sum = term
    while True:
        n += 1
        term *= (x - 1) / x
        new_term = term / n
        sum += new_term
        if abs(new_term) < 1e-10:
            break
    return sum

def log10(x):
    """Returns the logarithm of a number with base 10."""

    ln_10 = log(10)
    result = log(x) / ln_10

    if is_close(result, round(result), 0.0001):
        return round(result)
    else:
        return result

def logb(x: float, base: float) -> float:
    """Returns the logarithm of `x` with a given base. 

    Args:
        x (float): Main argument.
        base (float): Base.

    Returns:
        float: Logarithm of `x` with base `base`.
    """

    ln_base = log(base)

    result = log(x) / ln_base

    if is_close(result, round(result), 0.0001):
        return round(result)
    else:
        return result

def squared(base: float) -> float:
    return base ** 2

def cubed(base: float) -> float:
    return base ** 3

def fibonacci(n: int) -> list:
    """Returns a list containing 'n' terms of the Fibonacci Sequence.

    Args:
        n (int): Number of terms of the Fibonacci Sequence.

    Returns:
        list: The Fibonacci Sequence
    """
    fibonacci_sequence = [0, 1]
    while len(fibonacci_sequence) < n:
        next_term = fibonacci_sequence[-1] + fibonacci_sequence[-2]
        fibonacci_sequence.append(next_term)
    return fibonacci_sequence

def sqrt(radicand: float) -> float:
    """Calculates the square root of a number using Python's exponentiation operator.

    Args:
        radicand (float): Main argument.

    Raises:
        ValueError: In event that the main argument is a negative number.

    Returns:
        float: Square root of 
    """
    if radicand > 0:
        try: 
            return radicand ** (1/2)
        except TypeError:
            pass
    elif radicand < 0:
        raise ValueError("Cannot take square root of negative values.")
    
def nrsqrt(radicand: float) -> float:
    """Unlike the basic paramath.sqrt() function, paramath.nsqrt() uses the Newton-Raphson algorithm for finding square roots. Accuracy may differ in some calculations.

    Args: 
        radicand (float): Main argument.

    Raises:
        ValueError: In event that the main argument is a negative number.

    Returns:
        float: Square root using Newton-Raphson Algorithm.
    """

    if radicand < 0:
        raise ValueError("Cannot take square root of negative values.")
    else:
        x = radicand / 2
        while True:
            term = 0.5 * (x + radicand / x)
            if abs(x - term) < 1e-10:
                return term
            x = term

def cuberoot(radicand: float) -> float:
    """Returns the cuberoot of the radicand."""
    return radicand ** (1/3)
        
def nthroot(radicand: float, nthconst: float) -> float:
    """Returns the nthroot of the radicand.

    Args:
        radicand (float): Radicand (main argument)
        nthconst (float): Index (n, the number that the root will be taken of).

    Returns:
        float: Nthroot of the radicand.
    """
    return radicand ** (1/nthconst)

def factorial(constant: int) -> int:
    """Calculates the factorial by multiplying the constant by itself subtracted by 1 until it reaches 0."""
    try:
        factorial = 1
        for num in range(2, constant + 1):
            factorial *= num
        return factorial
    except TypeError as error:
        raise TypeError("factorial function cannot take in non-integer values.")
       
def permutation(total: int, objects: int) -> int:
    """Returns the number of orders/arrangements a set can be established in.

    Args:
        total (int): Total amount of items.
        objects (int): Number of unique items that exist.

    Returns:
        int: Calculated order/arrangements.
    """
    return factorial(objects) / factorial(objects - total)

def combination(total: int, choices: int) -> int:
    """Returns the number of different combinations of items.

    Args:
        total (int): Total amount of items.
        choices (int): Total amount of choices.

    Returns:
        int: The number of combinations.
    """
    return factorial(total) / (factorial(choices) * factorial(total - choices))    

def radians(degrees: float) -> float:
    """Converts degrees into radians.

    Args:
        degrees (float): Degrees

    Returns:
        float: Radians
    """
    return (pi/180) * degrees

def degrees(radians: float) -> float:
    """Converts radians into degrees

    Args:
        radians (float): Radians

    Returns:
        float: Degrees
    """
    return (180/pi) * radians

def hyp(a: float, b: float) -> float: 
    """Returns the hypotnuse of the triangle with given sides a and b.

    Args:
        a (float): Side length a.
        b (float): Side length b.

    Returns:
        float: Hypotnuse of the triangle. (returns square root of the sum of both sides.)
    """
    c = (a**2) + (b**2)
    return round(sqrt(c), 10)


def sin(x: float, mode=DEGREES) -> float:
    """Returns the sine of an angle

    Args:
        x (float): Angle.
        mode (_type_, optional): DEGREES, RADIANS. Defaults to DEGREES.

    Raises:
        NameError: This error occurs when an argument other than DEGREES & RADIANS are used.

    Returns:
        float: Sine of angle 'x'.
    """

    if mode == DEGREES:
        x %= 360
        x = radians(x)
    elif mode == RADIANS:
        x %= tau
    else: 
        raise NameError("Invalid mode name for function.")
    
    result = 0
    sign = 1
    for n in range(20):
        term = (x ** (2 * n + 1)) / factorial(2 * n + 1)
        result += sign * term
        sign *= -1
    
    return round(result, 10)

def cos(x: float, mode=DEGREES) -> float: 
    """Returns the cosine of an angle

    Args:
        x (float): Angle.
        mode (_type_, optional): DEGREES, RADIANS. Defaults to DEGREES.

    Raises:
        NameError: This error occurs when an argument other than DEGREES & RADIANS are used.

    Returns:
        float: Cosine of angle 'x'.
    """
    if mode == DEGREES:
        x %= 360
        x = radians(x)
    elif mode == RADIANS:
        x %= tau
    else: 
        raise NameError("Invalid mode name for function.")
    
    x %= 360
    x = radians(x)
    
    result = 0
    sign = 1
    for n in range(20):
        term = (x ** (2 * n)) / factorial(2 * n)
        result += sign * term
        sign *= -1

    return round(result, 10)

def tan(x: float, mode=DEGREES) -> float:

    """Returns the tangent of an angle

    Args:
        x (float): Angle.
        mode (_type_, optional): DEGREES, RADIANS. Defaults to DEGREES.

    Raises:
        NameError: This error occurs when an argument other than DEGREES & RADIANS are used.

    Returns:
        float: Tangent of angle 'x'.
    """

    if mode == DEGREES:
        if x == 90 or x == 270:
            return "Undefined"
        return sin(x) / cos(x)
    elif mode == RADIANS:
        if x == pi/4 or x == 3 * pi/4:
            return "Undefined"
        return sin(x, RADIANS) / cos(x, RADIANS)
    else: 
        raise NameError("Invalid mode name for function.")
    


def sec(x: float, mode=DEGREES) -> float: 

    """Returns the secant of an angle

    Args:
        x (float): Angle.
        mode (_type_, optional): DEGREES, RADIANS. Defaults to DEGREES.

    Raises:
        NameError: This error occurs when an argument other than DEGREES & RADIANS are used.

    Returns:
        float: Secant of angle 'x'.
    """

    if mode == DEGREES:
        if x == 90 or x == 270:
            return "Undefined"
        return 1 / cos(x)
    elif mode == RADIANS:
        if x == pi/4 or x == 3 * pi/4:
            return "Undefined"
        return 1 / cos(x, RADIANS)
    else: 
        raise NameError("Invalid mode name for function.")


def csc(x: float, mode=DEGREES) -> float:

    """Returns the cosecant of an angle

    Args:
        x (float): Angle.
        mode (_type_, optional): DEGREES, RADIANS. Defaults to DEGREES.

    Raises:
        NameError: This error occurs when an argument other than DEGREES & RADIANS are used.

    Returns:
        float: Cosecant of angle 'x'.
    """

    if mode == DEGREES:
        if x == 180 or x == 360 or x == 0:
            return "Undefined"
        return 1 / sin(x)
    elif mode == RADIANS:
        if x == pi/2 or x == 2 * pi or x == 0:
            return "Undefined"
        return 1 / sin(x, RADIANS)
    else: 
        raise NameError("Invalid mode name for function.")

def cot(x: float, mode=DEGREES) -> float:

    """Returns the cotangent of an angle

    Args:
        x (float): Angle.
        mode (_type_, optional): DEGREES, RADIANS. Defaults to DEGREES.

    Raises:
        NameError: This error occurs when an argument other than DEGREES & RADIANS are used.

    Returns:
        float: Cotangent of angle 'x'.
    """
    
    if mode == DEGREES:
        if x == 0 or x == 180 or x == 360:
            return "Undefined"
        return 1 / tan(x)
    elif mode == RADIANS:
        if x == pi/4 or x == 3 * pi/2:
            return 0
        return 1 / tan(x, RADIANS)
    else: 
        raise NameError("Invalid mode name for function.")


def sinc(x: float) -> float:
    """Returns the sine cardinal of x.

    Args:
        x (float): Input.

    Returns:
        float: Returns sine cardinal of x.
    """
    return sin(x) / radians(x)

def sinh(x: float) -> float: 
    """Returns the hyperbolic sine of x.

    Args:
        x (float): Input.

    Returns:
        float: Hyperbolic sine of x.
    """
    return ((e ** x) - (e ** -x)) / 2

def cosh(x: float) -> float: #hyperbolic cosine
    """Returns the hyperbolic cosine of x.

    Args:
        x (float): Input.

    Returns:
        float: Hyperbolic cosine of x.
    """    
    return ((e ** x) + (e ** -x)) / 2

def tanh(x: float) -> float: #hyperbolic tangent
    """Returns the hyperbolic tangent of x.

    Args:
        x (float): Input.

    Returns:
        float: Hyperbolic tangent of x.
    """    
    return sinh(x) / cosh(x)

def sech(x: float) -> float: #hyperbolic secant
    """Returns the hyperbolic secant of x.

    Args:
        x (float): Input.

    Returns:
        float: Hyperbolic secant of x.
    """    
    return 1 / cosh(x)

def csch(x: float) -> float: #hyperbolic cosecant
    """Returns the hyperbolic cosecant of x.

    Args:
        x (float): Input.

    Returns:
        float: Hyperbolic cosecant of x.
    """
    if x == 0:
        return "Undefined"    
    return 1 / sinh(x)

def coth(x: float) -> float: #hyperbolic cotangent
    """Returns the hyperbolic cotangent of x.

    Args:
        x (float): Input.

    Returns:
        float: Hyperbolic cotangent of x.
    """    
    if x == 0:
        return "undefined"
    return 1 / tanh(x)
    
def cis(x: float) -> complex:
    """Returns a complex number.

    Args:
        x (float): Angle in degrees.

    Returns:
        str: Returns complex number using the cis method otherwise known as e^ix.
    """
    
    return complex(cos(x), sin(x))

def mod(x: float, y: float) -> float:
    """Modulo

    Args:
        x (float): Numerator
        y (float): Denominator

    Returns:
        float: Remainder of x / y (if any)
    """
    return x % y

def limit(function: str, stepsize: float) -> float:
    supported_variables = {
        "x": stepsize,
        "y": stepsize,
        "z": stepsize,
        "a": stepsize,
        "b": stepsize,
        "c": stepsize,
        "d": stepsize,
        "t": stepsize
    }
    try:
        limit = eval(function, supported_variables)

        return limit
    except NameError:
        return f"Varriable not supported in function. Supported Variables: {supported_variables.keys()}"


def mean(array: list) -> float:
    """
    Returns the mean (average) of a list.
    """
    total = 0
    for number in range(0, len(array)):
        total = total + float(array[number])
    return total / len(array)  

def median(array: list) -> float:
    """ 
    Returns the median (middle number) of a list.
    """
    array.sort()
    mid_num = len(array) // 2
    median = (array[mid_num] + array[~mid_num]) / 2
    return median

def mode(array: list) -> float:
    """
    Returns the mode (most common) of a list.
    """
    num_list = Counter(array)
    return num_list.most_common(1)

def int_interval(a: int, b: int, increment: float=1.0) -> list:
    """Generates a list of numbers from `a` through `b` in specified increments (increment=1.0 by default).

    Args:
        a (int): Starting Value
        b (int): Ending Value
        increment (float): Step Value

    Returns:
        list: Interval of `a` through `b` with given increments of {increment}. This method is designed for {a} & {b} values to be positive or negative integer values.
    """
    interval_list = []
    current = a
    for _ in range(a, int(b/increment)+1):
        interval_list.append(current)
        current += increment
        for value in interval_list:
            if value > b:
                interval_list.remove(value)

    if interval_list[len(interval_list)-1] > b:
        interval_list.pop(len(interval_list)-1)

    return interval_list

def eval_interval(a: int, b: int, increment: float=1.0, function: str="x") -> list:
    """Like the `int_interval` function, the `eval_interval` function generates a list of numbers from `a` through `b` in specified increments (increment=1.0 by default) while also taking in a `function` arguement for the values in the interval to be evaluated at and then returned to a new list containing the evaluated values.\nExample of `function` usage: `function="x**2"`.

    Args:
        a (int): Starting Value
        b (int): Ending Value
        increment (float, optional): Increment. Defaults to 1.0.
        function (str, optional): Function that each value in interval will be evaulated to. Defaults to "x".

    Returns:
        list: A list containing values within the interval list that were evaluated with in the given `function`.
    """
    interval_list = []
    current = a
    for _ in range(a, int(eval(function, {"x": (b/increment)+1, "X": (b/increment)+1}))):
        interval_list.append(current)
        current += increment
        for value in interval_list:
            if value > b:
                interval_list.remove(value)

    if interval_list[len(interval_list)-1] > b:
        interval_list.pop(len(interval_list)-1)

    eval_interval_list = []
    for value in interval_list:
        supported_variables = {
            "x": value,
            "X": value
        }
        val = eval(function, supported_variables)
        eval_interval_list.append(val)

    return eval_interval_list

def summation(n: int=1, a: int=1) -> float:
    """Sum of numbers from `a` through `b` in specified increments, only adds the nth proceeding number of the series.\nFollows the series: ∑(n).

    Args:
        n (int, optional): Starting Value. Defaults to 1.
        a (int, optional): Ending Value. Defaults to 1.

    Returns:
        float: The sum of each number added together.
    """
    for i in range(n, a):
        i+=(n+1)
        n=i

    return i

def count(values: list[float], square: bool=False) -> float:
    """Returns the sum of all values in given list.

    Args:
        values (list): All values in list.
        square (bool, optional): Specifies if each value should be squared before counting. Defaults to False.

    Returns:
        float: Sum of all values.
    """
    if square is False:
        sum = 0
        length = len(values)
        for i in range(0, length):
            sum += values[i]
        return sum       
    elif square is True:
        sum = 0
        length = len(values)
        for i in range(0, length):
            sum += values[i] ** 2
        return sum  
    
#OBJECTS

class DataSet:
    """Create a DataSet object using a list[int] or list[float] values to perform operations to find things such as the mean, median, maximum, minimum, count, etc..

    Args:
        array: list[int] or list[float]
    """
    def __init__(self, array: list):
        self.array = array
        for n in self.array:
            if isinstance(n, int) or isinstance(n, float):
                pass
            else:
                raise ValueError("All elements must be integers or floating point numbers.")
                
    def mean(self) -> float:
        """
        Returns the mean (average) of the dataset.
        """
        total = 0
        for number in range(0, len(self.array)):
            total = total + float(self.array[number])
        return total / len(self.array)       
            
    def median(self) -> float:
        """
        Returns the median (middle) of the dataset.
        """
        self.array.sort()
        mid_num = len(self.array) // 2
        median = (self.array[mid_num] + self.array[~mid_num]) / 2
        return median
    
    def mode(self) -> float:
        """
        Returns the mode (most common) of the dataset.
        """
        num_list = Counter(self.array)
        return num_list.most_common(1)
    
    def num_counts(self) -> float: 
        """
        Returns all numbers and their number of occurances of the dataset.
        """
        num_list = Counter(self.array)
        return num_list.most_common()
    
    def sum(self) -> float:
        """
        Returns the sum of all numbers in the dataset.
        """
        total = 0
        for number in range(0, len(self.array)):
            total = total + float(self.array[number])
        return total 
    
    def max(self) -> float:
        """
        Returns the maximum value of the dataset.
        """
        self.array.sort(reverse=True)
        return self.array[0]
    
    def min(self) -> float:
        """
        Returns the minimum value of the dataset.
        """
        self.array.sort()
        return self.array[0]
    
    def evens(self) -> list: 
        """Returns all even numbers in the dataset.

        Returns:
            list: List of even numbers.
        """
        evens_list = []
        for n in self.array:
            if n % 2 == 0:
                evens_list.append(n)
            else: 
                pass

        return evens_list
    
    def odds(self) -> list: 
        """Returns all odd numbers in the dataset.

        Returns:
            list: List of odd numbers.
        """
        odds_list = []
        for n in self.array:
            if n % 2 != 0:
                odds_list.append(n)
            else: 
                pass

        return odds_list
    
    def sort_odd_even(self) -> dict:
        """Returns a dictionary containing two lists. One for odds, and one for evens.

        Returns:
            dict: Dictionary of even of odd numbers in respective list.
        """
        odds_list = []
        evens_list= []
        for n in self.array:
            if n % 2 == 0:
                evens_list.append(n)
            else: 
                odds_list.append(n)

        return {
            "Evens": evens_list,
            "Odds": odds_list
        }
    
    def primes(self) -> list:
        """Returns a list of prime numbers in the dataset.

        Returns:
            list: Prime numbers.
        """
        primes = []
        for n in self.array:
            if is_prime(n):
                primes.append(n)
        
        return primes
    
    def get_data_insights(self):
        """Returns a dictionary of common insights for the dataset."""

        insights = {
            "Mean": self.mean(),
            "Median": self.median(),
            "Mode": self.mode(),
            "Maximum Value": self.max(),
            "Minimum Value": self.min(),
            "Evens": self.evens(),
            "Odds": self.odds(),
            "Prime Numbers": self.primes(),
            "Occurrences": self.num_counts(),
            "Sum": self.sum()
        }

        return insights
    
class Parabola:
    """Create a Parabola object from given arguments of a, b, and c. 

    Args: 
    
    a: float
    b: float (None on default)
    c: float (None on default)

    """
    def __init__(self, a: float, b: float=None, c: float=None):
        self.a = a
        self.b = b
        self.c = c

    def solutions(self) -> tuple:
        """
        Returns 1 or 2 real solutions, if any. 
        Works the same as the quadratic function 
        already provided outside the Parabola object.
        """
        
        radicand = (self.b ** 2) - (4 * self.a * self.c)
        denominator = 2 * self.a

        try: 
            solution_one = ((-1 * self.b) + sqrt(radicand)) / denominator
        except TypeError:
            solution_one = "j"
        try:
            solution_two = ((-1 * self.b) - sqrt(radicand)) / denominator
        except TypeError:
            solution_two = "j"
        
        return (solution_one, solution_two)
    
    def vertex(self) -> tuple:
        """
        Calculates the vertex by finding the x-cordinate using the x = -b/2a formula, 
        then plugs in x into the equation to find the y-cordinate.
        """
        
        x = (-1 * (self.b)) / (2 * self.a)
        y = (self.a * (x ** 2)) + (self.b * x) + (self.c)
        
        return (x, y)
    
    def is_positive(self) -> bool:
        """
        Checks if the parabola is postive (opening upwards) or negative (opening downwards).
        """
        if self.a > 0:
            return True
        elif self.a < 0:
            return False
        
    def max(self) -> float:
        """
        Calculates the smallest y-value for a parabola. 
        """
        if self.a < 0:
            x = (-1 * (self.b)) / (2 * self.a)
            y = (self.a * (x ** 2)) + (self.b * x) + (self.c)
            
            return y
        else:
            return None       
         
    def min(self) -> float:
        """
        Calculates the largest y-value for a parabola. 
        """
        if self.a > 0:
            x = (-1 * (self.b)) / (2 * self.a)
            y = (self.a * (x ** 2)) + (self.b * x) + (self.c)
            
            return y
        else:
            return None

    def domain(self) -> tuple:
        """
        Finds the domain of a given parabola.
        """
        return ("-inf", "inf")

    
    def range(self) -> tuple:
        """
        Finds the range of a given parabola.
        """
        if self.is_positive() == True:
            return (self.min(), "inf")
        elif self.is_positive() == False:
            return ("-inf", self.max())
        
    def limit(self, h: float) -> float:
        """
        Finds the limit (lim) of a given parabola as x approaches h.
        """

        return self.a * (h ** 2) + (self.b * h) + (self.c)

class Table():
    """
    NEW!
    Create a table object by providing two lists (list[int] or list[float]) of X and Y values to perform operations such as finding the line of best fit, and more!.

    """
    def __init__(self, L1_X: list, L2_Y: list):
        self.L1_X = L1_X
        self.L2_Y = L2_Y

    def ordered_pairs(self) -> list[tuple]: 
        """Returns a list of tuples in which they hold each point of the plot in (x,y) pairs.

        Raises:
            ValueError: If there are more `x` values than `y` values and vise versa.

        Returns:
            list[tuple]: list of all paired points.
        """
        if len(self.L1_X) > len(self.L2_Y) or len(self.L2_Y) > len(self.L1_X):
            raise ValueError("Not all values have a corresponding pair, be sure to check if all values in list one have a pair in list two and vise versa.")
        n = 0
        points_list = []
        while n < len(self.L1_X):
            points_list.append((self.L1_X[n], self.L2_Y[n]))
            n+=1

        return points_list

    def linear_regression(self) -> str:
        """Generates a rough approximation of a linear function that best represents the trend with respect to x and y values. 
        Form: y=mx+b\n
        Returns:
            str: Rough approximated linear function of the plot.
        """
        index_count = len(self.L2_Y)
        sum_squared_x_values = count(self.L1_X, square=True)
        sum_squared_y_values = count(self.L2_Y, square=True)
        sum_x_y_values = 0
        for i in range(0, len(self.L1_X)):
            sum_x_y_values += (self.L1_X[i] * self.L2_Y[i])

        slope = ((sum_x_y_values - ((count(self.L1_X) * count(self.L2_Y)) / index_count)) / (sum_squared_x_values - ((count(self.L1_X) ** 2) / index_count)))
        constant = (count(self.L2_Y) - (slope * count(self.L1_X))) / index_count

        if constant < 0:
            sign = ""
        elif constant > 0:
            sign = "+"
        elif constant == 0:
            sign = "+"

        return f"y = {round(slope, 2)}x {sign} {round(constant, 2)}"

    def coefficient_of_determination(self) -> float:
        """Returns the coefficient of determination otherwise known as the R^2 value. \n

        Note: R^2 is a rough approximation that essentially tells you how well the regression fits and predicts values. (The closer R^2 is to 1.0, the better.)
        """
        index_count = len(self.L2_Y)
        sum_x = count(self.L1_X)
        sum_y = count(self.L2_Y)
        sum_squared_x_values = count(self.L1_X, square=True)
        sum_squared_y_values = count(self.L2_Y, square=True)
        sum_x_y_values = 0
        for i in range(0, len(self.L1_X)):
            sum_x_y_values += (self.L1_X[i] * self.L2_Y[i])

        r = ((index_count * (sum_x_y_values)) - (sum_x * sum_y))/(sqrt((index_count * sum_squared_x_values) - sum_x ** 2) * sqrt((index_count * sum_squared_y_values) - sum_y ** 2))

        return r**2

# Classes

class Volumetrics():
    """This class offers functions used for finding volumes of several common three-dimensional shapes and spaces."""
    
    @staticmethod
    def cuboid(length: float, width: float, height: float) -> float: 
        """Calculates the volume of the cuboid. 

        Args:
            length (float): Length of cuboid.
            width (float): Width of cuboid.
            height (float): Height of cuboid.

        Returns: 
            float: Volume of cuboid.
        """
        return length * width * height
    
    @staticmethod
    def Pyramid(length: float, width: float, height: float) -> float:
        """Calculates the volume of a pyramid

        Args:
            length (float): Length of pyramid.
            width (float): Width of pyramid.
            height (float): Height of pyramid.

        Returns:
            float: Volume of pyramid.
        """
        return (length * width * height) / 3
    
    @staticmethod
    def sphere(radius: float) -> float:
        """Calculates the volume of a sphere.

        Args:
            radius (float): Radius of sphere

        Returns:
            float: Volume of sphere
        """

        return (4/3 * pi) * (radius ** 3)
    
    @staticmethod
    def cylindroid(radius: float, height: float) -> float:
        """Calculates the volume of a cylinder

        Args:
            radius (float): Radius of cylinder.
            height (float): Height of cylinder.

        Returns:
            float: Volume of the cylinder.
        """

        return pi * (radius ** 2) * height
    
    @staticmethod
    def right_circular_cone(radius: float, height: float) -> float:
        """Calculate the volume of a right-angled circular cone.

        Args:
            radius (float): Radius of cone.
            height (float): Height of cone.

        Returns:
            float: Volume of cone.
        """

        return pi * (radius ** 2) * (height / 3)

    @staticmethod
    def oblique_cone(radius: float, height: float) -> float:
        """Calculate the volume of a oblique cone.

        Args:
            radius (float): Radius of cone.
            height (float): Height of cone.

        Returns:
            float: Volume of cone.
        """

        return (1 / 3) * pi * (radius ** 2) * height
    
class Security():
    """This class offers many functions and tools for passwords, hashes, etc.. No passwords or other sensitive information is ever collected when using this class."""
    
    @staticmethod
    def password_strength_score(password: str, common_passwords: list[str]=None) -> int:
        """
        Calculates the strength score of a password from 1 to 100. \nNote: No password is recorded when using this function. Additionally, you can set a common_passwords parameter to a list of common passwords that you want the score to deduct points if any common passwords are found. 
        """

        max_entropy_score: int = 30 

        # calculate length score

        password_length: int = len(password)

        if password_length <= 4:
            length_score: int = 0
        elif 5 <= password_length < 8:
            length_score: int = 10
        elif 8 <= password_length < 12:
            length_score: int = 20
        elif 12 <= password_length <= 16:
            length_score: int = 25
        elif password_length > 16:
            length_score: int = 30

        has_lower: bool = bool(search(r"[a-z]", password))
        has_upper: bool = bool(search(r"[A-Z]", password))
        has_digit: bool = bool(search(r"\d", password))
        has_special: bool = bool(search(r"[!@#$%^&*(),.?':{}}|<>]", password))

        complexity_score: int = (has_lower + has_upper + has_digit + has_special) * 10

        if common_passwords:
            for common_password in common_passwords:
                if common_password.lower() in password:
                    password_penalty = 50
            if password.lower() in common_passwords:
                password_penalty = 50
        else:
            password_penalty = 0

        if len(set(password)) == 1:  # All characters are the same
            entropy_score = 0
        else:
            entropy = 0
            pool_size = 0
            if has_lower:
                pool_size += 26
            if has_upper:
                pool_size += 26
            if has_digit:
                pool_size += 10
            if has_special:
                pool_size += len('!@#$%^&*(),.?":{}|<>')

            entropy = password_length * log2(pool_size)
            entropy_score = min(30, entropy / 4)

            if entropy_score > max_entropy_score:
                entropy_score = 30

        return round(length_score + complexity_score + entropy_score - password_penalty)
    
    @staticmethod
    def generate_password() -> str:
        """Generates a password with a length of 16-24 characters. Which is considered very strong by most applications."""

        chars: list[str] = list("QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm1234567890!@#$%^&*()-=_+{[}]:;'<.,>?\\/|")

        generated_password: str = ""

        for i in range(0, randint(16,24)):
            generated_password = generated_password + generated_password.join(choice(chars))

        return generated_password

class System():
    """
    # System Class
    The System class provides necessary functions for finding things such as memory size of Python objects and variables as well as tools for converting string to their hexidecimal, binary, octal representations, and as well as encoding and decoding. This class uses built-in tools from Python's base64 and sys libraries.
    """

    @staticmethod
    def memory(object) -> float:
        """Returns the memory size of a Python object or variable."""

        return getsizeof(object)

    @staticmethod
    def hexidecimal(object) -> str:
        """Returns the hexidecimal representation of a Python object or variable."""
        
        return str(object).encode("utf-8").hex()

    @staticmethod
    def binary(object) -> int:
        """Returns the binary representation of a Python object or variable."""
        
        binary_string = ""
        for byte in str(object).encode("utf-8"):
            binary_string += format(byte, "08b")

        return binary_string
    
    @staticmethod
    def octal(object) -> str:
        """Returns the octal representation of a Python object or variable."""
        
        octal_string = ""
        for byte in str(object).encode("utf-8"):
            octal_string += format(byte, "03o")  # '03o' ensures 3-digit octal representation

        return octal_string
    
    @staticmethod
    def encode(string: str, encoding: str) -> str:
        """Returns an encoded string representation. Supports Base 16, 32, 64, and 85.

        Args:
            string (str): String to encode.
            encoding (str): Encoding type. 

        Returns:
            str: Encoded string.
        """

        string = string.encode('utf-8')

        if encoding == BASE16:
            return b16encode(string).decode('utf-8')
        elif encoding == BASE32:
            return b32encode(string).decode('utf-8')
        elif encoding == BASE64:
            return b64encode(string).decode('utf-8')
        elif encoding == BASE85:
            return b85encode(string).decode('utf-8')
        else:
            raise ValueError("Invalid or unsuported encoding types.")

    @staticmethod
    def decode(encoded_string: str, encoding: str) -> str:
        """Returns a decoded string representation. Supports Base 16, 32, 64, and 85.

        Args:
            encoded_string (str): Encoded string to decode.
            encoding (str): Encoding type. 

        Returns:
            str: Decoded string.
        """
        if encoding == BASE16:
            return b16decode(encoded_string)
        elif encoding == BASE32:
            return b32decode(encoded_string)
        elif encoding == BASE64:
            return b64decode(encoded_string)
        elif encoding == BASE85:
            return b85decode(encoded_string)
        else:
            raise ValueError("Invalid or unsuported encoding types.")
