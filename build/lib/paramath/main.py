"""
# Paramath 0.1.3
## Powerful mathematical library for Python devlopers.
Paramath (coming from the words "parametric" and "mathematics" mixed together) is a powerful 3rd party library for Python 
devlopers. Paramath offers a wide variety of mathematical functions to help developers implement quick math in their projects.
Some of these functions include trigonometric and hyperbolic functions, functions that iterate through lists to find things 
such as mean, median, and mode, as well as functions for generating graphs based on scatter plots! Paramath also offers a 
variety of encoding functions for base16, 32, 64, etc.. as well as functions that can generate strong passwords, give 
strength scores to passwords, etc... 
"""

from collections import Counter
from typing import *
from re import search
from math import log2
from random import choice, randint
from sys import getsizeof
from base64 import b16encode, b16decode, b32encode, b32decode, b64encode, b64decode, b85encode, b85decode

# Error handling

class ParamathError(Exception):
    """Base class for all Paramath related exceptions."""
    pass

class TableInitalizationError(ParamathError):
    """Raised when a Table object's two lists do not have the same length"""
    def __init__(self, message="Your Table objects must have two lists of the same length."):
        self.message = message
        super().__init__(self.message)

class DomainError(ParamathError):
    """Raised whenever a value is found to be out of domain."""
    def __init__(self, message="Calculation out of domain or no solution available."):
        self.message = message
        super().__init__(self.message)

class InvalidLiteralError(ParamathError):
    """Raised when a incorrect literal is passed through a function."""
    def __init__(self, message="Literal is not supported."):
        self.message = message
        super().__init__(self.message)

# Variables

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

# ALGORITHMS

SQRT_NEWTON_RAPHSON: str = "new_rap"
SQRT_EXP_BASED: str = "exp_based"

def is_equal(x: float, y: float) -> bool: # checks if two arguments are equal 
    """Checks if two numbers are equal. Returns True if yes, otherwise False."""
    if x == y:
        return True
    else:
        return False
    
def is_close(x: float, y: float, threshold: float=0.25) -> bool:
    """Checks if numbers `x` and `y` are close to each other by a defined threshold.

    Args:
        x (float): `x` value.
        y (float): `y` value.
        threshold (float, optional): The threshold of how close two numbers can be. Defaults to 0.25 (1/4).

    Returns:
        bool: True -> if numbers `x` and `y` are close to each other, False -> if otherwise.
    """
    difference = x - y
    if abs(difference) > threshold:
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
    """Brings the base to the power of an exponent. (b^n)

    Args:
        base (float): Base or `b` value.
        exponent (float): Exponent/Power or `n` value.

    Returns:
        float: base ^ exponent. 
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

def square(base: float) -> float:
    return base ** 2

def cube(base: float) -> float:
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

def sqrt(radicand: float | int, algorithm: str=SQRT_EXP_BASED) -> float | int:
    if radicand < 0:
        return ((radicand * -1) ** (1/2)) * 1j
    match algorithm:
        case "exp_based":
            return radicand ** (1/2)
        case "new_rap":
            x = radicand / 2
            while True:
                term = 0.5 * (x + radicand / x)
                if abs(x - term) < 1e-10:
                    return term
                x = term
    
def nrsqrt(radicand: float) -> float:
    """Unlike the basic paramath.sqrt() function, paramath.nsqrt() uses the Newton-Raphson algorithm for finding square roots. Accuracy may differ in some calculations.

    Args: 
        radicand (float): Main argument.

    Raises:
        DomainError: In event that the main argument is a negative number.

    Returns:
        float: Square root using Newton-Raphson Algorithm.
    """

    if radicand < 0:
        raise DomainError()
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
        raise DomainError()
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
        raise InvalidLiteralError()
    
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
        raise InvalidLiteralError()
    
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
        raise InvalidLiteralError()
    


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
        raise InvalidLiteralError()


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
        raise InvalidLiteralError()

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
        raise InvalidLiteralError()


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

def interval(a: int, b: float, increment: float=1.0) -> list[int] | list[float]:
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
        interval_list.append(int(current))
        current += increment
        for value in interval_list:
            if value > b:
                interval_list.remove(value)

    if interval_list[len(interval_list)-1] > b:
        interval_list.pop(len(interval_list)-1)

    return interval_list

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
    if not square:
        sum = 0
        length = len(values)
        for i in range(0, length):
            sum += values[i]
        return sum       
    else:
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
            "evens": evens_list,
            "odds": odds_list
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
            "mean": self.mean(),
            "median": self.median(),
            "mode": self.mode(),
            "maximum_value": self.max(),
            "minimum_value": self.min(),
            "evens": self.evens(),
            "odds": self.odds(),
            "primes": self.primes(),
            "sum": self.sum()
        }

        return insights

class Table():
    """
    Create a table object by providing two lists (`list[int]` or `list[float]`) of `x` and `y` values to perform operations such as finding the line of best fit, and more!.
    """
    def __init__(self, L1_X: list, L2_Y: list):
        
        if len(L1_X) != len(L2_Y):
            raise TableInitalizationError()

        self.L1_X = L1_X
        self.L2_Y = L2_Y

    def ordered_pairs(self) -> list[tuple]: 
        """Returns a list of tuples in which they hold each point of the plot in (x,y) pairs.

        Raises:
            ValueError: If there are more `x` values than `y` values and vise versa.

        Returns:
            list[tuple]: list of all paired points.
        """
        n = 0
        points_list = []
        while n < len(self.L1_X):
            points_list.append((self.L1_X[n], self.L2_Y[n]))
            n+=1

        return points_list

    def linear_regression(self, notation: str=Y_EQUALS) -> str:
        """Returns a linear equation of the line of best fit.

        Args:
            notation (str, optional): `FUNCTION` or `Y_EQUALS` notation. Defaults to `Y_EQUALS`.

        Returns:
            str: Linear equation.
        """
        index_count = len(self.L2_Y)
        sum_squared_x_values = count(self.L1_X, square=True)
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

        match notation:
            case "y_equals":
                return f"y = {round(slope, 2)}x {sign} {round(constant, 2)}"
            case "function":
                return f"f(x) = {round(slope, 2)}x {sign} {round(constant, 2)}"
            case _:
                return None

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
    def generate_password(minimum_characters: int=16, maximum_characters: int=24) -> str:
        """_summary_

        Args:
            minimum_characters (int, optional): Minimum amount of characters. Defaults to 16.
            maximum_characters (int, optional): Maximum amount of characters. Defaults to 24.

        Returns:
            str: Randomly generated password.
        """

        chars: list[str] = list("QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm1234567890!@#$%^&*()-=_+{[}]:;'<.,>?\\/|")

        generated_password: str = ""

        for i in range(0, randint(minimum_characters, maximum_characters)):
            generated_password = generated_password + generated_password.join(choice(chars))

        return generated_password
