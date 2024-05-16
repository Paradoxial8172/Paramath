from collections import Counter
from typing import *

pi = 3.1415926535897
e = 2.7182818284590
tau = 2 * pi
phi = float(str((1 + 5**(1/2))/2)[:-2])

def is_equal(x: float, y: float) -> bool: #checks if two arguments are equal 
    """Checks if two numbers are equal."""
    if x == y:
        return True
    else:
        return False
    
def is_close(x: float, y: float, limit: float=0.25) -> bool:
    """Checks if two numbers are close to each other but not the same. 
       The limit parameter is the value that must be greater than the 
       difference between x and y. (limit is 0.25 on default)""" 
    difference = x - y
    if abs(difference) > limit:
        return False
    else:
        return True
    
def is_prime(n) -> bool:
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

def sqrt(radical: float) -> float:
    if radical > 0:
        return radical ** (1/2)
    elif radical < 0:
        raise ValueError("Domain Error: parameter must be greater than zero.")
    
def nthroot(radical: float, nthconst: float) -> float:
    return radical ** (1/nthconst)

def factorial(constant: int) -> int:
    try:
        factorial = 1
        for num in range(2, constant + 1):
            factorial *= num
        return factorial
    except TypeError:
        raise TypeError("Cannot take factorial of non-integer values.")
       
def permutation(total: int, objects: int) -> int:
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
    return sqrt((a ** 2) + (b ** 2))

def sin(x: float) -> float:
    """Returns the sine of an angle (x).

    Args:
        x (float): Angle in degrees.

    Returns:
        float: Sine of angle.
    """
    x %= 360
    x = radians(x)
    
    result = 0
    sign = 1
    for n in range(20):
        term = (x ** (2 * n + 1)) / factorial(2 * n + 1)
        result += sign * term
        sign *= -1
    
    return round(result, 10)

def cos(x: float) -> float: 
    """Returns the cosine of an angle (x).

    Args:
        x (float): Angle in degrees.

    Returns:
        float: Returns cosine of angle.
    """
    x %= 360
    x = radians(x)
    
    result = 0
    sign = 1
    for n in range(20):
        term = (x ** (2 * n)) / factorial(2 * n)
        result += sign * term
        sign *= -1

    return round(result, 10)

def tan(x: float) -> float:
    """Returns the tangent of an angle (x).

    Args:
        x (float): Angle in degrees.

    Returns:
        float: Returns tangent of angle. (calculated by sin(x) / cos(x))
    """
    if x == 90 or x == 270:
        return "Undefined"
    return sin(x) / cos(x)

def sec(x: float) -> float: 
    """Returns the secant of an angle (x).

    Args:
        x (float): Angle in degrees.

    Returns:
        float: Returns secant of angle. (calculated by 1 / cos(x))
    """
    if x == 90 or x == 270:
        return "Undefined"
    return 1 / cos(x)

def csc(x: float) -> float:
    """Returns the cosecant of an angle (x).

    Args:
        x (float): Angle in degrees.

    Returns:
        float: Returns cosecant of angle. (calculated by 1 / sin(x))
    """

    if x == 180 or x == 360:
        return "Undefined"
    return 1 / sin(x)

def cot(x: float) -> float:
    """Returns the cotangent of an angle (x).

    Args:
        x (float): Angle in degrees.

    Returns:
        float: Returns cotangent of angle. (calculated by 1 / tan(x))
    """
    if x == 0:
        return "Undefined"
    return 1 / tan(x)

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
    """Returns the hyperbolic cosine."""
    return ((e ** x) + (e ** -x)) / 2

def tanh(x: float) -> float: #hyperbolic tangent
    """Returns the hyperbolic tangent."""
    return sinh(x) / cosh(x)

def sech(x: float) -> float: #hyperbolic secant
    """Returns the hyperbolic secant."""
    return 1 / cosh(x)

def csch(x: float) -> float: #hyperbolic cosecant
    """Returns the hyperbolic cosecant."""
    return 1 / sinh(x)

def coth(x: float) -> float: #hyperbolic cotangent
    """Returns the hyperbolic cotangent."""
    return 1 / tanh(x)

def polynomial_integral(polynomial: list, upperbound: float, lowerbound: float) -> float:
    """Takes in a list and interprets it as a polynomial. 
       For example, if you have [3, 5, -2] as the polynomial,
       it will be interpreted as 3x^2 + 5x - 2, then integrated
       using the power rule x^n+1/n+1"""
    
def cis(x: float) -> str:
    """Returns a complex number.

    Args:
        x (float): Angle in degrees.

    Returns:
        str: Returns complex number using the cis method otherwise known as e^ix. The return value cannot be used in other methods.
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

def count(values: list, square=False) -> float:
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
    """Create a DataSet object with a given list of numbers.

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
        
        radical = (self.b ** 2) - (4 * self.a * self.c)
        denominator = 2 * self.a

        try: 
            solution_one = ((-1 * self.b) + sqrt(radical)) / denominator
        except TypeError:
            solution_one = "i"
        try:
            solution_two = ((-1 * self.b) - sqrt(radical)) / denominator
        except TypeError:
            solution_two = "i"
        
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
           Form: y=mx+b
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
