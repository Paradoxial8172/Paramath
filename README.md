# Paramath - Powerful mathematical library for Python developers.
Package Repository includes:
- Source code, .whl files, etc..
- Example Python files located ```/Paramath/examples/```.
- Directions on how to use this library.

## My Personal Motive Behind The Paramath Project.
Message from developer: 

I personally started this project to teach myself on how to implement powerful mathematical algorithms into any Python project. Over time, this project has grown to have nearly 100 different functions/methods. This project has also grown into something that I am quite proud of to have, and I hope those who use it, have good experiences. Something worth mentioning... Is that the Paramath project is constantly being developed and tested. There are chances where some newer functions or methods that are recently added may not be 100% accurate or may rely off of approximations. Of course though, this is subject to change. As the developer of this project I have had fun making, I want to ensure the best out of this project. If you happen to encounter any bugs or instances where a result is heavily incorrect, please let me know!

View [Paramath](https://pypi.org/project/paramath/) on PyPi!

## VERSION
0.1.2 | August 1 2024
## INSTALL
- Use package manager [pip](https://pip.pypa.io/en/stable/) and run the following command in your command prompt.
```bash
pip install paramath
```
## CHANGE LOG - 0.1.1 -> 0.1.2
- Added the following log functions: log, log10, logb (logarithmic functions are still being tested. results may not be 100% accurate)
- Added a new function nrsqrt() which works just the same as sqrt() but instead, it uses the Newton-Raphson algorithm.
- Edited all trignometric functions to allow user to choose between degrees (DEGREES) and radians (RADIANS). By default, the function is set to DEGREES.
- Added two classes, Volumetrics & Security. Volumetrics() contains functions for finding volumes of several 3D shapes. Security() contains two functions, a password generation function, and a function that gives a score based on the strength of a password. There are plans for creating hashing functions.
- Added constants: DEGRESS, RADIANS, FUNCTION, Y_EQUALS (FUNCTION & Y_EQUALS do not have any functionality as of right now.)
## BUG FIXES
- General bug fixes found during development.
## USAGE
Code:
```python
import paramath

# create x and y chart

x = [2,2,2,4,10,6,7,8,3,9]
y = [15,3,6,1,11,8,1,4,9,11]

my_table = paramath.Table(x, y)

print(f"Ordered Pairs: {my_table.ordered_pairs()}")
print(f"Graph: {my_table.linear_regression()}")
print(f"Coefficient of Determination (R^2): {my_table.coefficient_of_determination()}")
```
Output:
```
Ordered Pairs: [(2, 15), (2, 3), (2, 6), (4, 1), (10, 11), (6, 8), (7, 1), (8, 4), (3, 9), (9, 11)]
Graph: y = 0.11x + 6.33
Coefficient of Determination (R^2): 0.0050504254234526825
```
Code:
```python
import paramath

angle = pi/2 # 1.57079632679

sine_of_angle = paramath.sin(angle, RADIANS)

print(f"The sine of {angle} radians is {sine_of_angle}")
```
Output: 
```
The sine of 1.57079632679 radians is 1.0
```