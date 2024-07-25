# Paramath - Powerful mathematical library for Python developers.
Package Repository includes:
- Source code, .whl files, etc..
- Example Python files located ```/Paramath/examples/```.
- Directions on how to use this library.

View [Paramath](https://pypi.org/project/paramath/) on PyPi!

## VERSION
0.1.1 | May 16 2024
## INSTALL
- Use package manager [pip](https://pip.pypa.io/en/stable/) and run the following command in your command prompt.
```bash
pip install paramath
```
## CHANGE LOG - 0.1.0 -> 0.1.1
- General bug fixes
- Added new method to Table() class called "coefficient_of_determination()" which returns the R^2 value for the line of best fit.
- Rewrote a lot of functions' and methods' summaries and descriptions to give a better insight on what they do.
- Edited the hyp() method to ensure better accuracy of return value. 
- Removed "golden_ratio()" method as it is obsolete to the "phi" constant.
## BUG FIXES
- Fixed (as to my knowledge and testing) a LOT of bugs with the hyperbolic and regular trignometric functions. 
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

angle = 270

sine_of_angle = paramath.sin(angle)

print(f"The sine of {angle} degrees is {sine_of_angle}")
```
Output: 
```
The sine of 270 degrees is -1.0
```