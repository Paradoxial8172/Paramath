from build.lib.paramath.main import sin, cos, tan, sec, csc, cot, radians, degrees, sinh, cosh, tanh, sech, csch, coth, sinc, hyp, pi, tau, RADIANS, DEGREES
# from paramath import *

#TRIG AND TRIANGLES

# pi = approx 3.14
# tau = 2pi = approx 6.28

print("pi:", pi)
print("tau:", tau)

degrees(pi) # Converts radians into degrees. -> 180 degrees.
radians(360) # Converts degrees into radians -> 2 * pi or tau radians.

a = 5
b = 10
c = hyp(a, b) # -> hypotnuse of a triangle with sides 'a' and 'b'.

print(c) # -> approx 11.18

deg = float(input("Enter value in degrees: "))

# By default, trig functions are set to calculate with degrees.

print(f"sin({deg}) = {sin(deg)} \ncos({deg}) = {cos(deg)} \ntan({deg}) = {tan(deg)} \nsec({deg}) = {sec(deg)} \ncsc({deg}) = {csc(deg)} \ncot({deg}) = {cot(deg)}")

rad = float(input("Enter value in radians: "))

print(f"sin({rad}) = {sin(rad, RADIANS)} \ncos({rad}) = {cos(rad, RADIANS)} \ntan({rad}) = {tan(rad, RADIANS)} \nsec({rad}) = {sec(rad, RADIANS)} \ncsc({rad}) = {csc(rad, RADIANS)} \ncot({rad}) = {cot(rad, RADIANS)}")


# HYPERBOLIC TRIG

sinh(12) # -> 81377.3957064
cosh(12) # -> 81377.3957126

# as x -> infinity, sinh(x) ~ cosh(x)

tanh(12) # -> 0.99999999992
sech(12) # -> 0.0000122884247062
csch(12) # -> 0.0000122884247071
coth(12) # -> 1.00000000008 

# SPECIAL TRIG

# Sine-Cardinal

sinc(5 * pi) # -> 0