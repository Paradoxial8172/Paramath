from paramath import sin, cos, tan, sec, csc, cot, radians, degrees, sinh, cosh, tanh, sech, csch, coth, sinc, hyp, pi, tau

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

print(f"sin({deg}) = {sin(deg)} \ncos({deg}) = {cos(deg)} \ntan({deg}) = {tan(deg)} \nsec({deg}) = {sec(deg)} \ncsc({deg}) = {csc(deg)} \ncot({deg}) = {cot(deg)}")

# HYPERBOLIC TRIG

sinh(12) # -> 81377.3957064
cosh(12) # -> 81377.3957126
tanh(12) # -> 0.99999999992
sech(12) # -> 0.0000122884247062
csch(12) # -> 0.0000122884247071
coth(12) # -> 1.00000000008 

# SPECIAL TRIG

sinc(5 * pi) # -> 0