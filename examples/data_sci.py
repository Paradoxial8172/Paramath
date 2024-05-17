import paramath

# create x and y chart

x = [2,2,2,4,10,6,7,8,3,9]
y = [15,3,6,1,11,8,1,4,9,11]

my_table = paramath.Table(x, y)

print(f"Ordered Pairs: {my_table.ordered_pairs()}")
print(f"Graph: {my_table.linear_regression()}")
print(f"Coefficient of Determination (R^2): {my_table.coefficient_of_determination()}")

# output:
"""
Ordered Pairs: [(2, 15), (2, 3), (2, 6), (4, 1), (10, 11), (6, 8), (7, 1), (8, 4), (3, 9), (9, 11)]
Graph: y = 0.11x + 6.33
Coefficient of Determination (R^2): 0.0050504254234526825
"""
