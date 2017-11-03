from math import sqrt, acos, pi
from decimal import Decimal, getcontext

getcontext().prec = 30

class Vector(object):
    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple([Decimal(x) for x in coordinates])
            self.dimension = len(coordinates)

        except ValueError:
            raise ValueError('The coordinates must be nonempty')

        except TypeError:
            raise TypeError('The coordinates must be an iterable')

    def plus(self, v):
        new_coordinates = [x+y for x,y in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)

    def minus(self, v):
        new_coordinates = [x-y for x,y in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)

    def times_scalar(self, c):
        new_coordinates = [Decimal(c)*x for x in self.coordinates]
        return Vector(new_coordinates)

    def magnitude(self):
        coordinates_squared = [x**2 for x in self.coordinates]
        return Decimal(sqrt(sum(coordinates_squared)))

    def normalized(self):
        try:
            magnitude = self.magnitude()
            return self.times_scalar(Decimal('1.0')/magnitude)

        except ZeroDivisionError:
            raise Exception('Cannot normalize the zero vector')

    def dot(self, v):
        return sum([x*y for x,y in zip(self.coordinates, v.coordinates)])

    def angle_with(self, v, in_degrees=False):
        try:
            u1 = self.normalized()
            u2 = v.normalized()
            dot_product = (u1.dot(u2))
            if dot_product < -1:
                dot_product = -1
            if dot_product > 1:
                dot_product = 1
            angle_in_radians = acos(dot_product)

            if in_degrees:
                degrees_per_radian = 180. / pi
                return angle_in_radians * degrees_per_radian
            else:
                return angle_in_radians

        except Exception as e:
            if str(e) == 'Cannot normalize the zero vector':
                raise Exception('Cannot compute an angle with the zero vector')
            else:
                raise e

    def is_orthogonal_to(self, v, tolerance=1e-10):
        return abs(self.dot(v)) < tolerance

    def is_parallel_to(self, v):
        if (self.is_zero() or v.is_zero()):
            return True
        else:
            return (self.angle_with(v) == 0 or self.angle_with(v) == pi )

    def is_zero(self, tolerance=1e-10):
        return self.magnitude() < tolerance

    def projection_onto_base(self, base):
        unit_base = base.normalized()
        v_dot_base = self.dot(unit_base)
        return unit_base.times_scalar(v_dot_base)

    def component_orth_to_base(self, base):
        projection_onto_base = self.projection_onto_base(base)
        return self.minus(projection_onto_base)

    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)


    def __eq__(self, v):
        return self.coordinates == v.coordinates


a = Vector([3.039, 1.879])
b = Vector([0.825, 2.036])
c = Vector([-9.88, -3.264, -8.159])
d = Vector([-2.155, -9.353, -9.473])
# e = Vector([-2.328, -7.284, -1.214])
# f = Vector([-1.821, 1.072, -2.94])
# g = Vector([2.118, 4.827])
# h = Vector([0, 0])

# a = Vector([7.887, 4.138])
# b = Vector([-8.802, 6.776])
# c = Vector([-5.955, -4.904, -1.874])
# d = Vector([-4.496, -8.755, 7.103])
# e = Vector([3.183, -7.627])
# f = Vector([-2.668, 5.319])
# g = Vector([7.35, 0.221, 5.188])
# h = Vector([2.751, 8.259, 3.985])

print(a.projection_onto_base(b))
print(c.component_orth_to_base(d))
# print(c.is_parallel_to(d))
# print(c.is_orthogonal_to(d))
# print(e.is_parallel_to(f))
# print(e.is_orthogonal_to(f))
# print(g.is_parallel_to(h))
# print(g.is_orthogonal_to(h))

# print(c.dot(d))
# print(a.angle_with(b))
# print(g.angle_with(h, in_degrees=True))
#
#
# v = Vector([8.218, -9.341])
# w = Vector([-1.129, 2.111])
# print(v.plus(w))
# v = Vector([7.118, 8.215])
# w = Vector([-8.223, 0.878])
# print(v.minus(w))
# v = Vector([1.671, -1.012, -0.318])
# c = 7.41
# print(v.times_scalar(c))
