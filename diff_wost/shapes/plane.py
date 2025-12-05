from dataclasses import dataclass

from diff_wost.core.fwd import Array3, Float, dr


@dataclass
class Plane:
    p: Array3
    n: Array3

    @dr.syntax
    def bounding_point(self, x: Array3, v: Array3, rho_max: Float):
        """
        Find the closest point on the plane that has reflectance greater than rho_max
        """
        c = Array3(dr.inf, dr.inf, dr.inf)
        r = Float(dr.inf)
        p = Array3(dr.inf, dr.inf, dr.inf)
        if dr.abs(dr.dot(v, self.n)) > dr.epsilon(Float):
            d = dr.dot(self.p - x, self.n) / dr.dot(v, self.n)
            c = x + d * v
            # projection of x on the plane
            _x = x + dr.dot(self.p - x, self.n) * self.n
            # dr.abs is necessary to handle the case when d is negative
            r = rho_max * dr.abs(d)
            p = Array3(dr.inf, dr.inf, dr.inf)
            if dr.norm(_x - c) > dr.epsilon(Float):
                p = c + r * dr.normalize(_x - c)
            else:
                if dr.norm(self.n - Array3(1, 0, 0)) < 1e-4:
                    p = c + dr.cross(self.n, Array3(0, 1, 0)) * r
                else:
                    p = c + dr.cross(self.n, Array3(0, 0, 1)) * r
        return c, r, p
