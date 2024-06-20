#pragma once
#include <iostream>
#include <string>
#include <vector>
using namespace std;
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define _HOST_DEVICE_ __host__ __device__ 
#define _HASH_(i,j,k) ((i) * 18397 + (j) * 20483 + (k) * 29303)
#define _DIST_(P,Q,i) (P.i - Q.i) * (P.i - Q.i) 
#define _DISTANCE_(P,Q) _DIST_(P,Q,x) + _DIST_(P,Q,y) + _DIST_(P,Q,z) 
#define _DOT_(P,Q) P.x * Q.x + P.y * Q.y + P.z * Q.z
class Point {
public:
    double x = 0, y = 0, z = 0;
    _HOST_DEVICE_ Point(double X, double Y, double Z) : x(X), y(Y), z(Z) {}
    _HOST_DEVICE_ Point() { x = 0; y = 0; z = 0; }

    _HOST_DEVICE_ Point operator+(const Point& rhs) const { return Point(x + rhs.x, y + rhs.y, z + rhs.z); }
    _HOST_DEVICE_ Point operator-(const Point& rhs) const { return Point(x - rhs.x, y - rhs.y, z - rhs.z); }
    _HOST_DEVICE_ Point operator/(double d) { return Point(x / d, y / d, z / d); }
    _HOST_DEVICE_ Point operator*(double d) { return Point(x * d, y * d, z * d); }
    _HOST_DEVICE_ bool operator==(Point rhs) { return x == rhs.x && y == rhs.y && z == rhs.z; }
    _HOST_DEVICE_ Point operator*(Point rhs) {
        return Point(
            y * rhs.z - z * rhs.y,
            z * rhs.x - x * rhs.z,
            x * rhs.y - y * rhs.x
        );
    }

    void print() const { cout << "Point(" << x << "," << y << "," << z << ")" <<endl; }

    _HOST_DEVICE_ size_t get_hash(const double& map_size) const {
        return _HASH_(round(x / map_size), round(y / map_size), round(z / map_size));
    }

    //Vector2D rotate_90(Vector2D v) { return Vector2D(-v.y, v.x); }
    //Point get_normal(Point P1, Point P3) {        return (*this * 2 - P1 - P3) * sense(P1, P3) / 2;    }

    //int sense(Point P1, Point P3) {
    //    auto v1 = Vector2D(P1);
    //    auto v2 = Vector2D(*this );
    //    auto v3 = Vector2D(P3);
    //    return v2.sense(v1, v2);
    //}

    void translate(Point& v) {
        x += v.x;
        y += v.y;
        z += v.z;
    }

    Point normalize() {
        double d = sqrt(x * x + y * y + z * z);
        x = x / d;
        y = y / d;
        z = z / d;
        return Point(x, y, z);
    }

    double length() const { 
        return x * x + y * y + z * z; 
    };
    _HOST_DEVICE_
    double dist(const Point &p) const { 
        return (x - p.x) * (x - p.x) + (y - p.y) * (y - p.y) + (z - p.z) * (z - p.z);
    };

    _HOST_DEVICE_ double dist(const Point& p0, const Point& p1, const Point& p2) const;

    Point unity() {
        double d = sqrt(x * x + y * y + z * z);
        return Point(x / d, y / d, z / d);
    }

};

class AABB {
public:
    double xmin, ymin, zmin, xmax, ymax, zmax;
    AABB() { xmin = 0; ymin = 0; zmin = 0; xmax = 1; ymax = 1; zmax = 1; }
    AABB(const Point& Pmin, const Point& Pmax) {
        xmin = Pmin.x; ymin = Pmin.y; zmin = Pmin.z; 
        xmax = Pmax.x; ymax = Pmax.y; zmax = Pmax.z;

    }
    AABB(double _xmin, double _ymin, double _zmin, double _xmax, double _ymax, double _zmax) { xmin = _xmin; ymin = _ymin; zmin = _zmin; xmax = _xmax; ymax = _ymax; zmax = _zmax; }
    void print() const { printf("Bounding Box : Min(%f,%f,%f) , Max(%f,%f,%f)\n", xmin, ymin, zmin, xmax, ymax, zmax); }

    bool check_point_outside_beta_box(const Point& target, const double& beta) const {
        if (target.x + beta < xmin || target.y + beta < ymin || target.z + beta < zmin) return true;
        if (target.x - beta > xmax || target.y - beta > ymax || target.z - beta > zmax) return true;
        return false;
    }
};

typedef vector<Point> Points;

_HOST_DEVICE_ double get_nearest_point_dist(const Point& P0, const Point& P1, const Point& P2, const Point& target);
