#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Point.h"

#define _MAX2_(A,B) ((A > B) ? A : B) 
#define _MIN2_(A,B) ((A < B) ? A : B) 
#define _MAX3_(A,B,C) (A > B) ? _MAX2_(A,C) : _MAX2_(B,C)
#define _MIN3_(A,B,C) (A < B) ? _MIN2_(A,C) : _MIN2_(B,C)

class Face
{
public:
	int v[3];
	Face() { v[0] = -1; v[1] = -1; v[2] = -1;}
	Face(int i,int j,int k) { v[0] = i; v[1] = j; v[2] = k; }
	Face(int _v[3]) { v[0] = _v[0]; v[1] = _v[1]; v[2] = _v[2]; }
	void print() const { cout << "Face(" << v[0] << "," << v[1] << "," << v[2] << ")" << endl; }

	_HOST_DEVICE_ double dist(const Point& target, const Point* points) const;
	_HOST_DEVICE_ double dist(const Point& target, const Points& points) const;

		AABB get_box(const Points& points) const {
			AABB box;
			const Point& p0 = points[v[0]];
			const Point& p1 = points[v[1]];
			const Point& p2 = points[v[2]];
			return AABB(
				_MIN3_(p0.x, p1.x, p2.x), _MAX3_(p0.x, p1.x, p2.x),
				_MIN3_(p0.y, p1.y, p2.y), _MAX3_(p0.y, p1.y, p2.y),
				_MIN3_(p0.z, p1.z, p2.z), _MAX3_(p0.z, p1.z, p2.z)
			);
		}

		
};

typedef vector<Face> Faces;

