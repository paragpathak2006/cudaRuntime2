#pragma once
// unsigned_distance_function.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#pragma once
#include "../Geometry/Point.h"
#include "../Geometry/Face.h"

float unsigned_distance_brute_force(const Points& points, const Point& target, double beta, int& nearest_point);
double unsigned_distance_space_map2(const Points& points, const Point& target, double beta, double map_size, int& nearest_point);
double unsigned_distance_space_map_cuda(const Points& points, const Point& target, double beta, double map_size, int& nearest_point);


double unsigned_distance_brute_force(const Faces& faces, 
	const Points& points, const Point& target, double beta, int& nearest_point);

void print_output(double dist, int nearest_point, const Point& target, const Points& points);
void print_output(double dist, int nearest_point, const Point& target, const Points& points, const Faces& faces);
void test_local();

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
