#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <tuple>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "../Geometry/Point.h"
#include "../Geometry/Face.h"

class Point_index
{
public:
    int x, y, z, index = -1e5;
    Point_index(int X, int Y, int Z) : x(X), y(Y), z(Z) {}
    Point_index(int X, int Y, int Z, int _index) : x(X), y(Y), z(Z), index(_index) {}

    Point_index(const Point& P, const double& map_size)
    {
        x = round(P.x / map_size); 
        y = round(P.y / map_size);
        z = round(P.z / map_size);
    }

    Point_index(const Point& P, int i, const double& map_size)
    {
        x = round(P.x / map_size);
        y = round(P.y / map_size);
        z = round(P.z / map_size);
        index = i;
    }

    Point_index() { x = 0; y = 0; z = 0; index = -1e5; }
    bool operator==(const Point_index& rhs) const { return x == rhs.x && y == rhs.y && z == rhs.z; }
    void print() const { cout << "Point_index(" << x << "," << y << "," << z << ")" << endl; }
    void print2() const { cout << "Point_index(" << x << "," << y << "," << z << ")"; }
    size_t get_hash() const {return _HASH_(x, y, z);}
};

// Point Hash functor
struct Hash_of_Point_index {
    size_t operator() (const Point_index& P) const {
        return _HASH_(P.x, P.y, P.z);
    }
};

struct Bucket {
    int first = -1, count = -1;
    Bucket() {  first = -1; count = -1; }
    Bucket(int _first, int _count) : first(_first), count(_count) {}
};

typedef unordered_multimap< Point_index, int, Hash_of_Point_index> Point_Index_Map;
typedef vector<Bucket> Buckets;
typedef vector<Point_index> Point_indexes;

#define FOR_RANGE(it,range) for(auto& it = range.first; it != range.second; ++it)
#define FOR_ITER(local_it,point_map) for (auto local_it = point_map.begin(i); local_it != point_map.end(i); ++local_it)

class Space_map2
{
    double map_size;
public:
    Point_Index_Map point_map;
    Buckets buckets;
    Point_indexes point_indexes;

    Space_map2(const Points& points, const double& mapsize) {
        int num_points = points.size();
        map_size = mapsize;
        point_map.reserve(num_points * 2);

        for (size_t i = 0; i < num_points; i++)
            point_map.emplace(Point_index(points[i], i, map_size), i);
    }

    Space_map2(const Faces faces, const Points& points, const double& mapsize) {
        int num_faces = faces.size(), xmin, ymin, zmin, xmax, ymax, zmax;
        map_size = mapsize;
        point_map.reserve(num_faces * 5);

        for (size_t iFace = 0; iFace < num_faces; iFace++) {
                
            Point_index p0(points[faces[iFace].v[0]], map_size);
            Point_index p1(points[faces[iFace].v[1]], map_size);
            Point_index p2(points[faces[iFace].v[2]], map_size);

            xmin = _MIN3_(p0.x, p1.x, p2.x);            xmax = _MAX3_(p0.x, p1.x, p2.x);
            ymin = _MIN3_(p0.y, p1.y, p2.y);            ymax = _MAX3_(p0.y, p1.y, p2.y);
            zmin = _MIN3_(p0.z, p1.z, p2.z);            zmax = _MAX3_(p0.z, p1.z, p2.z);

            for (int i = xmin; i <= xmax; i++)
            for (int j = ymin; j <= ymax; j++)
            for (int k = zmin; k <= zmax; k++)
                point_map.emplace(Point_index(i,j,k, iFace), iFace);

        }
    }

    void get_cuda_hashmap() {
        int first_item = 0, count_items = 0;
        auto bucket_count = point_map.bucket_count();

        for (unsigned i = 0; i < bucket_count; ++i) {
            count_items = 0;
            for (auto iter = point_map.begin(i); iter != point_map.end(i); ++iter) {
                point_indexes.push_back(iter->first);
                count_items++;
            }
            buckets.push_back(Bucket(first_item, count_items));
            first_item = first_item + count_items;
        }

    }

    void lookup_point_region(const Point_index& P, vector<int>& point_indexes) {
        auto count = point_map.count(P);

        if (count > 0) {
            auto range = point_map.equal_range(P);
            FOR_RANGE(point, range) {
                int point_index = point->second;
                point_indexes.push_back(point_index);
            }
        }
    }


    void get_nearby_candidate_points(const Point& target, const double& beta, vector<int>& point_indexes) {
        int max_index = round(beta / map_size);
        Point_index target_index(target,map_size);
        point_indexes.reserve(50);
        int imin = target_index.x - max_index;        int imax = target_index.x + max_index;
        int jmin = target_index.y - max_index;        int jmax = target_index.y + max_index;
        int kmin = target_index.z - max_index;        int kmax = target_index.z + max_index;

        for (int i = imin; i <= imax; i++)
        for (int j = jmin; j <= jmax; j++)
        for (int k = kmin; k <= kmax; k++)
            lookup_point_region(Point_index( i, j, k), point_indexes);
    }

    double search_space_map(const Points& points, const Point& target, const double& beta, int& nearest_point ) {
        vector<int> point_indexes;
        double beta2 = beta * beta;
        get_nearby_candidate_points(target, beta, point_indexes);

        double min_dist = target.dist(points[0]);

        for (int i : point_indexes)
        {
            float dist = target.dist(points[i]);
            if (dist < min_dist && dist < beta2)
            {
                min_dist = dist;
                nearest_point = i;
            }
        }
        return _MIN2_(min_dist, beta2);
    }

    bool make_empty() {
        return point_map.empty();
    }

#define _CALC_BLOCK_DIM_(n,t) (n+t-1)/t
#define _MAP_INDEX_(x,y) round(x/y)
    void get_dim(const Point& target, const double& map_size, const double& beta, int& threads_dim, int& blocks_dim, int& dim, int& i0, int& j0, int& k0) {
        Point_index target_index(target, map_size);
        int max_size_index = _MAP_INDEX_(beta, map_size) + 1;

        max_size_index = max_size_index + max_size_index % 2;
        int num_threads = 2 * max_size_index;

        threads_dim = 4;
        blocks_dim = _CALC_BLOCK_DIM_(num_threads, threads_dim);
        dim = threads_dim * blocks_dim;

        i0 = target_index.x - dim / 2;
        j0 = target_index.y - dim / 2;
        k0 = target_index.z - dim / 2;

        cout << endl << "DIMs : " << endl << "max_size_index : " << max_size_index << " , Threads_dim : " << threads_dim << " , blocks_dim : " << blocks_dim << " , Dim : " << dim;
        cout << endl << "(i0,  j0, k0) : ( " << i0 << " , " << j0 << " , " << k0 << " )";
        cout << endl << "(i1,  j1, k1) : ( " << i0 + dim << " , " << j0 + dim << " , " << k0 + dim << " )" << endl;
    }

#define _MINIMUM_(A,B) A < B ? A : B
#define _LINEAR_INDEX_(i,j,k,dim) i + j * dim + k * dim * dim

#define _FOR_ITER_I_(threadId,x,threads_dim)for (threadId.x = 0; threadId.x < threads_dim; threadId.x++)
#define _FOR_ITER_XYZ_(threadId,threads_dim) _FOR_ITER_I_(threadId,x,threads_dim) _FOR_ITER_I_(threadId,y,threads_dim) _FOR_ITER_I_(threadId,z,threads_dim)

    double serial_calculate_min_dist(const Points& points, const Point& target, const double& beta)
    {
        int threads_dim, blocks_dim, dim, i0, j0, k0;
        get_dim(target, map_size, beta, threads_dim, blocks_dim, dim, i0, j0, k0);

        auto beta2 = beta * beta;
        auto bucket_count = point_map.bucket_count();
        vector<double> min_distances(dim * dim * dim);

        Point_index threadId, blockId;
        _FOR_ITER_XYZ_(blockId, blocks_dim)
        _FOR_ITER_XYZ_(threadId, threads_dim)
        {
            int i = threadId.x + blockId.x * threads_dim;
            int j = threadId.y + blockId.y * threads_dim;
            int k = threadId.z + blockId.z * threads_dim;

            min_distances[_LINEAR_INDEX_(i, j, k, dim)] = beta2;

            int bucket_index = _HASH_(i0 + i, j0 + j, k0 + k) % bucket_count;

            int first = buckets[bucket_index].first;
            int count = buckets[bucket_index].count;
            double min_distance = beta2, dist;
            //if(k==0 && j == 0) printf("bucket_index : %d ,bucket_count : %d, first : %d , count : %d\n",bucket_index,bucket_count,first,count);

            for (size_t iter = first; iter < first + count; iter++)
            {
                if (count == 0) break;
                //printf("count : %d, i : %d , j : %d , k : %d\n", count, i, j, k);
                const Point_index& p = point_indexes[iter];
                if (p.x == (i0 + i) && p.y == (j0 + j) && p.z == (k0 + k)) {
                    dist = _DISTANCE_(points[p.index], target);
                    min_distance = _MINIMUM_(min_distance, dist);
                    //printf("dist %f : %d,%d,%d\n", dist, i + i0, j + j0, k + k0);
                }

            }
            min_distances[_LINEAR_INDEX_(i, j, k, dim)] = min_distance;
        }
        auto min_dist_var = min_distances[0];
        for (auto dist : min_distances)
            min_dist_var = _MINIMUM_(min_dist_var, dist);

        return min_dist_var;

    }

    double serial_calculate_min_dist(const Faces& faces, const Points& points, const Point& target, const double& beta)
    {
        int threads_dim, blocks_dim, dim, i0, j0, k0;
        get_dim(target, map_size, beta, threads_dim, blocks_dim, dim, i0, j0, k0);
        auto beta2 = beta * beta;
        auto bucket_count = point_map.bucket_count();
        vector<double> min_distances(dim * dim * dim);

        Point_index threadId, blockId;
        _FOR_ITER_XYZ_(blockId, blocks_dim)
        _FOR_ITER_XYZ_(threadId, threads_dim)
        {
            int i = threadId.x + blockId.x * threads_dim;
            int j = threadId.y + blockId.y * threads_dim;
            int k = threadId.z + blockId.z * threads_dim;

            min_distances[_LINEAR_INDEX_(i, j, k, dim)] = beta2;

            int bucket_index = _HASH_(i0 + i, j0 + j, k0 + k) % bucket_count;

            int first = buckets[bucket_index].first;
            int count = buckets[bucket_index].count;
            double min_distance = beta2, dist;
            //if(k==0 && j == 0) printf("bucket_index : %d ,bucket_count : %d, first : %d , count : %d\n",bucket_index,bucket_count,first,count);

            for (size_t iter = first; iter < first + count; iter++)
            {
                if (count == 0) break;
                //printf("count : %d, i : %d , j : %d , k : %d\n", count, i, j, k);
                const Point_index& face_index = point_indexes[iter];
                if (face_index.x == (i0 + i) && face_index.y == (j0 + j) && face_index.z == (k0 + k)) {
                    dist = faces[face_index.index].dist(target, points);
                    min_distance = _MINIMUM_(min_distance, dist);
                    //printf("dist %f : %d,%d,%d\n", dist, i + i0, j + j0, k + k0);
                }

            }
            min_distances[_LINEAR_INDEX_(i, j, k, dim)] = min_distance;
        }
        auto min_dist_var = min_distances[0];
        for (auto dist : min_distances)
            min_dist_var = _MINIMUM_(min_dist_var, dist);

        return min_dist_var;

    }


    //double search_space_map_parallel(const Points& points, const Point& target, const double& beta, int& nearest_point) {
    //    vector<int> point_indexes;
    //    vector<double> dists;
    //    double beta2 = beta * beta;
    //    get_nearby_candidate_points(target, beta, point_indexes);
    //    Points points_filtered;
    //    points_filtered.reserve(point_indexes.size());

    //    for (int i : point_indexes)
    //        points_filtered.push_back(points[i]);

    //    lookup_region_parallel(points_filtered, target, beta2, dists);
    //}

    //void lookup_region_parallel(const Points& points, const Point& target, const double& beta2, vector<double>& dist) {
    //    int i = threadIdx.x;
    //    double min_dist = target.dist(points[i]);
    //    dist[i] = (min_dist > beta2) ? beta2 : min_dist;
    //}

};
