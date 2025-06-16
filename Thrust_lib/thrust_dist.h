#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "../Geometry/Point.h"
#include "../Containers/Space_map2.h"
typedef thrust::host_vector<double> Host_Vector;        typedef thrust::host_vector<Point> Host_Points;
typedef thrust::device_vector<double> Device_Vector;    typedef thrust::device_vector<Point> Device_Points;

#define _ITER_(Y) Y.begin(), Y.end()
#define _POINT_(X,Y,Z) thrust::make_zip_iterator(thrust::make_tuple(X.begin(), Y.begin(), Z.begin())) , thrust::make_zip_iterator(thrust::make_tuple(X.end(), Y.end(), Z.end()))

struct min_dist
{
    _HOST_DEVICE_
        double operator()(const double& Z, const double& Y) const {
        return (Z < Y) ? Z : Y;
    }
};

struct dist2_point
{
    const Point target;
    dist2_point(Point _target) : target(_target) {}

    _HOST_DEVICE_
        double operator()(const Point& P) {
        return _DISTANCE_(P,target);
    }
};

double min_dist_calculation2(const Points& candidate_points, const Point& target, const double& beta2) {
    Device_Points device_candidate_points = candidate_points;
    Device_Vector candidate_min_distances(candidate_points.size());

    // apply the transformation
    thrust::transform(_ITER_(device_candidate_points), candidate_min_distances.begin(), dist2_point(target));
    return thrust::reduce(_ITER_(candidate_min_distances), beta2, min_dist());
}

#define _MIN_BOX_(A,B) (A<B?A:B)
#define _MAX_BOX_(A,B) (A>B?A:B)
#define _MIN_BOX_POINT_(A,B) _MIN_BOX_(A.x,B.x) , _MIN_BOX_(A.y,B.y) , _MIN_BOX_(A.z,B.z) 
#define _MAX_BOX_POINT_(A,B) _MAX_BOX_(A.x,B.x) , _MAX_BOX_(A.y,B.y) , _MAX_BOX_(A.z,B.z) 

struct min_box
{
    _HOST_DEVICE_
        Point operator()(const Point& P, const Point& Q) const {
        return Point(_MIN_BOX_POINT_(P,Q));
    }
};
struct max_box
{
    _HOST_DEVICE_
        Point operator()(const Point& P, const Point& Q) const {
        return Point(_MAX_BOX_POINT_(P, Q));
    }
};
void cuda_bounding_box_calc(const Points& points,AABB &box) {
    Device_Points device_points = points;

    // apply the transformation
    Point Pmin = thrust::reduce(_ITER_(device_points), points[0], min_box());
    Point Pmax = thrust::reduce(_ITER_(device_points), points[0], max_box());
    box = AABB(Pmin, Pmax);
}

struct map_element2 {
    int bucket, index;
    _HOST_DEVICE_ map_element2() { bucket = -1; index = -1; }
    _HOST_DEVICE_ map_element2(int _bucket, int _index) { bucket = (_bucket); index = (_index); }
    _HOST_DEVICE_ bool operator<(const map_element2& y) {return bucket < y.bucket;}
    _HOST_DEVICE_ bool operator>(const map_element2& y) {return bucket > y.bucket;}
};
struct CompareBucketOrder {
    _HOST_DEVICE_
        bool operator()(const map_element2& p1, const map_element2& p2) const {
        return p1.bucket < p2.bucket;
    }
};
struct get_map_element
{
    const int bucket_count;
    const float map_size;
    get_map_element(int _bucket_count, float _map_size) : 
        bucket_count(_bucket_count) , map_size(_map_size) {}

    _HOST_DEVICE_
        map_element2 operator()(const Point& P, const int& p) const {

        int i = _MAP_INDEX_(P.x, map_size);
        int j = _MAP_INDEX_(P.y, map_size);
        int k = _MAP_INDEX_(P.z, map_size);

        int index = _HASH_(i, j, k) % bucket_count;
        return map_element2(index, p);
    }
};
struct get_hash
{
    const int bucket_count;
    const float map_size;

    get_hash(int _bucket_count, float _map_size) :
        bucket_count(_bucket_count), map_size(_map_size) {}

    _HOST_DEVICE_
        int operator()(const Point& P) const {
        int i = _MAP_INDEX_(P.x, map_size);
        int j = _MAP_INDEX_(P.y, map_size);
        int k = _MAP_INDEX_(P.z, map_size);
        return _HASH_(i, j, k) % bucket_count;
    }
};

struct get_bucket_indexes
{
    Bucket* buckets;
    const int* point_wise_bucket_indexes, * bucket_wise_point_indexes;

    get_bucket_indexes(Bucket* _buckets, const int* _point_wise_bucket_indexes, const int* _bucket_wise_point_indexes) :
        buckets(_buckets),
        point_wise_bucket_indexes(_point_wise_bucket_indexes),
        bucket_wise_point_indexes(_bucket_wise_point_indexes) {}

    _HOST_DEVICE_
        void operator()(int i) const {
        const auto& bucket1 = point_wise_bucket_indexes[bucket_wise_point_indexes[i]];
        const auto& bucket2 = point_wise_bucket_indexes[bucket_wise_point_indexes[i + 1]];

        if (bucket1 == bucket2)
            return;
        buckets[bucket1].count = i;
        buckets[bucket2].first = i + 1;
    }
};

struct get_bucket_indexes2
{
    int* buckets, * point_wise_bucket_indexes;
    int bucket_count;

    get_bucket_indexes2(int* _buckets, int* _point_wise_bucket_indexes,int _bucket_count) : 
        buckets(_buckets), 
        point_wise_bucket_indexes(_point_wise_bucket_indexes),
        bucket_count(_bucket_count){}

    _HOST_DEVICE_
        void operator()(int i) const {
        auto bucket1 = point_wise_bucket_indexes[buckets[i + bucket_count]];
        auto bucket2 = point_wise_bucket_indexes[buckets[i + 1 + bucket_count]];

        if (bucket1 == bucket2)
            return;
        buckets[2 * bucket1 + 1] = i;
        buckets[2 * bucket2]= i + 1;
    }
};

double unsigned_distance_space_map_cuda(const Points& points, const Point& target, double beta, double map_size, int& nearest_point) {

    nearest_point = -1;
    Space_map2 space_map(/* with input points as */ points, /* map_size as */ map_size);

    vector<int> candidate_point_indexes;
    double beta2 = beta * beta;

    space_map.get_nearby_candidate_points(target, beta, candidate_point_indexes);

    int num_candidate_points = candidate_point_indexes.size();
    Points candidate_points(num_candidate_points);
    for (const int& i : candidate_point_indexes)
        candidate_points.push_back(points[i]);

    return min_dist_calculation2(candidate_points, target, beta2);
}


#define _CAST_(P) thrust::raw_pointer_cast(P.data())
#define _CAST3_(X,Y,Z) _CAST_(X),_CAST_(Y),_CAST_(Z)


// This hashmap is generated entirely using thrust library
void cuda_parallel_hashmap_generation(
    const Points& points, float map_size, int bucket_count,
    thrust::device_vector<Bucket>& buckets, thrust::device_vector<int>& point_final_location) {

    const int n = points.size();
    Device_Points device_points = points;

    // Create a poitwise ordered vector of size n
    thrust::device_vector<int> point_bucket_number(n);
    auto& transformation_functor = get_hash(bucket_count, map_size);
    thrust::transform(_ITER_(device_points), point_bucket_number.begin(), transformation_functor);

    // Generate the Point bucket vector
    point_final_location.resize(n);
    thrust::sequence(_ITER_(point_final_location), 0);
    thrust::sort_by_key(_ITER_(point_final_location), point_bucket_number.begin());


    // Initialize the buckets ranges
    buckets.resize(bucket_count, Bucket(0, 0));
    buckets[bucket_count - 1] = Bucket(0, n - 1);

    // Generate the buckets ranges
    thrust::device_vector<int> i(n - 1,0);
    thrust::sequence(_ITER_(i),0);

    auto& hashing_functor = get_bucket_indexes(_CAST3_(
        buckets,
        point_bucket_number,
        point_final_location));
    thrust::for_each(_ITER_(i), hashing_functor);
}

// This hashmap is generated entirely using thrust library

#define _NULL_ -1
struct is_null_value {
    __host__ __device__
        bool operator()(const int& x) {
        return x == _NULL_;
    }
};

void cuda_parallel_hashmap_generation(
    const Points& points, float map_size, int bucket_count,
    thrust::device_vector<Bucket> & buckets, 
    thrust::device_vector<int>& point_final_location) {

    const int n = points.size();

    // Initialize the buckets ranges
    buckets.resize(bucket_count, Bucket(0, 0));
    buckets[bucket_count - 1] = Bucket(0, n - 1);

    thrust::device_vector<Point> device_points = points;
    thrust::device_vector<int> point_bucket_number(n);
    thrust::device_vector<int> point_bucket_counter(n, _NULL_);
    thrust::device_vector<int> bucket_counter(buckets.size(), 0);
    thrust::device_vector<int> pi(n, 0);
    thrust::sequence(_ITER_(pi), 0);

    auto& transformation_functor = get_hash(bucket_count, map_size);
    thrust::transform(_ITER_(device_points), point_bucket_number.begin(), transformation_functor);

    int* point_bucket_number_p = point_bucket_number.data().get();
    int* point_final_location_p = point_final_location.data().get();
    Bucket* bucket_p = buckets.data().get();

    int iter = 0;
    while(iter <5)
    {
        thrust::for_each(_ITER_(pi), [point_final_location_p , bucket_p, point_bucket_number_p](int index) {
            if (point_final_location_p[index] == _NULL_)
                bucket_p[point_bucket_number_p[index]].back_index = index;
        });

        thrust::for_each(_ITER_(buckets), [point_final_location_p] (Bucket &bucket) {
            if (point_final_location_p[bucket.back_index] == _NULL_) {
                point_final_location_p[bucket.back_index] = bucket.count;
                bucket.count++;
            }
        });

        auto it = thrust::find_if(_ITER_(point_final_location), is_null_value());
        if (it == point_final_location.end())
            break;

        iter++;
    }
    
    thrust::transform(_ITER_(buckets), bucket_counter.begin(), [](const Bucket &bucket) {return bucket.count;});
    thrust::inclusive_scan(_ITER_(bucket_counter), bucket_counter);

    int* bucket_counter_p = bucket_counter.data().get();
    int* point_final_location_p = point_final_location.data().get();

    thrust::for_each(_ITER_(pi), [point_final_location_p, bucket_counter_p, point_bucket_number_p](int index) {
        point_final_location_p[index] += bucket_counter_p[point_bucket_number_p[index]];
    });

    auto& hashing_functor = get_bucket_indexes(_CAST3_(
        buckets,
        point_bucket_number,
        point_final_location));
    thrust::for_each(_ITER_(pi), hashing_functor);
}




// test code only. not for use
// typedef vector<int> Index;
// typedef vector<Index>   Indexes;
//
// typedef thrust::host_vector<Index>   HIndexes;
// typedef thrust::device_vector<Index> DIndexes;

#define _px_ thrust::get<0>(t)
#define _py_ thrust::get<1>(t)
#define _pz_ thrust::get<2>(t)

#define _TGET_(i) thrust::get<i>
#define _SQ_DIFF_(P,Q) (P - Q)*(P - Q)
struct dist2_tuple
{
    const double x, y, z;
    dist2_tuple(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

    template <typename Tuple>
    _HOST_DEVICE_
        void operator()(Tuple t) {
        _py_ = _SQ_DIFF_(_px_, x) + _SQ_DIFF_(_py_, y) + _SQ_DIFF_(_pz_, z);
    }
};

#define CALC_DIST_TO_(target) dist2_tuple(target.x, target.y, target.z)
double min_dist_calculation(const Host_Vector& Px, const Host_Vector& Py, const Host_Vector& Pz, const Point& target, const double& beta2) {
    Device_Vector X = Px, Y = Py, Z = Pz;
    // apply the transformation
    thrust::for_each(_POINT_(X, Y, Z), CALC_DIST_TO_(target));
    return thrust::reduce(_ITER_(Y), beta2, min_dist());
}

    //Host_Vector X(n), Y(n), Z(n);
    //for (const int& i : candidate_point_indexes)
    //{
    //    X.push_back(points[i].x);
    //    Y.push_back(points[i].y);
    //    Z.push_back(points[i].z);
    //}
//    return min_dist_calculation(X, Y, Z, target, beta2);


//struct dist_sqxy
//{
//    const double x, y;
//    dist_sqxy(double _x, double _y) : x(_x), y(_y) {}
//    _HOST_DEVICE_
//        double operator()(const double& X, const double& Y) const {
//        return (X - x) * (X - x) + (Y - y) * (Y - y);
//    }
//};
//
//struct dist_sqz
//{
//    const double z;
//    dist_sqz(double _z) : z(_z) {}
//    _HOST_DEVICE_
//        double operator()(const double& Z, const double& Y) const {
//        return (Z - z) * (Z - z) + Y;
//    }
//};
