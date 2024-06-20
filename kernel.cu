//%%cuda
// #define GOOGLECOLLAB
// # prompt: write any cuda program using c++
// !pip install nvcc4jupyter
// %load_ext nvcc4jupyter

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#ifdef GOOGLECOLLAB

#include "/content/Containers/Space_map2.h"
#include "/content/Input_output/Loader.h"
#include "/content/Geometry/Point.h"
#include "/content/Geometry/Face.h"
#include "/content/Thrust_lib/thrust_dist.h"
#include "/content/Thrust_lib/unsigned_distance_function.h"
#include "/content/Input_output/Loader.cpp"
#include "/content/Thrust_lib/unsigned_distance_function.cpp"

#else

#include "Geometry/Point.h"
#include "Geometry/Face.h"
#include "Containers/Space_map2.h"
#include "Input_output/Loader.h"
#include "Thrust_lib/thrust_dist.h"
#include "Thrust_lib/unsigned_distance_function.h"

#endif

#define _DEVICE_RESET_FAILED_ if (cudaStatus != cudaSuccess) { fprintf(stderr, "\n\ncudaDeviceReset failed!\n\n");return 1;}

#define _SETUP_FAILED_ if (cudaStatus != cudaSuccess) {fprintf(stderr, "\n\ncudaSetDevice failed!  \n\nDo you have a CUDA-capable GPU installed?\n\n");goto Error;}
#define _MALLOC_FAILED_ if (cudaStatus != cudaSuccess) {fprintf(stderr, "\n\ncudaMalloc failed!\n\n");goto Error;}
#define _SYNC_FAILED_ if (cudaStatus != cudaSuccess) {fprintf(stderr, "\n\ncudaDeviceSynchronize returned error code %d after launching addKernel!\n\n");goto Error;}
#define _KERNAL_FAILED_ if (cudaStatus != cudaSuccess) {    fprintf(stderr, "\n\naddKernel launch failed: %s\n\n", cudaGetErrorString(cudaStatus));    goto Error;}

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
inline void cuda_begin_subroutine(cudaEvent_t& start, cudaEvent_t& stop);
inline int cuda_end_subroutine(cudaEvent_t& start, cudaEvent_t& stop);

_HOST_DEVICE_
double Face::dist(const Point& target, const Point* points) const {
    return get_nearest_point_dist(points[v[0]], points[v[1]], points[v[2]], target);
}

_HOST_DEVICE_
double Face::dist(const Point& target, const Points& points) const {
    return get_nearest_point_dist(points[v[0]], points[v[1]], points[v[2]], target);
}

_HOST_DEVICE_
double Point::dist(const Point& p0, const Point& p1, const Point& p2) const {
    return get_nearest_point_dist(p0, p1, p2, *this);
}

__global__
void addKernel(int* c, const int* a, const int* b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

#define _MAP_INDEX_(x,y) round(x/y)

__global__
void calculate_parallel_hash_map(const map_element2* point_iter, const int nmax, Bucket* bucket) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid > nmax || point_iter[tid].bucket == point_iter[tid + 1].bucket)
        return;

    bucket[point_iter[tid].bucket].count = tid;
    bucket[point_iter[tid + 1].bucket].first = tid + 1;
}
#define _CAST_(P) thrust::raw_pointer_cast(P.data())
#define _RAW_CAST_(P,Q,R,S) _CAST_(P) ,_CAST_(Q) , _CAST_(R) , _CAST_(S)
#define _RAW_CAST_5_(P,Q,R,S,T) _CAST_(P) ,_CAST_(Q) , _CAST_(R) , _CAST_(S) , _CAST_(T)

void cuda_parallel_hashmap(const Points& points, float map_size) {
    int n = points.size();
    Device_Points device_points = points;

    // Create a device vector of size n
    thrust::device_vector<int> indexes(n);
    thrust::device_vector<Bucket> buckets(n);
    thrust::device_vector<map_element2> map_elements(n, map_element2(-1, -1));

    // Generate sequence on the device, starting from 1
    thrust::sequence(_ITER_(indexes),0);

    // apply the transformation
    thrust::transform(_ITER_(device_points), indexes.begin(), map_elements.begin(), get_map_element(n, map_size));
    thrust::sort(_ITER_(map_elements), CompareBucketOrder());

    // Determine block size and grid size
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    calculate_parallel_hash_map << <blocksPerGrid, threadsPerBlock >> > 
        (_CAST_(map_elements), n - 2, _CAST_(buckets));
}

#define _CALC_BLOCK_DIM_(n,t) (n+t-1)/t
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

__global__
void calculate_min_dist(
    const Bucket* buckets, const Point_index* indexes, const Point* points, double* min_distances,
    Point target, double beta2, int bucket_count, int dim,
    int i0, int j0, int k0)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    // Print thread and block indices
   //if(k==0 && j == 0) printf("i : %d , j : %d , k : %d\n",i,j,k);
    if (i >= dim || j >= dim || k >= dim)
        return;

    min_distances[_LINEAR_INDEX_(i, j, k, dim)] = beta2;

    int bucket_index = _HASH_(i0 + i, j0 + j, k0 + k) % bucket_count;

    int first = buckets[bucket_index].first;
    int count = buckets[bucket_index].count;

    //if(k==0 && j == 0) printf("bucket_index : %d ,bucket_count : %d, first : %d , count : %d\n",bucket_index,bucket_count,first,count);

    double min_distance = beta2, dist;

    for (size_t iter = first; iter < first + count; iter++)
    {
        if (count == 0) break;
        //printf("count : %d, i : %d , j : %d , k : %d\n", count, i, j, k);
        const Point_index& p = indexes[iter];
        if (p.x == (i0 + i) && p.y == (j0 + j) && p.z == (k0 + k)) {
            dist = _DISTANCE_(points[p.index], target);
            min_distance = _MINIMUM_(min_distance, dist);
            //printf("dist %f : %d,%d,%d\n", dist, i + i0, j + j0, k + k0);
        }
    }
    min_distances[_LINEAR_INDEX_(i, j, k, dim)] = min_distance;
}

__global__
void calculate_min_dist(
    const Bucket* buckets, const Point_index* indexes, const Face* faces, const Point* points, double* min_distances,
    Point target, double beta2, int bucket_count, int dim,
    int i0, int j0, int k0)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= dim || j >= dim || k >= dim)
        return;

    min_distances[_LINEAR_INDEX_(i, j, k, dim)] = beta2;

    int bucket_index = _HASH_(i0 + i, j0 + j, k0 + k) % bucket_count;

    int first = buckets[bucket_index].first;
    int count = buckets[bucket_index].count;
    double min_distance = beta2, dist;

    for (size_t iter = first; iter < first + count; iter++)
    {
        const Point_index& p = indexes[iter];
        if (p.x == (i0 + i) && p.y == (j0 + j) && p.z == (k0 + k)) {
            dist = faces[p.index].dist(target, points);
            min_distance = _MINIMUM_(min_distance, dist);
            //printf("dist %f :Face%d(%d,%d,%d), target : (%f,%f,%f) ,Index(%d,%d,%d)\n",dist,p.index,faces[p.index].v[0],faces[p.index].v[1],faces[p.index].v[2],target.x,target.y,target.z,i + i0, j + j0, k + k0);
        }
    }
    min_distances[_LINEAR_INDEX_(i, j, k, dim)] = min_distance;
}

#define _DIM3_(x) x,x,x
#define _CUBE_(x) x*x*x
double serial_hash_map_implementation(const Points& points, const Point& target, double map_size, double beta)
{
    Space_map2 space_map(/* with input points as */ points, /* map_size as */ map_size);
    space_map.get_cuda_hashmap();
    return space_map.serial_calculate_min_dist(points, target, beta);
}

double cuda_hash_map_implementation(const Points& points, const Point& target, double map_size, double beta)
{
    Space_map2 space_map(/* with input points as */ points, /* map_size as */ map_size);
    space_map.get_cuda_hashmap();
    //return space_map.calculate_min_dist(points, target, beta);

    auto bucket_count = space_map.buckets.size();
    auto beta2 = beta * beta;

    int threads_dim, blocks_dim, dim, i0, j0, k0;
    get_dim(target, map_size, beta, threads_dim, blocks_dim, dim, i0, j0, k0);

    dim3 threads_per_block(_DIM3_(threads_dim));
    dim3 blocks_per_grid(_DIM3_(blocks_dim));

    thrust::device_vector<Bucket>       buckets = space_map.buckets;
    thrust::device_vector<Point_index>  point_indexes = space_map.point_indexes;
    thrust::device_vector<Point>        Dpoints = points;
    thrust::device_vector<double>       min_distances(_CUBE_(dim));


    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cuda_begin_subroutine(start, stop);

    calculate_min_dist << <blocks_per_grid, threads_per_block >> > (
        _RAW_CAST_(buckets, point_indexes, Dpoints, min_distances),
        target, beta2, bucket_count, threads_dim * blocks_dim, i0, j0, k0
        );

    cudaDeviceSynchronize();

    auto dist = thrust::reduce(_ITER_(min_distances), beta2, min_dist());
    cudaDeviceSynchronize();

    cuda_end_subroutine(start, stop);
    return dist;
}

double serial_hash_map_implementation(const Faces& faces, const Points& points, const Point& target, double map_size, double beta)
{
    Space_map2 space_map(/* with input Faces as */faces,/* with input points as */ points, /* map_size as */ map_size);
    space_map.get_cuda_hashmap();
    return space_map.serial_calculate_min_dist(faces, points, target, beta);
}

double cuda_hash_map_implementation(const Faces& faces, const Points& points, const Point& target, double map_size, double beta)
{
    Space_map2 space_map(/* with input Faces as */faces,/* with input points as */ points, /* map_size as */ map_size);
    space_map.get_cuda_hashmap();
    int bucket_count = space_map.buckets.size();
    double beta2 = beta * beta;

    int threads_dim, blocks_dim, dim, i0, j0, k0;
    get_dim(target, map_size, beta, threads_dim, blocks_dim, dim, i0, j0, k0);

    thrust::device_vector<Bucket>       buckets = space_map.buckets;
    thrust::device_vector<Point_index>  point_indexes = space_map.point_indexes;
    thrust::device_vector<Point>        Dpoints = points;
    thrust::device_vector<Face>         Dfaces = faces;
    thrust::device_vector<double>       min_distances(_CUBE_(dim), beta2);

    dim3 threads_per_block(_DIM3_(threads_dim));
    dim3 blocks_per_grid(_DIM3_(blocks_dim));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cuda_begin_subroutine(start, stop);

    calculate_min_dist << <blocks_per_grid, threads_per_block >> > (
        _RAW_CAST_5_(buckets, point_indexes, Dfaces, Dpoints, min_distances),
        target, beta2, bucket_count, threads_dim * blocks_dim, i0, j0, k0);
    cudaDeviceSynchronize();
    auto dist = thrust::reduce(_ITER_(min_distances), beta2, min_dist());
    cudaDeviceSynchronize();


    cuda_end_subroutine(start, stop);
    return dist;

}

inline void cuda_begin_subroutine(cudaEvent_t& start, cudaEvent_t& stop)
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start, 0);
}

inline int cuda_end_subroutine(cudaEvent_t& start, cudaEvent_t& stop)
{
    // Wait for the kernel to complete
    cudaDeviceSynchronize();

    // Record stop event
    cudaEventRecord(stop, 0);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // cudaDeviceReset must be called before exiting in order for profiling and tracing tools such as Nsight and Visual Profiler to show complete traces.
//    auto cudaStatus = cudaDeviceReset(); 
//    _DEVICE_RESET_FAILED_

    return 0;
}

int run_cuda_test(const AABB& box, const Faces& faces, const Points& points, const Point& target, const double& map_size, const double& beta);

int main()
{
    AABB box;

    #if defined(GOOGLECOLLAB)
        objl::Mesh mesh;    get_mesh("/content/3DObjects/cube.obj", mesh);
    #else
        objl::Mesh mesh;    get_mesh("3DObjects/cube.obj", mesh);
    #endif

    Points points;      get_points(mesh, points, box);
    Faces faces;        get_faces(mesh, faces);

    // Basecase
  //   Points points = { Point(0,0,0),Point(0,0,1),Point(0,1,1),Point(0,1,0) };

    Point* mypoints = new Point[4];
    Face* myfaces = new Face[2];
    mypoints[0] = Point(0, 0, 0);
    mypoints[1] = Point(0, 0, 1);
    mypoints[2] = Point(0, 1, 1);
    mypoints[3] = Point(0, 1, 0);
    myfaces[0] = Face(0, 1, 3);
    myfaces[1] = Face(2, 3, 1);
    //     Faces faces = { Face(0,1,3),Face(2,3,1) };
    //     box = AABB(-1,0,0,1,1,1);

    cout << "Num of points : " << points.size() << endl;
    cout << "Num of faces : " << faces.size() << endl;

    box.print();

    Point target = { 0,1,1.2 };
    double beta = 0.3;
    double map_size = 0.02;
    cout << "dist test : " << endl;
    cout << myfaces[0].dist(target, mypoints) << endl;
    cout << myfaces[1].dist(target, mypoints) << endl;

    auto res = run_cuda_test(box, faces, points, target, map_size, beta);
    return res;
}

int run_cuda_test(const AABB& box, const Faces& faces, const Points& points, const Point& target, const double& map_size, const double& beta) {
    int nearest_point = -1; int nearest_face = -1; float dist;

    Point_index(Point(box.xmin, box.ymin, box.zmin), map_size).print();
    Point_index(Point(box.xmax, box.ymax, box.zmax), map_size).print();
    cout << endl << endl;

    cout << "Target point : "; target.print();
    cout << "Target point index : "; Point_index(target, map_size).print();
    cout << "Beta : " << beta << endl;
    cout << "Map size : " << map_size << endl << endl;

    if (box.check_point_outside_beta_box(target, beta))
    {
        cout << "Target point is outside Beta box" << endl;;
        return 0;
    }

    cout << "---------------------Pointwise(brute force)-----------------" << endl;
    cout << "Unsigned distance for Points (brute force) => " << endl; nearest_point = -1;
    dist = unsigned_distance_brute_force(points, target, beta, nearest_point); cout << endl;
    print_output(dist, nearest_point, target, points);

    cout << "-----------------------Facewise(brute force)---------------------------" << endl;
    cout << "Unsigned distance for Faces (brute force) => " << endl; nearest_face = -1;
    dist = unsigned_distance_brute_force(faces, points, target, beta, nearest_face); cout << endl;
    print_output(dist, nearest_face, target, points, faces);

    cout << "---------------------Pointwise(local)---------------------------" << endl;
    cout << "Unsigned distance for Points (space map) => " << endl; nearest_point = -1;
    dist = unsigned_distance_space_map2(points, target, beta, map_size, nearest_point); cout << endl;
    print_output(dist, nearest_point, target, points);

    cout << "---------------------Pointwise(Serial)-----------------" << endl;
    cout << "Unsigned distance for Points (Serial) => " << endl; nearest_point = -1;
    dist = serial_hash_map_implementation(points, target, map_size, beta); cout << endl;
    print_output(dist, nearest_point, target, points);

    cout << "-----------------------Facewise(Serial)---------------------------" << endl;
    cout << "Unsigned distance for Faces (Serial) => " << endl; nearest_point = -1;
    dist = serial_hash_map_implementation(faces, points, target, map_size, beta); cout << endl;
    print_output(dist, nearest_face, target, points, faces);

    cout << endl << endl;
    cout << "**************************CUDA_TEST_BEGINS********************************" << endl;
    cout << "**************************CUDA_TEST_BEGINS********************************" << endl << endl;

    const int arraySize = 5, a[arraySize] = { 1, 2, 3, 4, 5 }, b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize); _DEVICE_RESET_FAILED_
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n", c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset(); _DEVICE_RESET_FAILED_

    cout << endl;
    cout << "**************************CUDA_TEST_SUCCESS********************************" << endl;
    cout << "**************************CUDA_TEST_SUCCESS********************************" << endl << endl;
    cout << "---------------------Pointwise(cuda)---------------------------" << endl;
    cout << "Unsigned_distance_cuda_hash_table with Points => " << endl; nearest_point = -1;
    dist = cuda_hash_map_implementation(points, target, map_size, beta); cout << endl;
    print_output(dist, nearest_point, target, points);
    cout << endl << endl;


    cout << "-----------------------Facewise(cuda)---------------------------" << endl;
    cout << "Unsigned_distance_cuda_hash_table with Faces => " << endl; nearest_point = -1;
    dist = cuda_hash_map_implementation(faces, points, target, map_size, beta); cout << endl;
    print_output(dist, nearest_point, target, points);
    cout << endl << endl;

    return 0;

    cout << "---------------------Pointwise---------------------------" << endl;
    cout << "Unsigned_distance_space_map Old" << endl; nearest_point = -1;
    dist = unsigned_distance_space_map_cuda(points, target, beta, map_size, nearest_point); cout << endl;
    print_output(dist, nearest_point, target, points);
    cout << endl << endl;

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0); _SETUP_FAILED_

        // Allocate GPU buffers for three vectors (two input, one output)    .
        cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int)); _MALLOC_FAILED_
        cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int)); _MALLOC_FAILED_
        cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int)); _MALLOC_FAILED_

        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice); _MALLOC_FAILED_
        cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice); _MALLOC_FAILED_

        // Launch a kernel on the GPU with one thread for each element.
        addKernel << <1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError(); _KERNAL_FAILED_

        // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize(); _SYNC_FAILED_

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost); _MALLOC_FAILED_

        Error :
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
#define __DIFF__(diff,P0,target) diff.x = P0.x - target.x; diff.y = P0.y - target.y; diff.z = P0.z - target.z;

__device__ __host__
double get_nearest_point_dist(const Point& P0, const Point& P1, const Point& P2, const Point& target) {
    //        printf("get_nearest_point_dist(begin) : target(%f,%f,%f) , P0(%f,%f,%f) , P1(%f,%f,%f) , P2(%f,%f,%f)\n",target.x,target.y,target.z,P0.x,P0.y,P0.z,P1.x,P1.y,P1.z,P2.x,P2.y,P2.z);

    double zero = 0.0;
    double one = 1.0;
    double two = 2.0;

    double3 diff; __DIFF__(diff, P0, target);
    double3 edge0; __DIFF__(edge0, P1, P0);
    double3 edge1; __DIFF__(edge1, P2, P0);
    double  a00 = _DOT_(edge0, edge0);
    double  a01 = _DOT_(edge0, edge1);
    double  a11 = _DOT_(edge1, edge1);
    double  b0 = _DOT_(diff, edge0);
    double  b1 = _DOT_(diff, edge1);
    //   printf("get_nearest_point_dist(begin),%f,%f,%f,%f,%f\n",a00,a01,a11,b0,b1);


//    double  det = std::max(a00 * a11 - a01 * a01, zero);
    double det = a00 * a11 - a01 * a01;
    det = det > zero ? det : zero;

    double  s = a01 * b1 - a11 * b0;
    double  t = a01 * b0 - a00 * b1;

    if (s + t <= det)
    {

        if (s < zero)
        {

            if (t < zero)  // region 4
            {
                //  printf("get_nearest_point_dist(region 4)\n");

                if (b0 < zero)
                {
                    t = zero;
                    if (-b0 >= a00)
                        s = one;
                    else
                        s = -b0 / a00;
                }
                else
                {
                    s = zero;
                    if (b1 >= zero)
                        t = zero;
                    else if (-b1 >= a11)
                        t = one;
                    else
                        t = -b1 / a11;
                }
            }
            else  // region 3
            {
                //  printf("get_nearest_point_dist(region 3)\n");

                s = zero;
                if (b1 >= zero)
                    t = zero;
                else if (-b1 >= a11)
                    t = one;
                else
                    t = -b1 / a11;
            }
        }
        else if (t < zero)  // region 5
        {
            //       printf("get_nearest_point_dist(region 5)\n");

            t = zero;
            if (b0 >= zero)
                s = zero;
            else if (-b0 >= a00)
                s = one;
            else
                s = -b0 / a00;
        }
        else  // region 0
        {

            //      printf("get_nearest_point_dist(region 0)\n");
          // minimum at interior point
            s /= det;
            t /= det;
        }
    }
    else
    {
        double tmp0{}, tmp1{}, numer{}, denom{};

        if (s < zero)  // region 2
        {
            //      printf("get_nearest_point_dist(region 2)\n");

            tmp0 = a01 + b0;
            tmp1 = a11 + b1;
            if (tmp1 > tmp0)
            {
                numer = tmp1 - tmp0;
                denom = a00 - two * a01 + a11;
                if (numer >= denom)
                {
                    s = one;
                    t = zero;
                }
                else
                {
                    s = numer / denom;
                    t = one - s;
                }
            }
            else
            {
                s = zero;
                if (tmp1 <= zero)
                    t = one;
                else if (b1 >= zero)
                    t = zero;
                else
                    t = -b1 / a11;
            }
        }
        else if (t < zero)  // region 6
        {
            //                printf("get_nearest_point_dist(region 6)\n");

            tmp0 = a01 + b1;
            tmp1 = a00 + b0;
            if (tmp1 > tmp0)
            {
                numer = tmp1 - tmp0;
                denom = a00 - two * a01 + a11;
                if (numer >= denom)
                {
                    t = one;
                    s = zero;
                }
                else
                {
                    t = numer / denom;
                    s = one - t;
                }
            }
            else
            {
                t = zero;
                if (tmp1 <= zero)
                    s = one;
                else if (b0 >= zero)
                    s = zero;
                else
                    s = -b0 / a00;
            }
        }
        else  // region 1
        {
            //             printf("get_nearest_point_dist(region 1)\n");

            numer = a11 + b1 - a01 - b0;
            if (numer <= zero)
            {
                s = zero;
                t = one;
            }
            else
            {
                denom = a00 - two * a01 + a11;
                if (numer >= denom)
                {
                    s = one;
                    t = zero;
                }
                else
                {
                    s = numer / denom;
                    t = one - s;
                }
            }
        }
    }

    double3 res;
    res.x = P0.x + edge0.x * s + edge1.x * t;
    res.y = P0.y + edge0.y * s + edge1.y * t;
    res.z = P0.z + edge0.z * s + edge1.z * t;
    auto ret = _DISTANCE_(res, target);
    //        printf("get_nearest_point_dist(ret : %f , s : %f , t : %f),P0(%f,%f,%f),Edge0(%f,%f,%f),Edge1(%f,%f,%f)\n",ret,s,t,P0.x,P0.y,P0.z,edge0.x,edge0.y,edge0.z,edge1.x,edge1.y,edge1.z);

    return ret;
}



