//
// Inspired and borrowd from
// website: https://coffeebeforearch.github.io/2020/06/23/mmul.html
// code:    https://github.com/CoffeeBeforeArch/mmul/tree/master
//

#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

typedef std::vector<double> vector;
typedef std::vector<std::vector<double>> matrix;

// Function to perform matrix multiplication C = A * B
// A: m x n matrix
// B: n x p matrix
// C: m x p matrix (result)
void serial_mmul_matrix(const matrix* A, const matrix* B, matrix* C)
{
    size_t m = A->size();
    size_t n = B->size();
    size_t p = B->at(0).size();

    // for each row in A
    for (size_t row = 0; row < m; row++) {
        // for each column in B
        for (size_t col = 0; col < p; col++) {
            // ideally dot product works, else use K to go over each element pairwise
            for (size_t k = 0; k < n; k++) {
                (*C)[row][col] += (*A)[row][k] * (*B)[k][col];
            }
        }
    }
}


// Function to perform matrix multiplication C = A * B
// A: N x N matrix
// B: N x N matrix
// C: N x N matrix (result)
void serial_mmul_linear(const double* A, const double* B, double* C, size_t N)
{
    // for each row in A
    for (size_t row = 0; row < N; row++) {
        // for each column in B
        for (size_t col = 0; col < N; col++) {
            // ideally dot product works, else use K to go over each element pairwise
            for (size_t k = 0; k < N; k++) {
                C[row * N + col] += A[row * N + k] * B[k * N + col];
            }
        }
    }
}


// Function to perform matrix multiplication C = A * B
// A: N x N matrix
// B: N x N matrix
// C: N x N matrix (result)
void parallel_mmul_linear(const double* A, const double* B, double* C, size_t N, size_t start_row, size_t end_row)
{
    // for each row in A
    for (size_t row = start_row; row < end_row; row++) {
        // for each column in B
        for (size_t col = 0; col < N; col++) {
            // ideally dot product works, else use K to go over each element pairwise
            for (size_t k = 0; k < N; k++) {
                C[row * N + col] += A[row * N + k] * B[k * N + col];
            }
        }
    }
}


static void benchmark_serial_mmul_matrix(benchmark::State& s) {

    std::size_t m = s.range(0);
    std::size_t n = s.range(1);
    std::size_t p = s.range(2);

    // Create our random number generators
    std::mt19937 rng;
    rng.seed(0);
    std::uniform_real_distribution<double> dist(-10.0,10.0);

    // create matrices A, B, C
    matrix A(m, vector(n));
    matrix B(n, vector(p));
    matrix C(m, vector(p, 0.0)); // Initialize C with zeros

    // Fill matrices A and B with random values
    std::for_each(A.begin(), A.end(), [&](vector& row) { std::generate(row.begin(), row.end(), [&]() { return dist(rng); }); });
    std::for_each(B.begin(), B.end(), [&](vector& row) { std::generate(row.begin(), row.end(), [&]() { return dist(rng); }); });

    // Main benchmark loop
    for (auto _ : s) {
        serial_mmul_matrix(&A, &B, &C);
    }
}


static void benchmark_serial_mmul_linear(benchmark::State& s) {

    std::size_t N = s.range(0); // Size of the matrix

    // Create our random number generators
    std::mt19937 rng;
    rng.seed(0);
    std::uniform_real_distribution<double> dist(-10.0,10.0);

    // create matrices A, B, C
    double* A = new double[N * N];
    double* B = new double[N * N];
    double* C = new double[N * N];

    // Fill matrices A and B with random values
    std::generate(A, A + N * N, [&] () { return dist(rng); });
    std::generate(B, B + N * N, [&] () { return dist(rng); });
    // Initialize matrix C to zero
    std::fill(C, C + N * N, 0.0);

    // Main benchmark loop
    for (auto _ : s) {
        serial_mmul_linear(A, B, C, N);
    }

    // Cleanup
    delete[] A;
    delete[] B;
    delete[] C;
}


static void benchmark_parallel_mmul_linear(benchmark::State& s) {

    std::size_t N = s.range(0); // Size of the matrix

    // Create our random number generators
    std::mt19937 rng;
    rng.seed(0);
    std::uniform_real_distribution<double> dist(-10.0,10.0);

    // create matrices A, B, C
    double* A = new double[N * N];
    double* B = new double[N * N];
    double* C = new double[N * N];

    // Fill matrices A and B with random values
    std::generate(A, A + N * N, [&] () { return dist(rng); });
    std::generate(B, B + N * N, [&] () { return dist(rng); });
    // Initialize matrix C to zero
    std::fill(C, C + N * N, 0.0);

    // Set up for launching threads
    std::size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    std::size_t n_rows = N / num_threads;
    // std::cout << "Number of threads: " << num_threads << std::endl;
    // std::cout << "Rows per thread: " << n_rows << std::endl;

    // Main benchmark loop
    for (auto _ : s) {
         for (size_t i = 0; i < num_threads; i++) {
            size_t start_row = i * n_rows;
            size_t end_row = start_row + n_rows;
            threads.emplace_back([&]() {parallel_mmul_linear(A, B, C, N, start_row, end_row);});
         }

         for (auto& t : threads) t.join();

         threads.clear();
    }

    // Cleanup
    delete[] A;
    delete[] B;
    delete[] C;
}


BENCHMARK(benchmark_serial_mmul_matrix)
    ->Args({256+16, 256+16, 256+16})
    ->Args({512+16, 512+16, 512+16})
    ->Args({1024+16, 1024+16, 1024+16})
    ->Unit(benchmark::kMillisecond);


BENCHMARK(benchmark_serial_mmul_linear)
    ->Arg(256+16)
    ->Arg(512+16)
    ->Arg(1024+16)
    ->Unit(benchmark::kMillisecond);


BENCHMARK(benchmark_parallel_mmul_linear)
    ->Arg(256+16)
    ->Arg(512+16)
    ->Arg(1024+16)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_MAIN();
