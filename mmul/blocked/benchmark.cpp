#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

typedef std::vector<double> vector;
typedef std::vector<std::vector<double>> matrix;


// Function to perform matrix multiplication C = A * B
// A: N x N matrix
// B: N x N matrix
// C: N x N matrix (result)
void serial_mmul_blocked(const double* A, const double* B, double* C, size_t N)
{
    // for each row in A
    for (size_t row = 0; row < N; row++) {
        // for each column in B
        for (size_t col = 0; col < N; col++) {
            //
            for (size_t block = 0; block < N; block+=64) {

                for (size_t chunk = 0; chunk < N; chunk+=64) {

                    for (size_t sub_chunk = 0; sub_chunk < 64; sub_chunk++) {

                        for (size_t k = 0; k < 64; k++) {
                            C[row * N + block + k] += 
                            A[row * N + chunk + sub_chunk] * 
                            B[chunk * N + sub_chunk * N + block + k];
                        }
                    }
                }
            }
        }
    }
}


// Function to perform matrix multiplication C = A * B
// A: N x N matrix
// B: N x N matrix
// C: N x N matrix (result)
void parallel_mmul_blocked(const double* A, const double* B, double* C, size_t N, size_t start_row, size_t end_row)
{
    // for each row in A
    for (size_t row = start_row; row < end_row; row++) {
        // for each column in B
        for (size_t col = 0; col < N; col++) {
            //
            for (size_t block = 0; block < N; block+=16) {

                for (size_t chunk = 0; chunk < N; chunk+=16) {

                    for (size_t sub_chunk = 0; sub_chunk < 16; sub_chunk++) {

                        for (size_t k = 0; k < 16; k++) {
                            C[row * N + block + k] += 
                            A[row * N + chunk + sub_chunk] * 
                            B[chunk * N + sub_chunk * N + block + k];
                        }
                    }
                }
            }
        }
    }
}


static void benchmark_serial_mmul_blocked(benchmark::State& s) {

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
        serial_mmul_blocked(A, B, C, N);
    }

    // Cleanup
    delete[] A;
    delete[] B;
    delete[] C;
}


static void benchmark_serial_mmul_blocked_aligned(benchmark::State& s) {

    std::size_t N = s.range(0); // Size of the matrix

    // Create our random number generators
    std::mt19937 rng;
    rng.seed(0);
    std::uniform_real_distribution<double> dist(-10.0,10.0);

    // create matrices A, B, C
    double *A = static_cast<double *>(aligned_alloc(64, N * N * sizeof(double)));
    double *B = static_cast<double *>(aligned_alloc(64, N * N * sizeof(double)));
    double *C = static_cast<double *>(aligned_alloc(64, N * N * sizeof(double)));

    // Fill matrices A and B with random values
    std::generate(A, A + N * N, [&] () { return dist(rng); });
    std::generate(B, B + N * N, [&] () { return dist(rng); });
    // Initialize matrix C to zero
    std::fill(C, C + N * N, 0.0);

    // Main benchmark loop
    for (auto _ : s) {
        serial_mmul_blocked(A, B, C, N);
    }

    // Cleanup
    delete[] A;
    delete[] B;
    delete[] C;
}


static void benchmark_parallel_mmul_blocked(benchmark::State& s) {

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
            threads.emplace_back([&]() {parallel_mmul_blocked(A, B, C, N, start_row, end_row);});
         }

         for (auto& t : threads) t.join();

         threads.clear();
    }

    // Cleanup
    delete[] A;
    delete[] B;
    delete[] C;
}


// BENCHMARK(benchmark_serial_mmul_blocked)
//     ->Arg(256+16)
//     ->Arg(512+16)
//     ->Arg(1024+16)
//     ->Unit(benchmark::kMillisecond);

BENCHMARK(benchmark_serial_mmul_blocked_aligned)
    ->Arg(256+16)
    ->Arg(512+16)
    ->Arg(1024+16)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(benchmark_parallel_mmul_blocked)
    ->Arg(256+16)
    ->Arg(512+16)
    ->Arg(1024+16)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_MAIN();
