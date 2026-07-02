// ============================================================================
// mpi_parallel.cpp (Full Code with Named Excel Exporter)
// ============================================================================
#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include "data_gen.hpp"

template <typename Func>
double time_it(int rank, const std::string& label, Func f) {
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    f();
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double ms = (t1 - t0) * 1000.0;
    if (rank == 0) {
        std::cout << std::left << std::setw(28) << label
                  << std::right << std::setw(12) << std::fixed << std::setprecision(3)
                  << ms << " ms\n";
    }
    return ms;
}

static void compute_counts_displs(uint64_t N, int size, std::vector<int>& counts, std::vector<int>& displs) {
    counts.assign(size, 0);
    displs.assign(size, 0);
    uint64_t base = N / size;
    uint64_t rem = N % size;
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        uint64_t c = base + (static_cast<uint64_t>(r) < rem ? 1 : 0);
        counts[r] = static_cast<int>(c);
        displs[r] = offset;
        offset += counts[r];
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uint64_t N = 10'000'000ULL;
    int bins = 20;
    int window = 100;
    if (argc >= 2) N = std::stoull(argv[1]);
    if (argc >= 3) bins = std::stoi(argv[2]);
    if (argc >= 4) window = std::stoi(argv[3]);

    std::string runSize = "Custom";
    if (N == 1'000'000ULL) runSize = "Small";
    else if (N == 10'000'000ULL) runSize = "Medium";
    else if (N == 100'000'000ULL) runSize = "Large";

    if (rank == 0) {
        std::cout << "=== MPI PARALLEL RUN (" << runSize << ") ===\n";
        std::cout << "Nodes = " << size << ", N = " << N
                  << ", bins = " << bins << ", window = " << window << "\n\n";
    }

    std::vector<int> counts, displs;
    compute_counts_displs(N, size, counts, displs);
    int localN = counts[rank];

    std::vector<double> fullA, fullB;
    std::vector<double> localA(localN), localB(localN);

    double t_gen = time_it(rank, "Data gen + Scatter", [&]() {
        if (rank == 0) {
            generate_dataset(fullA, fullB, N);
        }
        MPI_Scatterv(rank == 0 ? fullA.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, localA.data(), localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(rank == 0 ? fullB.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, localB.data(), localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    });

    double globalMean = 0.0, globalStd = 0.0, globalMin = 0.0, globalMax = 0.0;
    double t1 = time_it(rank, "1. Basic Statistics", [&]() {
        double localSum = 0.0, localSumSq = 0.0;
        double localMin = localA.empty() ? 0.0 : localA[0];
        double localMax = localA.empty() ? 0.0 : localA[0];
        for (double v : localA) {
            localSum += v; localSumSq += v * v;
            if (v < localMin) localMin = v;
            if (v > localMax) localMax = v;
        }
        double globalSum = 0.0, globalSumSq = 0.0;
        MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localSumSq, &globalSumSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            globalMean = globalSum / static_cast<double>(N);
            double variance = (globalSumSq / static_cast<double>(N)) - (globalMean * globalMean);
            globalStd = std::sqrt(variance);
        }
        MPI_Bcast(&globalMean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&globalStd, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    });

    std::vector<uint64_t> globalHist(bins, 0);
    double t2 = time_it(rank, "2. Histogram", [&]() {
        std::vector<uint64_t> localHist(bins, 0);
        double binWidth = (DATA_MAX - DATA_MIN) / bins;
        for (double v : localA) {
            int idx = static_cast<int>((v - DATA_MIN) / binWidth);
            if (idx < 0) idx = 0;
            if (idx >= bins) idx = bins - 1;
            localHist[idx]++;
        }
        MPI_Reduce(localHist.data(), globalHist.data(), bins, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    });

    std::vector<double> sortedLocal;
    double t3 = time_it(rank, "3. Sorting (Sample Sort)", [&]() {
        std::vector<double> local = localA;
        std::sort(local.begin(), local.end());
        std::vector<double> localSamples;
        if (!local.empty()) {
            for (int i = 1; i < size; ++i) {
                size_t idx = (local.size() * i) / size;
                if (idx >= local.size()) idx = local.size() - 1;
                localSamples.push_back(local[idx]);
            }
        } else { localSamples.assign(size - 1, 0.0); }

        std::vector<double> allSamples;
        if (rank == 0) allSamples.resize(localSamples.size() * size);
        MPI_Gather(localSamples.data(), static_cast<int>(localSamples.size()), MPI_DOUBLE, rank == 0 ? allSamples.data() : nullptr, static_cast<int>(localSamples.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        std::vector<double> splitters(size - 1);
        if (rank == 0) {
            std::sort(allSamples.begin(), allSamples.end());
            for (int i = 0; i < size - 1; ++i) {
                size_t idx = (allSamples.size() * (i + 1)) / size;
                if (idx >= allSamples.size()) idx = allSamples.size() - 1;
                splitters[i] = allSamples[idx];
            }
        }
        MPI_Bcast(splitters.data(), size - 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        std::vector<int> sendCounts(size, 0);
        std::vector<std::vector<double>> buckets(size);
        for (double v : local) {
            int b = static_cast<int>(std::upper_bound(splitters.begin(), splitters.end(), v) - splitters.begin());
            buckets[b].push_back(v);
        }
        std::vector<double> sendBuf;
        std::vector<int> sendDispls(size, 0);
        int off = 0;
        for (int b = 0; b < size; ++b) {
            sendCounts[b] = static_cast<int>(buckets[b].size());
            sendDispls[b] = off;
            sendBuf.insert(sendBuf.end(), buckets[b].begin(), buckets[b].end());
            off += sendCounts[b];
        }

        std::vector<int> recvCounts(size, 0);
        MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        std::vector<int> recvDispls(size, 0);
        int recvTotal = 0;
        for (int r = 0; r < size; ++r) { recvDispls[r] = recvTotal; recvTotal += recvCounts[r]; }
        std::vector<double> recvBuf(recvTotal);
        MPI_Alltoallv(sendBuf.data(), sendCounts.data(), sendDispls.data(), MPI_DOUBLE, recvBuf.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        std::sort(recvBuf.begin(), recvBuf.end());
        sortedLocal = std::move(recvBuf);
        int localSize = static_cast<int>(sortedLocal.size());
        std::vector<int> gatherCounts(size);
        MPI_Gather(&localSize, 1, MPI_INT, gatherCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    });

    double correlation = 0.0;
    double t4 = time_it(rank, "4. Pearson Correlation", [&]() {
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
        for (size_t i = 0; i < localA.size(); ++i) {
            sumX += localA[i]; sumY += localB[i]; sumXY += localA[i] * localB[i]; sumX2 += localA[i] * localA[i]; sumY2 += localB[i] * localB[i];
        }
        double local5[5] = {sumX, sumY, sumXY, sumX2, sumY2};
        double global5[5] = {0};
        MPI_Reduce(local5, global5, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            double n = static_cast<double>(N);
            correlation = (n * global5[2] - global5[0] * global5[1]) / std::sqrt((n * global5[3] - global5[0] * global5[0]) * (n * global5[4] - global5[1] * global5[1]));
        }
    });

    std::vector<double> localMavg;
    double t5 = time_it(rank, "5. Moving Average", [&]() {
        int haloSize = std::min(window - 1, localN);
        std::vector<double> haloRecv(haloSize, 0.0);
        int prev = rank - 1; int next = rank + 1;
        std::vector<double> haloSend(localA.end() - haloSize, localA.end());
        MPI_Request reqs[2]; int reqCount = 0;
        if (next < size) MPI_Isend(haloSend.data(), haloSize, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &reqs[reqCount++]);
        if (prev >= 0) MPI_Irecv(haloRecv.data(), haloSize, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &reqs[reqCount++]);
        if (reqCount > 0) MPI_Waitall(reqCount, reqs, MPI_STATUSES_IGNORE);

        std::vector<double> extended;
        if (prev >= 0) extended.insert(extended.end(), haloRecv.begin(), haloRecv.end());
        extended.insert(extended.end(), localA.begin(), localA.end());

        int haloOffset = (prev >= 0) ? haloSize : 0;
        localMavg.resize(localN); double runningSum = 0.0;
        for (int i = 0; i < haloOffset; ++i) runningSum += extended[i];
        for (int i = 0; i < localN; ++i) {
            int gIdx = haloOffset + i; runningSum += extended[gIdx];
            int windowCount = std::min(window, gIdx + 1);
            if (gIdx - window >= 0) runningSum -= extended[gIdx - window];
            localMavg[i] = runningSum / windowCount;
        }
    });

    uint64_t globalOutliers = 0;
    double t6 = time_it(rank, "6. Outlier Detection", [&]() {
        uint64_t localCount = 0;
        for (double v : localA) {
            double z = (v - globalMean) / globalStd;
            if (std::fabs(z) > 3.0) localCount++;
        }
        MPI_Reduce(&localCount, &globalOutliers, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    });

    if (rank == 0) {
        double totalTime = t1 + t2 + t3 + t4 + t5 + t6;
        std::cout << "\nTotal compute time (excludes data gen): " << totalTime << " ms\n";

        // Set clear names matching both the size and execution type
        std::string resultFilename = "custom_mpi_result.csv";
        if (runSize == "Small") resultFilename = "small_mpi_result.csv";
        else if (runSize == "Medium") resultFilename = "medium_mpi_result.csv";
        else if (runSize == "Large") resultFilename = "large_mpi_result.csv";

        // Save row to the specific MPI Excel file
        std::ifstream checkFile(resultFilename);
        bool exists = checkFile.good();
        checkFile.close();

        std::ofstream csvFile(resultFilename, std::ios::app);
        if (!exists) {
            csvFile << "Execution Type,Nodes,N,Bins,Window,Data Gen + Scatter (ms),Task 1 (ms),Task 2 (ms),Task 3 (ms),Task 4 (ms),Task 5 (ms),Task 6 (ms),Total Compute (ms)\n";
        }
        csvFile << "MPI Parallel," << size << "," << N << "," << bins << "," << window << ","
                << t_gen << "," << t1 << "," << t2 << "," << t3 << ","
                << t4 << "," << t5 << "," << t6 << "," << totalTime << "\n";
        csvFile.close();
        std::cout << "--> Results saved to '" << resultFilename << "'\n";
    }

    MPI_Finalize();
    return 0;
}