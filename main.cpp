#include <iostream>
#include <armadillo>
#include <iomanip>

arma::mat generateMatrix(int n, int m) {
    arma::arma_rng::set_seed_random();
    arma::mat A = arma::randi<arma::mat>(n, m, arma::distr_param(0, 99));
    return A;
}

bool isNashOptimal(const arma::mat &A, const arma::mat &B, int i, int j) {
    double a = A(i, j), b = B(i, j);
    for (int k = 0; k < int(A.n_rows); ++k) {
        if (A(k, j) > a) return false;
    }
    for (int k = 0; k < int(A.n_cols); ++k) {
        if (B(i, k) > b) return false;
    }
    return true;
}

bool isParetoOptimal(const arma::mat &A, const arma::mat &B, int i, int j) {
    double a = A(i, j), b = B(i, j);
    for (int k = 0; k < int(A.n_rows); ++k) {
        for (int l = 0; l < int(A.n_cols); ++l) {
            if (A(k, l) >= a && B(k, l) >= b && (A(k, l) > a || B(k, l) > b))
                return false;
        }
    }
    return true;
}


void printMatrix(const arma::mat &A, const arma::mat &B, bool flag = false) {
    if (!flag) {
        for (int i = 0; i < int(A.n_rows); ++i) {
            for (int j = 0; j < int(A.n_cols); ++j) {
                std::cout << "(" << std::setw(5) << std::fixed << std::setprecision(2) << A(i, j)
                          << ", " << std::setw(5) << std::fixed << std::setprecision(2) << B(i, j) << ")";
                if (j < int(A.n_cols) - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << std::endl;
        }
        return;
    }
    std::string reset = "\033[0m";
    std::string red = "\033[31m"; // Nash
    std::string green = "\033[32m"; // Pareto
    std::string yellow = "\033[33m";// Nash + Pareto

    for (int i = 0; i < int(A.n_rows); ++i) {
        for (int j = 0; j < int(A.n_cols); ++j) {
            std::string color = reset;
            if (isNashOptimal(A, B, i, j)) {
                color = red;
            }
            if (isParetoOptimal(A, B, i, j)) {
                color = green;
            }
            if (isNashOptimal(A, B, i, j) && isParetoOptimal(A, B, i, j)) {
                color = yellow;
            }
            std::cout << color << "(" << std::setw(5) << std::fixed << std::setprecision(2) << A(i, j)
                      << ", " << std::setw(5) << std::fixed << std::setprecision(2) << B(i, j) << ")" << reset;
            if (j < int(A.n_cols) - 1) {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;
    }
}


void findAnswer(const arma::mat &A, const arma::mat &B) {
    std::cout << "Nash optimal:" << std::endl;
    for (int i = 0; i < int(A.n_rows); ++i) {
        for (int j = 0; j < int(A.n_cols); ++j) {
            if (isNashOptimal(A, B, i, j))
                std::cout << "(" << i << ", " << j << ") -> (" << A(i, j) << ", " << B(i, j) << ")" << std::endl;
        }
    }
    std::cout << "Pareto optimal:" << std::endl;
    for (int i = 0; i < int(A.n_rows); ++i) {
        for (int j = 0; j < int(A.n_cols); ++j) {
            if (isParetoOptimal(A, B, i, j))
                std::cout << "(" << i << ", " << j << ") -> (" << A(i, j) << ", " << B(i, j) << ")" << std::endl;
        }
    }
    std::cout << "Intersection of Nash optimal and Pareto optimal situations:" << std::endl;
    for (int i = 0; i < int(A.n_rows); ++i) {
        for (int j = 0; j < int(A.n_cols); ++j) {
            if (isNashOptimal(A, B, i, j) && isParetoOptimal(A, B, i, j))
                std::cout << "(" << i << ", " << j << ") -> (" << A(i, j) << ", " << B(i, j) << ")" << std::endl;
        }
    }
}

void findMixedNash(const arma::mat &A, const arma::mat &B) {
    arma::vec u = {1, 1};
    arma::mat A_inv = arma::inv(A);
    arma::mat B_inv = arma::inv(B);
    arma::vec x = (1.0 / arma::dot(u, A_inv * u)) * B_inv * u;
    arma::vec y = (1.0 / arma::dot(u, B_inv * u)) * A_inv * u;
    double v_1 = 1.0 / arma::dot(u, A_inv * u);
    double v_2 = 1.0 / arma::dot(u, B_inv * u);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Mixed Nash:" << std::endl;
    std::cout << "Player 1 strategy: " << x.t() << std::endl;
    std::cout << "Player 2 strategy: " << y.t() << std::endl;
    std::cout << "Player 1 payoff: " << v_1 << std::endl;
    std::cout << "Player 2 payoff: " << v_2 << std::endl;
}


int main() {
    arma::mat A = generateMatrix(10, 10);
    arma::mat B = generateMatrix(10, 10);
//    arma::mat A = {{54, 84, 38, 86, 51, 75, 26, 4,  95, 21},
//                   {70, 25, 77, 95, 35, 16, 3,  97, 30, 71},
//                   {70, 26, 72, 16, 44, 0,  66, 49, 47, 43},
//                   {1,  70, 59, 45, 15, 80, 65, 87, 3,  14},
//                   {45, 47, 7,  94, 76, 9,  71, 64, 65, 99},
//                   {61, 17, 7,  53, 94, 89, 43, 24, 90, 37},
//                   {35, 91, 28, 84, 25, 90, 34, 79, 83, 87},
//                   {8,  54, 63, 84, 31, 93, 4,  50, 46, 76},
//                   {28, 93, 0,  57, 2,  78, 12, 89, 81, 32},
//                   {44, 16, 8,  7,  22, 76, 72, 48, 96, 32}};
//
//    arma::mat B = {{22, 65, 74, 4,  98, 97, 37, 52, 45, 55},
//                   {66, 74, 2,  92, 4,  75, 6,  46, 70, 53},
//                   {47, 73, 69, 92, 17, 41, 14, 9,  95, 71},
//                   {7,  54, 62, 2,  5,  19, 52, 47, 29, 36},
//                   {44, 5,  45, 34, 67, 0,  80, 85, 88, 73},
//                   {12, 32, 37, 30, 69, 16, 62, 22, 60, 5},
//                   {25, 21, 5,  67, 90, 5,  39, 92, 6,  12},
//                   {9,  52, 15, 50, 26, 18, 39, 10, 46, 70},
//                   {2,  39, 48, 6,  41, 74, 77, 49, 34, 68},
//                   {24, 34, 20, 3,  0,  19, 68, 81, 35, 40}};
    std::cout << "Random bimatrix game (10x10):" << std::endl;
    printMatrix(A, B, true);
    findAnswer(A, B);

    // Family Quarrel
    arma::mat FQ_A = {{4, 0},
                      {0, 1}};
    arma::mat FQ_B = {{1, 0},
                      {0, 4}};
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Family Quarrel game:" << std::endl;
    printMatrix(FQ_A, FQ_B);
    findAnswer(FQ_A, FQ_B);

    // Intersection Game
    arma::mat IG_A = {{0, 0.5},
                      {2, -1}};
    arma::mat IG_B = {{0,   2},
                      {0.7, -2}};
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Intersection game:" << std::endl;
    printMatrix(IG_A, IG_B);
    findAnswer(IG_A, IG_B);

    // Prisoner's Dilemma
    arma::mat PD_A = {{-1, -10},
                      {0,  -5}};
    arma::mat PD_B = {{-1,  0},
                      {-10, -5}};
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Prisoner's Dilemma game:" << std::endl;
    printMatrix(PD_A, PD_B);
    findAnswer(PD_A, PD_B);

    std::cout << "-------------------------------------------------------\n";
    std::cout << "4-th option game:" << std::endl;
    arma::mat VA = {{4, 5},
                    {0, 7}};
    arma::mat VB = {{7, 2},
                    {2, 3}};
    printMatrix(VA, VB);
    findAnswer(VA, VB);
    findMixedNash(VA, VB);
    return 0;
}

