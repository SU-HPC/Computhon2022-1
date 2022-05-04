#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <omp.h>
using namespace std;

/*
 * Parameters:
 *  [In]  nnz: unsigned int - number of non-zeros
 *  [In]  order: unsigned int - number of dimensions 
 *  [In]  dimensions: unsigned int* - an array of size `order` containing the dimension on each mode
 *  [In]  coord: unsigned int** - 2D structure of dimensions `order` x `nnz` with each row representing a non-zero
 *         ex: coord[i][j] = the i'th dimension value of the j'th non-zero in the tensor
 * 
 * Returns:
 *  unsigned long long - number of non-zero fibers in the tensor
*/
unsigned long long naive_count_coo(unsigned int nnz, unsigned int order, unsigned int* dimensions, unsigned int ** coord);

/*
 * Read a .tns file into a COO object
 * 
 * Parameters: 
 *  [In]  file_name: string
 *  [Out] nnz: unsigned int - number of non-zeros
 *  [Out] order: unsigned int& - number of dimensions (modes)
 *  [Out] dimensions: unsigned int*& - an array of size "order" containing the dimension on each order (mode)
 *  [Out] coord : unsigned int **& - 2D structure of dimensions order x num_non_zeros with each row representing a non-zero
 *                                   ex: coord[i][j] = the i'th mode value of the j'th non-zero in the tensor
 * 
 * Returns:
 * double - time in seconds taken to read .tns file and construct COO
*/
double read_tns_to_coo(string file_name, unsigned int& nnz, unsigned int& order, unsigned int*& dimensions, unsigned int **& coord);
/*
 * Get the upper limit on the total number of fibers that can exist in a tensor
 * 
 * Parameters:
 *  [In]  order: unsigned int - number of dimensions (modes)
 *  [In]  dimensions: unsigned int* - an array of size "order" containing the dimension on each order (mode)
 * 
 * Returns:
 *  unsigned long long - upper limit on the number of fibers
*/
unsigned long long fibers_upper_limit(unsigned int order, unsigned int* dimensions);

int main(int argc, char * argv[]){
    if (argc < 2){
        cout << "Usage:\n ./tensor <tensor_file.tns>\n";
        return 1;
    }
    string file_name = argv[1];
    unsigned int ** coord, *dimensions, order, nnz;
    cout << "General statistics: " << endl;
    cout << "-------------------" << endl;
    double time_to_read = read_tns_to_coo(file_name, nnz, order, dimensions, coord); 
    cout << "              Order = " << order << endl;
    cout << "         Dimensions = " << dimensions[0];
    for (int i = 1; i < order ; i++){
        cout << " x " << dimensions[i];
    }
    cout << endl;
    cout << "Number of non-zeros = " << nnz << endl;
    cout << "  Fiber upper limit = " << fibers_upper_limit(order, dimensions) << endl;
    cout << "-------------------" << endl;
    cout << endl << endl;
    
    unsigned long long number_of_fibers;

    double start = omp_get_wtime();

    //////////// YOUR CODE GOES HERE //////////// 
    
    number_of_fibers = naive_count_coo(nnz, order, dimensions, coord);
    
    /////////////////////////////////////////////

    double end = omp_get_wtime();

    cout << "Results: " << endl;
    cout << "-------------------" << endl;
    cout << "Number of fibers = " << number_of_fibers << endl;
    cout << "            Time = " << end - start << " s"  << endl;
    cout << "-------------------" << endl;
    return 0;
}


unsigned long long fibers_upper_limit(unsigned int order, unsigned int* dimensions){
    unsigned long long total_possible_fibers = 0;
    for (int i =0; i < order; i++) {
        unsigned long long mode_fibers = 1;
        for (int j = 0; j < order; j++){
            if (j == i) continue;
            else mode_fibers*=dimensions[j];
        }
        total_possible_fibers+=mode_fibers;
    }
    return total_possible_fibers;
}
/*
Functor for hashing the values of a vector.

* Used in `naive_count_coo`

*/
template <typename T>
struct vector_hash{
    size_t operator()(const vector<T>& vec) const {
        size_t hash = 0;
        for (int i =0; i< vec.size(); i++){
            hash+=(i+1)*vec[i];
        }
        return hash;
    }

};

/*
Functor for checking equality of a pair of vectors.

* Used in `naive_count_coo`
* Precondition: `lhs` and `rhs` have the same dimension 

*/
template <typename T>
struct vector_equal{
    bool operator()(const vector<T>& lhs, const vector<T>&rhs) const{
        for (int i =0; i< lhs.size(); i++){
            if (lhs[i] != rhs[i]) return false;
        }
        return true;
    }
};

unsigned long long naive_count_coo(unsigned int nnz, unsigned int order, unsigned int* dimensions, unsigned int ** coord){
    unsigned long long total_fibers = 0;
    // Go over each mode (find fibers on that mode)
    for (int m = 0; m < order; m++){
        // modes other than `m` (dimensions that will be constant for fibers on mode `m`)
        vector<int> modes_to_check;
        for (int x = 0; x < order; x++) {
            if (x == m) continue;
            modes_to_check.push_back(x);
        }
        // Set containing all the fibers on mode `m` based on the constant indices of each fiber
        unordered_set<vector<unsigned int>, vector_hash<unsigned int>, vector_equal<unsigned int>> fibers;
        vector<unsigned int> one_nnz(order-1);
        // Determine the fiber that each non-zero belongs to and add it to the set
        for (unsigned int i = 0; i < nnz; i++){
            // Construct the fiber ID using its dimensions on all modes except `m`
            for (int o = 0; o < order-1; o++){
                one_nnz[o] = coord[modes_to_check[o]][i];
            }
            fibers.insert(one_nnz);
        }
        // Add the fiber count in mode `m` to the total fiber count
        total_fibers+=fibers.size();
    }
    return total_fibers;
}


double read_tns_to_coo(string file_name, unsigned int& nnz, unsigned int& order, unsigned int*& dimensions, unsigned int **& coord){
    ifstream fin(file_name);
    string line;
    
    // Find order 
    getline(fin, line);
    stringstream ss(line);
    int num;
    order = 0;
    while(ss>>num) order++;
    order--;

    // Read non-zeros
    dimensions = new unsigned int[order]();
    fin.clear();
    fin.seekg(0);
    vector<unsigned int> non_zero_coordinate(order);
    vector<vector<unsigned int>> all_coords;
    double start = omp_get_wtime();
    while(getline(fin, line)){
        stringstream ss(line);
        for (int i =0; i < order; i++){
            ss >> non_zero_coordinate[i];
            // fix 1-indexing
            non_zero_coordinate[i]--;
            // find the maximum dimension of mode i
            dimensions[i] = max(dimensions[i], non_zero_coordinate[i]+1);
        }
        all_coords.push_back(non_zero_coordinate);
    }

    // Construct the COO
    coord = new unsigned int*[order];
    for (int i =0; i < order; i++){
        coord[i] = new unsigned int[all_coords.size()];
    }
    for (unsigned int i = 0; i < all_coords.size(); i++){
        for (int o = 0; o < order; o++){
            coord[o][i] = all_coords[i][o];
        }
    }
    nnz = all_coords.size();
    double end = omp_get_wtime();
    return end - start;
}