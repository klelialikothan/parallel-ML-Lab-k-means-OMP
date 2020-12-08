#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cmath>
#include <iostream>

#define N 100000
#define Nv 1000
#define Nc 100
#define THR_KMEANS 0.000001f

using std::memset;
using std::memcpy;
using std::srand;
using std::rand;
using std::sqrt;
using std::fabs;
using std::cout;
using std::endl;

// Define vector and center arrays as global
float Vec[N][Nv];
float Center[Nc][Nv];
int Classes[N];
int Observations[Nc];
float prev_dist_sum;
float curr_dist_sum;

// Vector initialisation
void init_vectors(void){  // TODO: parallel?

    srand(time(0));  // use current time as seed for random generator
    int upper_bound = 510001;
    float inverse_div = 0.001f;
    int temp;

    for (int i=0; i<N; i++){
        for (int j=0; j<Nv; j++){
            // create float in range [0, 510000]
            temp = rand() % upper_bound;
            // new range -> approximately [0.0, 510.0]
            Vec[i][j] = (float)temp * inverse_div;
            // final range -> approximately [-255.0, 255.0]
            Vec[i][j] -= 255.0f;
            // (approximate ranges as 0.001 cannot be exactly represented as a float)
        }
    }

}

// Pick initial centres
void init_centers(void){  // TODO: parallel?

    int indices[Nc];
    int i = 0;
    bool flag = false;
    indices[0] = rand() % N;  // randomly pick first vector

    while (i < Nc){
        indices[i] = rand() % N;  // randomly pick one vector
        for (int j=0; j<i; j++){  // check similarity with all other centres
            if (indices[i] == indices[j]){  // if not unique, stop checking
                flag = true;
                break;
            }
        }
        if(!flag){  // if unique, continue to next index
            i++;
        }
        flag = false; // else, try again
    }

    // populate Center array
    #pragma omp parallel for private(i)
    for (i=0; i<Nc; i++){
        // copy corresponding vector from Vec array
        memcpy(Center[i], Vec[indices[i]], sizeof(float) * Nv);
    }

}

// (Re)assignment of a cluster to each vector
void assign_clusters(void){

    // max distance: (255.0 - (-255.0))^2 foreach of the Nv dimensions
    float max_dist = Nv * 510.0 * 510.0;
    curr_dist_sum = 0.0f;

    #pragma omp parallel for
    for (int i=0; i<N; i++){  // foreach vector
        float min_dist = max_dist;
        int center_idx;
        // #pragma omp parallel for shared(min_dist, diff, temp_dist)
        for (int j=0; j<Nc; j++){  // foreach centre
            float diff;
            float temp_dist = 0.0f;
            // calculate euclidean distance
            for (int k=0; k<Nv; k++){
                diff = Center[j][k] - Vec[i][k];
                temp_dist += diff * diff;  // pow is slower
            }
            if (temp_dist < min_dist){  // better fit found
                min_dist = temp_dist;
                center_idx = j;
            }
        }
        min_dist = sqrt(min_dist);  // saves time and enables use of atomic below
        #pragma omp atomic
        curr_dist_sum += min_dist;
        Classes[i] = center_idx;  // assign centre to vector
    }

}

// Calculation of new cluster centres after (re)assignment
void update_centers(void){

    memset(Observations, 0, Nc * sizeof(int));  // number of vectors in each cluster
    memset(Center, 0, Nc * Nv * sizeof(float));  // set all centres to origin
    #pragma omp parallel for
    for (int i=0; i<N; i++){  // foreach vector
        for (int j=0; j<Nv; j++){  // foreach dimension
            // add coordinate to corresponding dimension of cluster centre
            #pragma omp atomic
            Center[Classes[i]][j] += Vec[i][j];
        }
        #pragma omp atomic
        Observations[Classes[i]] += 1;
    }

    float inverse_obs;
    #pragma omp parallel for private(inverse_obs)
    for (int i=0; i<Nc; i++){  // foreach centre
        // divide by number of observations in cluster to calculate the mean
        inverse_obs = 1.0f / (float)Observations[i];
        for (int j=0; j<Nv; j++){  // foreach dimension
            Center[i][j] *= inverse_obs;
        }
    }

}

// main function
int main (){

    cout<<"N = "<<N<<" | Nv = "<<Nv<<" | Nc = "<<Nc<<endl;

    init_vectors();
    init_centers();
    assign_clusters();
    update_centers();

    prev_dist_sum = 0.0f;
    int count = 1;
    float term = curr_dist_sum;
    while(term > THR_KMEANS){
        cout<<"term: "<<term<<" ";
        prev_dist_sum = curr_dist_sum;
        assign_clusters();
        update_centers();
        cout<<"Iteration #"<<count<<" distance sum: "<<curr_dist_sum<<endl;
        term = fabs(curr_dist_sum - prev_dist_sum) / prev_dist_sum;
        count++;
    }

    cout<<"Convergence!"<<endl;

    return 0;

}