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
using std::sqrtf;
using std::fabsf;
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
void init_vectors(void){

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
void init_centers(void){

    int indices[Nc];
    int i = 0;
    int flag = 0;
    indices[0] = rand() % N;  // randomly pick first vector

    while (i < Nc){
        indices[i] = rand() % N;  // randomly pick one vector
        for (int j=0; j<i; j++){  // check similarity with all other centres
            if (indices[i] == indices[j]){  // if not unique, stop checking
                flag = 1;
                break;
            }
        }
        if(!flag){  // if unique, continue to next index
            i++;
        }
        flag = 0; // else, try again
    }

    // populate Center array
    for (i=0; i<Nc; i++){
        // copy corresponding vector from Vec array
        memcpy(Center[i], Vec[indices[i]], sizeof(float) * Nv);
    }

}

// (Re)assignment of a cluster to each vector
void assign_clusters(void){

    // max distance: (255.0 - (-255.0))^2 foreach of the Nv dimensions
    float max_dist = Nv * 510.0 * 510.0;
    float min_dist, diff, temp_dist;
    int center_idx;
    curr_dist_sum = 0.0f;

    for (int i=0; i<N; i++){  // foreach vector
        min_dist = max_dist;
        for (int j=0; j<Nc; j++){  // foreach centre
            temp_dist = 0.0f;
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
        curr_dist_sum += sqrtf(min_dist);  // OPTIMISATION!
        Classes[i] = center_idx;  // assign centre to vector
    }

}

// Calculation of new cluster centres after (re)assignment
void update_centers(void){

    memset(Observations, 0, Nc * sizeof(int));  // number of vectors in each cluster
    memset(Center, 0, Nc * Nv * sizeof(float));  // set all centres to origin
    for (int i=0; i<N; i++){  // foreach vector
        for (int j=0; j<Nv; j++){  // foreach dimension
            // add coordinate to corresponding dimension of cluster centre
            Center[Classes[i]][j] += Vec[i][j];
        }
        Observations[Classes[i]] +=1;
    }

    float inverse_obs;
    for (int i=0; i<Nc; i++){  // foreach centre
        // divide by number of observations in cluster to calculate the mean
        inverse_obs = 1.0f / (float)Observations[i];
        for (int j=0; j<Nv; j++){  // foreach dimension
            Center[i][j] *= inverse_obs;
        }
    }

}

// main function
int main (void){

    cout<<"N = "<<N<<" | Nv = "<<Nv<<" | Nc = "<<Nc<<endl;

    init_vectors();
    init_centers();
    assign_clusters();
    update_centers();

    prev_dist_sum = 0.0f;
    int count = 1;
    while (fabsf(curr_dist_sum - prev_dist_sum)/prev_dist_sum >= THR_KMEANS){
        prev_dist_sum = curr_dist_sum;
        assign_clusters();
        update_centers();
        count++;
    }
    
    cout<<"Total iterations: "<<count<<endl;

    return 0;

}