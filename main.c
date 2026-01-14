/**
 *  main.c
 *
 *  Arxiu reutilitzat de l'assignatura de Computació d'Altes Prestacions de
 *  l'Escola d'Enginyeria de la Universitat Autònoma de Barcelona Created on: 31
 *  gen. 2019 Last modified: fall 24 (curs 24-25) Author: ecesar, asikora
 *  Modified: Blanca Llauradó, Christian Germer
 *
 *  Descripció:
 *  Funció que entrena la xarxa neuronal definida + Funció que fa el test del
 *  model entrenat + programa principal.
 *
 */

#include "main.h"

#include <mpi.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


int rank;
int num_procs;

//-----------FREE INPUT------------
void freeInput(int np, char** input) {
    for (int i = 0; i < np; i++)
        free(input[i]);
    free(input);
}

//-----------PRINTRECOGNIZED------------
void printRecognized(int p, layer Output) {
    int imax = 0;

    for (int i = 1; i < num_out_layer; i++)
        if (Output.actv[i] > Output.actv[imax])
            imax = i;

    if (imax == Validation[p])
        total++;

    if (debug == 1) {
        printf("El patró %d sembla un %c\t i és un %d", p, '0' + imax,
            Validation[p]);
        for (int k = 0; k < num_out_layer; k++)
            printf("\t%f\t", Output.actv[k]);
        printf("\n");
    }
}

/**
 * @brief Entrena la xarxa neuronal en base al conjunt d'entrenament
 *
 * @details Primer carrega tots els patrons d'entrenament (loadPatternSet)
 *          Després realitza num_epochs iteracions d'entrenament.
 *          Cada epoch fa:
 *              - Determina aleatòriament l'ordre en que es consideraran els
 * patrons (per evitar overfitting)
 *              - Per cada patró d'entrenament fa el forward_prop (reconeixament
 * del patró pel model actual) i el back_prop i update_weights (ajustament de
 * pesos i biaxos per provar de millorar la precisió del model)
 *
 * @see loadPatternSet, feed_input, forward_prop, back_prop, update_weights,
 * freeInput
 *
 */
void train_neural_net() {
    if (rank == 0) printf("\nTraining...\n");

    if ((input = loadPatternSet(num_training_patterns, dataset_training_path,
                                1)) == NULL) {
        printf("Loading Patterns: Error!!\n");
        exit(-1);
    }

    int ranpat[num_training_patterns];

    int base  = num_training_patterns / num_procs;
    int extra = num_training_patterns % num_procs;

    int from, to;

    if (rank < extra) {
        from = rank * (base + 1);
        to   = from + base + 1;
    } else {
        from = rank * base + extra;
        to   = from + base;
    }
    
    // Gradient Descent
    for (int it = 0; it < num_epochs; it++) {
        // Train patterns randomly

        double t_epoch_start = MPI_Wtime();

        if (rank == 0) {
            for (int p = 0; p < num_training_patterns; p++)
                ranpat[p] = p;

            for (int p = 0; p < num_training_patterns; p++) {
                int x = rando();
                int np = (x * x) % num_training_patterns;
                int tmp = ranpat[p];
                ranpat[p] = ranpat[np];
                ranpat[np] = tmp;
            }
        }

        MPI_Bcast(ranpat, num_training_patterns, MPI_INT, 0, MPI_COMM_WORLD);

        for (int i = from; i < to; i++) {
            int p = ranpat[i];
            feed_input(p);
            forward_prop();
            back_prop(p);
            update_weights();
        }

        for (int l = 0; l < num_layers - 1; l++) {

            int w_size = lay[l].num_neu * lay[l + 1].num_neu;

            MPI_Allreduce(MPI_IN_PLACE,
                          lay[l].out_weights,
                          w_size,
                          MPI_FLOAT,
                          MPI_SUM,
                          MPI_COMM_WORLD);

            for (int k = 0; k < w_size; k++)
                lay[l].out_weights[k] /= num_procs;

            MPI_Allreduce(MPI_IN_PLACE,
                          lay[l].bias,
                          lay[l].num_neu,
                          MPI_FLOAT,
                          MPI_SUM,
                          MPI_COMM_WORLD);

            for (int k = 0; k < lay[l].num_neu; k++)
                lay[l].bias[k] /= num_procs;
        }

        double t_epoch_end = MPI_Wtime();

        if (rank == 0) {
            printf("Epoch %d time: %f seconds\n",
                it, t_epoch_end - t_epoch_start);
        }
    }

    freeInput(num_training_patterns, input);
}

//-----------TEST THE TRAINED NETWORK------------
void test_nn() {
    char** rSet;
    double test_time_start = MPI_Wtime();

    if (rank == 0)
        printf("\nTesting...\n");

    if ((rSet = loadPatternSet(num_test_patterns, dataset_test_path, 0)) == NULL) {
        printf("Error loading test patterns!!\n");
        exit(-1);
    }

    int base  = num_test_patterns / num_procs;
    int extra = num_test_patterns % num_procs;
    int from, to;

    if (rank < extra) {
        from = rank * (base + 1);
        to   = from + base + 1;
    } else {
        from = rank * base + extra;
        to   = from + base;
    }

    total = 0;

    for (int i = from; i < to; i++) {
        for (int j = 0; j < num_neurons[0]; j++)
            lay[0].actv[j] = rSet[i][j];

        forward_prop();
        printRecognized(i, lay[num_layers - 1]);
    }

    int global_total = 0;
    MPI_Reduce(&total,
               &global_total,
               1,
               MPI_INT,
               MPI_SUM,
               0,
               MPI_COMM_WORLD);

    double test_time_end = MPI_Wtime();

    if (rank == 0) {
        printf("Test time: %f seconds\n",
               test_time_end - test_time_start);
        printf("\nTotal encerts = %d\n", global_total);
    }

    freeInput(num_test_patterns, rSet);
}


//-----------MAIN-----------//
int main(int argc, char** argv) {
    if (debug == 1)
        printf("argc = %d \n", argc);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc <= 1)
        readConfiguration("configuration/configfile.txt");
    else
        readConfiguration(argv[1]);

    if (debug == 1)
        printf("FINISH CONFIG \n");

    // Initialize the neural network module
    if (init() != SUCCESS_INIT) {
        printf("Error in Initialization...\n");
        exit(0);
    }

    if (debug == 1)
        printf("COST MALLOC \n");

    cost = (float*)malloc(num_neurons[num_layers - 1] * sizeof(float));

    // Start measuring time
    struct timeval begin, end;
    gettimeofday(&begin, 0);

    // Train
    train_neural_net();

    // Test
    test_nn();

    // Stop measuring time and calculate the elapsed time
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds * 1e-6;

    if (dinit() != SUCCESS_DINIT)
        printf("Error in Dinitialization...\n");

    free(cost);

    if (rank == 0) printf("\n\nGoodbye! (%f sec)\n\n", elapsed);

    MPI_Finalize();
    
    return 0;
}