#include <map>
#include <mpi.h>
#include <string>
#include <vector>
#include <iterator>
#include <iostream>
#include <algorithm>
#include <random>
#include <string_view>
#include <cmath>
#include <iomanip> // Para std::setprecision
using namespace std;

float t1,t2,t3,t4,t5,t6,t7,t8;
float t9, t10, t11, t12, t13, t14, t15, t16;
float t_inicial, t_final;

string generateRandomString(size_t length) {
    const string_view characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    random_device rd;
    mt19937 generator(rd());
    uniform_int_distribution<size_t> distribution(0, characters.size() - 1);

    string randomString;
    randomString.reserve(length);

    generate_n(back_inserter(randomString), length, [&]() {
        return characters[distribution(generator)];
    });

    return randomString;
}
string concatenar(const map<int, string>& data_by_rank) {
    string result;
    for (const auto& entry : data_by_rank) {
        result += entry.second;
    }
    return result;
}

vector<int> local_rank(const string& local_A, const string& A) {
    vector<int> rank_counts(A.size(), 0);

    for (size_t i = 0; i < A.size(); i++) {
        rank_counts[i] = lower_bound(local_A.begin(), local_A.end(), A[i]) - local_A.begin();
        }

    return rank_counts;
}
void gossip_step(int rank, int rows, int cols, int size, map<int, string>& local_data) {
    int row = rank / cols;
    int col = rank % cols;
    char recv_buffer[10000];

    for (int step = 0; step < rows - 1; ++step) {
        int send_to = ((row + 1) % rows) * cols + col;      // same col but below
        int receive_from = ((row + rows - 1) % rows) * cols + col; // above same col

        string send_buffer = concatenar(local_data);

        MPI_Request send_request, recv_request;
        MPI_Irecv(recv_buffer, 10000, MPI_CHAR, receive_from, 0, MPI_COMM_WORLD, &recv_request);
        MPI_Isend(send_buffer.c_str(), send_buffer.size() + 1, MPI_CHAR, send_to, 0, MPI_COMM_WORLD, &send_request);

        MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
        MPI_Wait(&send_request, MPI_STATUS_IGNORE);

        string received_string(recv_buffer);
        int source_rank = (rank - cols + size) % size; // calculate the rank of the data sender
        local_data[source_rank] = received_string;
    }
}

void reverse_broadcast_step(int rank, int rows, int cols, const string& starting_data, map<int, string>& resulting_data) {
    int row = rank / cols;
    int col = rank % cols;
    char recv_buffer[10000];

    if (col == row) {
        for (int c = 0; c < cols; ++c) {
            if (c != col) {
                int send_to = row * cols + c;
                MPI_Send(starting_data.c_str(), starting_data.size() + 1, MPI_CHAR, send_to, 0, MPI_COMM_WORLD);
            }
        }
        resulting_data[0] = starting_data;
    } else {
        int send_from = row * cols + row;
        MPI_Recv(recv_buffer, 10000, MPI_CHAR, send_from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        string received_string(recv_buffer);
        resulting_data[0] = received_string;
    }
}
string sort_and_print_by_rank(const vector<int>& aggregated_ranks, const string& result) {
    vector<pair<int, char>> rank_with_indices;

    for (size_t i = 0; i < result.size(); ++i) {
        rank_with_indices.emplace_back(aggregated_ranks[i], result[i]);
    }

    sort(rank_with_indices.begin(), rank_with_indices.end());
    string sorted_result;

    for(const auto& rank : rank_with_indices) {
        sorted_result += rank.second;
    }
    return sorted_result;
}

string calculate_and_print_ranks(int rank, int rows, int cols, const string& starting_data, const string& result) {
    string sorted_starting_data = starting_data;

    //SORT (4)

    t7 = MPI_Wtime();
    sort(sorted_starting_data.begin(), sorted_starting_data.end());
    t8 = MPI_Wtime();

    // LOCAL RANKING (5)

    t9 = MPI_Wtime();
    vector<int> local_ranking = local_rank(sorted_starting_data, result);
    t10 = MPI_Wtime();

    string sorted_result;

    MPI_Barrier(MPI_COMM_WORLD);

    int row = rank / cols;
    int col = rank % cols;
    int diagonal_process = row * cols + row;
    char recv_buffer[10000];
    string recv_word;

    // REDUCE (6)
    // Enviar los datos a las diagonales

    t11 = MPI_Wtime();
    if (col != row) 
    {
        MPI_Send(local_ranking.data(), local_ranking.size(), MPI_INT, diagonal_process, 0, MPI_COMM_WORLD);
        // cout << "Process " << rank << " sent ranks to diagonal process " << diagonal_process << endl; (COMENTADO)
    } 
    else 
    {
        // Los ranks de la diagonal
        vector<int> aggregated_ranks(local_ranking.size(), 0);
        for (int c = 0; c < cols; ++c) 
        {
            // recibir desde la diagonal por cada uno y sumarlo al proceso principal
            if (c != col) 
            {
                vector<int> received_ranks(local_ranking.size());
                MPI_Recv(received_ranks.data(), received_ranks.size(), MPI_INT, row * cols + c, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (size_t i = 0; i < aggregated_ranks.size(); ++i) 
                {
                    aggregated_ranks[i] += received_ranks[i];
                }
            } 
            else 
            {   // si es el diagonal, se le suma igual
                for (size_t i = 0; i < aggregated_ranks.size(); ++i) 
                {
                    aggregated_ranks[i] += local_ranking[i];
                }
            }
        }
        t12 = MPI_Wtime();

        // GATHER (6)
        // Después cada diagonal envía su ranking y string al proceso 0

        t13 = MPI_Wtime();
        if (rank != 0) 
        {
            MPI_Send(aggregated_ranks.data(), aggregated_ranks.size(), MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(result.c_str(), result.size() + 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
            // cout << "Diagonal process " << rank << " sent aggregated ranks to Process 0" << endl; (COMENTADO)
        } 
        else 
        {
            vector<int> global_ranks(aggregated_ranks);

            for (int r = 1; r < rows; ++r) 
            {
                int d_proc = r * cols + r; // proceso diagonal en base al iterador r
                vector<int> received_ranks(aggregated_ranks.size());
                // Recibe la info de cada diagonal
                MPI_Recv(received_ranks.data(), received_ranks.size(), MPI_INT, d_proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_buffer, 10000, MPI_CHAR, d_proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // Tranforma la información del buffer en una string
                string received_string(recv_buffer);
                recv_word += received_string;

                // cout << "Process 0 received ranks from diagonal process " << d_proc << endl; (COMENTADO)
                // Expande global ranks
                global_ranks.insert(global_ranks.end(), received_ranks.begin(), received_ranks.end());
            }

            t14 = MPI_Wtime();

            t15 = MPI_Wtime();
            sorted_result = sort_and_print_by_rank(global_ranks, (result + recv_word));
            t16 = MPI_Wtime();
        }
    }
    return sorted_result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sqrt_size = static_cast<int>(sqrt(size));
    if (sqrt_size * sqrt_size != size) {
        if (rank == 0) cerr << "Error: Number of processes must be a perfect square." << endl;
        MPI_Finalize();
        return 1;
    }

    const int rows = sqrt_size;
    const int cols = sqrt_size;

    if (argc < 2) {
        if (rank == 0) cerr << "Usage: mpiexec -n <num_processes> ./program <message_size>" << endl;
        MPI_Finalize();
        return 1;
    }

    int msg_size = atoi(argv[1])/size;
    if (msg_size <= 0) {
        if (rank == 0) cerr << "Error: Message size must be a positive integer." << endl;
        MPI_Finalize();
        return 1;
    }

    int total_elements = rows * cols;
    string input;

    if (rank == 0) {
        input = generateRandomString(msg_size * total_elements);
        if (input.size() % total_elements != 0) {
            cerr << "Input Size [" << input.size() << "] doesn't match row * col size [" << total_elements << "]" << endl;
            MPI_Finalize();
            return 1;
        }
    }

    char* local_string = new char[msg_size + 1];
    local_string[msg_size] = '\0';

    //SCATTER (1)

    t_inicial = MPI_Wtime();

    t1 = MPI_Wtime();
    MPI_Scatter(input.c_str(), msg_size, MPI_CHAR, local_string, msg_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    t2 = MPI_Wtime();

    string local_data_str(local_string);
    delete[] local_string;

    // cout << "Process " << rank << " received: " << local_data_str << endl;  (COMENTADO)

    map<int, string> local_data = {{rank, local_data_str}};
    map<int, string> resulting_data;

    // GOSSIP (2)

    t3 = MPI_Wtime();
    gossip_step(rank, rows, cols, size, local_data);
    t4 = MPI_Wtime();
    string gossip_result = concatenar(local_data);

    // BROADCAST (3)

    t5 = MPI_Wtime();
    reverse_broadcast_step(rank, rows, cols, gossip_result, resulting_data);
    t6 = MPI_Wtime();

    string result2 = concatenar(resulting_data);

    // SORT, LOCAL, RANKING, REDUCE Y GATHER
    string final_output = calculate_and_print_ranks(rank, rows, cols, gossip_result, result2);

    t_final = MPI_Wtime();

    // if (rank == 0) cout << "Final result: " << final_output << endl;

    if (rank == 0)
    {
        cout << fixed << setprecision(10);

        cout << "Ejecucion: " << ((t_final - t_inicial) - (t16 - t15)) << endl;
        cout << "Computo: " << ((t8 - t7) + (t10 - t9)) << endl;
        cout << "Comunicacion: " << ((t_final - t_inicial) - (t16 - t15) - ((t8 - t7) + (t10 - t9))) << endl;
    }

    MPI_Finalize();
    return 0;
}
