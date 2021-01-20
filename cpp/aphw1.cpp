#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <math.h>
#include <stdlib.h>

/*
getData: extrxct students features from file
Input: filename-> file path , add_bias -> add one to each student features
Output: student features
*/
std::vector<std::vector<double>> getData(char const *filename, bool add_bias = false)
{
    std::vector<std::vector<double>> data{};
    std::ifstream DataBaseFile{filename};

    if (!DataBaseFile.is_open()) //check if file opened or not
        throw std::runtime_error("Openning Error");

    else
    {
        double Feature{};
        char Trash_Comma{};
        for (size_t StudentCount{0}; !DataBaseFile.eof(); StudentCount++) //itterate through file
        {
            data.push_back(std::vector<double>());

            if (add_bias) // add one to each student features
                data[StudentCount].push_back(1);

            for (size_t LineCount{0}; LineCount < 6; LineCount++)
            {
                DataBaseFile >> Feature; //fetching data
                Trash_Comma = 'a';
                DataBaseFile >> Trash_Comma; //skipping commas
                data[StudentCount].push_back(Feature);
            }
            DataBaseFile >> Feature;
            data[StudentCount].push_back(Feature);
            if (Trash_Comma == 'a') // pop last vector if the last line of the file is empty
            {
                data.pop_back();
                DataBaseFile.close();
                return data;
            }
        }
    }
    DataBaseFile.close();
    return data;
}

/*
displayDataset: Show fetched data in a table
Input: data-> data fetched from file , has_bias -> whether to show bias or not
Output: void
*/
void displayDataset(std::vector<std::vector<double>> data, bool has_bias = false)
{
    std::cout << std::setiosflags(std::ios::left) << std::setw(17) << "No";
    if (has_bias)
        std::cout << std::setw(17) << "Bias";

    std::cout << std::setw(17) << "Class" << std::setw(17) << "TA" << std::setw(17) << "Coding" << std::setw(17) << "Studying"
              << std::setw(17) << "Background" << std::setw(17) << "Talent" << std::setw(17) << "Passed" << std::endl;

    std::cout << std::setw((7 + has_bias) * 17 + 7) << std::setfill('*') << '*'
              << std::setfill(' ') << std::endl; // fill one line with stars
    for (size_t i{0}; i < data.size(); i++)
    {
        std::cout << std::setiosflags(std::ios::left) << std::setw(17) << i + 1;

        for (size_t j{0}; j < data[i].size(); j++)
            std::cout << std::setw(17) << std::setprecision(2) << data[i][j];

        std::cout << std::setw(0) << std::endl;
    }
}

/*
h: Predict the probabilty of each student passing the course
Input: features -> student features , w -> weight of each feature
Output: Prediction Value
*/
double h(std::vector<double> features, std::vector<double> w)
{
    double z{0};

    for (size_t Counter{0}; Counter < features.size(); Counter++)
        z += features[Counter] * w[Counter]; // multiply each feature by its weight

    return 1 / (1 + exp((-1) * z)); // normalize probability between 0 -> 1
}

/*
j: average of loss functions of a bunch of students
Input: data -> data fetched from file , indices -> idicates which students will be involved in average,
w -> weight of each feature

Output: average of loss values
*/
double J(std::vector<std::vector<double>> data, std::vector<size_t> indices, std::vector<double> w)
{
    double j_value{}, y{};
    for (auto Indice_Holder : indices)
    {
        y = data[Indice_Holder].back(); // grab the passing state
        data[Indice_Holder].pop_back(); // pop the passing state because h() function just gets features not the passing state

        //calculate loss for each data
        j_value += ((-1) * y * log(h(data[Indice_Holder], w))) + ((-1) * (1 - y) * log(1 - h(data[Indice_Holder], w)));
    }
    return j_value / (indices.size()); //average
}

/*
fitOneEpoch: Finding Appropriate Weights
Input: data -> data fetched from file , W0 -> initialize weights, lr -> learning rate, 
batch_size -> number of students to perform each step on

Output: Appropriate weights that minimize the cost function
*/
std::vector<double> fitOneEpoch(std::vector<std::vector<double>> data, std::vector<double> w0, double lr = 0.01, size_t batch_size = 8)
{
    size_t last_number{};
    double calc{0};
    std::vector<double> w;

    for (size_t StCount{0}; StCount < data.size(); StCount += batch_size)
    {
        std::vector<double> y{};
        for (size_t i{0}; i < batch_size; i++)
        {
            if (i + StCount < data.size())
            {
                y.push_back(data[i + StCount].back()); // grab the passing state
                // pop the passing state because h() function just gets features not the passing state
                data[i + StCount].pop_back();
            }
        }
        w = w0; //saving the last weight till one batch is complete
        for (size_t j{0}; j < w0.size(); j++)
        {
            calc = 0;
            for (size_t i{0}; (i < batch_size) && (i + StCount < data.size()); i++)
            {
                last_number = i; // saving the size of each batch (some batches may be shorter)
                calc += (y[i] - h(data[i + StCount], w)) * data[i + StCount][j];
            }
            w0[j] += (lr / (last_number + 1)) * calc; // new (updated) weights
        }
    }

    return w0;
}

/*
fit: perform fitOneEpoch several times
Input: data -> data fetched from file , W0 -> initialize weights, lr -> learning rate, 
batch_size -> number of students to perform each step on, verbose -> whether to show loss values or not

Output: Appropriate weights that minimize the cost function
*/
std::vector<double> fit(std::vector<std::vector<double>> data, std::vector<double> w0, double lr = 0.01, size_t epochs = 10, size_t batch_size = 8, bool verbose = false)
{
    std::vector<size_t> indices{};
    for (size_t i = 0; i < data.size(); i++) // perform cost function on all students -> set indices to everyone
        indices.push_back(i);

    w0 = fitOneEpoch(data, w0, lr, batch_size);
    //forcefully print first epoch
    std::cout << "Epoch      1: J = " << std::setiosflags(std::ios::left)
              << std::setprecision(2) << J(data, indices, w0) << std::endl;

    for (size_t i{1}; i < epochs - 1; i++) // perform fitOneEpoch several times
    {
        w0 = fitOneEpoch(data, w0, lr, batch_size);
        //optionally print epochs
        if (verbose)
            std::cout << "Epoch " << std::setiosflags(std::ios::right) << std::setw(6) << i << ": J = "
                      << std::setiosflags(std::ios::left) << std::setprecision(2) << J(data, indices, w0)
                      << std::endl;
    }

    w0 = fitOneEpoch(data, w0, lr, batch_size);
    //forcefully print last epoch
    std::cout << "Epoch " << std::setiosflags(std::ios::right) << std::setw(6)
              << epochs << ": J = " << std::setiosflags(std::ios::left) << std::setprecision(2)
              << J(data, indices, w0) << std::endl;
    return w0;
}

/*
predict: Predict the probabilty of each student passing the course
Input: data -> data fetched from file, w -> minimized weight of each feature, verbose -> whether to show results or not
Output: Prediction Value
*/
std::vector<double> predict(std::vector<std::vector<double>> data, std::vector<double> w, bool verbose = false)
{
    double Pr_Result{}, Real_res{};
    std::vector<double> Result_Predictions{};

    if (verbose)
    {
        std::cout << std::endl
                  << std::left << std::setw(7) << "No" << std::setw(11) << "Passed" << std::setw(15) << "Prediction" << std::endl;
        std::cout << std::setw(29) << std::setfill('*') << '*' << std::endl;
        std::cout << std::setfill(' ');
    }
    for (size_t i = 0; i < data.size(); i++)
    {
        Real_res = data[i].back(); // grab the passing state
        data[i].pop_back();        // pop the passing state because h() function just gets features not the passing state
        Pr_Result = h(data[i], w); // predict function

        if (verbose) //whether to show results or not
            std::cout << std::left << std::setw(7) << i + 1 << std::setw(11)
                      << Real_res << std::setprecision(3) << std::setw(9) << Pr_Result * 100 << '%' << std::endl;

        Result_Predictions.push_back(Pr_Result);
    }
    return Result_Predictions;
}