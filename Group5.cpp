#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <algorithm>
#include <random>
#include <numeric>
#include <cstdint>

using namespace std;

#define IMAGE_SIZE 28
#define KERNEL_SIZE 3
#define STRIDE 1
#define POOL_SIZE 2
#define NUM_CLASSES 10
#define INPUT_SIZE 784 // for 28x28 mnist images when linear input
#define EPOCHS 20
#define LEARNING_RATE 0.001

double lambda = 0.01;

// Convolution Layer
vector<vector<double>> convlayer(vector<vector<double>> &input, vector<vector<double>> &kernel, int stride)
{
    int n = input.size();
    int k = kernel.size();
    int output_size = (n - k) / stride + 1;

    vector<vector<double>> output(output_size, vector<double>(output_size, 0.0));

    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < output_size; ++j)
        {
            double sum = 0.0;
            for (int m = 0; m < k; ++m)
            {
                for (int n = 0; n < k; ++n)
                {
                    sum += input[i * stride + m][j * stride + n] * kernel[m][n];
                }
            }
            output[i][j] = sum;
        }
    }
    return output;
}

// Flatten Layer
vector<double> Flatten(vector<vector<double>> &input)
{
    vector<double> output;
    for (auto &row : input)
    {
        for (auto &elem : row)
        {
            output.push_back(elem);
        }
    }
    return output;
}

// Fully Connected Layer
vector<double> Dense(vector<double> &input, vector<vector<double>> &weights, vector<double> &bias)
{
    int size_w = weights.size();
    vector<double> output(size_w, 0.0);

    for (int i = 0; i < size_w; ++i)
    {
        for (int j = 0; j < input.size(); ++j)
        {
            output[i] += input[j] * weights[i][j];
        }
        output[i] += bias[i]; 
    }
    return output;
}

// Max Pooling Layer
vector<vector<double>> MaxPool(vector<vector<double>> &input, int pool_size)
{
    int n = input.size();
    int output_size = n / pool_size;

    vector<vector<double>> output(output_size, vector<double>(output_size, 0.0));

    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < output_size; ++j)
        {
            double max_val = -1e9;
            for (int m = 0; m < pool_size; ++m)
            {
                for (int n = 0; n < pool_size; ++n)
                {
                    max_val = max(max_val, input[i * pool_size + m][j * pool_size + n]);
                }
            }
            output[i][j] = max_val;
        }
    }
    return output;
}

// ReLU Activation Function
vector<vector<double>> ReLU(vector<vector<double>> &input)
{
    int n = input.size();
    vector<vector<double>> output(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            output[i][j] = max(0.0, input[i][j]);
        }
    }
    return output;
}

// Softmax Activation Function
vector<double> softmax(vector<double> &input)
{
    double sum = 0.0;
    vector<double> output(input.size(), 0.0);

    for (double val : input)
    {
        sum += exp(val);
    }

    for (int i = 0; i < input.size(); ++i)
    {
        output[i] = exp(input[i]) / sum;
    }
    return output;
}

// cross entropy loss
double crossEntropyLoss(vector<double> &predicted, vector<double> &actual)
{
    double loss = 0.0;
    for (int i = 0; i < predicted.size(); ++i)
    {
        loss += actual[i] * log(predicted[i] + 1e-8);
    }
    return -loss;
}

// softmax gradient
vector<double> SoftmaxGradient_compute(vector<double> &predicted, vector<double> &actual)
{
    vector<double> dLoss(predicted.size(), 0.0);
    for (int i = 0; i < predicted.size(); ++i)
    {
        dLoss[i] = predicted[i] - actual[i];
    }
    return dLoss;
}

// fully connected layer gradient
void Dense_backward(vector<double> &denseLoss, vector<double> &input,
                   vector<vector<double>> &weights, vector<double> &bias,
                   vector<double> &denseInput, double learningRate)
{
    int numNeurons = weights.size();
    int inputSize = weights[0].size();

    denseInput.assign(inputSize, 0.0);

    for (int i = 0; i < numNeurons; ++i)
    {
        for (int j = 0; j < inputSize; ++j)
        {
            denseInput[j] += denseLoss[i] * weights[i][j];               
            weights[i][j] -= learningRate * (denseLoss[i] * input[j] + lambda * weights[i][j]); 
        }
        bias[i] -= learningRate * denseLoss[i]; 
    }
}

// backpropagation thru max pooling layer
vector<vector<double>> MaxPool_backward(vector<vector<double>> &input, vector<vector<double>> &dOutput)
{
    int n = input.size();
    int pool_size = POOL_SIZE;
    vector<vector<double>> dInput(n, vector<double>(n, 0.0));

    for (int i = 0; i < n / pool_size; ++i)
    {
        for (int j = 0; j < n / pool_size; ++j)
        {
            double maxVal = -1e9;
            int maxX = -1, maxY = -1;

            for (int m = 0; m < pool_size; ++m)
            {
                for (int n = 0; n < pool_size; ++n)
                {
                    int x = i * pool_size + m;
                    int y = j * pool_size + n;
                    if (input[x][y] > maxVal)
                    {
                        maxVal = input[x][y];
                        maxX = x;
                        maxY = y;
                    }
                }
            }

            dInput[maxX][maxY] = dOutput[i][j];
        }
    }
    return dInput;
}

// backpropagation thru relu layer
vector<vector<double>> ReLU_backward(vector<vector<double>> &input, vector<vector<double>> &dOutput)
{
    int n = input.size();
    vector<vector<double>> dInput(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            dInput[i][j] = (input[i][j] > 0) ? dOutput[i][j] : 0;
        }
    }
    return dInput;
}

// backprop thru conv layer
void Conv_backward(vector<vector<double>> &convInput, vector<vector<double>> &convKernel,
                  vector<vector<double>> &dConv, double learningRate,
                  vector<vector<double>> &dConvInput)
{

    int k = convKernel.size();
    int n = convInput.size();
    vector<vector<double>> dKernel(k, vector<double>(k, 0.0));
    dConvInput.assign(n, vector<double>(n, 0.0));

    for (int i = 0; i < dConv.size(); ++i)
    {
        for (int j = 0; j < dConv[i].size(); ++j)
        {
            for (int m = 0; m < k; ++m)
            {
                for (int n = 0; n < k; ++n)
                {
                    dKernel[m][n] += convInput[i + m][j + n] * dConv[i][j];     
                    dConvInput[i + m][j + n] += convKernel[m][n] * dConv[i][j]; 
                }
            }
        }
    }

    for (int i = 0; i < k; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            convKernel[i][j] -= learningRate * (dKernel[i][j] + lambda * convKernel[i][j]);
        }
    }
}

void Matrix_disp(vector<vector<double>> &matrix)
{
    for (auto &row : matrix)
    {
        for (auto &elem : row)
        {
            cout << elem << " ";
        }
        cout << endl;
    }
}

vector<vector<double>> Matrix_reshape(vector<double> &flattened_image, int size)
{
    vector<vector<double>> output(size, vector<double>(size, 0.0));
    int a = 0;
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            output[i][j] = flattened_image[a++];
        }
    }
    return output;
}


// for accuracy calculation
int Predict(const vector<double> &output)
{
    return max_element(output.begin(), output.end()) - output.begin();
}

int Actual(const vector<double> &label)
{
    return max_element(label.begin(), label.end()) - label.begin();
}

// training
void train(vector<vector<vector<double>>> &trainData, vector<vector<double>> &trainLabels,
           vector<vector<double>> &convKernel, vector<vector<double>> &denseWeights,
           vector<double> &denseBias, double learningRate, int epochs, vector<double> &epochLosses, vector<double> &epochAccuracies)
{
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double totalLoss = 0.0; 
        int correct = 0;
        vector<int> indices(trainData.size());
        iota(indices.begin(), indices.end(), 0);
        random_shuffle(indices.begin(), indices.end());

        for (int idx : indices)
        {
            auto &x = trainData[idx];
            auto &y = trainLabels[idx];

            vector<vector<double>> convOut = convlayer(x, convKernel, STRIDE);
            vector<vector<double>> reluOut = ReLU(convOut);
            vector<vector<double>> pooled = MaxPool(reluOut, POOL_SIZE);
            vector<double> flatOutput = Flatten(pooled);
            vector<double> prediction = Dense(flatOutput, denseWeights, denseBias);
            prediction = softmax(prediction);

            double loss = crossEntropyLoss(prediction, y);
            double l2Loss = 0.0;
            for (const auto &row : denseWeights)
                for (double w : row)
                    l2Loss += w * w;
            for (const auto &row : convKernel)
                for (double w : row)
                    l2Loss += w * w;

            loss += (lambda / 2.0) * l2Loss;
            totalLoss += loss;

            int predLabel = Predict(prediction);
            int trueLabel = Actual(y);
            if (predLabel == trueLabel)
                ++correct;

            // backprop
            vector<double> dLoss = SoftmaxGradient_compute(prediction, y);
            vector<double> dFlatten;
            Dense_backward(dLoss, flatOutput, denseWeights, denseBias, dFlatten, learningRate);
            vector<vector<double>> dPooled = Matrix_reshape(dFlatten, pooled.size());
            vector<vector<double>> dConvOut = MaxPool_backward(reluOut, dPooled);
            vector<vector<double>> dConv = ReLU_backward(convOut, dConvOut);
            vector<vector<double>> dInput;
            Conv_backward(x, convKernel, dConv, learningRate, dInput);
        }

        cout << "Epoch -> " << epoch + 1 << ", Loss: " << totalLoss / trainData.size()
             << ", Accuracy: " << 100.0 * correct / trainData.size() << "%" << endl;

        if(epoch%9==0 and epoch!=0) cout<<"--------------------------------------------------------------------------------------------------------------------------------"<<endl;
        epochLosses.push_back(totalLoss / trainData.size());
        epochAccuracies.push_back(100.0 * correct / trainData.size());

    }
}

// testing
void test(vector<vector<vector<double>>> &testData, vector<vector<double>> &testLabels,
          vector<vector<double>> &convKernel, vector<vector<double>> &denseWeights,
          vector<double> &denseBias)
{
    int correct = 0;
    int total = testData.size();

    for (int i = 0; i < total; ++i)
    {
        vector<vector<double>> convOut = convlayer(testData[i], convKernel, STRIDE);
        convOut = ReLU(convOut);
        vector<vector<double>> pooled = MaxPool(convOut, POOL_SIZE);
        vector<double> flatOutput = Flatten(pooled);
        vector<double> prediction = Dense(flatOutput, denseWeights, denseBias);
        prediction = softmax(prediction);

        int predClass = max_element(prediction.begin(), prediction.end()) - prediction.begin();
        int trueClass = max_element(testLabels[i].begin(), testLabels[i].end()) - testLabels[i].begin();

        if (predClass == trueClass)
        {
            correct++;
        }
    }

    double accuracy = (double)correct / total * 100.0;
    cout << "Test Accuracy: " << accuracy << "%" << endl;
}

vector<vector<vector<double>>> testAndGetFeatureMaps(
    vector<vector<vector<double>>> &testData,
    vector<vector<double>> &testLabels,
    vector<vector<double>> &convKernel,
    vector<vector<double>> &denseWeights,
    vector<double> &denseBias){
    int correct = 0;
    int total = testData.size();
    vector<vector<vector<double>>> featureMaps;

    for (int i = 0; i < total; ++i)
    {
        vector<vector<double>> convOut = convlayer(testData[i], convKernel, STRIDE);
        convOut = ReLU(convOut);
        featureMaps.push_back(convOut);  // collect feature map

        vector<vector<double>> pooled = MaxPool(convOut, POOL_SIZE);
        vector<double> flatOutput = Flatten(pooled);
        vector<double> prediction = Dense(flatOutput, denseWeights, denseBias);
        prediction = softmax(prediction);

        int predClass = max_element(prediction.begin(), prediction.end()) - prediction.begin();
        int trueClass = max_element(testLabels[i].begin(), testLabels[i].end()) - testLabels[i].begin();

        if (predClass == trueClass)
        {
            correct++;
        }
    }

    double accuracy = (double)correct / total * 100.0;
    cout << "Test Accuracy: " << accuracy << "%" << endl;

    return featureMaps;
 }

// load mnist images

vector<vector<vector<double>>> MNISTip_load(string filename, int numImages)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "Unable to open file " << filename << endl;
        exit(1);
    }

    uint32_t magicNumber = 0, numberOfImages = 0, rows = 0, cols = 0;
    file.read((char *)&magicNumber, 4);
    file.read((char *)&numberOfImages, 4);
    file.read((char *)&rows, 4);
    file.read((char *)&cols, 4);

    magicNumber = __builtin_bswap32(magicNumber);
    numberOfImages = __builtin_bswap32(numberOfImages);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    vector<vector<vector<double>>> images;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> flipDist(0, 1);
    uniform_int_distribution<> shiftDist(-2, 2);
    uniform_int_distribution<> cropOffset(0, 2); // for random crop offset (28 - 26 = 2)

    const int cropSize = 26; 

    for (int i = 0; i < numImages; ++i)
    {
        vector<vector<double>> image(rows, vector<double>(cols));
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                unsigned char pixel = 0;
                file.read((char *)&pixel, 1);
                image[r][c] = pixel / 255.0;
            }
        }
        images.push_back(image);
    }

    return images;
}

// load mnist labels
vector<vector<double>> MNISTop_load(string filename, int numLabels)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "Unable to open file " << filename << endl;
        exit(1);
    }

    uint32_t magicNumber = 0, numberOfLabels = 0;
    file.read((char *)&magicNumber, 4);
    file.read((char *)&numberOfLabels, 4);

    magicNumber = __builtin_bswap32(magicNumber);
    numberOfLabels = __builtin_bswap32(numberOfLabels);

    vector<vector<double>> labels;
    for (int i = 0; i < numLabels; ++i)
    {
        unsigned char label = 0;
        file.read((char *)&label, 1);
        // one hot encoding
        vector<double> oneHot(10, 0.0);
        oneHot[label] = 1.0;
        labels.push_back(oneHot);
    }

    return labels;
}

void saveAsPGM(const vector<vector<double>>& image, const string& filename)
{
    int rows = image.size();
    int cols = image[0].size();
    ofstream file(filename, ios::binary);
    file << "P5\n" << cols << " " << rows << "\n255\n";

    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            file.put(static_cast<unsigned char>(image[r][c] * 255));
}

vector<double> flatten_Image(const vector<vector<double>> &image)
{
    vector<double> flat;
    for (const auto &row : image)
    {
        for (double pixel : row)
        {
            flat.push_back(pixel);
        }
    }
    return flat;
}

// flatten all images
vector<vector<double>> flatten_Images(const vector<vector<vector<double>>> &images2D)
{
    vector<vector<double>> flatImages;
    for (const auto &img : images2D)
    {
        flatImages.push_back(flatten_Image(img));
    }
    return flatImages;
}

// initialise weights and biases
void initialize(vector<vector<double>> &weights, vector<double> &bias)
{
    srand(time(0));

    weights.resize(NUM_CLASSES, vector<double>(INPUT_SIZE, 0.0));
    bias.resize(NUM_CLASSES, 0.0);

    for (int i = 0; i < NUM_CLASSES; ++i)
    {
        for (int j = 0; j < INPUT_SIZE; ++j)
        {
            weights[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.01;
        }
    }
}

vector<double> linReg(const vector<double> &x, const vector<vector<double>> &weights, const vector<double> &bias)
{
    vector<double> output(NUM_CLASSES, 0.0);
    for (int i = 0; i < NUM_CLASSES; ++i)
    {
        for (int j = 0; j < INPUT_SIZE; ++j)
        {
            output[i] += weights[i][j] * x[j];
        }
        output[i] += bias[i];
    }
    return softmax(output);
}

double MSE(const vector<double> &predicted, const vector<double> &actual)
{
    double sum = 0.0;
    for (int i = 0; i < predicted.size(); ++i)
    {
        double diff = predicted[i] - actual[i];
        sum += diff * diff;
    }
    return sum / predicted.size();
}

void train_LR(const vector<vector<double>> &trainImages,
                    const vector<vector<double>> &trainLabels,
                    vector<vector<double>> &weights,
                    vector<double> &bias,
                    double learningRate, int epochs, vector<double> &epochLosses, vector<double> &epochAccuracies)
{
    int numSamples = trainImages.size();

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double totalLoss = 0.0;
        int correct = 0;

        for (int i = 0; i < numSamples; ++i)
        {
            const vector<double> &x = trainImages[i];
            const vector<double> &y_true = trainLabels[i];

            // Forward pass
            vector<double> y_pred(NUM_CLASSES, 0.0);
            for (int j = 0; j < NUM_CLASSES; ++j)
            {
                for (int k = 0; k < INPUT_SIZE; ++k)
                {
                    y_pred[j] += weights[j][k] * x[k];
                }
                y_pred[j] += bias[j];
            }

            totalLoss += MSE(y_pred, y_true);

            // Check accuracy
            int predLabel = distance(y_pred.begin(), max_element(y_pred.begin(), y_pred.end()));
            int trueLabel = distance(y_true.begin(), max_element(y_true.begin(), y_true.end()));
            if (predLabel == trueLabel)
                correct++;

            // Backpropagation
            for (int j = 0; j < NUM_CLASSES; ++j)
            {
                double error = y_pred[j] - y_true[j];
                for (int k = 0; k < INPUT_SIZE; ++k)
                {
                    weights[j][k] -= learningRate * error * x[k];
                }
                bias[j] -= learningRate * error;
            }
        }

        double avgLoss = totalLoss / numSamples;
        double accuracy = static_cast<double>(correct) / numSamples * 100.0;

        cout << "Epoch " << epoch + 1 << ": Loss = " << avgLoss << ", Accuracy = " << accuracy << "%" << endl;

        epochLosses.push_back(avgLoss);
        epochAccuracies.push_back(accuracy);
    }
}

void eval_LR(const vector<vector<double>> &testImages,
                       const vector<vector<double>> &testLabels,
                       const vector<vector<double>> &weights,
                       const vector<double> &bias)
{
    int correct = 0;
    int total = testImages.size();

    for (int i = 0; i < total; ++i)
    {
        const vector<double> &x = testImages[i];
        const vector<double> &y_true = testLabels[i];

        vector<double> y_pred(NUM_CLASSES, 0.0);
        for (int j = 0; j < NUM_CLASSES; ++j)
        {
            for (int k = 0; k < INPUT_SIZE; ++k)
            {
                y_pred[j] += weights[j][k] * x[k];
            }
            y_pred[j] += bias[j];
        }

        int predLabel = distance(y_pred.begin(), max_element(y_pred.begin(), y_pred.end()));
        int trueLabel = distance(y_true.begin(), max_element(y_true.begin(), y_true.end()));

        if (predLabel == trueLabel)
            correct++;
    }

    double accuracy = static_cast<double>(correct) / total * 100.0;
    cout << "Test Accuracy: " << accuracy << "%" << endl;
}

int main(int argc, char* argv[])
{
    int numTrainSamples = 15000;
    int numTestSamples = 4000;
    string trainImageFile = "train-images.idx3-ubyte";
    string trainLabelFile = "train-labels.idx1-ubyte";
    string testImageFile = "t10k-images.idx3-ubyte";
    string testLabelFile = "t10k-labels.idx1-ubyte";

    // Load training and test data using your custom loader
    auto trainData = MNISTip_load(trainImageFile, numTrainSamples);
    auto trainLabels = MNISTop_load(trainLabelFile, numTrainSamples);
    auto testData = MNISTip_load(testImageFile, numTestSamples);
    auto testLabels = MNISTop_load(testLabelFile, numTestSamples);

    saveAsPGM(trainData[0], "train_0.pgm");

    // Initialize weights
    vector<vector<double>> convKernel(KERNEL_SIZE, vector<double>(KERNEL_SIZE));
    auto K1_SIZE = KERNEL_SIZE;
    auto K2_SIZE = KERNEL_SIZE;
    vector<vector<double>> convKernel1(K1_SIZE, vector<double>(K1_SIZE));
    vector<vector<double>> convKernel2(K2_SIZE, vector<double>(K2_SIZE));

    vector<vector<double>> denseWeights(NUM_CLASSES, vector<double>((IMAGE_SIZE - KERNEL_SIZE + 1) / POOL_SIZE * (IMAGE_SIZE - KERNEL_SIZE + 1) / POOL_SIZE));
    vector<double> denseBias(NUM_CLASSES, 0.0);

    srand(time(0));

    int kernelSize = convKernel.size(); // assuming square kernel
    double fan_in = kernelSize * kernelSize;
    double stddev = sqrt(2.0 / fan_in);

    for (auto &row : convKernel)
        for (auto &val : row)
            val = ((double)rand() / RAND_MAX) * 2 * stddev - stddev;

    // He init - convKernel1
    int kernelSize1 = convKernel1.size(); // assuming square
    double fan_in1 = kernelSize1 * kernelSize1;
    double stddev1 = sqrt(2.0 / fan_in1);

    for (auto &row : convKernel1)
        for (auto &val : row)
            val = ((double)rand() / RAND_MAX) * 2 * stddev1 - stddev1;

    // He init - convKernel2
    int kernelSize2 = convKernel2.size();
    double fan_in2 = kernelSize2 * kernelSize2;
    double stddev2 = sqrt(2.0 / fan_in2);

    for (auto &row : convKernel2)
        for (auto &val : row)
            val = ((double)rand() / RAND_MAX) * 2 * stddev2 - stddev2;

    // He init - denseWeights
    int denseFanIn = denseWeights[0].size(); // number of inputs per neuron
    double denseStddev = sqrt(2.0 / denseFanIn);

    for (auto &row : denseWeights)
        for (auto &val : row)
            val = ((double)rand() / RAND_MAX) * 2 * denseStddev - denseStddev;

    // Bias
    for (auto &val : denseBias)
        val = 0.0;


    // Train and test
    vector<double> epochLosses_cnn;
    vector<double> epochAccuracies_cnn;
    auto featureMaps = vector<vector<vector<double>>>(3, vector<vector<double>>(IMAGE_SIZE, vector<double>(IMAGE_SIZE, 0.0)));
    cout << "Training CNN..." << endl;
    train(trainData, trainLabels, convKernel, denseWeights, denseBias, LEARNING_RATE, EPOCHS, epochLosses_cnn, epochAccuracies_cnn);
    featureMaps = testAndGetFeatureMaps(testData, testLabels, convKernel, denseWeights, denseBias);
    
    test(testData, testLabels, convKernel, denseWeights, denseBias);
    
    saveAsPGM(featureMaps[0], "feature_map_0.pgm"); 
    saveAsPGM(featureMaps[1], "feature_map_1.pgm");
    saveAsPGM(featureMaps[2], "feature_map_2.pgm");
    saveAsPGM(testData[0], "test_0.pgm");
    saveAsPGM(testData[1], "test_1.pgm");
    saveAsPGM(testData[2], "test_2.pgm");

    for (int i = 0; i < 20; ++i)
    {
        string filename = "image" + to_string(i) + ".pgm";
        saveAsPGM(trainData[i], filename);
    }

    vector<vector<double>> weights;
    vector<double> bias;
    initialize(weights, bias);

    vector<vector<double>> trainDataFlattened = flatten_Images(trainData);
    vector<vector<double>> testDataFlattened = flatten_Images(testData);

    cout << "Training Linear Regression..." << endl;

    vector<double> epochLosses_linear;
    vector<double> epochAccuracies_linear;

    train_LR(trainDataFlattened, trainLabels, weights, bias, LEARNING_RATE, EPOCHS, epochLosses_linear, epochAccuracies_linear);
    eval_LR(testDataFlattened, testLabels, weights, bias);

    ofstream lossFile("epochLosses.txt");
    if (!lossFile.is_open()) {
        cerr << "Error: Could not open epochLosses.txt" << endl;
        return 1;
    }
    ofstream accFile("epochAccuracies.txt");
    if (!accFile.is_open()) {
        cerr << "Error: Could not open epochAccuracies.txt" << endl;
        return 1;
    }
    for (const auto &loss : epochLosses_cnn) {
        lossFile << loss << endl;
    }
    for (const auto &acc : epochAccuracies_cnn) {
        accFile << acc << endl;
    }
    lossFile.close();
    accFile.close();

    ofstream lossFile_linear("epochLosses_linear.txt");
    if (!lossFile_linear.is_open()) {
        cerr << "Error: Could not open epochLosses_linear.txt" << endl;
        return 1;
    }
    ofstream accFile_linear("epochAccuracies_linear.txt");
    if (!accFile_linear.is_open()) {
        cerr << "Error: Could not open epochAccuracies_linear.txt" << endl;
        return 1;
    }
    for (const auto &loss : epochLosses_linear) {
        lossFile_linear << loss << endl;
    }
    for (const auto &acc : epochAccuracies_linear) {
        accFile_linear << acc << endl;
    }
    lossFile_linear.close();
    accFile_linear.close();

    return 0;

}
