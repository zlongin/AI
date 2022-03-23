clear all;
filename = "newsdataset.csv";
data = readtable("newsdataset.csv", 'TextType','string');
data.Category = categorical(data.Category);
cvp = cvpartition(data.Category,'Holdout',0.2);
dataTrain = data(training(cvp),:);
dataTest = data(test(cvp),:);

textDataTrain = dataTrain.Text;
textDataTest = dataTest.Text;

YTrain = dataTrain.Category;
YTest = dataTest.Category;

documentsTrain = preprocessText(textDataTrain);
documentsTest = preprocessText(textDataTest);

enc = wordEncoding(documentsTrain);
documentLengths = doclength(documentsTrain);


sequenceLength = 800;
XTrain = doc2sequence(enc,documentsTrain,'Length',sequenceLength);
XTest = doc2sequence(enc,documentsTest,'Length',sequenceLength);

inputSize = 1;
embeddingDimension = 50;
numHiddenUnits = 80;

numWords = enc.NumWords;
numClasses = numel(categories(YTrain));

layers = [ ...
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(embeddingDimension,numWords)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MiniBatchSize',16, ...
    'GradientThreshold',2, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XTest,YTest}, ...
    'Plots','training-progress', ...
    'Verbose',false);

%net = trainNetwork(XTrain,YTrain,layers,options);

nbtable = removevars(dataTrain,"ArticleId");
nbtable = removevars(nbtable,"Category");
treetraininput = str2double(textDataTrain);
tree = fitctree(textDataTrain,YTrain);

nbtesttable = removevars(dataTest,"ArticleId");
nbtesttable = removevars(nbtesttable,"Category");
NBTRAININGOUTPUT = predict(tree,nbtesttable);
% 
% bad = ~strcmp(NBTRAININGOUTPUT,YTest);
% error_training = sum(bad) / length(XTest);
% disp('Testing error is '), disp(error_training)


